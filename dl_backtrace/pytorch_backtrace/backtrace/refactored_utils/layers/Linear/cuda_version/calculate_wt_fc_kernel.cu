#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void calculate_wt_fc_fused_kernel(
    const float* row_specific_weights_ptr,
    const float* input_activations_ptr,    
    const float* weights_matrix_ptr,
    const float* bias_vector_ptr,
    float* final_output_ptr,
    const int D_in_kernel,
    const int D_out_kernel,
    const bool has_lower_bound,
    const float lower_threshold,
    const bool has_upper_bound, 
    const float upper_threshold,
    const bool is_non_mono,
    const int activation_func
) {

    int row = blockIdx.x;
    if (row >= D_in_kernel) return;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    
    // Shared memory for reductions
    extern __shared__ float sdata[];
    float* s_positive = sdata;
    float* s_negative = &sdata[block_size];
    
    const float ZERO = 0.0f;
    const float ONE = 1.0f;
    const float MINUS_ONE = -1.0f;
    
    // Get row-specific values
    float row_weight = row_specific_weights_ptr[row];
    float bias = bias_vector_ptr[row];
    
    // Phase 1: Compute positive and negative sums for this row
    float local_p_sum = ZERO;
    float local_n_sum = ZERO;
    
    // Each thread processes multiple columns
    for (int col = tid; col < D_out_kernel; col += block_size) {
        float scaled_weight = weights_matrix_ptr[row * D_out_kernel + col] * input_activations_ptr[row];
        
        if (scaled_weight > ZERO) {
            local_p_sum += scaled_weight;
        } else if (scaled_weight < ZERO) {
            local_n_sum += (-scaled_weight);  // Store absolute value
        }
    }
    
    // Store in shared memory
    s_positive[tid] = local_p_sum;
    s_negative[tid] = local_n_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_positive[tid] += s_positive[tid + stride];
            s_negative[tid] += s_negative[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 computes aggregate weights for this row
    float p_agg_wt = ZERO;
    float n_agg_wt = ZERO;
    float p_sum_div = ONE;
    float n_sum_div = ONE;
    
    if (tid == 0) {
        float p_sum = s_positive[0];
        float n_sum = s_negative[0];
        
        // Bias handling
        float pbias = fmaxf(ZERO, bias);
        float nbias = fmaxf(ZERO, -bias);
        
        // Total sum for activation checks
        float t_sum = p_sum + pbias - (n_sum + nbias);
        
        // Store original values for non-mono activation
        float p_sum_for_act = p_sum;
        float n_sum_for_act = n_sum;
        
        // Apply range bounds
        if (has_lower_bound && t_sum < lower_threshold) {
            p_sum = ZERO;
        }
        if (has_upper_bound && t_sum > upper_threshold) {
            n_sum = ZERO;
        }
        
        // Non-monotonic activation logic
        if (is_non_mono && p_sum > ZERO && n_sum > ZERO) {
            float t_act, p_act, n_act;
            
            switch (activation_func) {
                case 1: // ReLU
                    t_act = fmaxf(ZERO, t_sum);
                    p_act = fmaxf(ZERO, p_sum_for_act + pbias);
                    n_act = fmaxf(ZERO, MINUS_ONE * (n_sum_for_act + nbias));
                    break;
                case 2: // Sigmoid
                    t_act = ONE / (ONE + expf(-t_sum));
                    p_act = ONE / (ONE + expf(-(p_sum_for_act + pbias)));
                    n_act = ONE / (ONE + expf(-MINUS_ONE * (n_sum_for_act + nbias)));
                    break;
                default: // Identity
                    t_act = t_sum;
                    p_act = p_sum_for_act + pbias;
                    n_act = MINUS_ONE * (n_sum_for_act + nbias);
                    break;
            }
            
            // Use exact equality to match PyTorch behavior
            if (t_act == p_act) {
                n_sum = ZERO;
            } else if (t_act == n_act) {
                p_sum = ZERO;
            }
        }
        
        // Calculate aggregate weights
        float den1_common = p_sum + n_sum + pbias + nbias;
        
        if (p_sum > ZERO) {
            float ratio1_p = (p_sum + pbias) / den1_common;
            float ratio2_p_denom = p_sum + pbias;
            // Match PyTorch safe division logic: if denom is 0, use p_sum/1.0 = p_sum
            float ratio2_p = (ratio2_p_denom == ZERO) ? p_sum : p_sum / ratio2_p_denom;
            p_agg_wt = ratio1_p * ratio2_p;
        }
        
        if (n_sum > ZERO) {
            float ratio1_n = (n_sum + nbias) / den1_common;
            float ratio2_n_denom = n_sum + nbias;
            // Match PyTorch safe division logic
            float ratio2_n = (ratio2_n_denom == ZERO) ? n_sum : n_sum / ratio2_n_denom;
            n_agg_wt = ratio1_n * ratio2_n;
        }
        
        // Safe division denominators (match PyTorch logic)
        p_sum_div = (p_sum == ZERO) ? ONE : p_sum;
        n_sum_div = (n_sum == ZERO) ? ONE : n_sum;
        
        // Store computed values in shared memory
        s_positive[0] = p_agg_wt;
        s_positive[1] = n_agg_wt;
        s_positive[2] = p_sum_div;
        s_positive[3] = n_sum_div;
    }
    __syncthreads();
    
    // Phase 2: All threads compute contributions for their columns
    p_agg_wt = s_positive[0];
    n_agg_wt = s_positive[1];
    p_sum_div = s_positive[2];
    n_sum_div = s_positive[3];
    
    // Compute and accumulate contributions
    for (int col = tid; col < D_out_kernel; col += block_size) {
        float scaled_weight = weights_matrix_ptr[row * D_out_kernel + col] * input_activations_ptr[row];
        float contribution = ZERO;
        
        if (scaled_weight > ZERO) {
            float term_p_norm = scaled_weight / p_sum_div;
            contribution = term_p_norm * row_weight * p_agg_wt;
        } else if (scaled_weight < ZERO) {
            float term_n_norm = scaled_weight / n_sum_div;
            contribution = term_n_norm * row_weight * n_agg_wt * MINUS_ONE;
        }
        
        atomicAdd(&final_output_ptr[col], contribution);
    }
}

torch::Tensor calculate_wt_fc_cuda(
    const torch::Tensor& row_specific_weights,
    const torch::Tensor& input_activations,
    const torch::Tensor& weights_matrix,
    const torch::Tensor& bias_vector,
    const bool has_lower_bound,
    const float lower_threshold,
    const bool has_upper_bound,
    const float upper_threshold,
    const bool is_non_mono,
    const int activation_func
) {
    // Input validation
    TORCH_CHECK(row_specific_weights.dim() == 1, "row_specific_weights must be 1D");
    TORCH_CHECK(input_activations.dim() == 1, "input_activations must be 1D");
    TORCH_CHECK(weights_matrix.dim() == 2, "weights_matrix must be 2D");
    TORCH_CHECK(bias_vector.dim() == 1, "bias_vector must be 1D");
    
    int D_in_actual = weights_matrix.size(0);  // Correct: number of input features
    int D_out_actual = weights_matrix.size(1); // Correct: number of output features
    
    // TORCH_CHECK(row_specific_weights.size(0) == D_in_actual, "row_specific_weights size mismatch");
    // TORCH_CHECK(input_activations.size(0) == D_in_actual, "input_activations size mismatch");
    // TORCH_CHECK(bias_vector.size(0) == D_in_actual, "bias_vector size mismatch");
 
    // Create output tensor
    at::Tensor result = torch::zeros({D_out_actual}, weights_matrix.options());
    
    // Launch kernel
    const int BLOCK_SIZE_X = 16;
    const int BLOCK_SIZE_Y = 16;
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_dim(D_in_actual, 1);
    
    size_t shared_mem_size = 2 * BLOCK_SIZE_X * BLOCK_SIZE_Y * sizeof(float);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    calculate_wt_fc_fused_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        row_specific_weights.data_ptr<float>(),
        input_activations.data_ptr<float>(),
        weights_matrix.data_ptr<float>(),
        bias_vector.data_ptr<float>(),
        result.data_ptr<float>(),
        D_in_actual, D_out_actual,
        has_lower_bound, lower_threshold,
        has_upper_bound, upper_threshold,
        is_non_mono, activation_func,
        stream
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return result;
}
