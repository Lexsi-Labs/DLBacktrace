#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>

__device__ __forceinline__ float apply_activation(float x, int act_func) {
    switch (act_func) {
        case 1:  // Sigmoid
            return __fdividef(1.0f, 1.0f + __expf(-x));
        case 2:  // Swish
            return x * __fdividef(1.0f, 1.0f + __expf(-(0.75f * x)));
        case 3:  // Wave
            return __fdividef(x * __expf(1.0f), __expf(-x) + __expf(x));
        case 4:  // Pulse
            return 1.0f - __fmul_rn(tanhf(x), tanhf(x));
        case 5:  // Absolute
            return x * tanhf(x);
        case 6:  // Hard Sigmoid
            return fmaxf(fminf(__fmaf_rn(0.2f, x, 0.5f), 1.0f), 0.0f);
        case 7:  // Tanh
            return tanhf(x);
        default:  // Identity (0) or unsupported
            return x;
    }
}

__global__ void calculate_wt_fc_kernel(
    const float* __restrict__ relevance_y,
    const float* __restrict__ input_array,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ relevance_x,
    const int input_dim,
    const int output_dim,
    const int activation_kind, 
    const float act_lower_bound,
    const float act_upper_bound,
    const bool has_bias,
    const int act_func  
) {
    // Each block handles one output dimension for a batch element
    const int output_idx = blockIdx.x;
    
    if (output_idx >= output_dim) return;

    // Shared memory for reductions
    extern __shared__ float shared_mem[];
    float* contributions = shared_mem;
    __shared__ float p_sum_shared[256];
    __shared__ float n_sum_shared[256];
    contributions[threadIdx.x] = 0.0f;

    // Initialize thread-local accumulators
    float local_p_sum = 0.0f;
    float local_n_sum = 0.0f;

    int mul_val = output_idx * input_dim;

    for (int i = threadIdx.x; i < input_dim; i += blockDim.x) {
        contributions[i] = input_array[i] * w[mul_val + i];
        local_p_sum += fmaxf(contributions[i], 0.0f);
        local_n_sum += fmaxf(-contributions[i], 0.0f);
    }

    __syncthreads();

    p_sum_shared[threadIdx.x] = local_p_sum;
    n_sum_shared[threadIdx.x] = local_n_sum;
    __syncthreads();

    // Parallel reduction for positive sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            p_sum_shared[threadIdx.x] += p_sum_shared[threadIdx.x + stride];
            n_sum_shared[threadIdx.x] += n_sum_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float p_sum = p_sum_shared[0];
    float n_sum = n_sum_shared[0];
    
    // Handle bias
    float pbias = 0.0f, nbias = 0.0f;
    if (has_bias && b) {
        float bias_val = b[output_idx];
        pbias = fmaxf(bias_val, 0.0f);
        nbias = -fminf(bias_val, 0.0f);
    }
    
    // Compute total sum
    float t_sum = p_sum + pbias - n_sum - nbias;
    
    if (t_sum < act_lower_bound) p_sum = 0.0f;
    if (t_sum > act_upper_bound) n_sum = 0.0f;

    // Activation-specific handling
    if (activation_kind == 1) { // Non-monotonic activation
        
        // Apply activation functions
        float t_act = apply_activation(t_sum, act_func);
        float p_act = apply_activation(p_sum + pbias, act_func);
        float n_act = apply_activation(-(n_sum + nbias), act_func);
        
        // Check if both positive and negative contributions exist
        if (p_sum > 0.0f && n_sum > 0.0f) {
            if (t_act == p_act) {
                n_sum = 0.0f;  // Total matches positive part
            } else if (t_act == n_act) {
                p_sum = 0.0f;  // Total matches negative part
            }
        }
    }
    
    // Stabilize sums to avoid division by zero
    if (p_sum == 0.0f) p_sum = 1.0f;
    if (n_sum == 0.0f) n_sum = 1.0f;
    
    // Compute total sum with stabilization
    float total_sum = p_sum + n_sum + pbias + nbias;
    float total_psum = p_sum + pbias;
    float total_nsum = n_sum + nbias;

    // Optimized division
    float total_sum_inv = __fdividef(1.0f, total_sum);
    float total_psum_inv = __fdividef(1.0f, total_psum);
    float total_nsum_inv = __fdividef(1.0f, total_nsum);
    
    // Compute aggregated weights with stabilization
    float p_agg_wt = (p_sum > 0.0f) ? 
        (total_psum * total_sum_inv) * (p_sum * total_psum_inv) : 
        0.0f;
    
    float n_agg_wt = (n_sum > 0.0f) ? 
        (total_nsum * total_sum_inv) * (n_sum * total_nsum_inv) : 
        0.0f;
    
    // Get relevance value for this output
    float relevance_val = relevance_y[output_idx];
    float p_agg_wt_val = __fmul_rn(p_agg_wt, relevance_val);
    float n_agg_wt_val = __fmul_rn(n_agg_wt, relevance_val);
    
    // Compute and accumulate final weights
    for (int i = threadIdx.x; i < input_dim; i += blockDim.x) {
        float contrib = contributions[i];
        float weight = 0.0f;

        if (contrib > 0.0f) {
            weight = (contrib / p_sum) * p_agg_wt_val;
        } else if (contrib < 0.0f) {
            weight = (contrib / n_sum) * n_agg_wt_val * -1.0f;
        }
        
        // Atomic add to avoid write conflicts
        atomicAdd(&relevance_x[i], weight);
    }
}

extern "C" void launch_calculate_wt_fc_kernel(
    const float* relevance_y,
    const float* input_array,
    const float* w,
    const float* b,
    float* relevance_x,
    const int input_dim,
    const int output_dim,
    const int activation_kind,
    const float act_lower_bound,
    const float act_upper_bound,
    const bool has_bias,
    const int act_func
) {
    // Initialize output to zero
    cudaMemset(relevance_x, 0, input_dim * sizeof(float));
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    dim3 grid_dim(output_dim);  // x: output
    size_t shared_mem_size =  input_dim * sizeof(float);
    
    // Launch kernel
    calculate_wt_fc_kernel<<<grid_dim, threads_per_block, shared_mem_size>>>(
        relevance_y,
        input_array,
        w,
        b,
        relevance_x,
        input_dim,
        output_dim,
        activation_kind,
        act_lower_bound,
        act_upper_bound,
        has_bias,
        act_func
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}
