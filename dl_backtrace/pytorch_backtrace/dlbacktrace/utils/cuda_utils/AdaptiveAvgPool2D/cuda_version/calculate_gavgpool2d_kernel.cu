#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void sum_kernel(
    const float* __restrict__ inp_t_data,
    float* __restrict__ p_sums_data,
    float* __restrict__ n_sums_data,
    int spatial_size,
    int channels
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    float p_sum = 0.0f;
    float n_sum = 0.0f;

    for (int s = 0; s < spatial_size; ++s) {
        float val = inp_t_data[s * channels + c];
        if (val > 0) {
            p_sum += val;
        } else {
            n_sum += -val;
        }
    }
    p_sums_data[c] = p_sum;
    n_sums_data[c] = n_sum;
}

__global__ void apply_weights_kernel(
    const float* __restrict__ inp_t_data,
    const float* __restrict__ wts_data,
    const float* __restrict__ p_sums_data,
    const float* __restrict__ n_sums_data,
    float* __restrict__ output_data,
    int spatial_size,
    int channels
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;

    if (c >= channels || s >= spatial_size) return;

    float p_sum = p_sums_data[c];
    float n_sum = n_sums_data[c];

    float total_sum = p_sum + n_sum;
    float p_agg_wt = 0.0f;
    float n_agg_wt = 0.0f;

    if (total_sum > 0.0f) {
        p_agg_wt = p_sum / total_sum;
        n_agg_wt = n_sum / total_sum;
    }

    float p_sum_normalized = (p_sum == 0.0f) ? 1.0f : p_sum;
    float n_sum_normalized = (n_sum == 0.0f) ? 1.0f : n_sum;
    
    // Correct indexing for inp_t and wts.
    // inp_t is (spatial, channels), wts is (spatial, channels)
    int inp_t_idx = s * channels + c; 
    int wts_idx = s * channels + c;
    
    float val = inp_t_data[inp_t_idx];
    float weight = wts_data[wts_idx];

    float pos_contrib = 0.0f;
    float neg_contrib = 0.0f;

    if (val > 0) {
        pos_contrib = (val / p_sum_normalized) * weight * p_agg_wt;
    } else {
        neg_contrib = (val / n_sum_normalized) * weight * n_agg_wt * -1.0f;
    }
    
    output_data[inp_t_idx] = pos_contrib + neg_contrib;
}

torch::Tensor fused_weighted_gavgpool_cuda(
    const torch::Tensor& wts,
    const torch::Tensor& inp
) {
    // Transpose inputs to match PyTorch version's logic
    auto wts_t = wts.transpose(-2, -1).contiguous();
    auto inp_t = inp.transpose(-2, -1).contiguous();
    auto original_wts = wts.contiguous();

    // After transpose, last dimension is channels
    auto inp_sizes = inp_t.sizes();
    int inp_dim = inp_t.dim();
    TORCH_CHECK(inp_dim >= 1, "Input tensor must have at least 1 dimension");
    
    int channels = (inp_dim > 1) ? inp_sizes[inp_dim - 1] : 1;
    int spatial_size = inp_sizes[0]; // The first dimension of transposed input
    int batch_size = 1; // Simplified batch handling

    if (inp_dim > 2) {
        int total_elements = inp_t.numel();
        batch_size = total_elements / (spatial_size * channels);
    }

    auto output = torch::zeros_like(inp_t);

    // Per-channel sums (temporary storage on GPU)
    auto p_sums = torch::zeros({channels}, inp_t.options());
    auto n_sums = torch::zeros({channels}, inp_t.options());

    // Launch parameters for sum kernel
    dim3 block_sum(256);
    dim3 grid_sum((channels + block_sum.x - 1) / block_sum.x);

    // Kernel 1: Calculate sums
    sum_kernel<<<grid_sum, block_sum>>>(
        inp_t.data_ptr<float>(),
        p_sums.data_ptr<float>(),
        n_sums.data_ptr<float>(),
        spatial_size,
        channels
    );

    // Launch parameters for main kernel
    dim3 block(16, 16);
    dim3 grid(
        (channels + block.x - 1) / block.x,
        (spatial_size + block.y - 1) / block.y
    );

    // Kernel 2: Apply weights and calculate final output
    apply_weights_kernel<<<grid, block>>>(
        inp_t.data_ptr<float>(),
        original_wts.data_ptr<float>(), // Use original wts
        p_sums.data_ptr<float>(),
        n_sums.data_ptr<float>(),
        output.data_ptr<float>(),
        spatial_size,
        channels
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}
