#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_wt_max_unit_kernel(
    const float* __restrict__ patch_data,
    const float* __restrict__ wts_data,
    float* __restrict__ output_data,
    const int height,
    const int width,
    const int channels,
    const int total_elements
) {
    const int channel_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int spatial_size = height * width;
    const int block_size = blockDim.x;
    
    // Shared memory for reductions
    extern __shared__ float shared_data[];
    float* max_vals = shared_data;
    float* count_vals = &shared_data[block_size];
    
    // Phase 1: Find maximum value for this channel
    float local_max = -INFINITY;
    for (int i = tid; i < spatial_size; i += block_size) {
        int idx = i * channels + channel_idx;
        local_max = fmaxf(local_max, patch_data[idx]);
    }
    
    // Reduce to find global maximum
    max_vals[tid] = local_max;
    __syncthreads();
    
    // Block-wide reduction for maximum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + stride]);
        }
        __syncthreads();
    }
    
    float channel_max = max_vals[0];
    __syncthreads();
    
    // Phase 2: Count positions equal to maximum
    float local_count = 0.0f;
    for (int i = tid; i < spatial_size; i += block_size) {
        int idx = i * channels + channel_idx;
        if (patch_data[idx] == channel_max) {
            local_count += 1.0f;
        }
    }
    
    // Reduce to find total count
    count_vals[tid] = local_count;
    __syncthreads();
    
    // Block-wide reduction for count
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            count_vals[tid] += count_vals[tid + stride];
        }
        __syncthreads();
    }
    
    float total_count = count_vals[0];
    // Add epsilon protection for division by zero
    // float safe_count = (total_count > 0.0f) ? total_count : 1e-8f;
    float normalization_factor = (total_count > 0) ? (1.0f / total_count) : 0.0f;
    float weight = wts_data[channel_idx];
    __syncthreads();
    
    // Phase 3: Generate output
    for (int i = tid; i < spatial_size; i += block_size) {
        int idx = i * channels + channel_idx;
        float is_max = (patch_data[idx] == channel_max) ? 1.0f : 0.0f;
        output_data[idx] = is_max * normalization_factor * weight;
    }
}

torch::Tensor calculate_wt_max_unit_cuda(
    const torch::Tensor& patch,
    const torch::Tensor& wts,
    const int pool_size  // unused, maintained for compatibility
) {
    // Input validation
    TORCH_CHECK(patch.is_cuda(), "patch must be a CUDA tensor");
    TORCH_CHECK(wts.is_cuda(), "wts must be a CUDA tensor");
    TORCH_CHECK(patch.dtype() == torch::kFloat32, "patch must be float32");
    TORCH_CHECK(wts.dtype() == torch::kFloat32, "wts must be float32");
    
    // Extract dimensions
    const int height = patch.size(0);
    const int width = patch.size(1);
    const int channels = patch.size(2);
    const int total_elements = height * width * channels;
    
    // Create output tensor
    auto output = torch::zeros_like(patch);
    
    // Kernel launch parameters
    const int threads_per_block = std::min(1024, height * width);
    const int shared_memory_size = 2 * threads_per_block * sizeof(float);
    
    dim3 grid(channels);
    dim3 block(threads_per_block);
    
    // Launch kernel
    fused_wt_max_unit_kernel<<<grid, block, shared_memory_size>>>(
        patch.data_ptr<float>(),
        wts.data_ptr<float>(),
        output.data_ptr<float>(),
        height, width, channels, total_elements
    );
    
    // Check for errors
    cudaDeviceSynchronize();
    auto cuda_error = cudaGetLastError();
    TORCH_CHECK(cuda_error == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(cuda_error));
    
    return output;
}
