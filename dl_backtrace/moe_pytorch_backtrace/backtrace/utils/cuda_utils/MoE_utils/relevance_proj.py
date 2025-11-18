import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Warp-level reduction using shuffle
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Reduction kernel for computing statistics
__global__ void reduction_kernel(
    const float* __restrict__ output,
    const float* __restrict__ wts,
    int n_output,
    int n_wts,
    float* p_sum_out,
    float* n_sum_out,
    float* total_wt_out
) {
    __shared__ float shared_p[32];  // One per warp
    __shared__ float shared_n[32];
    __shared__ float shared_w[32];
    
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    // Compute partial sums for output tensor
    float p_sum = 0.0f;
    float n_sum = 0.0f;
    
    for (int idx = blockIdx.x * blockDim.x + tid; idx < n_output; idx += blockDim.x * gridDim.x) {
        float val = output[idx];
        if (val > 0.0f) {
            p_sum += val;
        } else if (val < 0.0f) {
            n_sum += fabsf(val);
        }
    }
    
    // Warp-level reduction
    p_sum = warp_reduce_sum(p_sum);
    n_sum = warp_reduce_sum(n_sum);
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        shared_p[warp_id] = p_sum;
        shared_n[warp_id] = n_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < 32) {
        p_sum = (tid < blockDim.x / 32) ? shared_p[tid] : 0.0f;
        n_sum = (tid < blockDim.x / 32) ? shared_n[tid] : 0.0f;
        
        p_sum = warp_reduce_sum(p_sum);
        n_sum = warp_reduce_sum(n_sum);
        
        if (tid == 0) {
            atomicAdd(p_sum_out, p_sum);
            atomicAdd(n_sum_out, n_sum);
        }
    }
    
    // Compute total weight (separate loop for weights tensor)
    float total_w = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + tid; idx < n_wts; idx += blockDim.x * gridDim.x) {
        total_w += wts[idx];
    }
    
    total_w = warp_reduce_sum(total_w);
    
    if (lane == 0) {
        shared_w[warp_id] = total_w;
    }
    __syncthreads();
    
    if (tid < 32) {
        total_w = (tid < blockDim.x / 32) ? shared_w[tid] : 0.0f;
        total_w = warp_reduce_sum(total_w);
        
        if (tid == 0) {
            atomicAdd(total_wt_out, total_w);
        }
    }
}

// Main kernel for element-wise transformation
__global__ void relevance_proj_kernel(
    const float* __restrict__ output,
    float* __restrict__ result,
    int n,
    float p_sum,
    float n_sum,
    float total_wt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Compute aggregate weights 
    float total_sum = p_sum + n_sum;
    float p_agg_wt = (total_sum > 0.0f && p_sum > 0.0f) ? p_sum / total_sum : 0.0f;
    float n_agg_wt = (total_sum > 0.0f && n_sum > 0.0f) ? n_sum / total_sum : 0.0f;
    
    // Safe denominators
    float p_sum_safe = (p_sum != 0.0f) ? p_sum : 1.0f;
    float n_sum_safe = (n_sum != 0.0f) ? n_sum : 1.0f;
    
    // Per-element transformation
    float val = output[idx];
    float res = 0.0f;
    
    if (val > 0.0f) {
        res = (p_agg_wt > 0.0f) ? (val / p_sum_safe) * total_wt * p_agg_wt : 0.0f;
    } else if (val < 0.0f) {
        res = (n_agg_wt > 0.0f) ? (val / n_sum_safe) * total_wt * n_agg_wt * -1.0f : 0.0f;
    }
    
    result[idx] = res;
}

// Host function to launch kernels
void launch_kernel(torch::Tensor wts, torch::Tensor output, torch::Tensor result) {
    const int n_output = output.numel();
    const int n_wts = wts.numel();
    
    const float* output_ptr = output.data_ptr<float>();
    const float* wts_ptr = wts.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();
    
    // Allocate device memory for reduction results
    float *d_p_sum, *d_n_sum, *d_total_wt;
    cudaMalloc(&d_p_sum, sizeof(float));
    cudaMalloc(&d_n_sum, sizeof(float));
    cudaMalloc(&d_total_wt, sizeof(float));
    
    cudaMemset(d_p_sum, 0, sizeof(float));
    cudaMemset(d_n_sum, 0, sizeof(float));
    cudaMemset(d_total_wt, 0, sizeof(float));
    
    // Launch reduction kernel
    const int block_size = 256;
    const int num_blocks = min(1024, (n_output + block_size - 1) / block_size);
    
    reduction_kernel<<<num_blocks, block_size>>>(
        output_ptr, wts_ptr, n_output, n_wts,
        d_p_sum, d_n_sum, d_total_wt
    );
    
    // Copy results back to host
    float p_sum, n_sum, total_wt;
    cudaMemcpy(&p_sum, d_p_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&n_sum, d_n_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_wt, d_total_wt, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Launch main kernel
    const int main_blocks = (n_output + block_size - 1) / block_size;
    relevance_proj_kernel<<<main_blocks, block_size>>>(
        output_ptr, result_ptr, n_output,
        p_sum, n_sum, total_wt
    );
    
    // Cleanup
    cudaFree(d_p_sum);
    cudaFree(d_n_sum);
    cudaFree(d_total_wt);
}
"""

cuda_declaration = r"""
void launch_kernel(torch::Tensor wts, torch::Tensor output, torch::Tensor result);
"""

def get_cuda_arch_flags():
    if not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    arch_flag = f"--generate-code=arch=compute_{major}{minor},code=sm_{major}{minor}"
    return [arch_flag]

extra_flags = [
    '-O3', 
    '--use_fast_math', 
    '-Xcompiler', '-fPIC',
]
extra_flags.extend(get_cuda_arch_flags())

cuda_ops = load_inline(
    name="relevance_proj_cuda",
    cpp_sources=cuda_declaration,
    cuda_sources=cuda_source,
    functions=["launch_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def calculate_relevance_proj_cuda(wts: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """CUDA-accelerated relevance projection calculation."""
    assert output.is_cuda and wts.is_cuda, "Tensors must be on GPU"
    assert output.dtype == torch.float32 and wts.dtype == torch.float32, "Only float32 supported"
    
    result = torch.zeros_like(output)
    cuda_ops.launch_kernel(wts, output, result)
    return result