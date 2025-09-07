import torch
from torch.utils.cpp_extension import load_inline
import numpy as np

mul_cuda_source = r"""
#include <torch/extension.h>
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void calculate_wt_mul_kernel(
    const float* __restrict__ R,
    float* __restrict__ R_x,
    float* __restrict__ R_y,
    int total_elements
) {
    // Calculate thread index for float4 operations (each thread processes 4 elements)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int base_idx = tid * 4;
    
    // Check bounds - ensure we don't go beyond array limits
    if (base_idx >= total_elements) return;
    
    // Load 4 float values as float4 for vectorized processing
    float4 r_vals;
    if (base_idx + 3 < total_elements) {
        // Safe to load full float4
        r_vals = *reinterpret_cast<const float4*>(&R[base_idx]);
    } else {
        // Handle remaining elements individually to avoid out-of-bounds access
        r_vals.x = (base_idx < total_elements) ? R[base_idx] : 0.0f;
        r_vals.y = (base_idx + 1 < total_elements) ? R[base_idx + 1] : 0.0f;
        r_vals.z = (base_idx + 2 < total_elements) ? R[base_idx + 2] : 0.0f;
        r_vals.w = (base_idx + 3 < total_elements) ? R[base_idx + 3] : 0.0f;
    }
    
    // Perform vectorized 50-50 split: multiply by 0.5f
    float4 half_vals;
    half_vals.x = r_vals.x * 0.5f;
    half_vals.y = r_vals.y * 0.5f;
    half_vals.z = r_vals.z * 0.5f;
    half_vals.w = r_vals.w * 0.5f;
    
    // Store results back to memory using vectorized stores
    if (base_idx + 3 < total_elements) {
        // Safe to store full float4
        *reinterpret_cast<float4*>(&R_x[base_idx]) = half_vals;
        *reinterpret_cast<float4*>(&R_y[base_idx]) = half_vals;
    } else {
        // Handle remaining elements individually
        if (base_idx < total_elements) {
            R_x[base_idx] = half_vals.x;
            R_y[base_idx] = half_vals.x;
        }
        if (base_idx + 1 < total_elements) {
            R_x[base_idx + 1] = half_vals.y;
            R_y[base_idx + 1] = half_vals.y;
        }
        if (base_idx + 2 < total_elements) {
            R_x[base_idx + 2] = half_vals.z;
            R_y[base_idx + 2] = half_vals.z;
        }
        if (base_idx + 3 < total_elements) {
            R_x[base_idx + 3] = half_vals.w;
            R_y[base_idx + 3] = half_vals.w;
        }
    }
}

torch::Tensor launch_calculate_wt_mul_kernel(const torch::Tensor& R) {
    // Ensure input is contiguous and on CUDA
    TORCH_CHECK(R.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(R.dtype() == torch::kFloat32, "Input tensor must be float32");
    
    auto R_contiguous = R.contiguous();
    
    // Get tensor properties
    int total_elements = R_contiguous.numel();
    
    // Create output tensors with same shape and properties as input
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(R_contiguous.device());
    torch::Tensor R_x = torch::empty_like(R_contiguous, options);
    torch::Tensor R_y = torch::empty_like(R_contiguous, options);
    
    // Calculate grid and block dimensions
    int threads_needed = (total_elements + 3) / 4;  // Round up division
    
    // optimal block size
    int block_size = 256;
    int grid_size = (threads_needed + block_size - 1) / block_size;  // Round up division
    
    // Launch kernel
    calculate_wt_mul_kernel<<<grid_size, block_size>>>(
        R_contiguous.data_ptr<float>(),
        R_x.data_ptr<float>(),
        R_y.data_ptr<float>(),
        total_elements
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    // Return both output tensors as a tuple (converted to Python tuple in the wrapper)
    return torch::stack({R_x, R_y}, 0);
}

"""

# Simplified C++ declaration
mul_cuda_declaration = r"""
torch::Tensor launch_calculate_wt_mul_kernel(
    const torch::Tensor& R
);
"""

def get_cuda_arch_flags():
    """
    Generate NVCC architecture flags for the current CUDA device.
    Returns an empty list if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return []
    
    major, minor = torch.cuda.get_device_capability()
    arch_flag = f"--generate-code=arch=compute_{major}{minor},code=sm_{major}{minor}"
    return [arch_flag]

extra_flags = [
    '-O3', 
    '--use_fast_math', 
    # '-Xcompiler', '-fPIC',
    # '-Xptxas', '-dlcm=cg',
    # '-Xptxas', '-dscm=wt',
]

extra_flags.extend(get_cuda_arch_flags())

custom_mul_cuda_ops = load_inline(
    name="mul_cuda_v1",
    cpp_sources=mul_cuda_declaration,
    cuda_sources=mul_cuda_source,
    functions=["launch_calculate_wt_mul_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def calculate_wt_mul(R):
    """
    CUDA-accelerated version of calculate_wt_mul with vectorized float4 operations.
    
    Args:
        R (torch.Tensor): Input relevance tensor, must be float32 and on CUDA
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: R_x and R_y, both equal to R * 0.5
    """
    # Ensure input is a CUDA tensor with float32 dtype
    if not isinstance(R, torch.Tensor):
        R = torch.tensor(R, dtype=torch.float32, device='cuda')
    elif not R.is_cuda:
        R = R.cuda()
    elif R.dtype != torch.float32:
        R = R.to(torch.float32)
    
    # Launch kernel and get stacked result
    result = custom_mul_cuda_ops.launch_calculate_wt_mul_kernel(R)
    
    # Split the stacked result back into two tensors
    R_x = result[0].cpu().numpy()
    R_y = result[1].cpu().numpy()
    
    return R_x, R_y