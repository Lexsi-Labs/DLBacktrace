import torch
from torch.utils.cpp_extension import load_inline
import torch.nn.functional as F
from typing import Tuple, Union, List

padding_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdio>

// CUDA kernel for applying padding to a 3D tensor (H, W, C)
__global__ void apply_padding_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int in_h, const int in_w, const int channels,
    const int out_h, const int out_w,
    const int pad_h_before, const int pad_h_after,
    const int pad_w_before, const int pad_w_after,
    const float const_val
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = out_h * out_w * channels;
    
    if (idx >= total_elements) return;
    
    // Calculate output position
    const int c = idx % channels;
    const int w_out = (idx / channels) % out_w;
    const int h_out = (idx / channels) / out_w;
    
    // Check if we're in padding region
    const int h_in = h_out - pad_h_before;
    const int w_in = w_out - pad_w_before;
    
    if (h_in < 0 || h_in >= in_h || w_in < 0 || w_in >= in_w) {
        // We're in padding region - fill with constant value
        output[idx] = const_val;
    } else {
        // Copy from input tensor
        const int input_idx = h_in * in_w * channels + w_in * channels + c;
        output[idx] = input[input_idx];
    }
}

std::vector<torch::Tensor> launch_calculate_padding_kernel(
    torch::Tensor input,
    const std::string& padding_mode,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int custom_pad_h, const int custom_pad_w,
    const float const_val
) {
    // Ensure input is contiguous and on CUDA
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (H, W, C)");
    input = input.contiguous();
    
    const int in_h = input.size(0);
    const int in_w = input.size(1);
    const int channels = input.size(2);
    
    int pad_h_before = 0, pad_h_after = 0;
    int pad_w_before = 0, pad_w_after = 0;
    
    // Calculate padding based on mode
    if (padding_mode == "valid") {
        // No padding needed
        torch::Tensor padding_info = torch::zeros({3, 2}, torch::dtype(torch::kInt32).device(input.device()));
        return {input, padding_info};
    }
    else if (padding_mode == "same") {
        // Calculate padding for 'same' mode
        int pad_h, pad_w;
        
        int height_remainder = in_h % stride_h;
        if (height_remainder == 0) {
            pad_h = std::max(0, kernel_h - stride_h);
        } else {
            pad_h = std::max(0, kernel_h - height_remainder);
        }
        
        int width_remainder = in_w % stride_w;
        if (width_remainder == 0) {
            pad_w = std::max(0, kernel_w - stride_w);
        } else {
            pad_w = std::max(0, kernel_w - width_remainder);
        }
        
        // Calculate asymmetric padding
        pad_h_before = pad_h / 2;
        pad_h_after = (pad_h + 1) / 2;
        pad_w_before = pad_w / 2;
        pad_w_after = (pad_w + 1) / 2;
    }
    else if (padding_mode == "custom" && (custom_pad_h > 0 || custom_pad_w > 0)) {
        // Custom padding
        pad_h_before = pad_h_after = custom_pad_h;
        pad_w_before = pad_w_after = custom_pad_w;
    }
    else {
        // No padding
        torch::Tensor padding_info = torch::zeros({3, 2}, torch::dtype(torch::kInt32).device(input.device()));
        return {input, padding_info};
    }
    
    // Calculate output dimensions
    const int out_h = in_h + pad_h_before + pad_h_after;
    const int out_w = in_w + pad_w_before + pad_w_after;
    
    // Allocate output tensor
    torch::Tensor output = torch::empty({out_h, out_w, channels}, 
                                       input.options());
    
    // Launch kernel
    const int total_elements = out_h * out_w * channels;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    apply_padding_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        in_h, in_w, channels,
        out_h, out_w,
        pad_h_before, pad_h_after,
        pad_w_before, pad_w_after,
        const_val
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(error));
    }
    
    // Create padding info tensor
    torch::Tensor padding_info = torch::zeros({3, 2}, torch::dtype(torch::kInt32).device(input.device()));
    padding_info[0][0] = pad_h_before;
    padding_info[0][1] = pad_h_after;
    padding_info[1][0] = pad_w_before;
    padding_info[1][1] = pad_w_after;
    
    return {output, padding_info};
}
"""

# Simplified C++ declaration
padding_cuda_declaration = r"""
std::vector<torch::Tensor> launch_calculate_padding_kernel(
    torch::Tensor input,
    const std::string& padding_mode,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int custom_pad_h, const int custom_pad_w,
    const float const_val
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
]

extra_flags.extend(get_cuda_arch_flags())

# Load the CUDA extension
custom_padding_cuda_ops = load_inline(
    name="padding_cuda",
    cpp_sources=padding_cuda_declaration,
    cuda_sources=padding_cuda_source,
    functions=["launch_calculate_padding_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=False
)

def calculate_padding_cuda(
    kernel_size: Tuple[int, int], 
    inp: torch.Tensor, 
    padding: Union[str, Tuple[Union[int, None], Union[int, None]]], 
    strides: Tuple[int, int], 
    const_val: float = 0.0
) -> Tuple[torch.Tensor, List[List[int]]]:
    """
    CUDA-optimized version of calculate_padding function.
    
    Args:
        kernel_size: Tuple of (height, width) representing the kernel dimensions
        inp: Input tensor of shape (height, width, channels) to be padded
        padding: Padding mode - 'valid', 'same', or tuple of (pad_h, pad_v)
        strides: Tuple of (stride_height, stride_width) for convolution strides
        const_val: Constant value used for padding (default: 0.0)
    
    Returns:
        Tuple containing:
            - Padded input tensor
            - List of padding values [[pad_h_before, pad_h_after], 
              [pad_v_before, pad_v_after], [0, 0]]
    """
    
    # Determine padding mode and values
    if isinstance(padding, str):
        padding_mode = padding
        custom_pad_h = custom_pad_w = 0
    elif isinstance(padding, tuple) and padding != (None, None):
        padding_mode = "custom"
        custom_pad_h = int(padding[0]) if padding[0] is not None else 0
        custom_pad_w = int(padding[1]) if padding[1] is not None else 0
    else:
        padding_mode = "valid"
        custom_pad_h = custom_pad_w = 0
    
    # Call CUDA kernel
    output, padding_info = custom_padding_cuda_ops.launch_calculate_padding_kernel(
        inp,
        padding_mode,
        int(kernel_size[0]), int(kernel_size[1]),
        int(strides[0]), int(strides[1]),
        custom_pad_h, custom_pad_w,
        float(const_val)
    )
    
    # Convert padding_info to list format
    padding_list = [
        [padding_info[0, 0].item(), padding_info[0, 1].item()],
        [padding_info[1, 0].item(), padding_info[1, 1].item()],
        [0, 0]
    ]
    
    return output, padding_list