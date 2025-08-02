import torch
from torch.utils.cpp_extension import load_inline
import numpy as np

conv2d_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#define MAX_KERNEL_SIZE 7

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

// SIMPLIFIED_DEVICE_FUNCTION: Process one output channel at a time to avoid race conditions
__device__ void calculate_wt_conv_unit_cuda_single_channel(
    const float* patch_chunk,     // Input patch chunk [kernel_h * kernel_w * chunk_size]
    float relevance_wt,           // Relevance weight for this output channel
    const float* w,               // Full kernel weights [kernel_h * kernel_w * in_channels * out_channels]
    float bias_val,               // Bias for this output channel
    int kernel_h, int kernel_w, int chunk_size, int in_channels, int out_channels,
    int channel_offset,           // Starting channel index for this chunk
    int oc,                       // Current output channel index
    int act_type, float act_range_l, float act_range_u, int act_func_int,
    float* output_chunk           // Output chunk [kernel_h * kernel_w * chunk_size]
) {
    float p_sum = 0.0f, n_sum = 0.0f;
    
    // Compute convolution for current chunk and this output channel
    for (int i = 0; i < kernel_h; i++) {
        for (int j = 0; j < kernel_w; j++) {
            for (int ic = 0; ic < chunk_size; ic++) {
                int chunk_patch_idx = (i * kernel_w + j) * chunk_size + ic;
                int global_ic = channel_offset + ic;
                int weight_idx = ((i * kernel_w + j) * in_channels + global_ic) * out_channels + oc;
                
                float conv_val = w[weight_idx] * patch_chunk[chunk_patch_idx];
                p_sum += fmaxf(conv_val, 0.0f);
                n_sum += fmaxf(-conv_val, 0.0f);
            }
        }
    }
    
    float t_sum = p_sum + n_sum;
    float denom_bias_term = 0.0f;
    float bias_pos = 0.0f;
    float bias_neg = 0.0f;
    
    if (bias_val != 0.0f) {
        bias_pos = fmaxf(bias_val, 0.0f);
        bias_neg = fmaxf(-bias_val, 0.0f);
        denom_bias_term = bias_pos + bias_neg;
    }
    else {
        bias_pos = 0.0f;
        bias_neg = 0.0f;
        denom_bias_term = 0.0f;
    }
    
    // Activation handling
    float p_saturate = (p_sum > 0.0f) ? 1.0f : 0.0f;
    float n_saturate = (n_sum > 0.0f) ? 1.0f : 0.0f;
    
    if (act_type == 0) { // mono
        if (!isinf(act_range_l)) {
            p_saturate = (t_sum > act_range_l) ? 1.0f : 0.0f;
        }
        if (!isinf(act_range_u)) {
            n_saturate = (t_sum < act_range_u) ? 1.0f : 0.0f;
        }
    }
    else if (act_type == 1) { // non-mono
        float t_act = apply_activation(t_sum, act_func_int);
        float p_act = apply_activation(p_sum + bias_pos, act_func_int);
        float n_act = apply_activation(-(n_sum + bias_neg), act_func_int);
        
        if (!isinf(act_range_l)) {
            float temp_ind = (t_sum > act_range_l) ? 1.0f : 0.0f;
            p_saturate = p_saturate * temp_ind;
        }
        if (!isinf(act_range_u)) {
            float temp_ind = (t_sum < act_range_u) ? 1.0f : 0.0f;
            n_saturate = n_saturate * temp_ind;
        }
        
        n_saturate = n_saturate * (fabsf(t_act - p_act) > 1e-5f);
        p_saturate = p_saturate * (fabsf(t_act - n_act) > 1e-5f);
    }
    
    // Numerical stabilization
    float denom = p_sum + n_sum + denom_bias_term;
    denom = (denom == 0.0f) ? 1e-12f : denom;
    
    float inv_denom = 1.0f / denom;
    float p_agg_wt = inv_denom * relevance_wt * p_saturate;
    float n_agg_wt = inv_denom * relevance_wt * n_saturate;
    
    // Apply weighted updates to output chunk (no race condition now)
    for (int i = 0; i < kernel_h; i++) {
        for (int j = 0; j < kernel_w; j++) {
            for (int ic = 0; ic < chunk_size; ic++) {
                int chunk_patch_idx = (i * kernel_w + j) * chunk_size + ic;
                int global_ic = channel_offset + ic;
                int weight_idx = ((i * kernel_w + j) * in_channels + global_ic) * out_channels + oc;
                
                float conv_val = w[weight_idx] * patch_chunk[chunk_patch_idx];
                float p_part = fmaxf(conv_val, 0.0f);
                float n_part = fmaxf(-conv_val, 0.0f);
                
                output_chunk[chunk_patch_idx] += p_part * p_agg_wt - n_part * n_agg_wt;
            }
        }
    }
}

// MAIN_KERNEL: Fused weighted convolution kernel with channel chunking optimization
__global__ void weighted_conv_kernel(
    const float* relevance_y,     // [batch_size, out_channels, out_h, out_w]
    const float* input_array,     // [batch_size, in_channels, in_h, in_w]
    const float* w,               // [out_channels, in_channels, kernel_h, kernel_w] -> transposed to [kernel_h, kernel_w, in_channels, out_channels]
    const float* b,               // [out_channels]
    float* relevance_x,           // [batch_size, in_channels, in_h, in_w]
    int batch_size, int in_channels, int out_channels,
    int in_h, int in_w, int out_h, int out_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w,
    int act_type, float act_range_l, float act_range_u, int act_func_int
) {
    // THREAD_MAPPING: Map threads to spatial output locations
    int batch_idx = blockIdx.z;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_y >= out_h || out_x >= out_w) return;
    
    // MEMORY_TILE: Calculate input region bounds with padding
    int in_y_start = out_y * stride_h - pad_h;
    int in_x_start = out_x * stride_w - pad_w;
    int in_y_end = in_y_start + kernel_h;
    int in_x_end = in_x_start + kernel_w;
    
    // BOUNDARY_HANDLING: Clamp to valid input bounds and handle partial patches
    int actual_y_start = max(0, in_y_start);
    int actual_x_start = max(0, in_x_start);
    int actual_y_end = min(in_h, in_y_end);
    int actual_x_end = min(in_w, in_x_end);
    
    // Calculate actual patch dimensions (may be smaller than kernel at boundaries)
    int actual_patch_h = actual_y_end - actual_y_start;
    int actual_patch_w = actual_x_end - actual_x_start;
    
    // Skip if patch is empty
    if (actual_patch_h <= 0 || actual_patch_w <= 0) return;
    
    // OPTIMIZED_CHANNEL_HANDLING: Use chunking for large channel counts
    const int MAX_LOCAL_CHANNELS = 4;  // Minimal to prevent stack overflow
    
    // Use minimal local memory to avoid stack overflow (4 channels max)
    float patch_chunk[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE * 4];
    float output_chunk[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE * 4];
    
    // RELEVANCE_LOAD: Load relevance weights for current output position
    float relevance_wts[512]; // Support up to 512 output channels
    for (int oc = 0; oc < out_channels && oc < 512; oc++) {
        int relevance_idx = batch_idx * (out_channels * out_h * out_w) + 
                           oc * (out_h * out_w) + 
                           out_y * out_w + out_x;
        relevance_wts[oc] = relevance_y[relevance_idx];
    }
    
    // CHANNEL_CHUNKING: Process channels in chunks to optimize memory usage
    for (int channel_start = 0; channel_start < in_channels; channel_start += MAX_LOCAL_CHANNELS) {
        int channel_end = min(channel_start + MAX_LOCAL_CHANNELS, in_channels);
        int current_chunk_size = channel_end - channel_start;
        int patch_chunk_size = kernel_h * kernel_w * current_chunk_size;
        
        // Initialize current chunk
        for (int i = 0; i < patch_chunk_size; i++) {
            patch_chunk[i] = 0.0f;
            output_chunk[i] = 0.0f;
        }
        
        // BOUNDARY_SAFE_LOAD: Load current channel chunk
    for (int ky = 0; ky < kernel_h; ky++) {
        for (int kx = 0; kx < kernel_w; kx++) {
                int input_y = in_y_start + ky;
                int input_x = in_x_start + kx;
                
                if (input_y >= 0 && input_y < in_h && input_x >= 0 && input_x < in_w) {
                    for (int ic = 0; ic < current_chunk_size; ic += blockDim.x) {
                        int local_channel_idx = ic + threadIdx.x;
                        if (local_channel_idx < current_chunk_size) {
                            int global_channel_idx = channel_start + local_channel_idx;
                            int input_idx = batch_idx * (in_channels * in_h * in_w) + 
                                           global_channel_idx * (in_h * in_w) + 
                                           input_y * in_w + input_x;
                            int patch_idx = (ky * kernel_w + kx) * current_chunk_size + local_channel_idx;
                            patch_chunk[patch_idx] = input_array[input_idx];
                        }
                    }
                }
            }
        }
        
        // CHUNKED_CONV_UNIT: Process current channel chunk for each output channel
        for (int oc = 0; oc < out_channels; oc++) {
            float relevance_wt = relevance_wts[oc];
            float bias_val = (b != nullptr) ? b[oc] : 0.0f;
            
            calculate_wt_conv_unit_cuda_single_channel(
                patch_chunk, relevance_wt, w, bias_val,
                kernel_h, kernel_w, current_chunk_size, in_channels, out_channels,
                channel_start, // offset for weight indexing
                oc, // current output channel
                act_type, act_range_l, act_range_u, act_func_int,
                output_chunk
            );
        }
        
        // CHUNKED_WRITEBACK: Write chunk results back to global memory
        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                int output_y = in_y_start + ky;
                int output_x = in_x_start + kx;
                
                if (output_y >= 0 && output_y < in_h && output_x >= 0 && output_x < in_w) {
                    for (int ic = 0; ic < current_chunk_size; ic += blockDim.x) {
                        int local_channel_idx = ic + threadIdx.x;
                        if (local_channel_idx < current_chunk_size) {
                            int global_channel_idx = channel_start + local_channel_idx;
                            int output_idx = batch_idx * (in_channels * in_h * in_w) + 
                                            global_channel_idx * (in_h * in_w) + 
                                            output_y * in_w + output_x;
                            int chunk_idx = (ky * kernel_w + kx) * current_chunk_size + local_channel_idx;
                            
                            atomicAdd(&relevance_x[output_idx], output_chunk[chunk_idx]);
                        }
                    }
                }
            }
        }
    } // End channel chunking loop
}

// HOST_LAUNCHER: PyTorch integration function
torch::Tensor weighted_conv_cuda(
    torch::Tensor relevance_y,
    torch::Tensor input_array,
    torch::Tensor w,
    torch::Tensor b,
    int padding_type,
    std::vector<int> custom_padding,
    std::vector<int> strides,
    int act_type,
    float act_range_l,
    float act_range_u,
    int act_func_int
) {
    
    // TENSOR_VALIDATION: Ensure tensors are on CUDA and contiguous
    TORCH_CHECK(relevance_y.is_cuda(), "relevance_y must be on CUDA");
    TORCH_CHECK(input_array.is_cuda(), "input_array must be on CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be on CUDA");
    if (b.defined()) {
        TORCH_CHECK(b.is_cuda(), "b must be on CUDA if provided");
    }
    
    // DTYPE_VALIDATION: Ensure all tensors are float32
    TORCH_CHECK(relevance_y.dtype() == torch::kFloat32, "relevance_y must be float32");
    TORCH_CHECK(input_array.dtype() == torch::kFloat32, "input_array must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    if (b.defined()) {
        TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    }
    
    relevance_y = relevance_y.contiguous();
    input_array = input_array.contiguous();
    w = w.contiguous();
    if (b.defined()) {
        b = b.contiguous();
    }
    
    // DIMENSION_EXTRACTION: Extract tensor dimensions
    int batch_size = input_array.size(0);
    int in_channels = input_array.size(1);
    int in_h = input_array.size(2);
    int in_w = input_array.size(3);
    
    int out_channels = relevance_y.size(1);
    int out_h = relevance_y.size(2);
    int out_w = relevance_y.size(3);
    
    int kernel_h = w.size(2);
    int kernel_w = w.size(3);
    
    // DIMENSION_VALIDATION: Validate tensor dimensions
    TORCH_CHECK(batch_size > 0, "batch_size must be positive, got ", batch_size);
    TORCH_CHECK(in_channels > 0, "in_channels must be positive, got ", in_channels);
    TORCH_CHECK(in_h > 0, "input height must be positive, got ", in_h);
    TORCH_CHECK(in_w > 0, "input width must be positive, got ", in_w);
    TORCH_CHECK(out_channels > 0, "out_channels must be positive, got ", out_channels);
    TORCH_CHECK(out_h > 0, "output height must be positive, got ", out_h);
    TORCH_CHECK(out_w > 0, "output width must be positive, got ", out_w);
    TORCH_CHECK(kernel_h > 0 && kernel_h <= MAX_KERNEL_SIZE, "kernel_h must be positive and <= ", MAX_KERNEL_SIZE, ", got ", kernel_h);
    TORCH_CHECK(kernel_w > 0 && kernel_w <= MAX_KERNEL_SIZE, "kernel_w must be positive and <= ", MAX_KERNEL_SIZE, ", got ", kernel_w);
    TORCH_CHECK(in_channels <= 1024, "in_channels must be <= 1024 for current implementation, got ", in_channels);
    TORCH_CHECK(out_channels <= 512, "out_channels must be <= 512 for current implementation, got ", out_channels);
    
    // COMPATIBILITY_VALIDATION: Check tensor dimension compatibility
    TORCH_CHECK(input_array.size(0) == relevance_y.size(0), "Batch size mismatch");
    TORCH_CHECK(w.size(1) == in_channels, "Weight input channels must match input channels");
    TORCH_CHECK(w.size(0) == out_channels, "Weight output channels must match relevance output channels");
    if (b.defined()) {
        TORCH_CHECK(b.size(0) == out_channels, "Bias size must match output channels");
    }
    
    // STRIDE_VALIDATION: Validate stride parameters
    TORCH_CHECK(strides.size() >= 2, "strides must have at least 2 elements");
    int stride_h = strides[0];
    int stride_w = strides[1];
    TORCH_CHECK(stride_h > 0, "stride_h must be positive, got ", stride_h);
    TORCH_CHECK(stride_w > 0, "stride_w must be positive, got ", stride_w);
    
    // PADDING_CALCULATION: Calculate padding based on padding_type
    int pad_h = 0, pad_w = 0;
    
    if (padding_type == 1) { // valid padding
        pad_h = 0;
        pad_w = 0;
    }
    else if (padding_type == 2) { // same padding
        // Calculate padding to maintain output size
        int height_remainder = in_h % stride_h;
        int width_remainder = in_w % stride_w;
        
        int total_pad_h = (height_remainder == 0) ? 
            std::max(0, kernel_h - stride_h) : 
            std::max(0, kernel_h - height_remainder);
        int total_pad_w = (width_remainder == 0) ? 
            std::max(0, kernel_w - stride_w) : 
            std::max(0, kernel_w - width_remainder);
            
        pad_h = total_pad_h / 2;
        pad_w = total_pad_w / 2;
    }
    else if (padding_type == 0 && custom_padding.size() >= 2) { // custom padding
        pad_h = custom_padding[0];
        pad_w = custom_padding[1]; 
        TORCH_CHECK(pad_h >= 0, "custom pad_h must be non-negative, got ", pad_h);
        TORCH_CHECK(pad_w >= 0, "custom pad_w must be non-negative, got ", pad_w);
    }
    else {
        TORCH_CHECK(false, "Invalid padding configuration: padding_type=", padding_type, 
                   ", custom_padding.size()=", custom_padding.size());
    }
    
    // ACTIVATION_VALIDATION: Validate activation parameters
    TORCH_CHECK(act_type == 0 || act_type == 1, "act_type must be 0 (mono) or 1 (non-mono), got ", act_type);
    TORCH_CHECK(act_func_int >= 0 && act_func_int <= 7, "act_func_int must be 0-7, got ", act_func_int);
    TORCH_CHECK(std::isfinite(act_range_l) || std::isinf(act_range_l), "act_range_l must be finite or inf");
    TORCH_CHECK(std::isfinite(act_range_u) || std::isinf(act_range_u), "act_range_u must be finite or inf");
    
    // WEIGHT_TRANSPOSE: Transpose weights to expected format [kernel_h, kernel_w, in_channels, out_channels]
    torch::Tensor w_transposed = w.permute({2, 3, 1, 0}).contiguous();
    
    // OUTPUT_ALLOCATION: Allocate output tensor
    torch::Tensor relevance_x = torch::zeros_like(input_array);
    
    // LAUNCH_CONFIGURATION: Configure CUDA kernel launch parameters
    dim3 block_size(16, 16);  // 16x16 threads per block
    dim3 grid_size(
        (out_w + block_size.x - 1) / block_size.x,
        (out_h + block_size.y - 1) / block_size.y,
        batch_size
    );
    
    // KERNEL_LAUNCH: Launch the weighted convolution kernel
    weighted_conv_kernel<<<grid_size, block_size>>>(
        relevance_y.data_ptr<float>(),
        input_array.data_ptr<float>(),
        w_transposed.data_ptr<float>(),
        b.defined() ? b.data_ptr<float>() : nullptr,
        relevance_x.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kernel_h, kernel_w, stride_h, stride_w,
        pad_h, pad_w,
        act_type, act_range_l, act_range_u, act_func_int
    );
    
    // ERROR_CHECK: Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Kernel launch successful, synchronizing..." << std::endl;
    }
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    // SYNCHRONIZE: Wait for kernel completion and check for runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "CUDA kernel execution error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Kernel execution completed successfully!" << std::endl;
    }
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel execution failed: ", cudaGetErrorString(err));
    
    return relevance_x;
}
"""

#C++ declaration
conv2d_cuda_declaration = r"""
torch::Tensor weighted_conv_cuda(
    torch::Tensor relevance_y,
    torch::Tensor input_array,
    torch::Tensor w,
    torch::Tensor b,
    int padding_type,
    std::vector<int> custom_padding,
    std::vector<int> strides,
    int act_type,
    float act_range_l,
    float act_range_u,
    int act_func_int
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
    '-Xcompiler', '-fPIC',
    # '-Xptxas', '-dlcm=cg',
    # '-Xptxas', '-dscm=wt',
]

extra_flags.extend(get_cuda_arch_flags())

custom_conv2d_cuda_ops = load_inline(
    name="custom_conv2d_cuda_v2",
    cpp_sources=conv2d_cuda_declaration,
    cuda_sources=conv2d_cuda_source,
    functions=["weighted_conv_cuda"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def calculate_wt_conv_cuda(relevance_y, input_array, w, b, padding, strides, act):
    """
    CUDA-accelerated version that maintains the original algorithm structure.
    Handles batch processing and multi-dimensional inputs in Python,
    delegates core computation to CUDA kernel.
    
    Args:
        relevance_y: relevance at the output (same shape as conv output)
        input_array: input to the conv layer (shape: [B, C, H, W])
        w: weight matrix of the conv layer (shape: [out_channels, in_channels, kernel_h, kernel_w])
        b: bias vector (shape: [out_channels]) or None
        padding: padding tuple (pad_h, pad_w) or "same" or "valid"
        strides: stride tuple (stride_h, stride_w) 
        act: dict containing activation info with keys: "type", "range", "func"

    Returns:
        relevance_x: relevance at the input, same shape as input_array
    """
    device = torch.device("cuda")
    
    # TENSOR_CONVERSION: Convert inputs to PyTorch tensors with proper validation
    if not isinstance(relevance_y, torch.Tensor):
        relevance_y = torch.tensor(relevance_y, dtype=torch.float32, device=device)
    else:
        relevance_y = relevance_y.to(device=device, dtype=torch.float32)
    
    if not isinstance(input_array, torch.Tensor):
        input_array = torch.tensor(input_array, dtype=torch.float32, device=device)
    else:
        input_array = input_array.to(device=device, dtype=torch.float32)
    
    if not isinstance(w, torch.Tensor):
        w = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        w = w.to(device=device, dtype=torch.float32)
    
    if b is not None:
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float32, device=device)
        else:
            b = b.to(device=device, dtype=torch.float32)
    else:
        # Create zero bias tensor when bias is None (CUDA function expects a tensor)
        out_channels = w.shape[0]
        b = torch.zeros(out_channels, dtype=torch.float32, device=device)
    
    # STRIDE_VALIDATION: Ensure strides are properly formatted
    if strides is None:
        strides = [1, 1]
    elif isinstance(strides, torch.Tensor):
        strides = strides.tolist()
    else:
        strides = list(strides)
    
    # Ensure strides has exactly 2 elements
    if len(strides) == 1:
        strides = [strides[0], strides[0]]
    elif len(strides) != 2:
        raise ValueError(f"strides must have 1 or 2 elements, got {len(strides)}")
    
    # SHAPE_VALIDATION: Validate tensor shapes
    if len(relevance_y.shape) != 4:
        raise ValueError(f"relevance_y must be 4D tensor [B, C, H, W], got shape {relevance_y.shape}")
    if len(input_array.shape) != 4:
        raise ValueError(f"input_array must be 4D tensor [B, C, H, W], got shape {input_array.shape}")
    if len(w.shape) != 4:
        raise ValueError(f"w must be 4D tensor [out_C, in_C, K_H, K_W], got shape {w.shape}")
    if b is not None and len(b.shape) != 1:
        raise ValueError(f"b must be 1D tensor [out_C], got shape {b.shape}")
    
    # DIMENSION_COMPATIBILITY: Check dimension compatibility
    if input_array.shape[0] != relevance_y.shape[0]:
        raise ValueError(f"Batch size mismatch: input {input_array.shape[0]} vs relevance {relevance_y.shape[0]}")
    if w.shape[1] != input_array.shape[1]:
        raise ValueError(f"Input channels mismatch: weight {w.shape[1]} vs input {input_array.shape[1]}")
    if w.shape[0] != relevance_y.shape[1]:
        raise ValueError(f"Output channels mismatch: weight {w.shape[0]} vs relevance {relevance_y.shape[1]}")
    if b is not None and b.shape[0] != w.shape[0]:
        raise ValueError(f"Bias size mismatch: bias {b.shape[0]} vs weight out_channels {w.shape[0]}")
    
    # PADDING_PROCESSING: Calculate padding type and custom values
    padding_type = 0
    custom_padding = [0, 0]
    
    if padding == "valid":
        padding_type = 1
        custom_padding = [0, 0]
    elif padding == "same":
        padding_type = 2
        custom_padding = [0, 0]  # Will be calculated in CUDA
    elif isinstance(padding, (tuple, list)) and len(padding) >= 2:
        padding_type = 0
        custom_padding = [int(padding[0]), int(padding[1])]
    
    # ACTIVATION_VALIDATION: Parse and validate activation parameters
    if not isinstance(act, dict):
        raise ValueError(f"act must be a dictionary, got {type(act)}")
    
    if "type" not in act:
        raise ValueError("act dictionary must contain 'type' key")
    if "range" not in act:
        raise ValueError("act dictionary must contain 'range' key")
    
    act_type = 0 if act["type"] == "mono" else 1
    
    if not isinstance(act["range"], dict):
        raise ValueError("act['range'] must be a dictionary")
    
    act_lower = -float('inf') if act["range"].get("l") is None else float(act["range"]["l"])
    act_upper = float('inf') if act["range"].get("u") is None else float(act["range"]["u"])
    
    # Convert activation function string to int
    act_func_int = 0  # default: identity
    if act.get("func") is not None:
        act_func_str = act["func"]
        if isinstance(act_func_str, str):
            if act_func_str == "sigmoid": act_func_int = 1
            elif act_func_str == "swish": act_func_int = 2
            elif act_func_str == "wave": act_func_int = 3
            elif act_func_str == "pulse": act_func_int = 4
            elif act_func_str == "absolute": act_func_int = 5
            elif act_func_str == "hard_sigmoid": act_func_int = 6
            elif act_func_str == "tanh": act_func_int = 7
            else:
                print(f"Warning: Unknown activation function '{act_func_str}', using identity")
        else:
            print(f"Warning: Activation function must be string, got {type(act_func_str)}, using identity")
    
    # CUDA_EXECUTION: Call CUDA kernel and return numpy array like PyTorch version
    try:
        result = custom_conv2d_cuda_ops.weighted_conv_cuda(
            relevance_y, input_array, w, b, padding_type, custom_padding, strides, 
            act_type, act_lower, act_upper, act_func_int
        )
        # Convert back to numpy for consistency with PyTorch version
        return result.cpu().numpy()
    except Exception as e:
        raise RuntimeError(f"CUDA kernel execution failed: {str(e)}")


#SCRAPPED WT_CONV_UNIT_CUDA

# // DEVICE_FUNCTION: Calculate weighted convolution unit
# __device__ void calculate_wt_conv_unit_cuda(
#     const float* patch,           // Input patch [kernel_h * kernel_w * in_channels]
#     const float* relevance_wts,   // Relevance weights [out_channels] 
#     const float* w,               // Kernel weights [kernel_h * kernel_w * in_channels * out_channels]
#     const float* b,               // Bias [out_channels]
#     int kernel_h, int kernel_w, int in_channels, int out_channels,
#     int act_type, float act_range_l, float act_range_u, int act_func_int,
#     float* output_patch           // Output [kernel_h * kernel_w * in_channels]
# ) {
#     // VECTORIZED_COMPUTATION: Process multiple output channels per thread
#     for (int oc = threadIdx.x; oc < out_channels; oc += blockDim.x) {
#         float p_sum = 0.0f, n_sum = 0.0f;
        
#         // MEMORY_COALESCING: Sequential access pattern for convolution computation  
#         for (int i = 0; i < kernel_h; i++) {
#             for (int j = 0; j < kernel_w; j++) {
#                 for (int ic = 0; ic < in_channels; ic++) {
#                     int patch_idx = (i * kernel_w + j) * in_channels + ic;
#                     int weight_idx = ((i * kernel_w + j) * in_channels + ic) * out_channels + oc;
                    
#                     float conv_val = w[weight_idx] * patch[patch_idx];
#                     p_sum += fmaxf(conv_val, 0.0f);
#                     n_sum += fmaxf(-conv_val, 0.0f);
#                 }
#             }
#         }
        
#         float t_sum = p_sum + n_sum;
#         float bias_pos = (b != nullptr) ? fmaxf(b[oc], 0.0f) : 0.0f;
#         float bias_neg = (b != nullptr) ? fmaxf(-b[oc], 0.0f) : 0.0f;
#         float denom_bias_term = bias_pos + bias_neg;
        
#         // ACTIVATION_HANDLING: Process activation function constraints
#         float p_saturate = (p_sum > 0.0f) ? 1.0f : 0.0f;
#         float n_saturate = (n_sum > 0.0f) ? 1.0f : 0.0f;
        
#         if (act_type == 0) { // mono
#             if (!isinf(act_range_l)) {
#                 p_saturate = (t_sum > act_range_l) ? 1.0f : 0.0f;
#             }
#             if (!isinf(act_range_u)) {
#                 n_saturate = (t_sum < act_range_u) ? 1.0f : 0.0f;
#             }
#         }
#         else if (act_type == 1) { // non-mono
#             float t_act = apply_activation(t_sum, act_func_int);
#             float p_act = apply_activation(p_sum + bias_pos, act_func_int);
#             float n_act = apply_activation(-(n_sum + bias_neg), act_func_int);
            
#             // Apply range constraints if specified (like PyTorch)
#             if (!isinf(act_range_l)) {
#                 float temp_ind = (t_sum > act_range_l) ? 1.0f : 0.0f;
#                 p_saturate = p_saturate * temp_ind;
#             }
#             if (!isinf(act_range_u)) {
#                 float temp_ind = (t_sum < act_range_u) ? 1.0f : 0.0f;
#                 n_saturate = n_saturate * temp_ind;
#             }
            
#             // Apply activation function difference thresholding (corrected order)
#             n_saturate = n_saturate * (fabsf(t_act - p_act) > 1e-5f);
#             p_saturate = p_saturate * (fabsf(t_act - n_act) > 1e-5f);
#         }
        
#         // NUMERICAL_STABILIZATION: Prevent division by zero
#         float denom = p_sum + n_sum + denom_bias_term;
#         denom = (denom == 0.0f) ? 1e-12f : denom;
        
#         float inv_denom = 1.0f / denom;
#         float relevance_wt = relevance_wts[oc];
#         float p_agg_wt = inv_denom * relevance_wt * p_saturate;
#         float n_agg_wt = inv_denom * relevance_wt * n_saturate;
        
#         // WEIGHT_ACCUMULATION: Apply weighted updates to output patch
#         for (int i = 0; i < kernel_h; i++) {
#             for (int j = 0; j < kernel_w; j++) {
#                 for (int ic = 0; ic < in_channels; ic++) {
#                     int patch_idx = (i * kernel_w + j) * in_channels + ic;
#                     int weight_idx = ((i * kernel_w + j) * in_channels + ic) * out_channels + oc;
                    
#                     float conv_val = w[weight_idx] * patch[patch_idx];
#                     float p_part = fmaxf(conv_val, 0.0f);
#                     float n_part = fmaxf(-conv_val, 0.0f);
                    
#                     atomicAdd(&output_patch[patch_idx], p_part * p_agg_wt - n_part * n_agg_wt);
#                 }
#             }
#         }
#     }
# }
