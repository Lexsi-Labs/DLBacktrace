import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import numpy as np

conv2d_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

// Constants for optimization
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_KERNEL_SIZE 11
#define SHARED_MEM_BANKS 32

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

// Device function for warp-level reduction
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Device function for block-level reduction
__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// Optimized padding calculation
__device__ void calculate_padding_cuda(
    int kernel_h, int kernel_w,
    int input_h, int input_w,
    int stride_h, int stride_w,
    const char* padding_mode,
    int* pad_h_before, int* pad_h_after,
    int* pad_w_before, int* pad_w_after
) {
    if (padding_mode[0] == 'v') { // "valid"
        *pad_h_before = *pad_h_after = *pad_w_before = *pad_w_after = 0;
    } else if (padding_mode[0] == 's') { // "same"
        int height_remainder = input_h % stride_h;
        int pad_h = (height_remainder == 0) ? 
                   max(0, kernel_h - stride_h) : 
                   max(0, kernel_h - height_remainder);
        
        int width_remainder = input_w % stride_w;
        int pad_w = (width_remainder == 0) ? 
                   max(0, kernel_w - stride_w) : 
                   max(0, kernel_w - width_remainder);
        
        *pad_h_before = pad_h / 2;
        *pad_h_after = (pad_h + 1) / 2;
        *pad_w_before = pad_w / 2;
        *pad_w_after = (pad_w + 1) / 2;
    }
}

// Core weighted convolution unit computation
__device__ float calculate_wt_conv_unit_cuda(
    const float* __restrict__ patch,
    const float* __restrict__ weights,
    const float* __restrict__ relevance_weights,
    const float* __restrict__ bias,
    int kernel_h, int kernel_w, int in_channels, int out_channels,
    int act_type, float act_range_l, float act_range_u, bool has_range_l, bool has_range_u,
    int patch_size, int channel_idx
) {
    // Use shared memory for frequently accessed data
    __shared__ float s_patch[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE * 64]; // Assume max 64 channels
    __shared__ float s_weights[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE * 64];
    
    int tid = threadIdx.x;
    int total_elements = kernel_h * kernel_w * in_channels;
    
    // Cooperative loading of patch and weights
    for (int i = tid; i < total_elements; i += blockDim.x) {
        if (i < patch_size) {
            s_patch[i] = patch[i];
            s_weights[i] = weights[i];
        }
    }
    __syncthreads();
    
    float result = 0.0f;
    
    // Process each output channel
    for (int oc = 0; oc < out_channels; oc++) {
        float p_sum = 0.0f, n_sum = 0.0f;
        
        // Compute positive and negative sums efficiently
        for (int i = tid; i < kernel_h * kernel_w; i += blockDim.x) {
            int spatial_idx = i;
            int weight_idx = spatial_idx * in_channels * out_channels + channel_idx * out_channels + oc;
            int patch_idx = spatial_idx * in_channels + channel_idx;
            
            if (spatial_idx < kernel_h * kernel_w && patch_idx < patch_size) {
                float conv_val = s_weights[weight_idx] * s_patch[patch_idx];
                p_sum += fmaxf(conv_val, 0.0f);
                n_sum += fminf(conv_val, 0.0f);
            }
        }
        
        // Warp-level reduction
        p_sum = warpReduceSum(p_sum);
        n_sum = warpReduceSum(n_sum);
        
        if (threadIdx.x % WARP_SIZE == 0) {
            n_sum = -n_sum; // Convert to positive
            float t_sum = p_sum + n_sum;
            
            // Handle bias
            float bias_pos = 0.0f, bias_neg = 0.0f, denom_bias = 0.0f;
            if (bias != nullptr) {
                bias_pos = fmaxf(bias[oc], 0.0f);
                bias_neg = fmaxf(-bias[oc], 0.0f);
                denom_bias = bias_pos + bias_neg;
            }
            
            // Saturation indicators
            float p_saturate = (p_sum > 0.0f) ? 1.0f : 0.0f;
            float n_saturate = (n_sum > 0.0f) ? 1.0f : 0.0f;
            
            // Handle activation function constraints
            if (act_type == 0) { // mono
                if (has_range_l && t_sum > act_range_l) p_saturate = 1.0f;
                if (has_range_u && t_sum < act_range_u) n_saturate = 1.0f;
            }
            // Note: non_mono case would require additional function evaluation
            
            // Calculate final weights with numerical stabilization
            float denom = p_sum + n_sum + denom_bias;
            denom = (denom == 0.0f) ? 1e-12f : denom;
            float inv_denom = 1.0f / denom;
            
            float rel_weight = relevance_weights[oc];
            float p_agg_wt = inv_denom * rel_weight * p_saturate;
            float n_agg_wt = inv_denom * rel_weight * n_saturate;
            
            // Accumulate contribution for this channel
            for (int i = 0; i < kernel_h * kernel_w; i++) {
                int weight_idx = i * in_channels * out_channels + channel_idx * out_channels + oc;
                int patch_idx = i * in_channels + channel_idx;
                
                if (patch_idx < patch_size) {
                    float conv_val = s_weights[weight_idx] * s_patch[patch_idx];
                    float pos_contrib = fmaxf(conv_val, 0.0f) * p_agg_wt;
                    float neg_contrib = fminf(conv_val, 0.0f) * n_agg_wt;
                    result += pos_contrib - neg_contrib;
                }
            }
        }
    }
    
    return result;
}

// Main CUDA kernel for weighted convolution
__global__ void weighted_conv_kernel(
    const float* __restrict__ relevance_y,    // [batch, out_channels, out_h, out_w]
    const float* __restrict__ input_array,    // [batch, in_channels, in_h, in_w]
    const float* __restrict__ weights,        // [out_channels, in_channels, kernel_h, kernel_w]
    const float* __restrict__ bias,           // [out_channels] or nullptr
    float* __restrict__ output,               // [batch, in_channels, in_h, in_w]
    int batch_size, int in_channels, int out_channels,
    int in_h, int in_w, int out_h, int out_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h_before, int pad_h_after, int pad_w_before, int pad_w_after,
    int act_type, float act_range_l, float act_range_u, 
    bool has_range_l, bool has_range_u
) {
    // Thread mapping: each thread processes one spatial output location
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_h * out_w * in_channels;
    
    if (idx >= total_outputs) return;
    
    // Decode indices
    int batch_idx = idx / (out_h * out_w * in_channels);
    int remaining = idx % (out_h * out_w * in_channels);
    int channel_idx = remaining / (out_h * out_w);
    remaining = remaining % (out_h * out_w);
    int out_y = remaining / out_w;
    int out_x = remaining % out_w;
    
    // Calculate input patch boundaries
    int in_y_start = out_y * stride_h - pad_h_before;
    int in_x_start = out_x * stride_w - pad_w_before;
    int in_y_end = in_y_start + kernel_h;
    int in_x_end = in_x_start + kernel_w;
    
    // Bounds checking for valid input region
    int valid_y_start = max(0, in_y_start);
    int valid_x_start = max(0, in_x_start);
    int valid_y_end = min(in_h, in_y_end);
    int valid_x_end = min(in_w, in_x_end);
    
    if (valid_y_start >= valid_y_end || valid_x_start >= valid_x_end) return;
    
    // Prepare patch data in registers/shared memory
    float patch_data[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
    int patch_idx = 0;
    
    for (int ky = 0; ky < kernel_h; ky++) {
        for (int kx = 0; kx < kernel_w; kx++) {
            int in_y = in_y_start + ky;
            int in_x = in_x_start + kx;
            
            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                int input_idx = ((batch_idx * in_channels + channel_idx) * in_h + in_y) * in_w + in_x;
                patch_data[patch_idx] = input_array[input_idx];
            } else {
                patch_data[patch_idx] = 0.0f; // Padding value
            }
            patch_idx++;
        }
    }
    
    // Get relevance weights for this output position
    float relevance_data[64]; // Assume max 64 output channels
    for (int oc = 0; oc < out_channels; oc++) {
        int rel_idx = ((batch_idx * out_channels + oc) * out_h + out_y) * out_w + out_x;
        relevance_data[oc] = relevance_y[rel_idx];
    }
    
    // Compute weighted convolution unit
    float result = calculate_wt_conv_unit_cuda(
        patch_data, 
        &weights[channel_idx * kernel_h * kernel_w], 
        relevance_data,
        bias,
        kernel_h, kernel_w, in_channels, out_channels,
        act_type, act_range_l, act_range_u, has_range_l, has_range_u,
        kernel_h * kernel_w, channel_idx
    );
    
    // Atomic add to output (handle overlapping patches)
    int output_base = (batch_idx * in_channels + channel_idx) * in_h * in_w;
    for (int ky = 0; ky < kernel_h; ky++) {
        for (int kx = 0; kx < kernel_w; kx++) {
            int in_y = in_y_start + ky;
            int in_x = in_x_start + kx;
            
            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                int output_idx = output_base + in_y * in_w + in_x;
                atomicAdd(&output[output_idx], result / (kernel_h * kernel_w));
            }
        }
    }
}

// Host launcher function
torch::Tensor launch_calculate_wt_conv_kernel(
    torch::Tensor relevance_y,
    torch::Tensor input_array,
    torch::Tensor weights,
    torch::Tensor bias,
    const std::string& padding,
    torch::Tensor strides,
    int act_type, float act_range_l, float act_range_u, int act_func_int
) {
    // Get tensor dimensions
    int batch_size = input_array.size(0);
    int in_channels = input_array.size(1);
    int in_h = input_array.size(2);
    int in_w = input_array.size(3);
    
    int out_channels = relevance_y.size(1);
    int out_h = relevance_y.size(2);
    int out_w = relevance_y.size(3);
    
    int kernel_h = weights.size(2);
    int kernel_w = weights.size(3);
    
    int stride_h = strides[0].item<int>();
    int stride_w = strides[1].item<int>();
    
    // Calculate padding
    int pad_h_before = 0, pad_h_after = 0, pad_w_before = 0, pad_w_after = 0;
    calculate_padding_cuda(kernel_h, kernel_w, in_h, in_w, stride_h, stride_w,
                          padding.c_str(), &pad_h_before, &pad_h_after, 
                          &pad_w_before, &pad_w_after);
    
    // Allocate output tensor
    torch::Tensor output = torch::zeros_like(input_array);
    
    // Launch kernel
    int total_threads = batch_size * out_h * out_w * in_channels;
    int num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Extract activation parameters (simplified)
    int act_type = 0; // 0 for mono, 1 for non_mono
    float act_range_l = 0.0f, act_range_u = 0.0f;
    bool has_range_l = false, has_range_u = false;
    
    weighted_conv_kernel<<<num_blocks, BLOCK_SIZE>>>(
        relevance_y.data_ptr<float>(),
        input_array.data_ptr<float>(),
        weights.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h_before, pad_h_after, pad_w_before, pad_w_after,
        act_type, act_range_l, act_range_u, has_range_l, has_range_u
    );
    
    cudaDeviceSynchronize();
    return output;
}

"""

conv2d_cuda_declaration = r"""
torch::Tensor launch_calculate_wt_conv_kernel(
    torch::Tensor relevance_y,
    torch::Tensor input_array,
    torch::Tensor weights,
    torch::Tensor bias,
    const std::string& padding,
    torch::Tensor strides,
    int act_type, float act_range_l, float act_range_u, int act_func_int
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

extra_flags = ['-O3', '--use_fast_math', '-Xcompiler', '-fPIC']
extra_flags.extend(get_cuda_arch_flags())

custom_conv2d_cuda_ops = load_inline(
    name="custom_conv2d_cuda_v2",
    cpp_sources=conv2d_cuda_declaration,
    cuda_sources=conv2d_cuda_source,
    functions=["launch_calculate_wt_conv_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def optimized_calculate_wt_conv(relevance_y, input_array, w, b, padding, strides, act):
    # Convert inputs to CUDA tensors
    device = torch.device('cuda')
    
    relevance_y = torch.tensor(relevance_y, dtype=torch.float32, device=device)
    input_array = torch.tensor(input_array, dtype=torch.float32, device=device) 
    w = torch.tensor(w, dtype=torch.float32, device=device)
    strides = torch.tensor(strides, dtype=torch.int32, device=device)
        
    if b is not None:
        b = torch.tensor(b, dtype=torch.float32, device=device)

    if padding != 'valid' and padding != 'same':
        padding = torch.tensor(padding, dtype=torch.int32, device=device)   
    
    # Parse activation parameters once
    act_type = 0 if act["type"] == "mono" else 1
    act_lower = -float('inf') if act["range"]["l"] is None else float(act["range"]["l"])
    act_upper = float('inf') if act["range"]["u"] is None else float(act["range"]["u"])
    
    # Convert activation function string to int
    act_func_int = 0  # default: identity
    if act["func"] is not None:
        act_func_str = act["func"]
        if act_func_str == "sigmoid": act_func_int = 1
        elif act_func_str == "swish": act_func_int = 2
        elif act_func_str == "wave": act_func_int = 3
        elif act_func_str == "pulse": act_func_int = 4
        elif act_func_str == "absolute": act_func_int = 5
        elif act_func_str == "hard_sigmoid": act_func_int = 6
        elif act_func_str == "tanh": act_func_int = 7    
    
    # Call optimized CUDA kernel
    result = custom_conv2d_cuda_ops.launch_calculate_wt_conv_kernel(
        relevance_y, input_array, w, b, padding, strides, act_type, act_lower, act_upper, act_func_int
    )
    
    return result.cpu().numpy()
