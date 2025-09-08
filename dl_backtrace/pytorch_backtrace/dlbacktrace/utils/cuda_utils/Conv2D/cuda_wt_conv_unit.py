import torch
from torch.utils.cpp_extension import load_inline
import numpy as np


wt_conv_unit_cuda_source = r"""
#include <torch/extension.h>
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

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

__global__ void calculate_wt_conv_unit_kernel(
    const float* __restrict__ patch,     // (i, j, k)
    const float* __restrict__ wts,       // (l,)
    const float* __restrict__ w,         // (i, j, k, l)
    const float* __restrict__ b,         // (l,) or nullptr
    float* __restrict__ output,          // (i, j, k)
    const int i_size,
    const int j_size, 
    const int k_size,
    const int l_size,
    const int act_type,                  // 0: mono, 1: non_mono
    const float act_lower,
    const float act_upper,
    const int act_func_int,
    const bool has_bias
) {
    // 3D thread mapping to spatial dimensions
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= i_size || j >= j_size || k >= k_size) return;
    
    // Calculate spatial position index
    int spatial_idx = i * j_size * k_size + j * k_size + k;
    
    // Get patch value at this spatial position
    float patch_val = patch[spatial_idx];
    
    // Shared memory for channel-wise reductions
    extern __shared__ float shared_mem[];
    float* s_p_sum = shared_mem;                    // l_size floats
    float* s_n_sum = s_p_sum + l_size;             // l_size floats
    float* s_conv_out = s_n_sum + l_size;          // l_size floats
    
    // Initialize shared memory for this thread block
    int tid = threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    
    // Each thread initializes part of shared memory
    for (int l = tid; l < l_size; l += threads_per_block) {
        s_p_sum[l] = 0.0f;
        s_n_sum[l] = 0.0f;
    }
    __syncthreads();
    
    // Compute convolution and accumulate positive/negative parts
    float result = 0.0f;
    
    // Phase 1: Compute convolution outputs and accumulate channel statistics
    for (int l = 0; l < l_size; l++) {
        int w_idx = spatial_idx * l_size + l;
        float conv_val = w[w_idx] * patch_val;
        
        // Store convolution output for later use
        if (tid == 0) s_conv_out[l] = conv_val;
        
        // Accumulate positive and negative parts atomically
        if (conv_val > 0.0f) {
            atomicAdd(&s_p_sum[l], conv_val);
        } else if (conv_val < 0.0f) {
            atomicAdd(&s_n_sum[l], -conv_val);  // Store as positive
        }
    }
    __syncthreads();
    
    // Phase 2: Only first thread computes activation logic and final weights
    if (tid == 0) {
        // Compute total sums and bias terms
        float p_sum_total = 0.0f, n_sum_total = 0.0f;
        float bias_pos_total = 0.0f, bias_neg_total = 0.0f;
        
        for (int l = 0; l < l_size; l++) {
            p_sum_total += s_p_sum[l];
            n_sum_total += s_n_sum[l];
            
            if (has_bias && b) {
                float bias_val = b[l];
                bias_pos_total += fmaxf(bias_val, 0.0f);
                bias_neg_total += fmaxf(-bias_val, 0.0f);
            }
        }
        
        float t_sum = p_sum_total + n_sum_total;
        float denom_bias_term = bias_pos_total + bias_neg_total;
        
        // Compute activation-specific saturation indicators
        float p_saturate_base = (p_sum_total > 0.0f) ? 1.0f : 0.0f;
        float n_saturate_base = (n_sum_total > 0.0f) ? 1.0f : 0.0f;
        
        for (int l = 0; l < l_size; l++) {
            float p_saturate = p_saturate_base;
            float n_saturate = n_saturate_base;
            
            if (act_type == 0) {  // mono
                if (act_lower > -FLT_MAX) {
                    float temp_ind = (t_sum > act_lower) ? 1.0f : 0.0f;
                    p_saturate = temp_ind;
                }
                if (act_upper < FLT_MAX) {
                    float temp_ind = (t_sum < act_upper) ? 1.0f : 0.0f;
                    n_saturate = temp_ind;
                }
            } else {  // non_mono
                float t_act = apply_activation(t_sum, act_func_int);
                float p_act = apply_activation(s_p_sum[l] + bias_pos_total, act_func_int);
                float n_act = apply_activation(-(s_n_sum[l] + bias_neg_total), act_func_int);
                
                // Apply range constraints
                if (act_lower > -FLT_MAX) {
                    float temp_ind = (t_sum > act_lower) ? 1.0f : 0.0f;
                    p_saturate *= temp_ind;
                }
                if (act_upper < FLT_MAX) {
                    float temp_ind = (t_sum < act_upper) ? 1.0f : 0.0f;
                    n_saturate *= temp_ind;
                }
                
                // Apply activation function difference thresholding
                float temp_ind = (fabsf(t_act - p_act) > 1e-5f) ? 1.0f : 0.0f;
                n_saturate *= temp_ind;
                temp_ind = (fabsf(t_act - n_act) > 1e-5f) ? 1.0f : 0.0f;
                p_saturate *= temp_ind;
            }
            
            // Calculate denominator with numerical stabilization
            float denom = s_p_sum[l] + s_n_sum[l] + denom_bias_term;
            if (denom == 0.0f) denom = 1e-12f;
            
            // Calculate aggregated weights
            float inv_denom = __fdividef(1.0f, denom);
            float p_agg_wt = inv_denom * wts[l] * p_saturate;
            float n_agg_wt = inv_denom * wts[l] * n_saturate;
            
            // Compute contribution to final result
            float p_contrib = (s_conv_out[l] > 0.0f) ? s_conv_out[l] * p_agg_wt : 0.0f;
            float n_contrib = (s_conv_out[l] < 0.0f) ? s_conv_out[l] * n_agg_wt : 0.0f;
            
            result += p_contrib - n_contrib;
        }
    }
    __syncthreads();
    
    // Write result to global memory
    if (tid == 0) {
        output[spatial_idx] = result;
    }
}

torch::Tensor launch_calculate_wt_conv_unit_kernel(
    const torch::Tensor& patch,
    const torch::Tensor& wts,
    const torch::Tensor& w,
    const torch::Tensor& b,
    int act_type,
    float act_lower,
    float act_upper,
    int act_func_int
) {
    // Validate inputs
    TORCH_CHECK(patch.is_cuda(), "patch must be CUDA tensor");
    TORCH_CHECK(wts.is_cuda(), "wts must be CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA tensor");
    TORCH_CHECK(patch.dtype() == torch::kFloat32, "patch must be float32");
    TORCH_CHECK(wts.dtype() == torch::kFloat32, "wts must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    
    // Get tensor dimensions
    auto patch_sizes = patch.sizes();
    auto w_sizes = w.sizes();
    
    int i_size = patch_sizes[0];
    int j_size = patch_sizes[1]; 
    int k_size = patch_sizes[2];
    int l_size = w_sizes[3];

    bool has_bias = b.defined() && b.numel() > 0;

    // Create output tensor
    auto output = torch::zeros({i_size, j_size, k_size}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(patch.device()));
    
    // Calculate grid and block dimensions
    dim3 threads_per_block(8, 8, 8);
    dim3 grid_dim(
        (i_size + threads_per_block.x - 1) / threads_per_block.x,
        (j_size + threads_per_block.y - 1) / threads_per_block.y,
        (k_size + threads_per_block.z - 1) / threads_per_block.z
    );
    
    // Calculate shared memory size
    int shared_mem_size = 3 * l_size * sizeof(float);
    
    // Get data pointers
    const float* patch_ptr = patch.data_ptr<float>();
    const float* wts_ptr = wts.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* b_ptr = has_bias ? b.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();
    
    // Launch kernel
    calculate_wt_conv_unit_kernel<<<grid_dim, threads_per_block, shared_mem_size>>>(
        patch_ptr, wts_ptr, w_ptr, b_ptr, output_ptr,
        i_size, j_size, k_size, l_size,
        act_type, act_lower, act_upper, act_func_int,
        has_bias
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    cudaDeviceSynchronize();

    return output;
}
"""

# Simplified C++ declaration
wt_conv_unit_cuda_declaration = r"""
torch::Tensor launch_calculate_wt_conv_unit_kernel(
    const torch::Tensor& patch,
    const torch::Tensor& wts,
    const torch::Tensor& w,
    const torch::Tensor& b,
    int act_type,
    float act_lower,
    float act_upper,
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
    # '-Xcompiler', '-fPIC',
    # '-Xptxas', '-dlcm=cg',
    # '-Xptxas', '-dscm=wt',
]

extra_flags.extend(get_cuda_arch_flags())

custom_wt_conv_unit_cuda_ops = load_inline(
    name="wt_conv_unit_cuda_v2",
    cpp_sources=wt_conv_unit_cuda_declaration,
    cuda_sources=wt_conv_unit_cuda_source,
    functions=["launch_calculate_wt_conv_unit_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def calculate_wt_conv_unit_cuda(patch, wts, w, b, act):
    """
    CUDA-optimized version of calculate_wt_conv_unit function.
    
    Args:
        patch: Input patch data of shape (i, j, k)
        wts: Weight values to be applied of shape (l,)
        w: Convolution kernel weights of shape (i, j, k, l)
        b: Optional bias tensor of shape (l,). If None, no bias is applied
        act: Dictionary containing activation function parameters
    
    Returns:
        torch::Tensor: Computed weight matrix of shape (i, j, k)
    """

    # Ensure tensors are contiguous
    patch = patch.contiguous()
    wts = wts.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()
    else:
        b = torch.empty(0, dtype=torch.float32)

    # Parse activation parameters once
    act_type = 0 if act["type"] == "mono" else 1
    act_lower = -float('inf') if act["range"].get("l") is None else float(act["range"]["l"])
    act_upper = float('inf') if act["range"].get("u") is None else float(act["range"]["u"])
    
    # Convert activation function to int
    act_func_int = 0  # default: identity
    if act.get("func") is not None:
        func_name = act["func"].__name__ if callable(act["func"]) else str(act["func"])
        if func_name == "sigmoid": act_func_int = 1
        elif func_name == "swish": act_func_int = 2  
        elif func_name == "wave": act_func_int = 3
        elif func_name == "pulse": act_func_int = 4
        elif func_name == "absolute": act_func_int = 5
        elif func_name == "hard_sigmoid": act_func_int = 6
        elif func_name == "tanh": act_func_int = 7

    return custom_wt_conv_unit_cuda_ops.launch_calculate_wt_conv_unit_kernel(
        patch, wts, w, b, act_type, act_lower, act_upper, act_func_int
    )