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

// Phase 1: Compute conv_out and extract p_ind/n_ind
__global__ void compute_conv_and_parts_kernel(
    const float* __restrict__ patch,     // (i, j, k)
    const float* __restrict__ w,         // (i, j, k, l)
    float* __restrict__ conv_out,        // (i, j, k, l)
    float* __restrict__ p_ind,           // (i, j, k, l)
    float* __restrict__ n_ind,           // (i, j, k, l)
    const int i_size,
    const int j_size,
    const int k_size,
    const int l_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = i_size * j_size * k_size * l_size;
    
    if (idx >= total_elements) return;
    
    // Decompose linear index
    int l = idx % l_size;
    int temp = idx / l_size;
    int k = temp % k_size;
    temp = temp / k_size;
    int j = temp % j_size;
    int i = temp / j_size;
    
    // Get indices
    int spatial_idx = i * j_size * k_size + j * k_size + k;
    int full_idx = spatial_idx * l_size + l;
    
    // Compute convolution: einsum("ijkl,ijk->ijkl")
    float patch_val = patch[spatial_idx];
    float w_val = w[full_idx];
    float conv_val = w_val * patch_val;
    
    // Store convolution output
    conv_out[full_idx] = conv_val;
    
    // Extract positive and negative parts
    p_ind[full_idx] = fmaxf(conv_val, 0.0f);
    n_ind[full_idx] = fminf(conv_val, 0.0f);
}

// Phase 2: Compute channel-wise sums (reduction over spatial dimensions)
__global__ void compute_channel_sums_kernel(
    const float* __restrict__ p_ind,     // (i, j, k, l)
    const float* __restrict__ n_ind,     // (i, j, k, l)
    float* __restrict__ p_sum,           // (l,)
    float* __restrict__ n_sum,           // (l,)
    const int i_size,
    const int j_size,
    const int k_size,
    const int l_size
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= l_size) return;
    
    float p_acc = 0.0f;
    float n_acc = 0.0f;
    
    // Sum over all spatial positions for this channel
    for (int i = 0; i < i_size; i++) {
        for (int j = 0; j < j_size; j++) {
            for (int k = 0; k < k_size; k++) {
                int idx = (i * j_size * k_size + j * k_size + k) * l_size + l;
                p_acc += p_ind[idx];
                n_acc += n_ind[idx];
            }
        }
    }
    
    p_sum[l] = p_acc;
    n_sum[l] = -n_acc;  // Convert to positive
}

// Phase 3: Compute weights and apply to get final output
__global__ void apply_weights_kernel(
    const float* __restrict__ p_ind,     // (i, j, k, l)
    const float* __restrict__ n_ind,     // (i, j, k, l)
    const float* __restrict__ p_sum,     // (l,)
    const float* __restrict__ n_sum,     // (l,)
    const float* __restrict__ wts,       // (l,)
    const float* __restrict__ b,         // (l,) or nullptr
    float* __restrict__ output,          // (i, j, k)
    const int i_size,
    const int j_size,
    const int k_size,
    const int l_size,
    const int act_type,
    const float act_lower,
    const float act_upper,
    const int act_func_int,
    const bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = i_size * j_size * k_size;
    
    if (idx >= total_spatial) return;
    
    float result = 0.0f;
    
    // Process each channel
    for (int l = 0; l < l_size; l++) {
        float p_sum_l = p_sum[l];
        float n_sum_l = n_sum[l];
        float t_sum_l = p_sum_l + n_sum_l;
        
        // Handle bias
        float bias_pos = 0.0f;
        float bias_neg = 0.0f;
        float denom_bias_term = 0.0f;
        
        if (has_bias && b != nullptr) {
            float bias_val = b[l];
            bias_pos = fmaxf(bias_val, 0.0f);
            bias_neg = fmaxf(-bias_val, 0.0f);
            denom_bias_term = bias_pos + bias_neg;
        }
        
        // Initialize saturation indicators
        float p_saturate = (p_sum_l > 0.0f) ? 1.0f : 0.0f;
        float n_saturate = (n_sum_l > 0.0f) ? 1.0f : 0.0f;
        
        // Handle activation function logic
        if (act_type == 0) {  // mono
            if (act_lower > -FLT_MAX) {
                float temp_ind = (t_sum_l > act_lower) ? 1.0f : 0.0f;
                p_saturate = temp_ind;
            }
            if (act_upper < FLT_MAX) {
                float temp_ind = (t_sum_l < act_upper) ? 1.0f : 0.0f;
                n_saturate = temp_ind;
            }
        } else if (act_type == 1) {  // non_mono
            float t_act = apply_activation(t_sum_l, act_func_int);
            float p_act = apply_activation(p_sum_l + bias_pos, act_func_int);
            float n_act = apply_activation(-(n_sum_l + bias_neg), act_func_int);
            
            // Apply range constraints
            if (act_lower > -FLT_MAX) {
                float temp_ind = (t_sum_l > act_lower) ? 1.0f : 0.0f;
                p_saturate = p_saturate * temp_ind;
            }
            if (act_upper < FLT_MAX) {
                float temp_ind = (t_sum_l < act_upper) ? 1.0f : 0.0f;
                n_saturate = n_saturate * temp_ind;
            }
            
            // Apply activation function difference thresholding
            float temp_ind = (fabsf(t_act - p_act) > 1e-5f) ? 1.0f : 0.0f;
            n_saturate = n_saturate * temp_ind;
            temp_ind = (fabsf(t_act - n_act) > 1e-5f) ? 1.0f : 0.0f;
            p_saturate = p_saturate * temp_ind;
        }
        
        // Calculate denominator with numerical stabilization
        float denom = p_sum_l + n_sum_l + denom_bias_term;
        if (denom == 0.0f) {
            denom = 1e-12f;
        }
        
        // Calculate aggregated weights
        float inv_denom = __fdividef(1.0f, denom);
        float p_agg_wt = inv_denom * wts[l] * p_saturate;
        float n_agg_wt = inv_denom * wts[l] * n_saturate;
        
        // Get the conv values for this spatial position and channel
        int full_idx = idx * l_size + l;
        float p_val = p_ind[full_idx];
        float n_val = n_ind[full_idx];
        
        // Compute weighted contribution
        float wt_contrib = p_val * p_agg_wt - n_val * n_agg_wt;
        result += wt_contrib;
    }
    
    output[idx] = result;
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
    
    auto patch_c = patch.contiguous();
    auto wts_c = wts.contiguous();
    auto w_c = w.contiguous();
    
    // Get tensor dimensions
    auto patch_sizes = patch_c.sizes();
    auto w_sizes = w_c.sizes();
    
    int i_size = patch_sizes[0];
    int j_size = patch_sizes[1];
    int k_size = patch_sizes[2];
    int l_size = w_sizes[3];
    
    bool has_bias = b.defined() && b.numel() > 0;
    
    // Allocate intermediate tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(patch.device());
    auto conv_out = torch::empty({i_size, j_size, k_size, l_size}, options);
    auto p_ind = torch::empty({i_size, j_size, k_size, l_size}, options);
    auto n_ind = torch::empty({i_size, j_size, k_size, l_size}, options);
    auto p_sum = torch::empty({l_size}, options);
    auto n_sum = torch::empty({l_size}, options);
    auto output = torch::zeros({i_size, j_size, k_size}, options);
    
    // Phase 1: Compute convolution and parts
    int total_elements = i_size * j_size * k_size * l_size;
    int threads1 = 256;
    int blocks1 = (total_elements + threads1 - 1) / threads1;
    
    compute_conv_and_parts_kernel<<<blocks1, threads1>>>(
        patch_c.data_ptr<float>(),
        w_c.data_ptr<float>(),
        conv_out.data_ptr<float>(),
        p_ind.data_ptr<float>(),
        n_ind.data_ptr<float>(),
        i_size, j_size, k_size, l_size
    );
    
    // Phase 2: Compute channel sums
    int threads2 = 256;
    int blocks2 = (l_size + threads2 - 1) / threads2;
    
    compute_channel_sums_kernel<<<blocks2, threads2>>>(
        p_ind.data_ptr<float>(),
        n_ind.data_ptr<float>(),
        p_sum.data_ptr<float>(),
        n_sum.data_ptr<float>(),
        i_size, j_size, k_size, l_size
    );
    
    // Phase 3: Apply weights and compute output
    int total_spatial = i_size * j_size * k_size;
    int threads3 = 256;
    int blocks3 = (total_spatial + threads3 - 1) / threads3;
    
    const float* b_ptr = has_bias ? b.data_ptr<float>() : nullptr;
    
    apply_weights_kernel<<<blocks3, threads3>>>(
        p_ind.data_ptr<float>(),
        n_ind.data_ptr<float>(),
        p_sum.data_ptr<float>(),
        n_sum.data_ptr<float>(),
        wts_c.data_ptr<float>(),
        b_ptr,
        output.data_ptr<float>(),
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

    b = b if b is not None else torch.empty(0, dtype=torch.float32)

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