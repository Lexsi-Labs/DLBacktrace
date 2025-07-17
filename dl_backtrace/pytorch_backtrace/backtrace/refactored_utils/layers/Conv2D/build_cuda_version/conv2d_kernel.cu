#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void calculate_wt_conv_cuda_kernel(
    const float* patches_ptr,           // (L, K_h, K_w, C)     Same as patch in the original implementation
    const float* kernel_weights_ptr,    // (1, K_h, K_w, C, F)  Same as w in the original implementation
    const float* bias_ptr,              // (F,)                 Same as b in the original implementation
    const float* grad_scales_ptr,       // (L, F)               Same as wts in the original implementation
    float* updates_ptr,                 // (L, K_h, K_w, C) - output
    const int L_patches,
    const int K_h, const int K_w, const int C_in, const int F_dim,
    const int act_type, 
    const int act_func,                // 0=mono, 1=non_mono
    const float act_range_l, const float act_range_u,
    const bool has_range_l, const bool has_range_u,
    const float epsilon = 1e-5f
) {
    // Thread organization: blockIdx.x = patch_idx, threadIdx.x = feature_idx
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = L_patches * F_dim;
    
    if (idx >= total_threads) return;

    const int patch_idx = idx / F_dim;
    const int feature_idx = idx % F_dim;
    
    // Per-thread accumulators for current patch and feature
    float sum_positive = 0.0f;
    float sum_abs_negative = 0.0f;
    
    // Compute convolution for current patch-feature pair
    for (int kh = 0; kh < K_h; kh++) {
        for (int kw = 0; kw < K_w; kw++) {
            for (int c = 0; c < C_in; c++) {
                const int patch_offset = patch_idx * (K_h * K_w * C_in) + 
                                       kh * (K_w * C_in) + kw * C_in + c;
                const int kernel_offset = (kh * K_w * C_in + kw * C_in + c) * F_dim + feature_idx;
                
                const float patch_val = patches_ptr[patch_offset];
                const float kernel_val = kernel_weights_ptr[kernel_offset];
                const float conv_val = patch_val * kernel_val;
                
                // Accumulate positive and negative parts
                if (conv_val > 0.0f) {
                    sum_positive += conv_val;
                } 
                else {
                    sum_abs_negative += -conv_val;
                }
            }
        }
    }
    
    const float sum_abs_total = sum_positive + sum_abs_negative;
    
    // Bias processing
    const float bias_val = bias_ptr[feature_idx];
    const float positive_bias = fmaxf(bias_val, 0.0f);
    const float abs_negative_bias = fmaxf(-bias_val, 0.0f);
    
    // Initialize masks based on sum values - following reference implementation
    bool pos_mask = sum_positive > 0.0f;
    bool neg_mask = sum_abs_negative > 0.0f;
    
    // Activation logic - match original implementation sequence exactly
    if (act_type == 0) { // mono
        // Range logic for mono type - REPLACE masks (not AND), following reference
        if (has_range_l) {
            pos_mask = sum_abs_total > act_range_l;  // REPLACE, not AND
        }
        if (has_range_u) {
            neg_mask = sum_abs_total < act_range_u;  // REPLACE, not AND
        }
    } 
    
    else if (act_type == 1) { // non_mono
        // Store original values for activation computation
        float orig_p_sum = sum_positive;
        float orig_n_sum = sum_abs_negative;
        
        // Compute activation functions with original values (like original implementation)
        float t_act, p_act, n_act;
        switch (act_func) {
            case 1: // ReLU
                t_act = fmaxf(0.0f, sum_abs_total);
                p_act = fmaxf(0.0f, orig_p_sum + positive_bias);
                // Fix: Use -1.0f * (orig_n_sum + abs_negative_bias), not -(orig_n_sum + abs_negative_bias)
                n_act = fmaxf(0.0f, -1.0f * (orig_n_sum + abs_negative_bias));
                break;
            case 2: // Sigmoid
                t_act = 1.0f / (1.0f + expf(-sum_abs_total));
                p_act = 1.0f / (1.0f + expf(-(orig_p_sum + positive_bias)));
                // Fix: Use -1.0f * (orig_n_sum + abs_negative_bias), not -(orig_n_sum + abs_negative_bias)
                n_act = 1.0f / (1.0f + expf(-(-1.0f * (orig_n_sum + abs_negative_bias))));
                break;
            default: // Identity
                t_act = sum_abs_total;
                p_act = orig_p_sum + positive_bias;
                // Fix: Use -1.0f * (orig_n_sum + abs_negative_bias), not -(orig_n_sum + abs_negative_bias)
                n_act = -1.0f * (orig_n_sum + abs_negative_bias);
                break;
        }
        
        // Apply range bounds first - AND with existing masks (following reference for non_mono)
        if (has_range_l) {
            bool range_mask_l = sum_abs_total > act_range_l;
            pos_mask = pos_mask && range_mask_l;
        }
        if (has_range_u) {
            bool range_mask_u = sum_abs_total < act_range_u;
            neg_mask = neg_mask && range_mask_u;
        }
        
        // Apply activation-based logic INDEPENDENTLY (not conditionally)
        // Following reference: temp_ind = np.abs(t_act - p_act)>1e-5; n_saturate = n_saturate*temp_ind
        bool neg_check = fabsf(t_act - p_act) > epsilon;
        neg_mask = neg_mask && neg_check;
        
        // Following reference: temp_ind = np.abs(t_act - n_act)>1e-5; p_saturate = p_saturate*temp_ind  
        bool pos_check = fabsf(t_act - n_act) > epsilon;
        pos_mask = pos_mask && pos_check;
    }
    
    // Weight computation with numerical stability
    const float denominator = sum_abs_total + positive_bias + abs_negative_bias;
    const float inv_denom = 1.0f / denominator;
    const float grad_scale = grad_scales_ptr[patch_idx * F_dim + feature_idx];
    
    // Check for numerical stability in weight calculations
    const float pos_weight = inv_denom * grad_scale * pos_mask;
    const float neg_weight = inv_denom * grad_scale * neg_mask;
    
    // Compute and store updates for current patch
    for (int kh = 0; kh < K_h; kh++) {
        for (int kw = 0; kw < K_w; kw++) {
            for (int c = 0; c < C_in; c++) {
                const int patch_offset = patch_idx * (K_h * K_w * C_in) + 
                                       kh * (K_w * C_in) + kw * C_in + c;
                const int kernel_offset = (kh * K_w * C_in + kw * C_in + c) * F_dim + feature_idx;
                const int update_offset = patch_idx * (K_h * K_w * C_in) + 
                                        kh * (K_w * C_in) + kw * C_in + c;
                
                const float patch_val = patches_ptr[patch_offset];
                const float kernel_val = kernel_weights_ptr[kernel_offset];
                const float conv_val = patch_val * kernel_val;
                
                float update_val = 0.0f;
                if (conv_val > 0.0f) {
                    update_val = conv_val * pos_weight;
                } else {
                    update_val = -conv_val * neg_weight; // Note: conv_val is negative
                }
                
                atomicAdd(&updates_ptr[update_offset], update_val);
            }
        }
    }
}

torch::Tensor calculate_wt_conv_cuda(
    const torch::Tensor& patches,
    const torch::Tensor& kernel_weights,
    const torch::Tensor& bias,
    const torch::Tensor& grad_scales,
    const int L_patches,
    const int K_h, const int K_w, const int C_in_dim, const int F_dim,
    const int act_type,                 // 0=mono, 1=non_mono
    const int act_func,
    const float act_range_l, const float act_range_u,
    const bool has_range_l, const bool has_range_u
) {
    // Input validation
    TORCH_CHECK(patches.dim() == 4, "patches must be 4D");
    TORCH_CHECK(kernel_weights.dim() == 5, "kernel_weights must be 5D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(grad_scales.dim() == 2, "grad_scales must be 2D");

    // Create output tensor
    auto result = torch::zeros_like(patches);

    // Launch kernel
    const int total_threads = L_patches * F_dim;
    const int threads_per_block = 32;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    calculate_wt_conv_cuda_kernel<<<blocks, threads_per_block>>>(
        patches.data_ptr<float>(),
        kernel_weights.data_ptr<float>(),
        bias.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        result.data_ptr<float>(),
        L_patches, K_h, K_w, C_in_dim, F_dim,
        act_type, act_func, act_range_l, act_range_u, has_range_l, has_range_u
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return result;   
}
