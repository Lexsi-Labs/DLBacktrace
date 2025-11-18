import torch
from torch.utils.cpp_extension import load_inline
import numpy as np


linear_layer_cuda_source = r"""
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

// Warp-level reduction for better performance
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using warp primitives
__device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    // Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Write reduced value to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Final reduction for first warp
    if (threadIdx.x < blockDim.x / 32) {
        val = shared[threadIdx.x];
    } else {
        val = 0.0f;
    }
    
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

__global__ void calculate_wt_fc_kernel(
    const float* __restrict__ relevance_y,
    const float* __restrict__ input_array,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ relevance_x,
    const int input_dim,
    const int output_dim,
    const int activation_kind, 
    const float act_lower_bound,
    const float act_upper_bound,
    const bool has_bias,
    const int act_func  
) {
    // Grid-stride loop for better occupancy and coalescing
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    
    // Shared memory for reductions (sized for max 32 warps)
    __shared__ float reduction_shared[32];
    __shared__ float p_sum_final;
    __shared__ float n_sum_final;
    __shared__ float relevance_val_shared;
    __shared__ float p_agg_wt_shared;
    __shared__ float n_agg_wt_shared;
    
    // Process multiple outputs per block using grid-stride loop
    for (int output_idx = bid; output_idx < output_dim; output_idx += gridDim.x) {
        
        // Initialize accumulators
        float local_p_sum = 0.0f;
        float local_n_sum = 0.0f;
        
        const int weight_offset = output_idx * input_dim;
        
        // Vectorized memory access when possible
        const int vec_size = 4;
        const int vec_input_dim = input_dim / vec_size;
        
        // Process vectorized elements
        if ((input_dim % vec_size == 0) && (tid * vec_size < input_dim)) {
            for (int i = tid; i < vec_input_dim; i += block_size) {
                int idx = i * vec_size;
                if (idx < input_dim) {
                    // Load 4 elements at once using float4
                    float4 input_vec = *reinterpret_cast<const float4*>(&input_array[idx]);
                    float4 weight_vec = *reinterpret_cast<const float4*>(&w[weight_offset + idx]);
                    
                    // Process vectorized data
                    #pragma unroll
                    for (int j = 0; j < vec_size; j++) {
                        float contrib = ((float*)&input_vec)[j] * ((float*)&weight_vec)[j];
                        local_p_sum += fmaxf(contrib, 0.0f);
                        local_n_sum += fmaxf(-contrib, 0.0f);
                    }
                }
            }
        } else {
            // Fallback for non-aligned data
            for (int i = tid; i < input_dim; i += block_size) {
                float input_val = input_array[i];
                float weight_val = w[weight_offset + i];
                float contrib = input_val * weight_val;
                local_p_sum += fmaxf(contrib, 0.0f);
                local_n_sum += fmaxf(-contrib, 0.0f);
            }
        }
        
        // Efficient block reduction
        p_sum_final = block_reduce_sum(local_p_sum, reduction_shared);
        __syncthreads();
        n_sum_final = block_reduce_sum(local_n_sum, reduction_shared);
        __syncthreads();
        
        // Thread 0 computes final values
        if (tid == 0) {
            float p_sum = p_sum_final;
            float n_sum = n_sum_final;
            
            // Handle bias
            float pbias = 0.0f, nbias = 0.0f;
            if (has_bias && b) {
                float bias_val = b[output_idx];
                pbias = fmaxf(bias_val, 0.0f);
                nbias = -fminf(bias_val, 0.0f);
            }
            
            // Compute total sum
            float t_sum = p_sum + pbias - n_sum - nbias;
            
            if (activation_kind == 0) {
                if (t_sum < act_lower_bound) p_sum = 0.0f;
                if (t_sum > act_upper_bound) n_sum = 0.0f;
            } 
            
            else if (activation_kind == 1) {
                // Activation-specific handling
                float t_act = apply_activation(t_sum, act_func);
                float p_act = apply_activation(p_sum + pbias, act_func);
                float n_act = apply_activation(-(n_sum + nbias), act_func);

                if (t_sum < act_lower_bound) p_sum = 0.0f;
                if (t_sum > act_upper_bound) n_sum = 0.0f;

                if (p_sum > 0.0f && n_sum > 0.0f) {
                    if (t_act == p_act) {
                        n_sum = 0.0f;
                    } else if (t_act == n_act) {
                        p_sum = 0.0f;
                    }
                }
            }
            
            float p_agg_wt = 0.0f;
            float n_agg_wt = 0.0f;
            
            if (p_sum > 0.0f) {
                float total_sum = p_sum + n_sum + pbias + nbias;
                float total_psum = p_sum + pbias;
                p_agg_wt = (total_psum / total_sum) * (p_sum / total_psum);
            }
            
            if (n_sum > 0.0f) {
                float total_sum = p_sum + n_sum + pbias + nbias;
                float total_nsum = n_sum + nbias;
                n_agg_wt = (total_nsum / total_sum) * (n_sum / total_nsum);
            }
            
            float p_sum_div = (p_sum == 0.0f) ? 1.0f : p_sum;
            float n_sum_div = (n_sum == 0.0f) ? 1.0f : n_sum;
            
            // Store in shared memory for all threads
            relevance_val_shared = relevance_y[output_idx];
            p_agg_wt_shared = p_agg_wt * relevance_val_shared;
            n_agg_wt_shared = n_agg_wt * relevance_val_shared;
            p_sum_final = p_sum_div;
            n_sum_final = n_sum_div;
        }
        __syncthreads();
        
        // Load shared values
        float p_agg_wt_val = p_agg_wt_shared;
        float n_agg_wt_val = n_agg_wt_shared;
        float p_sum_div = p_sum_final;
        float n_sum_div = n_sum_final;
        
        // Second pass with coalesced writes
        for (int i = tid; i < input_dim; i += block_size) {
            float input_val = input_array[i];
            float weight_val = w[weight_offset + i];
            float contrib = input_val * weight_val;
            float weight = 0.0f;
            
            if (contrib > 0.0f) {
                weight = (contrib / p_sum_div) * p_agg_wt_val;
            } else if (contrib < 0.0f) {
                weight = (-contrib / n_sum_div) * n_agg_wt_val;
            }
            
            if (weight != 0.0f) {
                atomicAdd(&relevance_x[i], weight);
            }
        }
        __syncthreads();
    }
}

torch::Tensor launch_calculate_wt_fc_kernel(
    const torch::Tensor& relevance_y,
    const torch::Tensor& input_array,
    const torch::Tensor& w,
    const torch::Tensor& b,
    int act_type,
    float act_lower,
    float act_upper,
    int act_func_int
) {
    // Input validation
    TORCH_CHECK(relevance_y.is_cuda(), "relevance_y must be CUDA tensor");
    TORCH_CHECK(input_array.is_cuda(), "input_array must be CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA tensor");
    TORCH_CHECK(relevance_y.dim() == 1, "relevance_y must be 1D tensor");
    TORCH_CHECK(input_array.dim() == 1, "input_array must be 1D tensor");
    TORCH_CHECK(w.dim() == 2, "w must be 2D tensor");
    
    auto relevance_y_c = relevance_y.contiguous();
    auto input_array_c = input_array.contiguous();
    auto w_c = w.contiguous();

    auto input_dim = input_array_c.size(0);
    auto output_dim = relevance_y_c.size(0);

    TORCH_CHECK(w_c.size(0) == output_dim, "w must have output_dim rows");
    TORCH_CHECK(w_c.size(1) == input_dim, "w must have input_dim columns");
    
    // Allocate output tensor
    auto relevance_x = torch::zeros({input_dim}, 
        torch::TensorOptions().dtype(input_array.dtype()).device(input_array.device()));
    
    bool has_bias = b.defined() && b.numel() > 0;
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    dim3 grid_dim(output_dim);  // x: output
    
    // Launch kernel
    calculate_wt_fc_kernel<<<grid_dim, threads_per_block>>>(
        relevance_y.data_ptr<float>(),
        input_array.data_ptr<float>(),
        w.data_ptr<float>(),
        has_bias ? b.contiguous().data_ptr<float>() : nullptr,
        relevance_x.data_ptr<float>(),
        input_dim,
        output_dim,
        act_type,
        act_lower,
        act_upper,
        has_bias,
        act_func_int
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    cudaDeviceSynchronize();

    return relevance_x;
}
"""

# Simplified C++ declaration (no pybind11 includes needed)
linear_layer_cuda_declaration = r"""
torch::Tensor launch_calculate_wt_fc_kernel(
    const torch::Tensor& relevance_y,
    const torch::Tensor& input_array,
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

custom_linear_layer_cuda_ops = load_inline(
    name="linear_layer_cuda_v3",
    cpp_sources=linear_layer_cuda_declaration,
    cuda_sources=linear_layer_cuda_source,
    functions=["launch_calculate_wt_fc_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def calculate_wt_fc_cuda(relevance_y, input_array, w, b, act):
    """
    CUDA-accelerated version that maintains the original algorithm structure.
    Handles batch processing and multi-dimensional inputs in Python,
    delegates core computation to CUDA kernel.
    
    Args:
        relevance_y: relevance at the output (same shape as linear output)
        input_array: input to the linear layer (can be any shape: [B, D], [B, T, D], etc.)
        w: weight matrix of the linear layer (shape: [out_dim, in_dim])
        b: bias vector (shape: [out_dim]) or None
        act: dict containing activation info with keys: "type", "range", "func"

    Returns:
        relevance_x: relevance at the input, same shape as input_array
    """
    # Validate inputs
    if relevance_y is None or input_array is None or w is None:
        print(f"[CUDA ERROR] One or more inputs is None")
        return None
        
    # Flatten input except for last dim (same as original)
    original_shape = input_array.shape
    batch_dims = original_shape[:-1]
    feature_dim = original_shape[-1]

    input_flat = input_array.reshape(-1, feature_dim)
    relevance_flat = relevance_y.reshape(-1, relevance_y.shape[-1])

    # Process each batch element individually (maintains original logic)
    relevance_x_flat = []
    cuda_device = torch.device("cuda")
    
    # Convert weights to CUDA once (they're the same for all batch elements)
    w_torch = torch.tensor(w, dtype=torch.float32, device=cuda_device)
    b_torch = torch.tensor(b, dtype=torch.float32, device=cuda_device) if b is not None else torch.empty(0, device=cuda_device)
    
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
    
    for i in range(input_flat.shape[0]):
        inp = input_flat[i]            # shape: (input_dim,)
        wts = relevance_flat[i]        # shape: (output_dim,)
        
        # Convert to CUDA tensors for single batch element
        inp_torch = torch.tensor(inp, dtype=torch.float32, device=cuda_device)
        wts_torch = torch.tensor(wts, dtype=torch.float32, device=cuda_device)
        
        cuda_function = custom_linear_layer_cuda_ops.launch_calculate_wt_fc_kernel
        
        # Call CUDA kernel for single batch element with simplified parameters
        try:
            result = cuda_function(
                wts_torch, inp_torch, w_torch, b_torch,
                act_type, act_lower, act_upper, act_func_int
            )

            torch.cuda.synchronize()
            
            if result is None:
                print(f"[CUDA ERROR] Kernel returned None for batch element {i}")
                return None
                
            relevance_x_flat.append(result.cpu().numpy())
        except Exception as e:
            print(f"[CUDA ERROR] Kernel failed for batch element {i}: {e}")
            return None
    
    # Reshape back to original dimensions
    relevance_x_flat = np.array(relevance_x_flat)
    relevance_x = relevance_x_flat.reshape(*batch_dims, feature_dim)
    return relevance_x
