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
    // Each block handles one output dimension for a batch element
    const int output_idx = blockIdx.x;
    
    if (output_idx >= output_dim) return;

    // Use only fixed-size shared memory for reductions
    __shared__ float p_sum_shared[256];
    __shared__ float n_sum_shared[256];

    // Initialize thread-local accumulators
    float local_p_sum = 0.0f;
    float local_n_sum = 0.0f;

    int mul_val = output_idx * input_dim;

    // First pass: compute positive and negative sums
    for (int i = threadIdx.x; i < input_dim; i += blockDim.x) {
        float contrib = input_array[i] * w[mul_val + i];
        local_p_sum += fmaxf(contrib, 0.0f);
        local_n_sum += fmaxf(-contrib, 0.0f);
    }

    p_sum_shared[threadIdx.x] = local_p_sum;
    n_sum_shared[threadIdx.x] = local_n_sum;
    __syncthreads();

    // Parallel reduction for positive and negative sums
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            p_sum_shared[threadIdx.x] += p_sum_shared[threadIdx.x + stride];
            n_sum_shared[threadIdx.x] += n_sum_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float p_sum = p_sum_shared[0];
    float n_sum = n_sum_shared[0];
    
    // Handle bias
    float pbias = 0.0f, nbias = 0.0f;
    if (has_bias && b) {
        float bias_val = b[output_idx];
        pbias = fmaxf(bias_val, 0.0f);
        nbias = -fminf(bias_val, 0.0f);
    }
    
    // Compute total sum
    float t_sum = p_sum + pbias - n_sum - nbias;
    
    if (t_sum < act_lower_bound) p_sum = 0.0f;
    if (t_sum > act_upper_bound) n_sum = 0.0f;

    // Activation-specific handling
    if (activation_kind == 1) { // Non-monotonic activation
        
        // Apply activation functions
        float t_act = apply_activation(t_sum, act_func);
        float p_act = apply_activation(p_sum + pbias, act_func);
        float n_act = apply_activation(-(n_sum + nbias), act_func);
        
        // Check if both positive and negative contributions exist
        if (p_sum > 0.0f && n_sum > 0.0f) {
            if (t_act == p_act) {
                n_sum = 0.0f;  // Total matches positive part
            } else if (t_act == n_act) {
                p_sum = 0.0f;  // Total matches negative part
            }
        }
    }
    
    // Stabilize sums to avoid division by zero
    if (p_sum == 0.0f) p_sum = 1.0f;
    if (n_sum == 0.0f) n_sum = 1.0f;
    
    // Compute total sum with stabilization
    float total_sum = p_sum + n_sum + pbias + nbias;
    float total_psum = p_sum + pbias;
    float total_nsum = n_sum + nbias;

    // Optimized division
    float total_sum_inv = __fdividef(1.0f, total_sum);
    float total_psum_inv = __fdividef(1.0f, total_psum);
    float total_nsum_inv = __fdividef(1.0f, total_nsum);
    
    // Compute aggregated weights with stabilization
    float p_agg_wt = (p_sum > 0.0f) ? 
        (total_psum * total_sum_inv) * (p_sum * total_psum_inv) : 
        0.0f;
    
    float n_agg_wt = (n_sum > 0.0f) ? 
        (total_nsum * total_sum_inv) * (n_sum * total_nsum_inv) : 
        0.0f;
    
    // Get relevance value for this output
    float relevance_val = relevance_y[output_idx];
    float p_agg_wt_val = __fmul_rn(p_agg_wt, relevance_val);
    float n_agg_wt_val = __fmul_rn(n_agg_wt, relevance_val);
    
    // Second pass: compute and accumulate final weights (recompute contributions on-the-fly)
    for (int i = threadIdx.x; i < input_dim; i += blockDim.x) {
        float contrib = input_array[i] * w[mul_val + i];
        float weight = 0.0f;

        if (contrib > 0.0f) {
            weight = (contrib / p_sum) * p_agg_wt_val;
        } else if (contrib < 0.0f) {
            weight = (contrib / n_sum) * n_agg_wt_val * -1.0f;
        }
        
        // Atomic add to avoid write conflicts
        atomicAdd(&relevance_x[i], weight);
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
    '-Xcompiler', '-fPIC',
    # '-Xptxas', '-dlcm=cg',
    # '-Xptxas', '-dscm=wt',
]

extra_flags.extend(get_cuda_arch_flags())

custom_linear_layer_cuda_ops = load_inline(
    name="custom_linear_layer_cuda_v2",
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
