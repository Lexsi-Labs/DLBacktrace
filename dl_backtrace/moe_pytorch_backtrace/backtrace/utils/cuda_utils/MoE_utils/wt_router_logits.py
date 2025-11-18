import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void wt_router_logits_kernel(
    const float* __restrict__ wts,
    const float* __restrict__ inp,
    const float* __restrict__ W_router,
    float* __restrict__ output,
    const int n_samples,
    const int n_features
) {
    const int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (sample_idx >= n_samples || feat_idx >= n_features) return;
    
    // Constants
    const float act_range_lower = -1.0f;
    const float act_range_upper = 2.0f;
    
    // Load sample weight once
    const float wt_val = wts[sample_idx];
    
    // Shared memory for W_router column (optional optimization)
    extern __shared__ float s_W_router[];
    
    // Cooperative load of W_router column into shared memory
    for (int k = threadIdx.y * blockDim.x + threadIdx.x; 
         k < n_features; 
         k += blockDim.x * blockDim.y) {
        if (blockIdx.x * blockDim.x < n_features) {
            int col_idx = min(blockIdx.x * blockDim.x + k % blockDim.x, n_features - 1);
            s_W_router[k] = W_router[k * n_features + col_idx];
        }
    }
    __syncthreads();
    
    // Step 1 & 2: Compute contributions and accumulate sums
    float p_sum = 0.0f;
    float n_sum = 0.0f;
    float t_sum = 0.0f;
    
    const int inp_offset = sample_idx * n_features;
    
    for (int k = 0; k < n_features; ++k) {
        // Compute contribution on-the-fly
        const float contrib = inp[inp_offset + k] * W_router[k * n_features + feat_idx];
        
        if (contrib > 0.0f) {
            p_sum += contrib;
            t_sum += contrib;
        } else if (contrib < 0.0f) {
            n_sum -= contrib;  // Store as positive
            t_sum += contrib;
        }
    }
    
    // Step 3: Apply activation range filtering
    if (t_sum < act_range_lower) {
        p_sum = 0.0f;
    }
    if (t_sum > act_range_upper) {
        n_sum = 0.0f;
    }
    
    // Step 4: Calculate aggregation weights
    const float total_sum = p_sum + n_sum;
    const float p_agg_wt = (p_sum > 0.0f && total_sum > 0.0f) ? p_sum / total_sum : 0.0f;
    const float n_agg_wt = (n_sum > 0.0f && total_sum > 0.0f) ? n_sum / total_sum : 0.0f;
    
    // Step 5: Normalize sums (avoid division by zero)
    const float p_sum_norm = (p_sum != 0.0f) ? p_sum : 1.0f;
    const float n_sum_norm = (n_sum != 0.0f) ? n_sum : 1.0f;
    
    // Step 6: Calculate weight components
    const float p_weight = wt_val * p_agg_wt / p_sum_norm;
    const float n_weight = wt_val * n_agg_wt / n_sum_norm;
    
    // Step 7: Compute weighted output
    float result = 0.0f;
    
    for (int k = 0; k < n_features; ++k) {
        const float contrib = inp[inp_offset + k] * W_router[k * n_features + feat_idx];
        
        if (contrib > 0.0f) {
            result += contrib * p_weight;
        } else if (contrib < 0.0f) {
            result -= contrib * n_weight;  // Double negation: -(-contrib) * weight
        }
    }
    
    output[sample_idx * n_features + feat_idx] = result;
}

void launch_kernel(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor W_router,
    torch::Tensor output
) {
    const int n_samples = inp.size(0);
    const int n_features = inp.size(1);
    
    // Thread configuration
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (n_features + blockDim.x - 1) / blockDim.x,
        (n_samples + blockDim.y - 1) / blockDim.y
    );
    
    // Shared memory size
    const int shared_mem_size = n_features * sizeof(float);
    
    wt_router_logits_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        wts.data_ptr<float>(),
        inp.data_ptr<float>(),
        W_router.data_ptr<float>(),
        output.data_ptr<float>(),
        n_samples,
        n_features
    );
}
"""

cuda_declaration = r"""
void launch_kernel(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor W_router,
    torch::Tensor output
);
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
    name="wt_router_logits",
    cpp_sources=cuda_declaration,
    cuda_sources=cuda_source,
    functions=["launch_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def calculate_wt_router_logits_cuda(
    wts: torch.Tensor,
    inp: torch.Tensor,
    W_router: torch.Tensor
) -> torch.Tensor:
    """
    CUDA-accelerated version of calculate_wt_router_logits.
    
    Args:
        wts: (n_samples,) sample weights
        inp: (n_samples, n_features) input features
        W_router: (n_features, n_features) router weights
    
    Returns:
        output: (n_samples, n_features) weighted router logits
    """
    # Input validation
    assert wts.is_cuda and inp.is_cuda and W_router.is_cuda, "All tensors must be on CUDA"
    assert wts.dtype == torch.float32, "wts must be float32"
    assert inp.dtype == torch.float32, "inp must be float32"
    assert W_router.dtype == torch.float32, "W_router must be float32"
    assert inp.is_contiguous() and W_router.is_contiguous(), "Tensors must be contiguous"
    
    n_samples, n_features = inp.shape
    assert W_router.shape == (n_features, n_features), "W_router shape mismatch"
    assert wts.shape == (n_samples,), "wts shape mismatch"
    
    # Allocate output
    output = torch.empty((n_samples, n_features), dtype=torch.float32, device=inp.device)
    
    # Launch kernel
    cuda_ops.launch_kernel(wts, inp, W_router, output)
    
    return output
