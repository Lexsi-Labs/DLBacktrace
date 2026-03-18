import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_lm_head_relevance_kernel(
    const float* __restrict__ wts,           // [B, S, V]
    const float* __restrict__ inp,           // [B, S, H]
    const float* __restrict__ W_lm_head,     // [V, H]
    float* __restrict__ relevance_input,     // [B, S, H]
    int B, int S, int V, int H
) {
    // Thread index maps to output position (b, s, h)
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    int b = blockIdx.z;
    
    if (h >= H) return;
    
    // Linear indices for input and output
    int inp_idx = b * S * H + s * H + h;
    float inp_val = inp[inp_idx];
    
    // First pass: compute positive_sum and negative_sum
    float positive_sum = 0.0f;
    float negative_sum = 0.0f;
    
    for (int v = 0; v < V; v++) {
        float w_val = W_lm_head[v * H + h];  // Coalesced access across h
        float contrib = w_val * inp_val;
        
        if (contrib > 0.0f) {
            positive_sum += contrib;
        } else if (contrib < 0.0f) {
            negative_sum -= contrib;  // Make positive
        }
    }
    
    // Compute normalization factors
    float total_sum = positive_sum + negative_sum;
    
    float positive_agg_wt = (positive_sum > 0.0f) ? (positive_sum / total_sum) : 0.0f;
    float negative_agg_wt = (negative_sum > 0.0f) ? (negative_sum / total_sum) : 0.0f;
    
    float positive_sum_safe = (positive_sum != 0.0f) ? positive_sum : 1.0f;
    float negative_sum_safe = (negative_sum != 0.0f) ? negative_sum : 1.0f;
    
    // Second pass: compute weighted relevance
    float positive_relevance = 0.0f;
    float negative_relevance = 0.0f;
    
    for (int v = 0; v < V; v++) {
        float w_val = W_lm_head[v * H + h];
        float contrib = w_val * inp_val;
        float wt = wts[b * S * V + s * V + v];
        
        if (contrib > 0.0f) {
            positive_relevance += (contrib / positive_sum_safe) * wt * positive_agg_wt;
        } else if (contrib < 0.0f) {
            negative_relevance += (contrib / negative_sum_safe) * wt * negative_agg_wt * (-1.0f);
        }
    }
    
    // Write output
    relevance_input[inp_idx] = positive_relevance + negative_relevance;
}

void launch_kernel(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor W_lm_head,
    torch::Tensor relevance_input
) {
    int B = wts.size(0);
    int S = wts.size(1);
    int V = wts.size(2);
    int H = inp.size(2);
    
    const int threads = 256;
    dim3 block(threads);
    dim3 grid((H + threads - 1) / threads, S, B);
    
    fused_lm_head_relevance_kernel<<<grid, block>>>(
        wts.data_ptr<float>(),
        inp.data_ptr<float>(),
        W_lm_head.data_ptr<float>(),
        relevance_input.data_ptr<float>(),
        B, S, V, H
    );
}
"""

cuda_declaration = r"""
void launch_kernel(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor W_lm_head,
    torch::Tensor relevance_input
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
    name="lm_head_relevance_kernel",
    cpp_sources=cuda_declaration,
    cuda_sources=cuda_source,
    functions=["launch_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def calculate_wt_lm_head_cuda(
    wts: torch.Tensor,
    inp: torch.Tensor,
    w: dict
) -> torch.Tensor:
    """
    CUDA-accelerated version of calculate_wt_lm_head_vectorized
    
    Args:
        wts: [B, S, V] weight tensor
        inp: [B, S, H] input tensor
        w: dict containing 'W_lm_head' [V, H]
    
    Returns:
        relevance_input: [B, S, H] relevance scores
    """
    B, S, H = inp.shape
    V = wts.size(2)
    
    # Allocate output
    relevance_input = torch.empty_like(inp)
    
    # Launch kernel
    cuda_ops.launch_kernel(
        wts.contiguous(),
        inp.contiguous(),
        w['W_lm_head'].contiguous(),
        relevance_input
    )
    
    return relevance_input