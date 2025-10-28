import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void calculate_wt_lm_head_chunk_kernel(
    const float* __restrict__ inp_flat,
    const float* __restrict__ W_chunk,
    const float* __restrict__ R_chunk,
    float* __restrict__ relevance_flat,
    const int BT,
    const int C,
    const int D,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BT * D) return;
    
    int bt = idx / D;
    int d = idx % D;
    
    float inp_val = inp_flat[bt * D + d];
    float accum = 0.0f;
    
    // Process all vocabulary items in chunk
    for (int c = 0; c < C; ++c) {
        float w_val = W_chunk[c * D + d];
        float r_val = R_chunk[bt * C + c];
        
        // Compute p_sum and n_sum by reading all D dimensions
        float p_sum = 0.0f;
        float n_sum = 0.0f;
        
        for (int d_inner = 0; d_inner < D; ++d_inner) {
            float inp_inner = inp_flat[bt * D + d_inner];
            float w_inner = W_chunk[c * D + d_inner];
            float L_inner = inp_inner * w_inner;
            p_sum += fmaxf(L_inner, 0.0f);
            n_sum += fmaxf(-L_inner, 0.0f);
        }
        
        // Denominator with epsilon
        float denom = p_sum + n_sum + eps;
        
        // Aggregate weights (conditional)
        float p_agg = (p_sum > 0.0f) ? (p_sum / denom) : 0.0f;
        float n_agg = (n_sum > 0.0f) ? (n_sum / denom) : 0.0f;
        
        // Element-wise product for this d
        float L = inp_val * w_val;
        float L_pos = fmaxf(L, 0.0f);
        float L_neg = fminf(L, 0.0f);
        
        // Normalized contributions
        float p_norm = (p_sum > 0.0f) ? (L_pos / (p_sum + eps)) : 0.0f;
        float n_norm = (n_sum > 0.0f) ? (L_neg / (n_sum + eps)) : 0.0f;
        
        // Weighted contribution
        float contrib = p_norm * (r_val * p_agg) + (-n_norm) * (r_val * n_agg);
        accum += contrib;
    }
    
    // Atomic add for accumulation across chunks
    atomicAdd(&relevance_flat[bt * D + d], accum);
}

void launch_kernel(
    torch::Tensor inp_flat,
    torch::Tensor W_chunk,
    torch::Tensor R_chunk,
    torch::Tensor relevance_flat,
    float eps
) {
    const int BT = inp_flat.size(0);
    const int D = inp_flat.size(1);
    const int C = W_chunk.size(0);
    
    const int threads = 256;
    const int blocks = (BT * D + threads - 1) / threads;
    
    calculate_wt_lm_head_chunk_kernel<<<blocks, threads>>>(
        inp_flat.data_ptr<float>(),
        W_chunk.data_ptr<float>(),
        R_chunk.data_ptr<float>(),
        relevance_flat.data_ptr<float>(),
        BT, C, D, eps
    );
}
"""

cuda_declaration = r"""
void launch_kernel(
    torch::Tensor inp_flat,
    torch::Tensor W_chunk,
    torch::Tensor R_chunk,
    torch::Tensor relevance_flat,
    float eps
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
    name="wt_lm_head_cuda",
    cpp_sources=cuda_declaration,
    cuda_sources=cuda_source,
    functions=["launch_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def calculate_wt_lm_head_cuda(
    wts: torch.Tensor,
    inp: torch.Tensor,
    w: dict,
    chunk_rows: int = 4096,
    eps: float = 1e-12
) -> torch.Tensor:
    """CUDA-optimized version of calculate_wt_lm_head"""
    
    W = w['W_lm_head']
    B, T, V = wts.shape
    _, _, D = inp.shape
    
    # Ensure float32 and contiguous
    W = W.to(dtype=torch.float32).contiguous()
    wts = wts.to(dtype=torch.float32).contiguous()
    inp = inp.to(dtype=torch.float32).contiguous()
    
    # Flatten
    wts_flat = wts.reshape(B * T, V)
    inp_flat = inp.reshape(B * T, D)
    
    # Pre-allocate output
    relevance_flat = torch.zeros(B * T, D, dtype=torch.float32, device=inp.device)
    
    # Process in chunks
    for start in range(0, V, chunk_rows):
        end = min(start + chunk_rows, V)
        
        W_chunk = W[start:end, :].contiguous()
        R_chunk = wts_flat[:, start:end].contiguous()
        
        # Launch CUDA kernel
        cuda_ops.launch_kernel(inp_flat, W_chunk, R_chunk, relevance_flat, eps)
    
    # Reshape and return
    return relevance_flat.reshape(B, T, D)