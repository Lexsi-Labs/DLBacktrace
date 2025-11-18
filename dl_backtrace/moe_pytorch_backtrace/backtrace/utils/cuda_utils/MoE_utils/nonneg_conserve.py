import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void dlb_nonneg_conserve_kernel(
    const float* __restrict__ wts,
    const float* __restrict__ inp,
    float* __restrict__ out,
    const float eps,
    const int batch_size,
    const int channels,
    const int height,
    const int width
) {
    // Each block handles one (batch, channel) combination
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || channel_idx >= channels) return;
    
    int spatial_size = height * width;
    int offset = (batch_idx * channels + channel_idx) * spatial_size;
    
    // Shared memory for reductions
    __shared__ float s_p_sum[256];
    __shared__ float s_n_sum[256];
    __shared__ float s_M[256];
    
    int tid = threadIdx.x;
    
    // Phase 1: Compute reductions
    float local_p = 0.0f;
    float local_n = 0.0f;
    float local_m = 0.0f;
    
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        float inp_val = inp[offset + i];
        float wts_val = wts[offset + i];
        
        if (inp_val > 0.0f) local_p += inp_val;
        if (inp_val < 0.0f) local_n += inp_val;
        local_m += fabsf(wts_val);
    }
    
    s_p_sum[tid] = local_p;
    s_n_sum[tid] = local_n;
    s_M[tid] = local_m;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_p_sum[tid] += s_p_sum[tid + stride];
            s_n_sum[tid] += s_n_sum[tid + stride];
            s_M[tid] += s_M[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute derived values
    float p_sum = s_p_sum[0];
    float n_sum = -s_n_sum[0];
    float M = s_M[0];
    float denom = p_sum + n_sum + eps;
    
    float p_share = (p_sum > 0.0f) ? (p_sum / denom) : 0.0f;
    float n_share = (n_sum > 0.0f) ? (n_sum / denom) : 0.0f;
    float p_div = (p_sum == 0.0f) ? 1.0f : p_sum;
    float n_div = (n_sum == 0.0f) ? 1.0f : n_sum;
    
    // Phase 2: Compute outputs
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        float inp_val = inp[offset + i];
        float result = 0.0f;
        
        if (inp_val > 0.0f) {
            result = (inp_val / p_div) * (p_share * M);
        } else if (inp_val < 0.0f) {
            result = (inp_val / n_div) * (n_share * M) * (-1.0f);
        }
        
        out[offset + i] = result;
    }
}

void launch_kernel(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor out,
    float eps
) {
    const int batch_size = wts.size(0);
    const int channels = wts.size(1);
    const int height = wts.size(2);
    const int width = wts.size(3);
    
    dim3 grid(batch_size, channels);
    dim3 block(256);
    
    dlb_nonneg_conserve_kernel<<<grid, block>>>(
        wts.data_ptr<float>(),
        inp.data_ptr<float>(),
        out.data_ptr<float>(),
        eps,
        batch_size,
        channels,
        height,
        width
    );
}
"""

cuda_declaration = r"""
void launch_kernel(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor out,
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
    name="dlb_conserve_cuda",
    cpp_sources=cuda_declaration,
    cuda_sources=cuda_source,
    functions=["launch_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def dlb_style_nonneg_conserve_cuda(
    wts: torch.Tensor,
    inp: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    CUDA-accelerated version of dlb_style_nonneg_conserve.
    
    Args:
        wts: Weight tensor of shape (B, C, H, W)
        inp: Input tensor of shape (B, C, H, W)
        eps: Epsilon for numerical stability
    
    Returns:
        Output tensor of shape (B, C, H, W)
    """
    assert wts.shape == inp.shape
    assert wts.is_cuda and inp.is_cuda
    assert wts.is_contiguous() and inp.is_contiguous()
    assert wts.dtype == torch.float32 and inp.dtype == torch.float32
    
    out = torch.zeros_like(inp)
    cuda_ops.launch_kernel(wts, inp, out, eps)
    return out

def dlb_style_signed_conserve_cuda(
    wts: torch.Tensor,
    inp: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    CUDA-accelerated version of dlb_style_signed_conserve.
    
    Args:
        wts: Weight tensor of shape (B, C, H, W)
        inp: Input tensor of shape (B, C, H, W)
        eps: Epsilon for numerical stability
    
    Returns:
        Output tensor of shape (B, C, H, W)
    """
    Rp = torch.clamp(wts, min=0.0)
    Rn = torch.clamp(-wts, min=0.0)
    
    P = dlb_style_nonneg_conserve_cuda(Rp, inp, eps)
    N = dlb_style_nonneg_conserve_cuda(Rn, inp, eps)
    
    return P - N
    