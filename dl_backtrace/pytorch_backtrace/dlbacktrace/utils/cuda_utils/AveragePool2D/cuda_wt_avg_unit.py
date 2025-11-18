import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Warp-level reduction helper
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void calculate_wt_avg_kernel(
    const float* __restrict__ patch,
    const float* __restrict__ wts,
    float* __restrict__ output,
    const int H,
    const int W,
    const int C,
    const int spatial_size
) {
    const int channel = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Shared memory for reductions
    __shared__ float s_p_sum[256];
    __shared__ float s_n_sum[256];
    
    // Phase 1: Parallel reduction per channel
    float p_local = 0.0f;
    float n_local = 0.0f;
    
    // Each thread processes multiple elements (grid-stride loop)
    for (int idx = tid; idx < spatial_size; idx += block_size) {
        int i = idx / W;
        int j = idx % W;
        int index = i * W * C + j * C + channel;
        
        float val = patch[index];
        
        // Separate positive and negative (replicates p_ind/n_ind logic)
        if (val > 0.0f) {
            p_local += val;
        } else if (val < 0.0f) {
            n_local += -val;  // Store absolute value (replicates *-1.0)
        }
    }
    
    // Warp-level reduction
    p_local = warp_reduce_sum(p_local);
    n_local = warp_reduce_sum(n_local);
    
    // First thread in each warp writes to shared memory
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    
    if (lane == 0) {
        s_p_sum[warp_id] = p_local;
        s_n_sum[warp_id] = n_local;
    }
    
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < 32) {
        p_local = (tid < (block_size + 31) / 32) ? s_p_sum[tid] : 0.0f;
        n_local = (tid < (block_size + 31) / 32) ? s_n_sum[tid] : 0.0f;
        
        p_local = warp_reduce_sum(p_local);
        n_local = warp_reduce_sum(n_local);
    }
    
    // Broadcast reduced values to shared memory
    __shared__ float channel_p_sum;
    __shared__ float channel_n_sum;
    __shared__ float channel_t_sum_safe;
    __shared__ float channel_p_agg_wt;
    __shared__ float channel_n_agg_wt;
    __shared__ bool channel_p_saturate;
    __shared__ bool channel_n_saturate;
    
    if (tid == 0) {
        channel_p_sum = p_local;
        channel_n_sum = n_local;
        
        float t_sum = p_local + n_local;
        
        // Safe division: replicate torch.where(t_sum == 0, 1.0, t_sum)
        channel_t_sum_safe = (t_sum == 0.0f) ? 1.0f : t_sum;
        
        // Saturation masks: replicate (p_sum > 0), (n_sum > 0)
        channel_p_saturate = (p_local > 0.0f);
        channel_n_saturate = (n_local > 0.0f);
        
        float wt = wts[channel];
        
        // Compute aggregated weights
        // p_agg_wt = (1.0 / t_sum_safe) * wts * p_saturate
        channel_p_agg_wt = (1.0f / channel_t_sum_safe) * wt * (channel_p_saturate ? 1.0f : 0.0f);
        
        // n_agg_wt = (1.0 / t_sum_safe) * wts * n_saturate
        channel_n_agg_wt = (1.0f / channel_t_sum_safe) * wt * (channel_n_saturate ? 1.0f : 0.0f);
    }
    
    __syncthreads();
    
    // Phase 2: Apply weights element-wise
    for (int idx = tid; idx < spatial_size; idx += block_size) {
        int i = idx / W;
        int j = idx % W;
        int index = i * W * C + j * C + channel;
        
        float val = patch[index];
        float result;
        
        // Replicate: wt_mat = p_ind * p_agg_wt + n_ind * n_agg_wt * -1.0
        if (val > 0.0f) {
            // Positive component: p_ind * p_agg_wt
            result = val * channel_p_agg_wt;
        } else if (val < 0.0f) {
            // Negative component: n_ind * n_agg_wt * -1.0
            result = val * channel_n_agg_wt * -1.0f;
        } else {
            // Zero values remain zero
            result = 0.0f;
        }
        
        output[index] = result;
    }
}

void launch_kernel(torch::Tensor patch, torch::Tensor wts, torch::Tensor output) {
    const int H = patch.size(0);
    const int W = patch.size(1);
    const int C = patch.size(2);
    const int spatial_size = H * W;
    
    const int block_size = 256;
    const int num_blocks = C;
    
    dim3 grid(num_blocks);
    dim3 block(block_size);
    
    calculate_wt_avg_kernel<<<grid, block>>>(
        patch.data_ptr<float>(),
        wts.data_ptr<float>(),
        output.data_ptr<float>(),
        H, W, C, spatial_size
    );
}
"""

cuda_declaration = r"""
void launch_kernel(torch::Tensor patch, torch::Tensor wts, torch::Tensor output);
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
    name="wt_avg_unit_kernel",
    cpp_sources=cuda_declaration,
    cuda_sources=cuda_source,
    functions=["launch_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def calculate_wt_avg_unit_cuda(patch: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
    """
    CUDA-accelerated version of calculate_wt_avg_unit.
    
    Args:
        patch: Input tensor of shape (H, W, C)
        wts: Weight tensor of shape (C,)
    
    Returns:
        Weighted matrix of shape (H, W, C)
    """
    assert patch.is_cuda, "Input must be on CUDA device"
    assert wts.is_cuda, "Weights must be on CUDA device"
    assert patch.dtype == torch.float32, "Only float32 supported"
    assert wts.dtype == torch.float32, "Only float32 supported"
    assert patch.dim() == 3, "Patch must be 3D (H, W, C)"
    assert wts.dim() == 1, "Weights must be 1D (C,)"
    assert patch.size(2) == wts.size(0), "Channel dimensions must match"
    
    output = torch.empty_like(patch)
    cuda_ops.launch_kernel(patch, wts, output)
    return output