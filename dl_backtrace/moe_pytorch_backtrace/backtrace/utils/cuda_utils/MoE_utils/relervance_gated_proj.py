import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Swish activation with beta scaling and clamping
__device__ __forceinline__ float swish_activation(float x, float beta) {
    float x_beta = beta * x;
    // Clamp to prevent overflow in exp
    x_beta = fmaxf(-500.0f, fminf(500.0f, x_beta));
    float sigmoid = 1.0f / (1.0f + expf(-x_beta));
    return x * sigmoid;
}

// Kernel 1: Compute statistics (pos_sum, neg_sum, total_weight)
__global__ void compute_statistics_kernel(
    const float* __restrict__ output,
    const float* __restrict__ wts,
    const int N,
    float* pos_sum_out,
    float* neg_sum_out,
    float* total_weight_out
) {
    __shared__ float shared_pos[256];
    __shared__ float shared_neg[256];
    __shared__ float shared_wt[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Thread-local accumulators
    float local_pos = 0.0f;
    float local_neg = 0.0f;
    float local_wt = 0.0f;
    
    // Grid-stride loop for large tensors
    for (int i = idx; i < N; i += stride) {
        float val = output[i];
        if (val > 0.0f) {
            local_pos += val;
        } else if (val < 0.0f) {
            local_neg += val;
        }
        local_wt += wts[i];
    }
    
    // Store to shared memory
    shared_pos[tid] = local_pos;
    shared_neg[tid] = local_neg;
    shared_wt[tid] = local_wt;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_pos[tid] += shared_pos[tid + s];
            shared_neg[tid] += shared_neg[tid + s];
            shared_wt[tid] += shared_wt[tid + s];
        }
        __syncthreads();
    }
    
    // First thread writes block result
    if (tid == 0) {
        atomicAdd(pos_sum_out, shared_pos[0]);
        atomicAdd(neg_sum_out, shared_neg[0]);
        atomicAdd(total_weight_out, shared_wt[0]);
    }
}

// Kernel 2: Apply weighted gating
__global__ void apply_gating_kernel(
    const float* __restrict__ output,
    const int N,
    float pos_sum_base,
    float neg_sum_base,
    float total_weight,
    float* __restrict__ wt_mat_total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N) return;
    
    const float beta = 0.75f;
    const float eps = 1e-6f;
    
    // Make neg_sum_base positive for calculations
    neg_sum_base = -neg_sum_base;
    
    // Calculate activations
    float t_sum_base = pos_sum_base - neg_sum_base;
    float t_act = swish_activation(t_sum_base, beta);
    float p_act = swish_activation(pos_sum_base, beta);
    float n_act = swish_activation(-neg_sum_base, beta);
    
    // Apply conditional logic
    float pos_sum = pos_sum_base;
    float neg_sum = neg_sum_base;
    
    // Threshold condition
    if (t_sum_base < -6.0f) {
        pos_sum = 0.0f;
    }
    
    // Both positive condition with activation equality checks
    if (pos_sum_base > 0.0f && neg_sum_base > 0.0f) {
        if (fabsf(t_act - p_act) < eps) {
            neg_sum = 0.0f;
        } else if (fabsf(t_act - n_act) < eps) {
            pos_sum = 0.0f;
        }
    }
    
    // Calculate aggregation weights
    float denominator = pos_sum + neg_sum;
    float pos_agg_wt = (pos_sum > 0.0f && denominator > 0.0f) ? (pos_sum / denominator) : 0.0f;
    float neg_agg_wt = (neg_sum > 0.0f && denominator > 0.0f) ? (neg_sum / denominator) : 0.0f;
    
    // Normalization denominators
    float pos_sum_norm = (pos_sum != 0.0f) ? pos_sum : 1.0f;
    float neg_sum_norm = (neg_sum != 0.0f) ? neg_sum : 1.0f;
    
    // Compute output
    float val = output[idx];
    float result = 0.0f;
    
    if (val > 0.0f && pos_agg_wt != 0.0f) {
        result = (val / pos_sum_norm) * pos_agg_wt * total_weight;
    } else if (val < 0.0f && neg_agg_wt != 0.0f) {
        result = (val / neg_sum_norm) * neg_agg_wt * total_weight * -1.0f;
    }
    
    wt_mat_total[idx] = result;
}

void launch_kernel(
    torch::Tensor output,
    torch::Tensor wts,
    torch::Tensor wt_mat_total
) {
    const int N = output.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    // Allocate device memory for scalar results
    float *d_pos_sum, *d_neg_sum, *d_total_weight;
    cudaMalloc(&d_pos_sum, sizeof(float));
    cudaMalloc(&d_neg_sum, sizeof(float));
    cudaMalloc(&d_total_weight, sizeof(float));
    
    cudaMemset(d_pos_sum, 0, sizeof(float));
    cudaMemset(d_neg_sum, 0, sizeof(float));
    cudaMemset(d_total_weight, 0, sizeof(float));
    
    // Kernel 1: Compute statistics
    compute_statistics_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        wts.data_ptr<float>(),
        N,
        d_pos_sum,
        d_neg_sum,
        d_total_weight
    );
    
    // Copy results back to host
    float pos_sum_base, neg_sum_base, total_weight;
    cudaMemcpy(&pos_sum_base, d_pos_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&neg_sum_base, d_neg_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_weight, d_total_weight, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Kernel 2: Apply gating
    apply_gating_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        N,
        pos_sum_base,
        neg_sum_base,
        total_weight,
        wt_mat_total.data_ptr<float>()
    );
    
    // Cleanup
    cudaFree(d_pos_sum);
    cudaFree(d_neg_sum);
    cudaFree(d_total_weight);
}
"""

cuda_declaration = r"""
void launch_kernel(torch::Tensor output, torch::Tensor wts, torch::Tensor wt_mat_total);
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
    name="relevance_gating_kernel",
    cpp_sources=cuda_declaration,
    cuda_sources=cuda_source,
    functions=["launch_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def calculate_relevance_gated_proj_cuda(
    wts: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA-accelerated version of calculate_relevance_gated_proj
    
    Args:
        wts: Weight tensor (same shape as output)
        output: Output tensor to compute gating on
    
    Returns:
        Weighted gated projection tensor
    """
    assert output.is_cuda, "Input must be on CUDA device"
    assert wts.is_cuda, "Weights must be on CUDA device"
    assert output.dtype == torch.float32, "Only float32 supported"
    assert wts.dtype == torch.float32, "Only float32 supported"
    assert output.shape == wts.shape, "Shapes must match"
    
    # Allocate output tensor
    wt_mat_total = torch.zeros_like(output)
    
    # Launch CUDA kernel
    cuda_ops.launch_kernel(output, wts, wt_mat_total)
    
    return wt_mat_total
