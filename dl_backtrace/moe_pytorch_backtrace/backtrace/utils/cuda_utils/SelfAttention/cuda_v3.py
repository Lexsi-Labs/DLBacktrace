import torch
from torch.utils.cpp_extension import load_inline
from typing import Tuple, Optional

# Kernel 1: Fused Softmax
fused_softmax_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fused_softmax_kernel(
    const float* __restrict__ logits,
    float* __restrict__ output,
    const int T_k,
    const float epsilon
) {
    // Each block handles one row (attention scores for one query position)
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_idx = blockIdx.x;
    
    const int row_offset = ((batch_idx * gridDim.y + head_idx) * gridDim.x + query_idx) * T_k;
    
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = shared_mem + blockDim.x;
    
    // Phase 1: Find maximum
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < T_k; i += blockDim.x) {
        thread_max = fmaxf(thread_max, logits[row_offset + i]);
    }
    
    // Warp-level reduction
    thread_max = warp_reduce_max(thread_max);
    if (threadIdx.x % 32 == 0) {
        shared_max[threadIdx.x / 32] = thread_max;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (threadIdx.x < 32) {
        float val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_max[threadIdx.x] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (threadIdx.x == 0) shared_max[0] = val;
    }
    __syncthreads();
    float row_max = shared_max[0];
    
    // Phase 2: Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < T_k; i += blockDim.x) {
        float exp_val = expf(logits[row_offset + i] - row_max);
        output[row_offset + i] = exp_val;
        thread_sum += exp_val;
    }
    
    // Warp-level sum reduction
    thread_sum = warp_reduce_sum(thread_sum);
    if (threadIdx.x % 32 == 0) {
        shared_sum[threadIdx.x / 32] = thread_sum;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        float val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_sum[threadIdx.x] : 0.0f;
        val = warp_reduce_sum(val);
        if (threadIdx.x == 0) shared_sum[0] = val + epsilon;
    }
    __syncthreads();
    float row_sum = shared_sum[0];
    
    // Phase 3: Normalize
    for (int i = threadIdx.x; i < T_k; i += blockDim.x) {
        output[row_offset + i] /= row_sum;
    }
}

void launch_fused_softmax(
    torch::Tensor logits,
    torch::Tensor output,
    float epsilon
) {
    const int B = logits.size(0);
    const int H = logits.size(1);
    const int T_q = logits.size(2);
    const int T_k = logits.size(3);
    
    const int threads = 256;
    const int shared_mem_size = 2 * threads * sizeof(float);
    
    dim3 grid(T_q, H, B);
    dim3 block(threads);
    
    fused_softmax_kernel<<<grid, block, shared_mem_size>>>(
        logits.data_ptr<float>(),
        output.data_ptr<float>(),
        T_k,
        epsilon
    );
}
"""

fused_softmax_declaration = r"""
void launch_fused_softmax(torch::Tensor logits, torch::Tensor output, float epsilon);
"""

# Kernel 2: Fused Stabilize and Normalize
fused_stabilize_normalize_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_stabilize_normalize_kernel(
    const float* __restrict__ numerator,
    const float* __restrict__ denominator,
    float* __restrict__ output,
    const int total_elements,
    const float epsilon
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float denom_val = denominator[idx] * 2.0f;  // Apply the *2 from original code
        
        // Stabilize denominator
        float abs_denom = fabsf(denom_val);
        float sign_denom = (denom_val == 0.0f) ? 1.0f : copysignf(1.0f, denom_val);
        float stable_denom = (abs_denom < epsilon) ? (epsilon * sign_denom) : denom_val;
        
        // Normalize
        output[idx] = numerator[idx] / stable_denom;
    }
}

void launch_fused_stabilize_normalize(
    torch::Tensor numerator,
    torch::Tensor denominator,
    torch::Tensor output,
    float epsilon
) {
    const int total_elements = numerator.numel();
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    fused_stabilize_normalize_kernel<<<blocks, threads>>>(
        numerator.data_ptr<float>(),
        denominator.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        epsilon
    );
}
"""

fused_stabilize_normalize_declaration = r"""
void launch_fused_stabilize_normalize(
    torch::Tensor numerator, 
    torch::Tensor denominator, 
    torch::Tensor output,
    float epsilon
);
"""

# Kernel 3: Fused Conservation
fused_conservation_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum_conserve(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fused_conservation_kernel(
    const float* __restrict__ wts,
    const float* __restrict__ inp,
    float* __restrict__ output,
    const int B,
    const int H,
    const int T,
    const int D,
    const float eps
) {
    // Each block handles one [T, D] slice
    const int slice_offset = blockIdx.y * T * D;
    
    extern __shared__ float shared_data[];
    float* s_p_sum = shared_data;
    float* s_n_sum = shared_data + blockDim.x;
    float* s_abs_sum = shared_data + 2 * blockDim.x;
    
    // Process both positive and negative parts of weights
    for (int sign_pass = 0; sign_pass < 2; sign_pass++) {
        // Get appropriate weight component (Rp or Rn)
        float thread_p_sum = 0.0f;
        float thread_n_sum = 0.0f;
        float thread_abs_sum = 0.0f;
        
        for (int i = threadIdx.x; i < T * D; i += blockDim.x) {
            float wt_val = wts[slice_offset + i];
            float wt_component = (sign_pass == 0) ? fmaxf(wt_val, 0.0f) : fmaxf(-wt_val, 0.0f);
            float inp_val = inp[slice_offset + i];
            
            // Accumulate sums for this weight component
            if (inp_val > 0.0f) {
                thread_p_sum += inp_val;
            } else if (inp_val < 0.0f) {
                thread_n_sum += -inp_val;
            }
            thread_abs_sum += fabsf(wt_component);
        }
        
        // Reduce within warp
        thread_p_sum = warp_reduce_sum_conserve(thread_p_sum);
        thread_n_sum = warp_reduce_sum_conserve(thread_n_sum);
        thread_abs_sum = warp_reduce_sum_conserve(thread_abs_sum);
        
        if (threadIdx.x % 32 == 0) {
            int warp_id = threadIdx.x / 32;
            s_p_sum[warp_id] = thread_p_sum;
            s_n_sum[warp_id] = thread_n_sum;
            s_abs_sum[warp_id] = thread_abs_sum;
        }
        __syncthreads();
        
        // Final reduction
        if (threadIdx.x < 32) {
            int num_warps = (blockDim.x + 31) / 32;
            float p_val = (threadIdx.x < num_warps) ? s_p_sum[threadIdx.x] : 0.0f;
            float n_val = (threadIdx.x < num_warps) ? s_n_sum[threadIdx.x] : 0.0f;
            float abs_val = (threadIdx.x < num_warps) ? s_abs_sum[threadIdx.x] : 0.0f;
            
            p_val = warp_reduce_sum_conserve(p_val);
            n_val = warp_reduce_sum_conserve(n_val);
            abs_val = warp_reduce_sum_conserve(abs_val);
            
            if (threadIdx.x == 0) {
                s_p_sum[0] = p_val;
                s_n_sum[0] = n_val;
                s_abs_sum[0] = abs_val;
            }
        }
        __syncthreads();
        
        float p_sum = s_p_sum[0];
        float n_sum = s_n_sum[0];
        float M = s_abs_sum[0];
        float denom = p_sum + n_sum + eps;
        
        float p_share = (p_sum > 0.0f) ? (p_sum / denom) : 0.0f;
        float n_share = (n_sum > 0.0f) ? (n_sum / denom) : 0.0f;
        
        float p_div = (p_sum == 0.0f) ? 1.0f : p_sum;
        float n_div = (n_sum == 0.0f) ? 1.0f : n_sum;
        
        for (int i = threadIdx.x; i < T * D; i += blockDim.x) {
            float wt_val = wts[slice_offset + i];
            float wt_component = (sign_pass == 0) ? fmaxf(wt_val, 0.0f) : fmaxf(-wt_val, 0.0f);
            float inp_val = inp[slice_offset + i];
            
            float contribution = 0.0f;
            if (inp_val > 0.0f) {
                contribution = (inp_val / p_div) * (p_share * M);
            } else if (inp_val < 0.0f) {
                contribution = (inp_val / n_div) * (n_share * M) * (-1.0f);
            }
            
            if (sign_pass == 0) {
                output[slice_offset + i] = contribution;
            } else {
                output[slice_offset + i] -= contribution;
            }
        }
        __syncthreads();
    }
}

void launch_fused_conservation(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor output,
    float eps
) {
    const int B = wts.size(0);
    const int H = wts.size(1);
    const int T = wts.size(2);
    const int D = wts.size(3);
    
    const int threads = 256;
    const int shared_mem_size = 3 * threads * sizeof(float);
    
    dim3 grid(1, B * H);
    dim3 block(threads);
    
    fused_conservation_kernel<<<grid, block, shared_mem_size>>>(
        wts.data_ptr<float>(),
        inp.data_ptr<float>(),
        output.data_ptr<float>(),
        B, H, T, D,
        eps
    );
}
"""

fused_conservation_declaration = r"""
void launch_fused_conservation(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor output,
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

# Compile kernels
softmax_ops = load_inline(
    name="fused_softmax",
    cpp_sources=fused_softmax_declaration,
    cuda_sources=fused_softmax_source,
    functions=["launch_fused_softmax"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

stabilize_ops = load_inline(
    name="fused_stabilize_normalize",
    cpp_sources=fused_stabilize_normalize_declaration,
    cuda_sources=fused_stabilize_normalize_source,
    functions=["launch_fused_stabilize_normalize"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

conservation_ops = load_inline(
    name="fused_conservation",
    cpp_sources=fused_conservation_declaration,
    cuda_sources=fused_conservation_source,
    functions=["launch_fused_conservation"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def calculate_wt_self_attention_cuda(
    R_out: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    masked_fill: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    epsilon: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    B, H, T_q, D = Q.shape
    T_k = K.shape[2]
    
    if scale is None:
        scale = float(D) ** 0.5

    # Step 1: Raw attention logits
    QK_output = torch.matmul(Q, K.transpose(-2, -1))
    logits_unmasked = QK_output / scale

    # Step 2: Fused softmax for unmasked
    A = torch.empty_like(logits_unmasked)
    softmax_ops.launch_fused_softmax(logits_unmasked, A, epsilon)
    torch.cuda.synchronize()  

    # Step 3: Apply mask
    masked_fill = None
    if masked_fill is not None:
        logits_masked = logits_unmasked + masked_fill
        # Step 4: Fused softmax for masked
        A_masked = torch.empty_like(logits_masked)
        softmax_ops.launch_fused_softmax(logits_masked, A_masked, epsilon)
        torch.cuda.synchronize()
    else:
        # No mask applied
        A_masked = A

    # Step 5: Attention output
    attention_output = torch.matmul(A_masked, V)

    # Step 6: Fused stabilize + normalize for relevance propagation
    relevance_norm_attn_out = torch.empty_like(R_out)
    stabilize_ops.launch_fused_stabilize_normalize(
        R_out, attention_output, relevance_norm_attn_out, epsilon
    )
    torch.cuda.synchronize()

    # Compute R_QK and R_V_raw
    R_QK = torch.matmul(relevance_norm_attn_out, V.transpose(-2, -1)) * A
    R_V_raw = torch.matmul(A.transpose(-2, -1), relevance_norm_attn_out) * V

    # Fused conservation for R_V
    R_V = torch.empty_like(V)
    conservation_ops.launch_fused_conservation(R_V_raw, V, R_V, epsilon)
    torch.cuda.synchronize()

    # Fused stabilize + normalize for QK
    relevance_norm_QK_out = torch.empty_like(R_QK)
    stabilize_ops.launch_fused_stabilize_normalize(
        R_QK, QK_output, relevance_norm_QK_out, epsilon
    )
    torch.cuda.synchronize()

    # Compute R_Q_raw and R_K_raw
    R_Q_raw = torch.matmul(relevance_norm_QK_out, K) * Q
    R_K_raw = torch.matmul(Q.transpose(-2, -1), relevance_norm_QK_out).transpose(-2, -1) * K

    # Fused conservation for R_Q and R_K
    R_Q = torch.empty_like(Q)
    R_K = torch.empty_like(K)
    conservation_ops.launch_fused_conservation(R_Q_raw, Q, R_Q, epsilon)
    conservation_ops.launch_fused_conservation(R_K_raw, K, R_K, epsilon)
    torch.cuda.synchronize()

    # Mask relevance
    delta_A = A - A_masked
    R_blocked_per_qk = torch.einsum('bhqk,bhkd->bhqk', delta_A, V)
    R_masked_fill = R_blocked_per_qk.sum(dim=1, keepdim=True)

    return R_Q, R_K, R_V, R_masked_fill