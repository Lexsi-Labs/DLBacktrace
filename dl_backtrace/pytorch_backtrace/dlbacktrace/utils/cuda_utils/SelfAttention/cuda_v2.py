import torch
from torch.utils.cpp_extension import load_inline
import numpy as np

attention_cuda_source = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <vector>

__device__ __forceinline__ float stabilize_value(float x, float epsilon) {
    float abs_x = fabsf(x);
    float sign_x = (x == 0.0f) ? 1.0f : ((x > 0.0f) ? 1.0f : -1.0f);
    return (abs_x < epsilon) ? (epsilon * sign_x) : x;
}

__device__ void warp_reduce_max(float& val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float tmp = __shfl_down_sync(0xffffffff, val, offset);
        if (tmp > val) val = tmp;
    }
}

__device__ void warp_reduce_sum(float& val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
}

// Kernel 1: Compute QK scores and attention weights
__global__ void compute_attention_weights_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ mask,
    float* __restrict__ qk_scores,
    float* __restrict__ unmasked_attn,
    float* __restrict__ masked_attn,
    const int B, const int H, const int T_q, const int T_k, const int D,
    const float scale, const float epsilon, const bool has_mask, const int mask_stride
) {
    extern __shared__ float smem[];
    
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int t_q = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (t_q >= T_q || head >= H || batch >= B) return;

    // Shared memory layout
    float* s_qk = smem;                    // [T_k]
    float* s_logits = s_qk + T_k;          // [T_k]
    float* s_temp = s_logits + T_k;        // [32] - one per warp for reductions
    
    const int query_offset = (((batch * H) + head) * T_q + t_q) * D;
    const int attn_offset = (((batch * H) + head) * T_q + t_q) * T_k;
    const int mask_base = has_mask ? (batch * mask_stride) + t_q * T_k : 0;
    
    // ==================== 1. Compute QK scores ====================
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        float qk_raw = 0.0f;
        const int key_offset = (((batch * H) + head) * T_k + t_k) * D;
        
        // Compute Q @ K^T
        for (int d = 0; d < D; d++) {
            qk_raw += Q[query_offset + d] * K[key_offset + d];
        }
        
        s_qk[t_k] = qk_raw;
        qk_scores[attn_offset + t_k] = qk_raw;  // Store for later kernels
        s_logits[t_k] = qk_raw / scale;
    }
    __syncthreads();
    
    // ==================== 2. Compute unmasked attention ====================
    // Find max value for numerical stability
    float thread_max = -CUDART_INF_F;
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        if (s_logits[t_k] > thread_max) thread_max = s_logits[t_k];
    }
    
    // Warp-level reduction for max
    warp_reduce_max(thread_max);
    if (lane_id == 0) s_temp[warp_id] = thread_max;
    __syncthreads();
    
    // Block-level reduction for max
    float block_max = -CUDART_INF_F;
    if (tid < (blockDim.x + 31) / 32) {
        float val = (tid < (blockDim.x + 31) / 32) ? s_temp[tid] : -CUDART_INF_F;
        warp_reduce_max(val);
        if (tid == 0) block_max = val;
    }
    block_max = __shfl_sync(0xffffffff, block_max, 0);
    
    // Compute exp and sum
    float thread_sum = 0.0f;
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        float exp_val = expf(s_logits[t_k] - block_max);
        s_logits[t_k] = exp_val;
        thread_sum += exp_val;
    }
    
    // Warp-level reduction for sum
    warp_reduce_sum(thread_sum);
    if (lane_id == 0) s_temp[warp_id] = thread_sum;
    __syncthreads();
    
    // Block-level reduction for sum
    float block_sum = 0.0f;
    if (tid < (blockDim.x + 31) / 32) {
        float val = (tid < (blockDim.x + 31) / 32) ? s_temp[tid] : 0.0f;
        warp_reduce_sum(val);
        if (tid == 0) block_sum = val;
    }
    block_sum = __shfl_sync(0xffffffff, block_sum, 0) + epsilon;
    
    // Normalize to get attention weights
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        float attn_weight = s_logits[t_k] / block_sum;
        unmasked_attn[attn_offset + t_k] = attn_weight;
    }
    __syncthreads();
    
    // ==================== 3. Compute masked attention (if needed) ====================
    if (has_mask) {
        // Recompute logits with mask
        for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
            s_logits[t_k] = (s_qk[t_k] / scale) + mask[mask_base + t_k];
        }
        __syncthreads();
        
        // Find max with mask
        thread_max = -CUDART_INF_F;
        for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
            if (s_logits[t_k] > thread_max) thread_max = s_logits[t_k];
        }
        
        warp_reduce_max(thread_max);
        if (lane_id == 0) s_temp[warp_id] = thread_max;
        __syncthreads();
        
        block_max = -CUDART_INF_F;
        if (tid < (blockDim.x + 31) / 32) {
            float val = (tid < (blockDim.x + 31) / 32) ? s_temp[tid] : -CUDART_INF_F;
            warp_reduce_max(val);
            if (tid == 0) block_max = val;
        }
        block_max = __shfl_sync(0xffffffff, block_max, 0);
        
        // Compute exp and sum with mask
        thread_sum = 0.0f;
        for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
            float exp_val = expf(s_logits[t_k] - block_max);
            s_logits[t_k] = exp_val;
            thread_sum += exp_val;
        }
        
        warp_reduce_sum(thread_sum);
        if (lane_id == 0) s_temp[warp_id] = thread_sum;
        __syncthreads();
        
        block_sum = 0.0f;
        if (tid < (blockDim.x + 31) / 32) {
            float val = (tid < (blockDim.x + 31) / 32) ? s_temp[tid] : 0.0f;
            warp_reduce_sum(val);
            if (tid == 0) block_sum = val;
        }
        block_sum = __shfl_sync(0xffffffff, block_sum, 0) + epsilon;
        
        // Store masked attention weights
        for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
            masked_attn[attn_offset + t_k] = s_logits[t_k] / block_sum;
        }
    } else {
        // Copy unmasked to masked
        for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
            masked_attn[attn_offset + t_k] = unmasked_attn[attn_offset + t_k];
        }
    }
}

// Kernel 2: Compute attention output
__global__ void compute_attention_output_kernel(
    const float* __restrict__ V,
    const float* __restrict__ masked_attn,
    float* __restrict__ attention_out,
    const int B, const int H, const int T_q, const int T_k, const int D
) {
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int t_q = blockIdx.x;
    const int d = threadIdx.x;

    if (t_q >= T_q || head >= H || batch >= B || d >= D) return;

    const int out_offset = (((batch * H) + head) * T_q + t_q) * D + d;
    const int attn_offset = (((batch * H) + head) * T_q + t_q) * T_k;
    
    float output_val = 0.0f;
    for (int t_k = 0; t_k < T_k; t_k++) {
        const int v_offset = (((batch * H) + head) * T_k + t_k) * D + d;
        output_val += masked_attn[attn_offset + t_k] * V[v_offset];
    }
    
    attention_out[out_offset] = output_val;
}

// Kernel 3: Compute relevance propagation
__global__ void compute_relevance_kernel(
    const float* __restrict__ R_out,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ qk_scores,
    const float* __restrict__ unmasked_attn,
    const float* __restrict__ masked_attn,
    const float* __restrict__ attention_out,
    float* __restrict__ R_Q,
    float* __restrict__ R_K,
    float* __restrict__ R_V,
    float* __restrict__ R_mask,
    const int B, const int H, const int T_q, const int T_k, const int D,
    const float epsilon, const bool has_mask
) {
    extern __shared__ float smem[];
    
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int t_q = blockIdx.x;
    const int tid = threadIdx.x;

    if (t_q >= T_q || head >= H || batch >= B) return;

    // Shared memory for this block's computations
    float* s_rel_norm_attn = smem;          // [D]
    float* s_R_QK = s_rel_norm_attn + D;    // [T_k]
    float* s_rel_norm_QK = s_R_QK + T_k;    // [T_k]
    
    const int base_offset = (((batch * H) + head) * T_q + t_q) * D;
    const int attn_offset = (((batch * H) + head) * T_q + t_q) * T_k;
    
    // ==================== 1. Stabilize attention output ====================
    if (tid < D) {
        float attn_out = attention_out[base_offset + tid] * 2.0f;
        float stab_attn = stabilize_value(attn_out, epsilon);
        s_rel_norm_attn[tid] = R_out[base_offset + tid] / stab_attn;
    }
    __syncthreads();
    
    // ==================== 2. Compute R_QK and stabilize ====================
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        float r_qk = 0.0f;
        const int v_offset = (((batch * H) + head) * T_k + t_k) * D;
        
        for (int d = 0; d < D; d++) {
            r_qk += s_rel_norm_attn[d] * V[v_offset + d];
        }
        s_R_QK[t_k] = r_qk * unmasked_attn[attn_offset + t_k];
        
        // Stabilize QK output
        float qk_val = qk_scores[attn_offset + t_k] * 2.0f;
        float stab_qk = stabilize_value(qk_val, epsilon);
        s_rel_norm_QK[t_k] = s_R_QK[t_k] / stab_qk;
    }
    __syncthreads();
    
    // ==================== 3. Compute R_Q ====================
    if (tid < D) {
        float r_q = 0.0f;
        for (int t_k = 0; t_k < T_k; t_k++) {
            const int k_offset = (((batch * H) + head) * T_k + t_k) * D + tid;
            r_q += s_rel_norm_QK[t_k] * K[k_offset];
        }
        R_Q[base_offset + tid] = r_q * Q[base_offset + tid];
    }
    
    // ==================== 4. Compute R_K and R_V ====================
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        const int kv_offset = (((batch * H) + head) * T_k + t_k) * D;
        
        for (int d = 0; d < D; d++) {
            // R_K computation
            float r_k = Q[base_offset + d] * s_rel_norm_QK[t_k];
            atomicAdd(&R_K[kv_offset + d], r_k * K[kv_offset + d]);
            
            // R_V computation  
            float r_v = unmasked_attn[attn_offset + t_k] * s_rel_norm_attn[d];
            atomicAdd(&R_V[kv_offset + d], r_v * V[kv_offset + d]);
        }
    }
    
    // ==================== 5. Compute R_mask ====================
    if (has_mask) {
        for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
            float delta_attn = unmasked_attn[attn_offset + t_k] - masked_attn[attn_offset + t_k];
            float v_sum = 0.0f;
            const int v_offset = (((batch * H) + head) * T_k + t_k) * D;
                
            // Sum over D dimension
            for (int d = 0; d < D; d++) {
                v_sum += V[v_offset + d];
            }
                
            int mask_idx = batch * T_q * T_k + t_q * T_k + t_k;
            atomicAdd(&R_mask[mask_idx], delta_attn * v_sum);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>> 
launch_multi_kernel_attention_relevance(
    const torch::Tensor& R_out,
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::optional<torch::Tensor>& mask,
    const torch::optional<float>& scale
) {
    // Validate inputs
    TORCH_CHECK(R_out.is_cuda() && Q.is_cuda() && K.is_cuda() && V.is_cuda(), 
                "All inputs must be CUDA tensors");
    TORCH_CHECK(R_out.dtype() == torch::kFloat32, "Only float32 supported");
    
    const auto sizes = Q.sizes();
    const int B = sizes[0];
    const int H = sizes[1];
    const int T_q = sizes[2];
    const int D = sizes[3];
    const int T_k = K.size(2);
    
    // Get scale from input or default to sqrt(D)
    float scale_val = scale.has_value() ? scale.value() : std::sqrt(static_cast<float>(D));
    float epsilon_val = 1e-9;

    // Prepare outputs
    auto R_Q = torch::zeros_like(Q);
    auto R_K = torch::zeros_like(K);
    auto R_V = torch::zeros_like(V);
    auto R_mask = torch::zeros({B, 1, T_q, T_k}, Q.options());

    // Intermediate tensors
    auto qk_scores = torch::empty({B, H, T_q, T_k}, Q.options());
    auto unmasked_attn = torch::empty({B, H, T_q, T_k}, Q.options());
    auto masked_attn = torch::empty({B, H, T_q, T_k}, Q.options());
    auto attention_out = torch::empty_like(Q);

    // Mask parameters
    bool has_mask = mask.has_value();
    int mask_stride = 0;
    if (has_mask) {
        auto mask_fill = mask.value();
        TORCH_CHECK(mask_fill.size(0) == B, "Mask batch size mismatch");
        mask_stride = mask_fill.size(1) == 1 ? T_q * T_k : H * T_q * T_k;
    }

    // Kernel 1: Compute attention weights
    {
        const int threads = min(256, ((T_k + 31) / 32) * 32);
        const dim3 blocks(T_q, H, B);
        const size_t smem_size = (2 * T_k + 32) * sizeof(float);  // qk + logits + temp
        
        compute_attention_weights_kernel<<<blocks, threads, smem_size>>>(
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            has_mask ? mask.value().data_ptr<float>() : nullptr,
            qk_scores.data_ptr<float>(),
            unmasked_attn.data_ptr<float>(),
            masked_attn.data_ptr<float>(),
            B, H, T_q, T_k, D, scale_val, epsilon_val, has_mask, mask_stride
        );
    }

    // Kernel 2: Compute attention output
    {
        const int threads = min(D, 1024);
        const dim3 blocks(T_q, H, B);
        
        compute_attention_output_kernel<<<blocks, threads>>>(
            V.data_ptr<float>(),
            masked_attn.data_ptr<float>(),
            attention_out.data_ptr<float>(),
            B, H, T_q, T_k, D
        );
    }

    // Kernel 3: Compute relevance
    {
        const int threads = min(256, max(D, ((T_k + 31) / 32) * 32));
        const dim3 blocks(T_q, H, B);
        const size_t smem_size = (D + 2 * T_k) * sizeof(float);  // rel_norm_attn + R_QK + rel_norm_QK
        
        compute_relevance_kernel<<<blocks, threads, smem_size>>>(
            R_out.data_ptr<float>(),
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            V.data_ptr<float>(),
            qk_scores.data_ptr<float>(),
            unmasked_attn.data_ptr<float>(),
            masked_attn.data_ptr<float>(),
            attention_out.data_ptr<float>(),
            R_Q.data_ptr<float>(),
            R_K.data_ptr<float>(),
            R_V.data_ptr<float>(),
            R_mask.data_ptr<float>(),
            B, H, T_q, T_k, D, epsilon_val, has_mask
        );
    }
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA kernel execution failed: ", cudaGetErrorString(error));

    cudaDeviceSynchronize();

    return std::make_tuple(R_Q, R_K, R_V, R_mask);
}
"""

attention_cuda_declaration = r"""
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>> 
launch_multi_kernel_attention_relevance(
    const torch::Tensor& R_out,
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::optional<torch::Tensor>& mask,
    const torch::optional<float>& scale
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
]

extra_flags.extend(get_cuda_arch_flags())

multi_kernel_attention_cuda_ops = load_inline(
    name="multi_kernel_attention_cuda",
    cpp_sources=attention_cuda_declaration,
    cuda_sources=attention_cuda_source,
    functions=["launch_multi_kernel_attention_relevance"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def calculate_wt_self_attention_multi_kernel(R_out, Q, K, V, mask, scale):
    """
    Multi-kernel implementation for attention relevance computation.
    
    This approach splits the computation into three specialized kernels:
    1. Attention weights computation (handles softmax)
    2. Attention output computation (matrix multiplication)  
    3. Relevance propagation (backpropagation through attention)
    
    B : batch size (Can change)
    H : number of heads (Fixed according to the model architecture, might go as high as 96)
    T_q : query length (depends on query sequence length)
    T_k : key length (depends on key sequence length)
    D : dimension of the query and key (Fixed according to the model architecture, might go as high as 128)

    Benefits:
    - Reduced shared memory usage per kernel
    - Better memory access patterns
    - Scalable to much longer sequences
    - Easier to optimize each stage independently
    
    Args:
        R_out: Relevance tensor of shape [B, H, T_q, D] 
        Q: Query tensor of shape [B, H, T_q, D]         
        K: Key tensor of shape [B, H, T_k, D]          
        V: Value tensor of shape [B, H, T_k, D]         
        mask: Optional additive mask of shape [B, 1, T_q, T_k] or [B, H, T_q, T_k] 
        scale: Optional scaling factor (default: sqrt(D))
        
    Returns:
        Tuple containing:
            - R_Q: Query relevance tensor, same shape as Q
            - R_K: Key relevance tensor, same shape as K
            - R_V: Value relevance tensor, same shape as V
            - R_mask: Mask relevance tensor, same shape as mask (or zeros)
    """
    D = Q.size(3)
    device = torch.device("cuda")
    scale = torch.sqrt(torch.tensor(D, dtype=torch.float32, device=device)) if scale is None else scale

    return multi_kernel_attention_cuda_ops.launch_multi_kernel_attention_relevance(
        R_out, Q, K, V, mask, scale
    )    