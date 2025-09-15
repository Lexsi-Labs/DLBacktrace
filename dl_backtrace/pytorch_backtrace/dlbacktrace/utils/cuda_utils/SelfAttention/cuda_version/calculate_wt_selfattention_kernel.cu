#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

__device__ __forceinline__ float stabilize_value(float x, float epsilon) {
    float abs_x = fabsf(x);
    if (abs_x < epsilon) {
        // Handle the sign logic: sign(x + (x == 0))
        // This means: if x == 0, use +1, otherwise use sign(x)
        float sign_val = (x == 0.0f) ? 1.0f : ((x > 0.0f) ? 1.0f : -1.0f);
        return epsilon * sign_val;
    }
    return x;
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

__global__ void fused_attention_relevance_kernel(
    const float* __restrict__ R_out,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ mask,
    float* __restrict__ R_Q,
    float* __restrict__ R_K,
    float* __restrict__ R_V,
    float* __restrict__ R_mask,
    const int B, const int H, const int T_q, const int T_k, const int D,
    const float scale, const float epsilon,
    const bool has_mask, const int mask_stride
) {
    extern __shared__ float smem[];
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int t_q = blockIdx.x;
    const int tid = threadIdx.x;

    if (t_q >= T_q || head >= H || batch >= B) return;
    
    // Shared memory pointers
    float* s_qk_raw = smem;                     // [T_k] - Store raw QK output
    float* s_logits = s_qk_raw + T_k;           // [T_k]
    float* s_unmasked_attn = s_logits + T_k;   // [T_k]
    float* s_masked_attn = s_unmasked_attn + T_k; // [T_k]
    float* s_attention_out = s_masked_attn + T_k; // [D]
    float* s_rel_norm_attn = s_attention_out + D; // [D]
    float* s_R_QK = s_rel_norm_attn + D;        // [T_k]
    float* s_rel_norm_QK = s_R_QK + T_k;        // [T_k]
    float* s_query = s_rel_norm_QK + T_k;        // [D]
    
    // Global memory offsets
    const int base_offset = (((batch * H) + head) * T_q + t_q) * D;
    const int mask_base = (batch * mask_stride) + t_q * T_k;
    
    // Load query vector into shared memory
    if (tid < D) {
        s_query[tid] = Q[base_offset + tid];
    }
    __syncthreads();
    
    // ==================== 1. Compute attention logits ====================
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        float qk_raw = 0.0f;
        const int k_offset = (((batch * H) + head) * T_k + t_k) * D;
        
        #pragma unroll
        for (int d = 0; d < D; d++) {
            qk_raw += s_query[d] * K[k_offset + d];
        }
        s_qk_raw[t_k] = qk_raw;
        s_logits[t_k] = qk_raw / scale;
    }
    __syncthreads();
    
    // ==================== 2. Compute attention weights ====================
    // Unmasked attention
    float max_val = -CUDART_INF_F;
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        if (s_logits[t_k] > max_val) max_val = s_logits[t_k];
    }
    warp_reduce_max(max_val);
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    
    float sum_exp = 0.0f;
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        float val = expf(s_logits[t_k] - max_val);
        s_unmasked_attn[t_k] = val;
        sum_exp += val;
    }
    warp_reduce_sum(sum_exp);
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0) + epsilon;
    
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        s_unmasked_attn[t_k] /= sum_exp;
    }
    
    // Masked attention (if applicable)
    if (has_mask) {
        for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
            s_logits[t_k] = (s_qk_raw[t_k] / scale) + mask[mask_base + t_k];
        }
        __syncthreads();
        
        max_val = -CUDART_INF_F;
        for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
            if (s_logits[t_k] > max_val) max_val = s_logits[t_k];
        }
        warp_reduce_max(max_val);
        max_val = __shfl_sync(0xffffffff, max_val, 0);
        
        sum_exp = 0.0f;
        for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
            float val = expf(s_logits[t_k] - max_val);
            s_masked_attn[t_k] = val;
            sum_exp += val;
        }
        warp_reduce_sum(sum_exp);
        sum_exp = __shfl_sync(0xffffffff, sum_exp, 0) + epsilon;
        
        for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
            s_masked_attn[t_k] /= sum_exp;
        }
    } else {
        for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
            s_masked_attn[t_k] = s_unmasked_attn[t_k];
        }
    }
    __syncthreads();
    
    // ==================== 3. Compute attention output ====================
    for (int d = tid; d < D; d += blockDim.x) {
        float out_val = 0.0f;
        for (int t_k = 0; t_k < T_k; t_k++) {
            const int v_offset = (((batch * H) + head) * T_k + t_k) * D + d;
            out_val += s_masked_attn[t_k] * V[v_offset];
        }
        s_attention_out[d] = out_val;
    }
    __syncthreads();
    
    // ==================== 4. Stabilize attention output ====================
    for (int d = tid; d < D; d += blockDim.x) {
        float attn_out = s_attention_out[d] * 2.0f;
        float stab_attn = stabilize_value(attn_out, epsilon);
        s_rel_norm_attn[d] = R_out[base_offset + d] / stab_attn;
    }
    __syncthreads();
    
    // ==================== 5. Compute R_QK and stabilize ====================
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        float r_qk = 0.0f;
        const int v_offset = (((batch * H) + head) * T_k + t_k) * D;
        
        #pragma unroll
        for (int d = 0; d < D; d++) {
            r_qk += s_rel_norm_attn[d] * V[v_offset + d];
        }
        s_R_QK[t_k] = r_qk * s_unmasked_attn[t_k];
    }

    __syncthreads();

    // Stabilize QK output
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        float qk_val = s_qk_raw[t_k] * 2.0f;
        float stab_qk = stabilize_value(qk_val, epsilon);
        s_rel_norm_QK[t_k] = s_R_QK[t_k] / stab_qk;
    }
    __syncthreads();
    
    // ==================== 6. Compute R_Q ====================
    for (int d = tid; d < D; d += blockDim.x) {
        float r_q = 0.0f;
        for (int t_k = 0; t_k < T_k; t_k++) {
            const int k_offset = (((batch * H) + head) * T_k + t_k) * D + d;
            r_q += s_rel_norm_QK[t_k] * K[k_offset];
        }
        R_Q[base_offset + d] = r_q * s_query[d];
    }
    
    // ==================== 7. Compute R_K and R_V ====================
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        const int k_offset = (((batch * H) + head) * T_k + t_k) * D;
        const int v_offset = k_offset;
        
        for (int d = 0; d < D; d++) {
            // R_K
            float r_k = s_query[d] * s_rel_norm_QK[t_k];
            atomicAdd(&R_K[k_offset + d], r_k * K[k_offset + d]);
            
            // R_V (use unmasked attention like PyTorch version)
            float r_v = s_unmasked_attn[t_k] * s_rel_norm_attn[d];
            atomicAdd(&R_V[v_offset + d], r_v * V[v_offset + d]);
        }
    }
    
    // ==================== 8. Compute R_mask ====================
    // Implement einsum('bhqk,bhkd->bhqk', delta_A, V) and sum over heads
    for (int t_k = tid; t_k < T_k; t_k += blockDim.x) {
        float delta_attn = s_unmasked_attn[t_k] - s_masked_attn[t_k];
        float v_sum = 0.0f;
        const int v_offset = (((batch * H) + head) * T_k + t_k) * D;
            
        // Sum over D dimension: einsum bhqk,bhkd->bhqk means sum over d
        #pragma unroll
        for (int d = 0; d < D; d++) {
            v_sum += V[v_offset + d];
        }
            
        int mask_idx;
        if (has_mask) {
            mask_idx = batch * T_q * T_k + t_q * T_k + t_k;
        } else {
            mask_idx = (((batch * H) + head) * T_q + t_q) * T_k + t_k;
        }
        atomicAdd(&R_mask[mask_idx], delta_attn * v_sum);
    }
}

// Host function to launch the kernel
extern "C" void launch_fused_attention_relevance(
    const float* R_out, const float* Q, const float* K, const float* V,
    const float* mask, float* R_Q, float* R_K, float* R_V, float* R_mask,
    int B, int H, int T_q, int T_k, int D, float scale, float epsilon,
    bool has_mask, int mask_stride
) {
    
    // Kernel configuration
    const int threads = 256;
    const dim3 blocks(T_q, H, B);
    
    // Calculate shared memory size
    const size_t smem_size = 
        (T_k * 6  +  // qk_raw, logits, unmasked_attn, masked_attn, R_QK, rel_norm_QK
        D * 3  + // attention_out, rel_norm_attn, query 
        256) * sizeof(float);     
    
    fused_attention_relevance_kernel<<<blocks, threads, smem_size>>>(
        R_out, Q, K, V, mask, R_Q, R_K, R_V, R_mask,
        B, H, T_q, T_k, D, scale, epsilon, has_mask, mask_stride
    );
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(error));
    }
}
