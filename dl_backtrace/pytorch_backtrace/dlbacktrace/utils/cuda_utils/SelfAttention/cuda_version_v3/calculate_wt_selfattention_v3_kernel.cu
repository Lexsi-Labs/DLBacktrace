#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Fused Softmax
// ═══════════════════════════════════════════════════════════════════

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
    
    thread_max = warp_reduce_max(thread_max);
    if (threadIdx.x % 32 == 0) {
        shared_max[threadIdx.x / 32] = thread_max;
    }
    __syncthreads();
    
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
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    
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

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Fused Stabilize and Normalize
// ═══════════════════════════════════════════════════════════════════

__global__ void fused_stabilize_normalize_kernel(
    const float* __restrict__ numerator,
    const float* __restrict__ denominator,
    float* __restrict__ output,
    const int total_elements,
    const float epsilon
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        float denom_val = denominator[idx] * 2.0f;
        
        float abs_denom = fabsf(denom_val);
        float sign_denom = (denom_val == 0.0f) ? 1.0f : copysignf(1.0f, denom_val);
        float stable_denom = (abs_denom < epsilon) ? (epsilon * sign_denom) : denom_val;
        
        output[idx] = numerator[idx] / stable_denom;
    }
}

void launch_fused_stabilize_normalize(
    torch::Tensor numerator,
    torch::Tensor denominator,
    torch::Tensor output,
    float epsilon
) {
    TORCH_CHECK(numerator.is_cuda(), "numerator must be a CUDA tensor");
    TORCH_CHECK(denominator.is_cuda(), "denominator must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    
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

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Fused Conservation
// ═══════════════════════════════════════════════════════════════════

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
    const int slice_offset = blockIdx.y * T * D;
    
    extern __shared__ float shared_data[];
    float* s_p_sum = shared_data;
    float* s_n_sum = shared_data + blockDim.x;
    float* s_abs_sum = shared_data + 2 * blockDim.x;
    
    for (int sign_pass = 0; sign_pass < 2; sign_pass++) {
        float thread_p_sum = 0.0f;
        float thread_n_sum = 0.0f;
        float thread_abs_sum = 0.0f;
        
        for (int i = threadIdx.x; i < T * D; i += blockDim.x) {
            float wt_val = wts[slice_offset + i];
            float wt_component = (sign_pass == 0) ? fmaxf(wt_val, 0.0f) : fmaxf(-wt_val, 0.0f);
            float inp_val = inp[slice_offset + i];
            
            if (inp_val > 0.0f) {
                thread_p_sum += inp_val;
            } else if (inp_val < 0.0f) {
                thread_n_sum += -inp_val;
            }
            thread_abs_sum += fabsf(wt_component);
        }
        
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
    TORCH_CHECK(wts.is_cuda(), "wts must be a CUDA tensor");
    TORCH_CHECK(inp.is_cuda(), "inp must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    
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
