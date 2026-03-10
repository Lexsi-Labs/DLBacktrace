#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float swish_activation(float x, float beta) {
    float x_beta = fmaxf(-500.0f, fminf(500.0f, beta * x));
    return x / (1.0f + expf(-x_beta));
}

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

    int tid    = threadIdx.x;
    int idx    = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_pos = 0.0f, local_neg = 0.0f, local_wt = 0.0f;
    for (int i = idx; i < N; i += stride) {
        float val = output[i];
        if (val > 0.0f)       local_pos += val;
        else if (val < 0.0f)  local_neg += val;
        local_wt += wts[i];
    }
    shared_pos[tid] = local_pos;
    shared_neg[tid] = local_neg;
    shared_wt[tid]  = local_wt;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_pos[tid] += shared_pos[tid + s];
            shared_neg[tid] += shared_neg[tid + s];
            shared_wt[tid]  += shared_wt[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(pos_sum_out,      shared_pos[0]);
        atomicAdd(neg_sum_out,      shared_neg[0]);
        atomicAdd(total_weight_out, shared_wt[0]);
    }
}

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
    const float eps  = 1e-6f;

    neg_sum_base = -neg_sum_base;

    float t_sum_base = pos_sum_base - neg_sum_base;
    float t_act = swish_activation(t_sum_base,  beta);
    float p_act = swish_activation(pos_sum_base, beta);
    float n_act = swish_activation(-neg_sum_base, beta);

    float pos_sum = pos_sum_base;
    float neg_sum = neg_sum_base;

    if (t_sum_base < -6.0f) pos_sum = 0.0f;

    if (pos_sum_base > 0.0f && neg_sum_base > 0.0f) {
        if (fabsf(t_act - p_act) < eps)       neg_sum = 0.0f;
        else if (fabsf(t_act - n_act) < eps)  pos_sum = 0.0f;
    }

    float denominator = pos_sum + neg_sum;
    float pos_agg_wt  = (pos_sum > 0.0f && denominator > 0.0f) ? pos_sum / denominator : 0.0f;
    float neg_agg_wt  = (neg_sum > 0.0f && denominator > 0.0f) ? neg_sum / denominator : 0.0f;
    float pos_sum_norm = (pos_sum != 0.0f) ? pos_sum : 1.0f;
    float neg_sum_norm = (neg_sum != 0.0f) ? neg_sum : 1.0f;

    float val    = output[idx];
    float result = 0.0f;
    if (val > 0.0f && pos_agg_wt != 0.0f)
        result = (val / pos_sum_norm) * pos_agg_wt * total_weight;
    else if (val < 0.0f && neg_agg_wt != 0.0f)
        result = (val / neg_sum_norm) * neg_agg_wt * total_weight * -1.0f;
    wt_mat_total[idx] = result;
}

void launch_kernel(
    torch::Tensor output,
    torch::Tensor wts,
    torch::Tensor wt_mat_total
) {
    const int N       = output.numel();
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    float *d_pos_sum, *d_neg_sum, *d_total_weight;
    cudaMalloc(&d_pos_sum,      sizeof(float));
    cudaMalloc(&d_neg_sum,      sizeof(float));
    cudaMalloc(&d_total_weight, sizeof(float));
    cudaMemset(d_pos_sum,      0, sizeof(float));
    cudaMemset(d_neg_sum,      0, sizeof(float));
    cudaMemset(d_total_weight, 0, sizeof(float));

    compute_statistics_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), wts.data_ptr<float>(), N,
        d_pos_sum, d_neg_sum, d_total_weight);

    float pos_sum_base, neg_sum_base, total_weight;
    cudaMemcpy(&pos_sum_base,  d_pos_sum,      sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&neg_sum_base,  d_neg_sum,      sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_weight,  d_total_weight, sizeof(float), cudaMemcpyDeviceToHost);

    apply_gating_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), N,
        pos_sum_base, neg_sum_base, total_weight,
        wt_mat_total.data_ptr<float>());

    cudaFree(d_pos_sum); cudaFree(d_neg_sum); cudaFree(d_total_weight);
}
