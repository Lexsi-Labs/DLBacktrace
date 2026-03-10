#include <torch/extension.h>
#include <cuda_runtime.h>

// Warp-level reduction using shuffle
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduction_kernel(
    const float* __restrict__ output,
    const float* __restrict__ wts,
    int n_output,
    int n_wts,
    float* p_sum_out,
    float* n_sum_out,
    float* total_wt_out
) {
    __shared__ float shared_p[32];
    __shared__ float shared_n[32];
    __shared__ float shared_w[32];

    int tid    = threadIdx.x;
    int lane   = tid % 32;
    int warp_id = tid / 32;

    float p_sum = 0.0f, n_sum = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + tid; idx < n_output; idx += blockDim.x * gridDim.x) {
        float val = output[idx];
        if (val > 0.0f) p_sum += val;
        else if (val < 0.0f) n_sum += fabsf(val);
    }
    p_sum = warp_reduce_sum(p_sum);
    n_sum = warp_reduce_sum(n_sum);
    if (lane == 0) { shared_p[warp_id] = p_sum; shared_n[warp_id] = n_sum; }
    __syncthreads();
    if (tid < 32) {
        p_sum = (tid < blockDim.x / 32) ? shared_p[tid] : 0.0f;
        n_sum = (tid < blockDim.x / 32) ? shared_n[tid] : 0.0f;
        p_sum = warp_reduce_sum(p_sum);
        n_sum = warp_reduce_sum(n_sum);
        if (tid == 0) { atomicAdd(p_sum_out, p_sum); atomicAdd(n_sum_out, n_sum); }
    }

    float total_w = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + tid; idx < n_wts; idx += blockDim.x * gridDim.x)
        total_w += wts[idx];
    total_w = warp_reduce_sum(total_w);
    if (lane == 0) shared_w[warp_id] = total_w;
    __syncthreads();
    if (tid < 32) {
        total_w = (tid < blockDim.x / 32) ? shared_w[tid] : 0.0f;
        total_w = warp_reduce_sum(total_w);
        if (tid == 0) atomicAdd(total_wt_out, total_w);
    }
}

__global__ void relevance_proj_kernel(
    const float* __restrict__ output,
    float* __restrict__ result,
    int n,
    float p_sum,
    float n_sum,
    float total_wt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float total_sum = p_sum + n_sum;
    float p_agg_wt  = (total_sum > 0.0f && p_sum > 0.0f) ? p_sum / total_sum : 0.0f;
    float n_agg_wt  = (total_sum > 0.0f && n_sum > 0.0f) ? n_sum / total_sum : 0.0f;
    float p_sum_safe = (p_sum != 0.0f) ? p_sum : 1.0f;
    float n_sum_safe = (n_sum != 0.0f) ? n_sum : 1.0f;

    float val = output[idx];
    float res = 0.0f;
    if (val > 0.0f)
        res = (p_agg_wt > 0.0f) ? (val / p_sum_safe) * total_wt * p_agg_wt : 0.0f;
    else if (val < 0.0f)
        res = (n_agg_wt > 0.0f) ? (val / n_sum_safe) * total_wt * n_agg_wt * -1.0f : 0.0f;
    result[idx] = res;
}

void launch_kernel(torch::Tensor wts, torch::Tensor output, torch::Tensor result) {
    const int n_output = output.numel();
    const int n_wts    = wts.numel();

    float *d_p_sum, *d_n_sum, *d_total_wt;
    cudaMalloc(&d_p_sum,    sizeof(float));
    cudaMalloc(&d_n_sum,    sizeof(float));
    cudaMalloc(&d_total_wt, sizeof(float));
    cudaMemset(d_p_sum,    0, sizeof(float));
    cudaMemset(d_n_sum,    0, sizeof(float));
    cudaMemset(d_total_wt, 0, sizeof(float));

    const int block_size = 256;
    const int num_blocks = min(1024, (n_output + block_size - 1) / block_size);
    reduction_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(), wts.data_ptr<float>(),
        n_output, n_wts, d_p_sum, d_n_sum, d_total_wt);

    float p_sum, n_sum, total_wt;
    cudaMemcpy(&p_sum,    d_p_sum,    sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&n_sum,    d_n_sum,    sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_wt, d_total_wt, sizeof(float), cudaMemcpyDeviceToHost);

    const int main_blocks = (n_output + block_size - 1) / block_size;
    relevance_proj_kernel<<<main_blocks, block_size>>>(
        output.data_ptr<float>(), result.data_ptr<float>(),
        n_output, p_sum, n_sum, total_wt);

    cudaFree(d_p_sum); cudaFree(d_n_sum); cudaFree(d_total_wt);
}
