#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void wt_router_logits_kernel(
    const float* __restrict__ wts,
    const float* __restrict__ inp,
    const float* __restrict__ W_router,
    float* __restrict__ output,
    const int n_samples,
    const int n_features
) {
    const int feat_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (sample_idx >= n_samples || feat_idx >= n_features) return;

    const float act_range_lower = -1.0f;
    const float act_range_upper =  2.0f;

    const float wt_val = wts[sample_idx];

    extern __shared__ float s_W_router[];

    for (int k = threadIdx.y * blockDim.x + threadIdx.x;
         k < n_features;
         k += blockDim.x * blockDim.y) {
        if (blockIdx.x * blockDim.x < n_features) {
            int col_idx = min(blockIdx.x * blockDim.x + k % blockDim.x, n_features - 1);
            s_W_router[k] = W_router[k * n_features + col_idx];
        }
    }
    __syncthreads();

    float p_sum = 0.0f;
    float n_sum = 0.0f;
    float t_sum = 0.0f;

    const int inp_offset = sample_idx * n_features;

    for (int k = 0; k < n_features; ++k) {
        const float contrib = inp[inp_offset + k] * W_router[k * n_features + feat_idx];
        if (contrib > 0.0f) {
            p_sum += contrib;
            t_sum += contrib;
        } else if (contrib < 0.0f) {
            n_sum -= contrib;
            t_sum += contrib;
        }
    }

    if (t_sum < act_range_lower) p_sum = 0.0f;
    if (t_sum > act_range_upper) n_sum = 0.0f;

    const float total_sum = p_sum + n_sum;
    const float p_agg_wt  = (p_sum > 0.0f && total_sum > 0.0f) ? p_sum / total_sum : 0.0f;
    const float n_agg_wt  = (n_sum > 0.0f && total_sum > 0.0f) ? n_sum / total_sum : 0.0f;

    const float p_sum_norm = (p_sum != 0.0f) ? p_sum : 1.0f;
    const float n_sum_norm = (n_sum != 0.0f) ? n_sum : 1.0f;

    const float p_weight = wt_val * p_agg_wt / p_sum_norm;
    const float n_weight = wt_val * n_agg_wt / n_sum_norm;

    float result = 0.0f;
    for (int k = 0; k < n_features; ++k) {
        const float contrib = inp[inp_offset + k] * W_router[k * n_features + feat_idx];
        if (contrib > 0.0f)  result += contrib * p_weight;
        else if (contrib < 0.0f) result -= contrib * n_weight;
    }

    output[sample_idx * n_features + feat_idx] = result;
}

void launch_kernel(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor W_router,
    torch::Tensor output
) {
    const int n_samples  = inp.size(0);
    const int n_features = inp.size(1);

    dim3 blockDim(16, 16);
    dim3 gridDim(
        (n_features + blockDim.x - 1) / blockDim.x,
        (n_samples  + blockDim.y - 1) / blockDim.y
    );
    const int shared_mem_size = n_features * sizeof(float);

    wt_router_logits_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        wts.data_ptr<float>(),
        inp.data_ptr<float>(),
        W_router.data_ptr<float>(),
        output.data_ptr<float>(),
        n_samples,
        n_features
    );
}
