#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relevance_propagation_kernel(
    const float* __restrict__ wts,
    const float* __restrict__ inp,
    const float* __restrict__ w,
    float* __restrict__ relevance_input,
    const int batch_size,
    const int seq_len,
    const int output_features,
    const int input_features
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (b >= batch_size || s >= seq_len || i >= input_features) return;

    const int inp_base = b * seq_len * input_features + s * input_features;
    const int wts_base = b * seq_len * output_features + s * output_features;
    const float inp_val = inp[inp_base + i];

    float relevance_sum = 0.0f;

    for (int o = 0; o < output_features; ++o) {
        const float wts_val     = wts[wts_base + o];
        const float contribution = w[o * input_features + i] * inp_val;

        float p_sum = 0.0f, n_sum = 0.0f;
        for (int j = 0; j < input_features; ++j) {
            const float contrib_j = w[o * input_features + j] * inp[inp_base + j];
            if (contrib_j > 0.0f)       p_sum += contrib_j;
            else if (contrib_j < 0.0f)  n_sum -= contrib_j;
        }

        const float total_sum = p_sum + n_sum;
        const float p_agg_wt  = (p_sum > 0.0f && total_sum != 0.0f) ? p_sum / total_sum : 0.0f;
        const float n_agg_wt  = (n_sum > 0.0f && total_sum != 0.0f) ? n_sum / total_sum : 0.0f;

        float relevance_contrib = 0.0f;
        if      (contribution > 0.0f && p_sum != 0.0f)
            relevance_contrib =  (contribution / p_sum) * wts_val * p_agg_wt;
        else if (contribution < 0.0f && n_sum != 0.0f)
            relevance_contrib = -(contribution / n_sum) * wts_val * n_agg_wt;
        relevance_sum += relevance_contrib;
    }

    relevance_input[inp_base + i] = relevance_sum;
}

void launch_kernel(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor w,
    torch::Tensor relevance_input
) {
    const int batch_size      = wts.size(0);
    const int seq_len         = wts.size(1);
    const int output_features = wts.size(2);
    const int input_features  = inp.size(2);

    const dim3 threads(16, 8, 1);
    const dim3 blocks(
        (input_features  + threads.x - 1) / threads.x,
        (seq_len         + threads.y - 1) / threads.y,
        batch_size
    );

    relevance_propagation_kernel<<<blocks, threads>>>(
        wts.data_ptr<float>(),
        inp.data_ptr<float>(),
        w.data_ptr<float>(),
        relevance_input.data_ptr<float>(),
        batch_size, seq_len, output_features, input_features
    );
}
