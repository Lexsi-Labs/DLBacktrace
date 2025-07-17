#pragma once

#include <torch/extension.h>
#include <vector>
#include <tuple>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#ifdef __cplusplus
extern "C" {
#endif

void launch_calculate_wt_fc_kernel(
    const float* relevance_y,
    const float* input_array,
    const float* w,
    const float* b,
    float* relevance_x,
    const int input_dim,
    const int output_dim,
    const int activation_kind,
    const float act_lower_bound,
    const float act_upper_bound,
    const bool has_bias,
    const int act_func
);

#ifdef __cplusplus
}
#endif

torch::Tensor calculate_wt_fc_interface(
    const torch::Tensor& row_specific_weights,
    const torch::Tensor& input_activations,
    const torch::Tensor& weights_matrix,
    const torch::Tensor& bias_vector,
    py::dict act
);


