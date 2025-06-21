#pragma once

#include <torch/extension.h>
#include <map>
#include <string>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor calculate_wt_fc_cuda(
    const torch::Tensor& row_specific_weights,
    const torch::Tensor& input_activations,
    const torch::Tensor& weights_matrix,
    const torch::Tensor& bias_vector,
    const bool has_lower_bound,
    const c10::optional<float>& lower_threshold,
    const bool has_upper_bound,
    const c10::optional<float>& upper_threshold,
    const bool is_non_mono,
    const int activation_func
);
