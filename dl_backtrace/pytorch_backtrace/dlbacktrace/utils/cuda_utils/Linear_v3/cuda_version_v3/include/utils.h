#pragma once

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor launch_calculate_wt_fc_kernel(
    const torch::Tensor& relevance_y,
    const torch::Tensor& input_array,
    const torch::Tensor& w,
    const torch::Tensor& b,
    int act_type,
    float act_lower,
    float act_upper,
    int act_func_int
);
