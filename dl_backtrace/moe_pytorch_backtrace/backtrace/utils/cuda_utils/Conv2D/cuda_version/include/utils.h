#pragma once

#include <torch/extension.h>
#include <map>
#include <string>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor calculate_wt_conv_cuda(
    const torch::Tensor& patches,
    const torch::Tensor& kernel_weights,
    const torch::Tensor& bias,
    const torch::Tensor& grad_scales,
    const int L_patches,
    const int K_h, const int K_w, const int C_in_dim, const int F_dim,
    const int act_type,                 // 0=mono, 1=non_mono
    const int act_func,
    const float act_range_l, const float act_range_u,
    const bool has_range_l, const bool has_range_u
);
