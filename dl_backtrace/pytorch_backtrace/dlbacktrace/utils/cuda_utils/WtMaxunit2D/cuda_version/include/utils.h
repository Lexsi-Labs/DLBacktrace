#pragma once

#include <torch/extension.h>
#include <cfloat>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor calculate_wt_max_unit_cuda(
    const torch::Tensor& patch,
    const torch::Tensor& wts,
    const int pool_size  // unused, maintained for compatibility
);
