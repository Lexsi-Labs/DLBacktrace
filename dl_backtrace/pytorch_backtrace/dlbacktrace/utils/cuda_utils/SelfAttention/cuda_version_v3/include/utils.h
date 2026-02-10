#pragma once

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void launch_fused_softmax(
    torch::Tensor logits,
    torch::Tensor output,
    float epsilon
);

void launch_fused_stabilize_normalize(
    torch::Tensor numerator,
    torch::Tensor denominator,
    torch::Tensor output,
    float epsilon
);

void launch_fused_conservation(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor output,
    float eps
);
