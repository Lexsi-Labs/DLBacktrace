#pragma once

#include <torch/extension.h>
#include <vector>
#include <tuple>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Declare the CUDA kernel launch function
extern "C" void launch_fused_attention_relevance(
    const float* R_out, const float* Q, const float* K, const float* V,
    const float* mask, float* R_Q, float* R_K, float* R_V, float* R_mask,
    int B, int H, int T_q, int T_k, int D, float scale, float epsilon,
    bool has_mask, int mask_stride
);

// PyTorch interface function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>> 
wt_selfattention_cuda(
    const torch::Tensor& R_out,
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::optional<torch::Tensor>& mask,
    const torch::optional<double>& scale
);
