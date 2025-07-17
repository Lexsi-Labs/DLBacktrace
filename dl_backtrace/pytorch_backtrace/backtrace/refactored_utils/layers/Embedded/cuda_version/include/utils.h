#pragma once

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor wt_embedding_cuda(
    const torch::Tensor& R_out,
    const torch::Tensor& input_ids, 
    const int vocab_size,
    const std::string& aggregate
);
