#pragma once

#include <torch/extension.h>
#include <vector>
#include <tuple>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

extern "C" void launch_wt_add_equal_fused_kernel(
    const float* R,
    float** outputs,
    const int* R_strides,
    const int* R_sizes,
    const int* all_inp_strides,
    const int* all_inp_sizes,
    const int* inp_numel,
    const int R_ndim,
    const int num_inputs,
    const int batch_size,
    const int max_output_elements
);

std::vector<torch::Tensor> wt_add_equal_cuda(
    const torch::Tensor& R,
    const std::vector<torch::Tensor>& inp
);
