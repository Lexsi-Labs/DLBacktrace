#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>
#include <tuple>
#include "utils.h"

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>> 
wt_selfattention_cuda(
    const torch::Tensor& R_out,
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::optional<torch::Tensor>& mask,
    const torch::optional<double>& scale
) {
    // Validate inputs
    TORCH_CHECK(R_out.is_cuda() && Q.is_cuda() && K.is_cuda() && V.is_cuda(), 
                "All inputs must be CUDA tensors");
    TORCH_CHECK(R_out.dtype() == torch::kFloat32, "Only float32 supported");
    
    const auto sizes = Q.sizes();
    const int B = sizes[0];
    const int H = sizes[1];
    const int T_q = sizes[2];
    const int D = sizes[3];
    const int T_k = K.size(2);
    
    // Get scale from input or default to sqrt(D)
    double scale_val = scale.has_value() ? scale.value() : std::sqrt(static_cast<float>(D));
    double epsilon_val = 1e-9;

    // Prepare outputs
    auto R_Q = torch::zeros_like(Q);
    auto R_K = torch::zeros_like(K);
    auto R_V = torch::zeros_like(V);
    auto R_mask = torch::zeros({B, 1, T_q, T_k}, Q.options());

    // Mask parameters
    bool has_mask = mask.has_value();
    int mask_stride = 0;
    if (has_mask) {
        auto mask_fill = mask.value();
        TORCH_CHECK(mask_fill.size(0) == B, "Mask batch size mismatch");
        // Calculate stride based on mask dimensions
        // If mask is [B, 1, T_q, T_k], stride is T_q * T_k (broadcast across heads)
        // If mask is [B, H, T_q, T_k], stride is H * T_q * T_k
        mask_stride = mask_fill.size(1) == 1 ? T_q * T_k : H * T_q * T_k;
    }

    // Launch CUDA kernel
    launch_fused_attention_relevance(
        R_out.data_ptr<float>(),
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        (has_mask) ? mask.value().data_ptr<float>() : nullptr,
        R_Q.data_ptr<float>(),
        R_K.data_ptr<float>(),
        R_V.data_ptr<float>(),
        R_mask.data_ptr<float>(),  // Always provide R_mask pointer
        B, H, T_q, T_k, D,
        scale_val, epsilon_val, has_mask, mask_stride
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA kernel execution failed: ", cudaGetErrorString(error));
    
    return std::make_tuple(R_Q, R_K, R_V, R_mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wt_selfattention_cuda", &wt_selfattention_cuda, 
          "Self-Attention Relevance Propagation CUDA implementation",
          py::arg("R_out"), py::arg("Q"), py::arg("K"), py::arg("V"), 
          py::arg("mask") = py::none(), py::arg("scale") = py::none());
}
