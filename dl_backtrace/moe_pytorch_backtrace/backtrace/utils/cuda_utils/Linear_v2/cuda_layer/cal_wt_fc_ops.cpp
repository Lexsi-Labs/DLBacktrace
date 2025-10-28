#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>
#include <tuple>
#include <cfloat>
#include "utils.h"

namespace py = pybind11;

torch::Tensor calculate_wt_fc_interface(
    const torch::Tensor& relevance_y,
    const torch::Tensor& input_array,
    const torch::Tensor& w,
    const torch::Tensor& b,
    py::dict act
) {
    // Input validation
    TORCH_CHECK(relevance_y.is_cuda(), "relevance_y must be CUDA tensor");
    TORCH_CHECK(input_array.is_cuda(), "input_array must be CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA tensor");
    TORCH_CHECK(relevance_y.dim() == 1, "relevance_y must be 1D tensor");
    TORCH_CHECK(input_array.dim() == 1, "input_array must be 1D tensor");
    TORCH_CHECK(w.dim() == 2, "w must be 2D tensor");
    
    auto relevance_y_c = relevance_y.contiguous();
    auto input_array_c = input_array.contiguous();
    auto w_c = w.contiguous();

    auto input_dim = input_array_c.size(0);
    auto output_dim = relevance_y_c.size(0);

    TORCH_CHECK(w_c.size(0) == output_dim, "w must have output_dim rows");
    TORCH_CHECK(w_c.size(1) == input_dim, "w must have input_dim columns");
    
    // Allocate output tensor
    auto relevance_x = torch::zeros({input_dim}, 
        torch::TensorOptions().dtype(input_array.dtype()).device(input_array.device()));
    
    // Parse activation parameters
    int act_type = (act["type"].cast<std::string>() == "mono") ? 0 : 1;
    float act_lower = act["range"]["l"].is_none() ? -FLT_MAX : 
                      act["range"]["l"].cast<float>();
    float act_upper = act["range"]["u"].is_none() ? FLT_MAX : 
                      act["range"]["u"].cast<float>();
    
    // Convert activation function string to int
    int act_func_int = 0;  // default: identity
    if (!act["func"].is_none()) {
        auto act_func_str = act["func"].cast<std::string>();
        if (act_func_str == "sigmoid") act_func_int = 1;
        else if (act_func_str == "swish") act_func_int = 2;
        else if (act_func_str == "wave") act_func_int = 3;
        else if (act_func_str == "pulse") act_func_int = 4;
        else if (act_func_str == "absolute") act_func_int = 5;
        else if (act_func_str == "hard_sigmoid") act_func_int = 6;
        else if (act_func_str == "tanh") act_func_int = 7;
    }
    
    bool has_bias = b.defined() && b.numel() > 0;
    
    launch_calculate_wt_fc_kernel(
        relevance_y_c.data_ptr<float>(),
        input_array_c.data_ptr<float>(),
        w_c.data_ptr<float>(),
        has_bias ? b.contiguous().data_ptr<float>() : nullptr,
        relevance_x.data_ptr<float>(),
        input_dim, output_dim,
        act_type, act_lower, act_upper, has_bias, act_func_int
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    cudaDeviceSynchronize();

    return relevance_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_wt_fc_interface", &calculate_wt_fc_interface, "CUDA implementation of calculate_wt_fc",
          py::arg("relevance_y"), py::arg("input_array"), py::arg("w"), py::arg("b"), py::arg("act"));
}
