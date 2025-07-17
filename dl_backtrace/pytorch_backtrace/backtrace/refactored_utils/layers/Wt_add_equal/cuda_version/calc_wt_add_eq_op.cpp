#include <torch/extension.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <vector>
#include "utils.h"

namespace py = pybind11;

std::vector<torch::Tensor> wt_add_equal_cuda(
    const torch::Tensor& R,
    const std::vector<torch::Tensor>& inp
) {
    // Input validation
    TORCH_CHECK(R.is_cuda(), "R must be CUDA tensor");
    TORCH_CHECK(R.is_contiguous(), "R must be contiguous");
    
    const int num_inputs = inp.size();
    const int R_ndim = R.dim();
    const int batch_size = R.size(0);
    
    // Prepare output tensors
    std::vector<torch::Tensor> outputs;
    std::vector<float*> output_ptrs;
    
    for (int i = 0; i < num_inputs; i++) {
        TORCH_CHECK(inp[i].is_cuda(), "Input tensors must be CUDA");
        TORCH_CHECK(inp[i].size(0) == batch_size, "Batch size mismatch");
        
        torch::Tensor output = torch::empty_like(inp[i]);
        outputs.push_back(output);
        output_ptrs.push_back(output.data_ptr<float>());
    }
    
    // Prepare metadata
    std::vector<int> R_strides, R_sizes;
    for (int i = 0; i < R_ndim; i++) {
        R_strides.push_back(R.stride(i));
        R_sizes.push_back(R.size(i));
    }
    
    // Prepare input metadata - flatten all strides and sizes
    std::vector<int> all_inp_strides, all_inp_sizes;
    std::vector<int> inp_numel_vec;
    
    for (int i = 0; i < num_inputs; i++) {
        inp_numel_vec.push_back(inp[i].numel());
        for (int d = 0; d < R_ndim; d++) {
            all_inp_strides.push_back(inp[i].stride(d));
            all_inp_sizes.push_back(inp[i].size(d));
        }
    }
    
    // Allocate device memory for arrays
    float** d_outputs;
    int* d_R_strides;
    int* d_R_sizes;
    int* d_all_inp_strides;
    int* d_all_inp_sizes;
    int* d_inp_numel;
    
    cudaMalloc(&d_outputs, num_inputs * sizeof(float*));
    cudaMalloc(&d_R_strides, R_ndim * sizeof(int));
    cudaMalloc(&d_R_sizes, R_ndim * sizeof(int));
    cudaMalloc(&d_all_inp_strides, num_inputs * R_ndim * sizeof(int));
    cudaMalloc(&d_all_inp_sizes, num_inputs * R_ndim * sizeof(int));
    cudaMalloc(&d_inp_numel, num_inputs * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_outputs, output_ptrs.data(), num_inputs * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R_strides, R_strides.data(), R_ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R_sizes, R_sizes.data(), R_ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_inp_strides, all_inp_strides.data(), num_inputs * R_ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_inp_sizes, all_inp_sizes.data(), num_inputs * R_ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inp_numel, inp_numel_vec.data(), num_inputs * sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate max elements on host
    int max_output_elements = 0;
    for (int i = 0; i < num_inputs; i++) {
        max_output_elements = std::max(max_output_elements, inp_numel_vec[i]);
    }
    
    // Launch kernel
    launch_wt_add_equal_fused_kernel(
        R.data_ptr<float>(),
        d_outputs,
        d_R_strides, d_R_sizes,
        d_all_inp_strides, d_all_inp_sizes,
        d_inp_numel,
        R_ndim, num_inputs, batch_size,
        max_output_elements
    );
    
    // Clean up device memory
    cudaFree(d_outputs);
    cudaFree(d_R_strides);
    cudaFree(d_R_sizes);
    cudaFree(d_all_inp_strides);
    cudaFree(d_all_inp_sizes);
    cudaFree(d_inp_numel);
    
    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wt_add_equal_cuda", &wt_add_equal_cuda, "Optimized wt_add_equal CUDA implementation", py::arg("R"), py::arg("inp"));
}
