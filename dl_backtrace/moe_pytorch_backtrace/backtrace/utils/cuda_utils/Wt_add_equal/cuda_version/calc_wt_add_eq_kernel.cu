#include <cuda_runtime.h>
#include <torch/extension.h>

#define MAX_DIMS 8

__global__ void wt_add_equal_fused_kernel(
    const float* __restrict__ R,
    float** __restrict__ outputs,
    const int* __restrict__ R_strides,
    const int* __restrict__ R_sizes,
    const int* __restrict__ all_inp_strides,
    const int* __restrict__ all_inp_sizes,
    const int* __restrict__ inp_numel,
    const int R_ndim,
    const int num_inputs,
    const int batch_size
) {
    int input_idx = blockIdx.y;  // Which input tensor
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (input_idx >= num_inputs) return;
    
    // Calculate output element count for this input tensor
    int output_numel = inp_numel[input_idx];
    if (thread_idx >= output_numel) return;
    
    // Get pointers to this input's strides and sizes
    const int* inp_strides = all_inp_strides + input_idx * R_ndim;
    const int* inp_sizes = all_inp_sizes + input_idx * R_ndim;
    
    // Convert linear index to multi-dimensional coordinates for output
    int out_coords[MAX_DIMS];
    int remaining = thread_idx;
    for (int d = R_ndim - 1; d >= 0; d--) {
        int inp_size = inp_sizes[d];
        out_coords[d] = remaining % inp_size;
        remaining /= inp_size;
    }
    
    // Calculate sum over reduced dimensions
    float sum = 0.0f;
    
    // Determine which dimensions need reduction
    bool reduce_dim[MAX_DIMS];
    int reduction_size = 1;
    for (int d = 0; d < R_ndim; d++) {
        reduce_dim[d] = (inp_sizes[d] == 1) && (R_sizes[d] > 1);
        if (reduce_dim[d]) {
            reduction_size *= R_sizes[d];
        }
    }
    
    // Perform reduction by iterating over all elements that map to this output
    for (int r = 0; r < reduction_size; r++) {
        int R_coords[MAX_DIMS];
        int temp_r = r;
            
        for (int d = 0; d < R_ndim; d++) {
            if (reduce_dim[d]) {
                R_coords[d] = temp_r % R_sizes[d];
                temp_r /= R_sizes[d];
            } else {
                R_coords[d] = out_coords[d];
            }
        }
            
        // Calculate linear index in R tensor
        int R_idx = 0;
        for (int d = 0; d < R_ndim; d++) {
            R_idx += R_coords[d] * R_strides[d];
        }
            
        // Add to sum with division
        sum += R[R_idx] / num_inputs;
    }
    
    // Write result to output tensor
    outputs[input_idx][thread_idx] = sum;
}

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
) {
    dim3 blockDim(256, 1, 1);
    dim3 gridDim(
        (max_output_elements + blockDim.x - 1) / blockDim.x,
        num_inputs,
        1
    );

    wt_add_equal_fused_kernel<<<gridDim, blockDim>>>(
        R, outputs, R_strides, R_sizes, all_inp_strides, all_inp_sizes, inp_numel, R_ndim, num_inputs, batch_size
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(error));
    }
}
