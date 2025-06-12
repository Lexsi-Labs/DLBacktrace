#include <torch/extension.h>
#include "utils.h"
#include <limits>
#include <map>
#include <string>

torch::Tensor calculate_wt_conv2d_interface(
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
) {
    // std::cout << "--- C++ Input Shapes ---" << std::endl;
    // std::cout << "patches.shape: " << patches.sizes() << "device: " << patches.device() << std::endl;
    // std::cout << "kernel_weights.shape: " << kernel_weights.sizes() << "device: " << kernel_weights.device() << std::endl;
    // std::cout << "bias.shape: " << bias.sizes() << "device: " << bias.device() << std::endl;
    // std::cout << "grad_scales.shape: " << grad_scales.sizes() << "device: " << grad_scales.device() << std::endl;
        
    // --- Tensor Checks ---
    TORCH_CHECK(patches.dim() == 4, "patches must be 4D");
    TORCH_CHECK(kernel_weights.dim() == 5, "kernel_weights must be 5D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(grad_scales.dim() == 2, "grad_scales must be 2D");
    
    return calculate_wt_conv_cuda(
        patches,
        kernel_weights,
        bias,
        grad_scales,
        L_patches, K_h, K_w, C_in_dim, F_dim,
        act_type, act_func, act_range_l, act_range_u, has_range_l, has_range_u
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_wt_conv2d_interface", &calculate_wt_conv2d_interface, "Fused Weighted Convolutional Layer with activation controls");
}
