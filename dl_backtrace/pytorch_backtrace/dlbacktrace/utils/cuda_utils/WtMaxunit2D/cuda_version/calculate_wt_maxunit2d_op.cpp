#include <torch/extension.h>
#include "utils.h"
#include <limits>
#include <string>

torch::Tensor calculate_wt_maxunit2d_interface(
    const torch::Tensor& patch,
    const torch::Tensor& wts,
    const int pool_size  // unused, maintained for compatibility
) {
    // std::cout << "--- C++ Input Shapes ---" << std::endl;
    // std::cout << "patch.shape: " << patch.sizes() << "device: " << patch.device() << std::endl;
    // std::cout << "wts.shape: " << wts.sizes() << "device: " << wts.device() << std::endl;

    // --- Tensor Checks (already present and good) ---
    TORCH_CHECK(patch.is_cuda(), "patch must be a CUDA tensor");
    TORCH_CHECK(wts.is_cuda(), "wts must be a CUDA tensor");
    
    return calculate_wt_max_unit_cuda(patch, wts, pool_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_wt_maxunit2d_interface", &calculate_wt_maxunit2d_interface, "Fused Weighted Max Unit 2D Layer with activation controls");
}
