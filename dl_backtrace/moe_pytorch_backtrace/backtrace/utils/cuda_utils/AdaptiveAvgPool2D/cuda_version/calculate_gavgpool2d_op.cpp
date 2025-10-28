#include <torch/extension.h>
#include "utils.h"

torch::Tensor calculate_wt_gavgpool_interface(
    const torch::Tensor& wts,
    const torch::Tensor& inp
) {
    TORCH_CHECK(wts.is_cuda(), "Weights must be CUDA tensor");
    TORCH_CHECK(inp.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(wts.dtype() == torch::kFloat32, "Weights must be float32");
    TORCH_CHECK(inp.dtype() == torch::kFloat32, "Input must be float32");

    return fused_weighted_gavgpool_cuda(wts, inp);

}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_weighted_gavgpool", &fused_weighted_gavgpool_cuda, 
          "Fused weighted global average pooling CUDA implementation");
}
