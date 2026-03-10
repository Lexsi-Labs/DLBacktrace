#include <torch/extension.h>

void launch_kernel(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor W_router,
    torch::Tensor output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_kernel", &launch_kernel,
          "Weighted router-logits kernel",
          py::arg("wts"),
          py::arg("inp"),
          py::arg("W_router"),
          py::arg("output"));
}
