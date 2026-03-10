#include <torch/extension.h>

void launch_kernel(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor out,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_kernel", &launch_kernel,
          "Non-negative conserve kernel",
          py::arg("wts"),
          py::arg("inp"),
          py::arg("out"),
          py::arg("eps"));
}
