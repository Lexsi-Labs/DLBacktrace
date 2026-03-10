#include <torch/extension.h>

void launch_kernel(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor w,
    torch::Tensor relevance_input
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_kernel", &launch_kernel,
          "Relevance propagation kernel",
          py::arg("wts"),
          py::arg("inp"),
          py::arg("w"),
          py::arg("relevance_input"));
}
