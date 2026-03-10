#include <torch/extension.h>

void launch_kernel(torch::Tensor wts, torch::Tensor output, torch::Tensor result);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_kernel", &launch_kernel,
          "Relevance projection kernel",
          py::arg("wts"),
          py::arg("output"),
          py::arg("result"));
}
