#include <torch/extension.h>

void launch_kernel(
    torch::Tensor output,
    torch::Tensor wts,
    torch::Tensor wt_mat_total
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_kernel", &launch_kernel,
          "Relevance gated-projection kernel",
          py::arg("output"),
          py::arg("wts"),
          py::arg("wt_mat_total"));
}
