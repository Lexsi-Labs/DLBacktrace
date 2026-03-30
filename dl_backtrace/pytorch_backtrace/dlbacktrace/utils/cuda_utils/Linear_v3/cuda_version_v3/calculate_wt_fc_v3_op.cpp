#include <torch/extension.h>
#include "utils.h"

torch::Tensor launch_calculate_wt_fc_kernel(
    const torch::Tensor& relevance_y,
    const torch::Tensor& input_array,
    const torch::Tensor& w,
    const torch::Tensor& b,
    int act_type,
    float act_lower,
    float act_upper,
    int act_func_int
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_calculate_wt_fc_kernel", &launch_calculate_wt_fc_kernel,
        "Fused Weighted Fully Connected Layer v3 with activation controls",
        py::arg("relevance_y"),
        py::arg("input_array"),
        py::arg("w"),
        py::arg("b"),
        py::arg("act_type"),
        py::arg("act_lower"),
        py::arg("act_upper"),
        py::arg("act_func_int")
    );
}
