#include <torch/extension.h>
#include "utils.h"

void launch_fused_softmax(
    torch::Tensor logits,
    torch::Tensor output,
    float epsilon
);

void launch_fused_stabilize_normalize(
    torch::Tensor numerator,
    torch::Tensor denominator,
    torch::Tensor output,
    float epsilon
);

void launch_fused_conservation(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor output,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_softmax", &launch_fused_softmax,
        "Fused Softmax kernel for attention scores",
        py::arg("logits"),
        py::arg("output"),
        py::arg("epsilon")
    );
    m.def("launch_fused_stabilize_normalize", &launch_fused_stabilize_normalize,
        "Fused Stabilize and Normalize kernel for relevance propagation",
        py::arg("numerator"),
        py::arg("denominator"),
        py::arg("output"),
        py::arg("epsilon")
    );
    m.def("launch_fused_conservation", &launch_fused_conservation,
        "Fused Conservation kernel for relevance redistribution",
        py::arg("wts"),
        py::arg("inp"),
        py::arg("output"),
        py::arg("eps")
    );
}
