#include <torch/extension.h>
#include "utils.h"

torch::Tensor wt_embedding_cuda_v2(
    const torch::Tensor& R_out,
    const torch::Tensor& input_ids, 
    const int vocab_size,
    const std::string& aggregate
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wt_embedding_cuda_v2", &wt_embedding_cuda_v2,
        "Weighted Embedding v2 with optimized kernel selection",
        py::arg("R_out"),
        py::arg("input_ids"),
        py::arg("vocab_size"),
        py::arg("aggregate")
    );
}
