#include <torch/extension.h>
#include "utils.h"
#include <string>

torch::Tensor calculate_wt_embedding_interface(
    const torch::Tensor& R_out,
    const torch::Tensor& input_ids, 
    const int vocab_size,
    const std::string& aggregate
) {
    return wt_embedding_cuda(R_out, input_ids, vocab_size, aggregate);
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wt_embedding_cuda", &wt_embedding_cuda, "WT Embedding CUDA implementation");
}
