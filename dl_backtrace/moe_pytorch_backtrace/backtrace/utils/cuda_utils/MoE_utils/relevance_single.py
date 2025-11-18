import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relevance_propagation_kernel(
    const float* __restrict__ wts,
    const float* __restrict__ inp,
    const float* __restrict__ w,
    float* __restrict__ relevance_input,
    const int batch_size,
    const int seq_len,
    const int output_features,
    const int input_features
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    
    if (b >= batch_size || s >= seq_len || i >= input_features) return;
    
    const int inp_base = b * seq_len * input_features + s * input_features;
    const int wts_base = b * seq_len * output_features + s * output_features;
    const float inp_val = inp[inp_base + i];
    
    float relevance_sum = 0.0f;
    
    for (int o = 0; o < output_features; ++o) {
        const float wts_val = wts[wts_base + o];
        const float w_val = w[o * input_features + i];
        const float contribution = w_val * inp_val;
        
        // Compute p_sum and n_sum
        float p_sum = 0.0f;
        float n_sum = 0.0f;
        
        for (int j = 0; j < input_features; ++j) {
            const float inp_j = inp[inp_base + j];
            const float w_oj = w[o * input_features + j];
            const float contrib_j = w_oj * inp_j;
            
            if (contrib_j > 0.0f) {
                p_sum += contrib_j;
            } else if (contrib_j < 0.0f) {
                n_sum -= contrib_j;
            }
        }
        
        // Calculate aggregation weights
        const float total_sum = p_sum + n_sum;
        const float p_agg_wt = (p_sum > 0.0f && total_sum != 0.0f) ? p_sum / total_sum : 0.0f;
        const float n_agg_wt = (n_sum > 0.0f && total_sum != 0.0f) ? n_sum / total_sum : 0.0f;
        
        // Calculate relevance contribution
        float relevance_contrib = 0.0f;
        
        if (contribution > 0.0f && p_sum != 0.0f) {
            relevance_contrib = (contribution / p_sum) * wts_val * p_agg_wt;
        } else if (contribution < 0.0f && n_sum != 0.0f) {
            relevance_contrib = -(contribution / n_sum) * wts_val * n_agg_wt;
        }
        
        relevance_sum += relevance_contrib;
    }
    
    relevance_input[inp_base + i] = relevance_sum;
}

void launch_kernel(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor w,
    torch::Tensor relevance_input
) {
    const int batch_size = wts.size(0);
    const int seq_len = wts.size(1);
    const int output_features = wts.size(2);
    const int input_features = inp.size(2);
    
    const dim3 threads(16, 8, 1);
    const dim3 blocks(
        (input_features + threads.x - 1) / threads.x,
        (seq_len + threads.y - 1) / threads.y,
        batch_size
    );
    
    relevance_propagation_kernel<<<blocks, threads>>>(
        wts.data_ptr<float>(),
        inp.data_ptr<float>(),
        w.data_ptr<float>(),
        relevance_input.data_ptr<float>(),
        batch_size,
        seq_len,
        output_features,
        input_features
    );
}
"""

cuda_declaration = r"""
void launch_kernel(
    torch::Tensor wts,
    torch::Tensor inp,
    torch::Tensor w,
    torch::Tensor relevance_input
);
"""

def get_cuda_arch_flags():
    if not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    arch_flag = f"--generate-code=arch=compute_{major}{minor},code=sm_{major}{minor}"
    return [arch_flag]

extra_flags = [
    '-O3',
    '--use_fast_math',
    '-Xcompiler', '-fPIC',
]
extra_flags.extend(get_cuda_arch_flags())

cuda_ops = load_inline(
    name="relevance_propagation",
    cpp_sources=cuda_declaration,
    cuda_sources=cuda_source,
    functions=["launch_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)

def calculate_relevance_cuda(
    wts: torch.Tensor,
    inp: torch.Tensor,
    w: torch.Tensor
) -> torch.Tensor:
    """
    CUDA-accelerated relevance propagation.
    
    Args:
        wts: (batch_size, seq_len, output_features)
        inp: (batch_size, seq_len, input_features)
        w: (output_features, input_features)
    
    Returns:
        relevance_input: (batch_size, seq_len, input_features)
    """
    batch_size, seq_len, output_features = wts.shape
    input_features = inp.size(2)
    
    # Allocate output tensor
    relevance_input = torch.empty(
        batch_size, seq_len, input_features,
        device=wts.device,
        dtype=wts.dtype
    )
    
    # Ensure contiguous memory layout
    wts = wts.contiguous()
    inp = inp.contiguous()
    w = w.contiguous()
    
    # Launch kernel
    cuda_ops.launch_kernel(wts, inp, w, relevance_input)
    
    return relevance_input