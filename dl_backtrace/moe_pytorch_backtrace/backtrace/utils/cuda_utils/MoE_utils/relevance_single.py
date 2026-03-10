import os
import importlib.util
import torch

# ─── Import precompiled CUDA extension ───
_this_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_version_dir = os.path.join(_this_dir, "cuda_version", "relevance_single")

try:
    import relevance_single_ops as cuda_ops
except ImportError:
    _so_files = (
        [f for f in os.listdir(_cuda_version_dir) if f.endswith('.so')]
        if os.path.isdir(_cuda_version_dir) else []
    )
    if _so_files:
        _spec = importlib.util.spec_from_file_location(
            "relevance_single_ops",
            os.path.join(_cuda_version_dir, _so_files[0])
        )
        cuda_ops = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(cuda_ops)
    else:
        raise ImportError(
            "Precompiled CUDA extension 'relevance_single_ops' not found. "
            "Please run 'bash compile_moe_layers.sh' from the project root, "
            "or run 'python setup.py develop' inside "
            f"{_cuda_version_dir} to build it."
        )


def calculate_relevance_cuda(
    wts: torch.Tensor,
    inp: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA-accelerated relevance propagation.

    Args:
        wts: (batch_size, seq_len, output_features)
        inp: (batch_size, seq_len, input_features)
        w:   (output_features, input_features)

    Returns:
        relevance_input: (batch_size, seq_len, input_features)
    """
    batch_size, seq_len, input_features = inp.shape

    relevance_input = torch.empty(
        batch_size, seq_len, input_features,
        device=wts.device,
        dtype=wts.dtype,
    )

    wts = wts.contiguous()
    inp = inp.contiguous()
    w   = w.contiguous()

    cuda_ops.launch_kernel(wts, inp, w, relevance_input)
    return relevance_input