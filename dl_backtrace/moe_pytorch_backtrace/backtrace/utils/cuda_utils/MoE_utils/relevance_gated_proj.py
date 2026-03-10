import os
import importlib.util
import torch

# ─── Import precompiled CUDA extension ───
_this_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_version_dir = os.path.join(_this_dir, "cuda_version", "relevance_gated_proj")

try:
    import relevance_gated_proj_ops as cuda_ops
except ImportError:
    _so_files = (
        [f for f in os.listdir(_cuda_version_dir) if f.endswith('.so')]
        if os.path.isdir(_cuda_version_dir) else []
    )
    if _so_files:
        _spec = importlib.util.spec_from_file_location(
            "relevance_gated_proj_ops",
            os.path.join(_cuda_version_dir, _so_files[0])
        )
        cuda_ops = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(cuda_ops)
    else:
        raise ImportError(
            "Precompiled CUDA extension 'relevance_gated_proj_ops' not found. "
            "Please run 'bash compile_moe_layers.sh' from the project root, "
            "or run 'python setup.py develop' inside "
            f"{_cuda_version_dir} to build it."
        )


def calculate_relevance_gated_proj_cuda(
    wts: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA-accelerated version of calculate_relevance_gated_proj.

    Args:
        wts:    Weight tensor (same shape as output)
        output: Output tensor to compute gating on

    Returns:
        Weighted gated projection tensor
    """
    assert output.is_cuda and wts.is_cuda, "Input must be on CUDA device"
    assert output.dtype == wts.dtype == torch.float32, "Only float32 supported"
    assert output.shape == wts.shape, "Shapes must match"

    wt_mat_total = torch.zeros_like(output)
    cuda_ops.launch_kernel(output, wts, wt_mat_total)
    return wt_mat_total
