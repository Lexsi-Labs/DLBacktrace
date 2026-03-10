import os
import importlib.util
import torch

# ─── Import precompiled CUDA extension ───
_this_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_version_dir = os.path.join(_this_dir, "cuda_version", "relevance_proj")

try:
    import relevance_proj_ops as cuda_ops
except ImportError:
    _so_files = (
        [f for f in os.listdir(_cuda_version_dir) if f.endswith('.so')]
        if os.path.isdir(_cuda_version_dir) else []
    )
    if _so_files:
        _spec = importlib.util.spec_from_file_location(
            "relevance_proj_ops",
            os.path.join(_cuda_version_dir, _so_files[0])
        )
        cuda_ops = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(cuda_ops)
    else:
        raise ImportError(
            "Precompiled CUDA extension 'relevance_proj_ops' not found. "
            "Please run 'bash compile_moe_layers.sh' from the project root, "
            "or run 'python setup.py develop' inside "
            f"{_cuda_version_dir} to build it."
        )


def calculate_relevance_proj_cuda(wts: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """CUDA-accelerated relevance projection calculation."""
    assert output.is_cuda and wts.is_cuda, "Tensors must be on GPU"
    assert output.dtype == wts.dtype == torch.float32, "Only float32 supported"

    result = torch.zeros_like(output)
    cuda_ops.launch_kernel(wts, output, result)
    return result