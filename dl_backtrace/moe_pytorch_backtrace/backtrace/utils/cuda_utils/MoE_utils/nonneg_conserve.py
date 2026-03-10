import os
import importlib.util
import torch

# ─── Import precompiled CUDA extension ───
_this_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_version_dir = os.path.join(_this_dir, "cuda_version", "nonneg_conserve")

try:
    import nonneg_conserve_ops as cuda_ops
except ImportError:
    _so_files = (
        [f for f in os.listdir(_cuda_version_dir) if f.endswith('.so')]
        if os.path.isdir(_cuda_version_dir) else []
    )
    if _so_files:
        _spec = importlib.util.spec_from_file_location(
            "nonneg_conserve_ops",
            os.path.join(_cuda_version_dir, _so_files[0])
        )
        cuda_ops = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(cuda_ops)
    else:
        raise ImportError(
            "Precompiled CUDA extension 'nonneg_conserve_ops' not found. "
            "Please run 'bash compile_moe_layers.sh' from the project root, "
            "or run 'python setup.py develop' inside "
            f"{_cuda_version_dir} to build it."
        )


def dlb_style_nonneg_conserve_cuda(
    wts: torch.Tensor,
    inp: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    CUDA-accelerated version of dlb_style_nonneg_conserve.

    Args:
        wts: Weight tensor of shape (B, C, H, W)
        inp: Input tensor of shape (B, C, H, W)
        eps: Epsilon for numerical stability

    Returns:
        Output tensor of shape (B, C, H, W)
    """
    assert wts.shape == inp.shape
    assert wts.is_cuda and inp.is_cuda
    assert wts.is_contiguous() and inp.is_contiguous()
    assert wts.dtype == inp.dtype == torch.float32

    out = torch.zeros_like(inp)
    cuda_ops.launch_kernel(wts, inp, out, eps)
    return out


def dlb_style_signed_conserve_cuda(
    wts: torch.Tensor,
    inp: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    CUDA-accelerated version of dlb_style_signed_conserve.

    Args:
        wts: Weight tensor of shape (B, C, H, W)
        inp: Input tensor of shape (B, C, H, W)
        eps: Epsilon for numerical stability

    Returns:
        Output tensor of shape (B, C, H, W)
    """
    Rp = torch.clamp(wts,  min=0.0)
    Rn = torch.clamp(-wts, min=0.0)
    P  = dlb_style_nonneg_conserve_cuda(Rp, inp, eps)
    N  = dlb_style_nonneg_conserve_cuda(Rn, inp, eps)
    return P - N