import os
import importlib.util
import torch

# ─── Import precompiled CUDA extension ───
_precompiled = False
_this_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_version_dir = os.path.join(_this_dir, "cuda_version", "wt_router_logits")

try:
    import wt_router_logits_ops as cuda_ops
    _precompiled = True
except ImportError:
    _so_files = (
        [f for f in os.listdir(_cuda_version_dir) if f.endswith('.so')]
        if os.path.isdir(_cuda_version_dir) else []
    )
    if _so_files:
        _spec = importlib.util.spec_from_file_location(
            "wt_router_logits_ops",
            os.path.join(_cuda_version_dir, _so_files[0])
        )
        cuda_ops = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(cuda_ops)
        _precompiled = True
    else:
        raise ImportError(
            "Precompiled CUDA extension 'wt_router_logits_ops' not found. "
            "Please run 'bash compile_moe_layers.sh' from the project root, "
            "or run 'python setup.py develop' inside "
            f"{_cuda_version_dir} to build it."
        )


def calculate_wt_router_logits_cuda(
    wts: torch.Tensor,
    inp: torch.Tensor,
    W_router: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA-accelerated weighted router-logits calculation.

    Args:
        wts:      (n_samples,) sample weights
        inp:      (n_samples, n_features) input features
        W_router: (n_features, n_features) router weights

    Returns:
        output: (n_samples, n_features) weighted router logits
    """
    assert wts.is_cuda and inp.is_cuda and W_router.is_cuda, \
        "All tensors must be on CUDA"
    assert wts.dtype == inp.dtype == W_router.dtype == torch.float32, \
        "Only float32 supported"
    assert inp.is_contiguous() and W_router.is_contiguous(), \
        "Tensors must be contiguous"

    n_samples, n_features = inp.shape
    assert W_router.shape == (n_features, n_features), "W_router shape mismatch"
    assert wts.shape == (n_samples,), "wts shape mismatch"

    output = torch.empty((n_samples, n_features), dtype=torch.float32, device=inp.device)
    cuda_ops.launch_kernel(wts, inp, W_router, output)
    return output
