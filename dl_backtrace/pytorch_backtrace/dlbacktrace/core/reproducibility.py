"""
Exact reproducibility utilities for PyTorch backtrace.
This module ensures exact reproducibility for DL-Backtrace PyTorch operations.
"""

# --- IMPORTANT: set env vars BEFORE importing torch ---------------------------
import os

# Ensure deterministic cuBLAS (CUDA >= 10.2). Must be set before importing torch.
# Keep any user-provided value; otherwise default to ":4096:8".
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Optional: keep Python hash seeding stable if supplied later by setup()
# (We don't set PYTHONHASHSEED here to avoid surprising global effects.)
# -----------------------------------------------------------------------------


import random
import numpy as np
import warnings
from typing import Optional, Dict, Any, Tuple

import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_exact_reproducibility(
    seed: int = 42,
    disable_optimizations: bool = True,
    verbose: bool = True,
) -> None:
    """
    Set up exact-ish reproducibility for PyTorch backtrace operations.

    Notes:
    - CUBLAS_WORKSPACE_CONFIG has already been set at import time to ensure
      cuBLAS determinism before torch initializes CUDA handles.
    - We keep torch.use_deterministic_algorithms(warn_only=True) by default to
      avoid hard errors on unsupported kernels; switch to warn_only=False if you
      want strict failures.

    Args:
        seed: Random seed to use for all RNGs.
        disable_optimizations: Whether to disable speed-oriented features that
            can introduce nondeterminism or numeric drift.
        verbose: Print configuration summary.
    """
    if verbose:
        print(f"🔧 Setting up exact PyTorch reproducibility with seed={seed}")

    # Python / NumPy seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if disable_optimizations:
        # Enforce deterministic algorithms where supported.
        # Use warn_only=True to log instead of raising on unsupported paths.
        torch.use_deterministic_algorithms(True, warn_only=True)

        # cuDNN determinism and autotuning
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Disable TF32 to reduce numeric divergence on Ampere+
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False

        # Keep a consistent default dtype
        torch.set_default_dtype(torch.float32)

        # Prefer math (deterministic) SDPA path; disable flash/mem-effic kernels
        try:
            from torch.backends.cuda import sdp_kernel  # PyTorch 2.x
            sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            pass

        if verbose:
            print("🔧 PyTorch optimizations disabled for exact reproducibility")
    else:
        if verbose:
            print("⚠️  PyTorch optimizations enabled — results may not be exactly reproducible")

    # Tracing / export-related toggles (only if we’re aiming for determinism)
    if disable_optimizations:
        # Limit aggressive graph/execution rewrites that may vary by run/hardware
        os.environ["TORCH_EXPORT_DISABLE_OPTIMIZATIONS"] = "1"
        os.environ["TORCH_EXPORT_DETERMINISTIC"] = "1"
        os.environ["TORCH_FX_DISABLE_OPTIMIZATIONS"] = "1"
        if verbose:
            print("🔧 torch.export / FX optimizations constrained for reproducibility")

    # Helpful logging for dynamic shapes (optional)
    os.environ["TORCH_LOGS"] = os.environ.get("TORCH_LOGS", "+dynamic")
    os.environ["TORCH_DYNAMIC_SHAPES_DETERMINISTIC"] = "1"

    if verbose:
        print("✅ Exact PyTorch reproducibility configured")


def create_deterministic_dynamic_shapes(
    input_shapes: Dict[str, Tuple[int, ...]],
    seed: int = 42,
) -> Dict[str, Dict[int, "Dim"]]:
    """
    Create deterministic dynamic shapes for torch.export.

    Args:
        input_shapes: Mapping input-name -> static shape tuple (ints).
        seed: Seed used to generate deterministic Dim names/ranges.

    Returns:
        Mapping suitable for torch.export's dynamic_shapes argument.
    """
    from torch.export import Dim

    # Seed just to keep naming/ranges deterministic
    random.seed(seed)
    np.random.seed(seed)

    dynamic_shapes: Dict[str, Dict[int, Dim]] = {}

    for input_name, shape in input_shapes.items():
        dynamic_shapes[input_name] = {}
        for dim_idx, dim_size in enumerate(shape):
            # Only make dimensions >1 dynamic (size-1 dims usually act as constants)
            if dim_size > 1:
                dim_name = f"{input_name}_dim_{dim_idx}"
                # Provide conservative, deterministic bounds
                min_val = max(1, dim_size // 2)
                max_val = max(dim_size * 2, min_val + 1)
                dynamic_shapes[input_name][dim_idx] = Dim(dim_name, min=min_val, max=max_val)

    return dynamic_shapes


def export_model_deterministically(
    model: torch.nn.Module,
    sample_inputs: Tuple[torch.Tensor, ...],
    dynamic_shapes: Optional[Dict] = None,
    seed: int = 42,
) -> "torch.export.ExportedProgram":
    """
    Export a model deterministically for reproducible tracing.

    Args:
        model: PyTorch model to export.
        sample_inputs: Inputs used for tracing/export.
        dynamic_shapes: Optional dynamic shape constraints (from create_deterministic_dynamic_shapes).
        seed: Seed to enforce before export.

    Returns:
        torch.export.ExportedProgram with deterministic configuration applied.
    """
    from torch.export import export, export_for_training

    # Configure reproducibility (does not change CUBLAS_WORKSPACE_CONFIG order)
    setup_exact_reproducibility(seed, disable_optimizations=True, verbose=False)

    # Ensure deterministic evaluation mode
    model.eval()
    model.requires_grad_(False)
    for module in model.modules():
        if hasattr(module, "training"):
            module.training = False
        if hasattr(module, "eval"):
            module.eval()

    try:
        if dynamic_shapes:
            ep = export_for_training(model, sample_inputs, dynamic_shapes=dynamic_shapes)
        else:
            ep = export_for_training(model, sample_inputs)

        # Run decompositions with an explicit (empty) table to avoid hidden variance
        if hasattr(ep, "run_decompositions"):
            ep = ep.run_decompositions(decomp_table={})

        print("✅ Model exported deterministically")
        return ep

    except Exception as e:
        print(f"❌ Error during deterministic export: {e}")
        raise


def verify_exact_reproducibility(
    test_function,
    num_runs: int = 3,
    tolerance: float = 1e-6,
    seed: int = 42,
) -> bool:
    """
    Verify that a function produces reproducible results across runs.

    Args:
        test_function: Callable producing a numeric-like result (array/tensor/list).
        num_runs: How many runs to compare.
        tolerance: atol/rtol for equality; set to 0 for bitwise equality if feasible.
        seed: Seed to reset before each run.

    Returns:
        True if all runs are equal within tolerance; False otherwise.
    """
    results = []
    for _ in range(num_runs):
        setup_exact_reproducibility(seed, disable_optimizations=True, verbose=False)
        out = test_function()
        # Normalize to numpy array
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        elif not isinstance(out, np.ndarray):
            out = np.asarray(out)
        results.append(out)

    if len(results) < 2:
        return True

    ref = results[0]
    for i in range(1, len(results)):
        if not np.allclose(ref, results[i], atol=tolerance, rtol=tolerance):
            print("❌ Exact reproducibility test failed: results differ between runs")
            print(f"   Run 0: {ref}")
            print(f"   Run {i}: {results[i]}")
            return False

    print(f"✅ Exact reproducibility test passed: {num_runs} runs matched (tol={tolerance})")
    return True


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Return current reproducibility-related configuration and runtime flags.
    """
    info: Dict[str, Any] = {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_export_disable_optimizations": os.environ.get("TORCH_EXPORT_DISABLE_OPTIMIZATIONS"),
        "torch_export_deterministic": os.environ.get("TORCH_EXPORT_DETERMINISTIC"),
        "torch_fx_disable_optimizations": os.environ.get("TORCH_FX_DISABLE_OPTIMIZATIONS"),
        "torch_logs": os.environ.get("TORCH_LOGS"),
        "torch_dynamic_shapes_deterministic": os.environ.get("TORCH_DYNAMIC_SHAPES_DETERMINISTIC"),
        "pytorch_version": torch.__version__,
        "pytorch_cudnn_deterministic": torch.backends.cudnn.deterministic,
        "pytorch_cudnn_benchmark": torch.backends.cudnn.benchmark,
        "pytorch_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        info["pytorch_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    return info


def print_reproducibility_info() -> None:
    """Print current reproducibility configuration."""
    info = get_reproducibility_info()
    print("🔍 Current Exact Reproducibility Configuration")
    print("=" * 60)
    for k, v in info.items():
        print(f"{k}: {v}")
    print("=" * 60)


# Auto-setup when imported (can be disabled by env)
if os.environ.get("DL_BACKTRACE_AUTO_REPRODUCIBILITY", "true").lower() == "true":
    # Keep verbose=False to avoid noisy imports.
    setup_exact_reproducibility(verbose=False)


__all__ = [
    "setup_exact_reproducibility",
    "create_deterministic_dynamic_shapes",
    "export_model_deterministically",
    "verify_exact_reproducibility",
    "get_reproducibility_info",
    "print_reproducibility_info",
]
