import os
import torch
from typing import Tuple

# ─── Import precompiled CUDA extension ───
_precompiled = False

try:
    import wt_embedding_v2_ops as embedding_cuda_ops
    _precompiled = True
except ImportError:
    # Try importing from the cuda_version_v2 directory (in-tree build)
    _cuda_version_dir = os.path.join(os.path.dirname(__file__), "cuda_version_v2")
    import importlib.util
    _so_files = [f for f in os.listdir(_cuda_version_dir) if f.endswith('.so')] if os.path.isdir(_cuda_version_dir) else []
    if _so_files:
        _spec = importlib.util.spec_from_file_location("wt_embedding_v2_ops", os.path.join(_cuda_version_dir, _so_files[0]))
        embedding_cuda_ops = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(embedding_cuda_ops)
        _precompiled = True
    else:
        raise ImportError(
            "Precompiled CUDA extension 'wt_embedding_v2_ops' not found. "
            "Please run 'bash compile_layers.sh' or 'python setup.py develop' "
            "in the cuda_version_v2 directory to build it."
        )


def calculate_wt_embedding_cuda(
    R_out: torch.Tensor,
    input_ids: torch.Tensor,
    vocab_size: int,
    aggregate: str = "sum"
) -> Tuple[torch.Tensor]:
    """
    CUDA-accelerated weighted embedding calculation with optimized kernels.
    
    Uses three specialized kernels:
    1. Optimized kernel: 1 thread per input element, good for large embeddings
    2. Warp kernel: Warp-level optimization for medium embeddings  
    3. Mean kernel: Vectorized mean aggregation with fast math
    
    Args:
        R_out: Relevance tensor [B, T, D] (float32, CUDA)
        input_ids: Token indices [B, T] (int64, CUDA)
        vocab_size: Size of vocabulary
        aggregate: "sum" or "mean" aggregation mode
        
    Returns:
        Tuple containing the output relevance tensor
    """
    # Ensure inputs are on CUDA
    if not R_out.is_cuda:
        R_out = R_out.cuda()
    if not input_ids.is_cuda:
        input_ids = input_ids.cuda()
    
    # Ensure correct dtypes
    R_out = R_out.float()
    input_ids = input_ids.long()
    
    # Handle dimensionality
    if R_out.dim() == 2:
        R_out = R_out.unsqueeze(0)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    result = embedding_cuda_ops.wt_embedding_cuda_v2(R_out, input_ids, vocab_size, aggregate)
    
    return (result,)