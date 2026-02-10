import os
import torch
from typing import Tuple, Optional

# ─── Import precompiled CUDA extension ───
_precompiled = False
_ops_module = None

try:
    import wt_selfattention_v3_ops as _ops_module
    _precompiled = True
except ImportError:
    # Try importing from the cuda_version_v3 directory (in-tree build)
    _cuda_version_dir = os.path.join(os.path.dirname(__file__), "cuda_version_v3")
    import importlib.util
    _so_files = [f for f in os.listdir(_cuda_version_dir) if f.endswith('.so')] if os.path.isdir(_cuda_version_dir) else []
    if _so_files:
        _spec = importlib.util.spec_from_file_location("wt_selfattention_v3_ops", os.path.join(_cuda_version_dir, _so_files[0]))
        _ops_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_ops_module)
        _precompiled = True
    else:
        raise ImportError(
            "Precompiled CUDA extension 'wt_selfattention_v3_ops' not found. "
            "Please run 'bash compile_layers.sh' or 'python setup.py develop' "
            "in the cuda_version_v3 directory to build it."
        )

# Expose kernel functions from the precompiled module
softmax_ops = _ops_module
stabilize_ops = _ops_module
conservation_ops = _ops_module


def calculate_wt_self_attention_cuda(
    R_out: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    masked_fill: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    epsilon: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    B, H, T_q, D = Q.shape
    T_k = K.shape[2]
    
    if scale is None:
        scale = float(D) ** 0.5

    # Step 1: Raw attention logits
    QK_output = torch.matmul(Q, K.transpose(-2, -1))
    logits_unmasked = QK_output / scale

    # Step 2: Fused softmax for unmasked
    A = torch.empty_like(logits_unmasked)
    softmax_ops.launch_fused_softmax(logits_unmasked, A, epsilon)
    torch.cuda.synchronize()  

    # Step 3: Apply mask
    masked_fill = None
    if masked_fill is not None:
        logits_masked = logits_unmasked + masked_fill
        # Step 4: Fused softmax for masked
        A_masked = torch.empty_like(logits_masked)
        softmax_ops.launch_fused_softmax(logits_masked, A_masked, epsilon)
        torch.cuda.synchronize()
    else:
        # No mask applied
        A_masked = A

    # Step 5: Attention output
    attention_output = torch.matmul(A_masked, V)

    # Step 6: Fused stabilize + normalize for relevance propagation
    relevance_norm_attn_out = torch.empty_like(R_out)
    stabilize_ops.launch_fused_stabilize_normalize(
        R_out, attention_output, relevance_norm_attn_out, epsilon
    )
    torch.cuda.synchronize()

    # Compute R_QK and R_V_raw
    R_QK = torch.matmul(relevance_norm_attn_out, V.transpose(-2, -1)) * A
    R_V_raw = torch.matmul(A.transpose(-2, -1), relevance_norm_attn_out) * V

    # Fused conservation for R_V
    R_V = torch.empty_like(V)
    conservation_ops.launch_fused_conservation(R_V_raw, V, R_V, epsilon)
    torch.cuda.synchronize()

    # Fused stabilize + normalize for QK
    relevance_norm_QK_out = torch.empty_like(R_QK)
    stabilize_ops.launch_fused_stabilize_normalize(
        R_QK, QK_output, relevance_norm_QK_out, epsilon
    )
    torch.cuda.synchronize()

    # Compute R_Q_raw and R_K_raw
    R_Q_raw = torch.matmul(relevance_norm_QK_out, K) * Q
    R_K_raw = torch.matmul(Q.transpose(-2, -1), relevance_norm_QK_out).transpose(-2, -1) * K

    # Fused conservation for R_Q and R_K
    R_Q = torch.empty_like(Q)
    R_K = torch.empty_like(K)
    conservation_ops.launch_fused_conservation(R_Q_raw, Q, R_Q, epsilon)
    conservation_ops.launch_fused_conservation(R_K_raw, K, R_K, epsilon)
    torch.cuda.synchronize()

    # Mask relevance
    delta_A = A - A_masked
    R_blocked_per_qk = torch.einsum('bhqk,bhkd->bhqk', delta_A, V)
    R_masked_fill = R_blocked_per_qk.sum(dim=1, keepdim=True)

    return R_Q, R_K, R_V, R_masked_fill