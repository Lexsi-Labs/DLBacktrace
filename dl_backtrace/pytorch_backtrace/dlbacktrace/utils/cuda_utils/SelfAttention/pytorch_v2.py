import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def dlb_style_nonneg_conserve(wts: torch.Tensor, inp: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Non-negative, mass-preserving DLB on |wts| (conserves L1).
    Keeps your current behavior: sum(out) == sum(abs(wts)) per (B,H).
    """
    assert wts.shape == inp.shape
    
    # Use boolean masks for efficiency
    pos = inp > 0
    neg = inp < 0

    # Vectorized sum operations with keepdims
    p_sum = torch.sum(torch.where(pos, inp, 0.0), dim=(-2, -1), keepdim=True)
    n_sum = -torch.sum(torch.where(neg, inp, 0.0), dim=(-2, -1), keepdim=True)
    denom = p_sum + n_sum + eps

    # Conditional shares using torch.where
    p_share = torch.where(p_sum > 0, p_sum / denom, 0.0)
    n_share = torch.where(n_sum > 0, n_sum / denom, 0.0)

    # Mass to conserve
    M = torch.sum(torch.abs(wts), dim=(-2, -1), keepdim=True)

    # Division stabilization
    p_div = torch.where(p_sum == 0, 1.0, p_sum)
    n_div = torch.where(n_sum == 0, 1.0, n_sum)

    # Vectorized output computation
    out = torch.zeros_like(inp)
    out = out + torch.where(pos, (inp / p_div) * (p_share * M), 0.0)
    out = out + torch.where(neg, (inp / n_div) * (n_share * M) * (-1.0), 0.0)
    
    return out


def dlb_style_signed_conserve(wts: torch.Tensor, inp: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Signed, mass-preserving DLB:
    For each (B,H), sum over (T,D) of out == sum over (T,D) of wts (signed).
    Entries may be negative (as they should be if wts has negatives).
    """
    # Use torch.clamp for consistent behavior with numpy.maximum
    Rp = torch.clamp(wts, min=0.0)
    Rn = torch.clamp(-wts, min=0.0)

    P = dlb_style_nonneg_conserve(Rp, inp, eps)
    N = dlb_style_nonneg_conserve(Rn, inp, eps)

    return P - N


def stabilize(matrix: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Stabilize matrix values by setting small values to epsilon while preserving sign.
    Exactly match NumPy's np.sign(matrix + (matrix == 0)) behavior.
    """
    # Create sign tensor: for zeros, use +1; for non-zeros, use actual sign
    abs_matrix = torch.abs(matrix)
    sign_matrix = torch.where(matrix == 0, 
                             torch.ones_like(matrix), 
                             torch.sign(matrix))
    
    return torch.where(abs_matrix < epsilon, 
                      epsilon * sign_matrix, 
                      matrix)

@torch.compile
def calculate_wt_self_attention(
    R_out: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    masked_fill: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    epsilon: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate relevance propagation through self-attention mechanism with exact NumPy matching.
    """
    B, H, T_q, D = Q.shape
    T_k = K.shape[2]
    
    # Match NumPy's scale calculation exactly
    if scale is None:
        scale = float(D) ** 0.5

    # Step 1: Raw attention logits
    QK_output = torch.matmul(Q, K.transpose(-2, -1))
    logits_unmasked = QK_output / scale

    # Step 2: Softmax over unmasked logits - manual implementation to match NumPy exactly
    logits_max_unmasked = torch.max(logits_unmasked, dim=-1, keepdim=True)[0]
    A_exp = torch.exp(logits_unmasked - logits_max_unmasked)
    A = A_exp / (torch.sum(A_exp, dim=-1, keepdim=True) + epsilon)

    # Step 3: Apply additive attention mask (optional)
    if masked_fill is not None:
        logits_masked = logits_unmasked + masked_fill
    else:
        logits_masked = logits_unmasked.clone()

    # Step 4: Softmax over masked logits - manual implementation
    logits_max_masked = torch.max(logits_masked, dim=-1, keepdim=True)[0]
    A_masked_exp = torch.exp(logits_masked - logits_max_masked)
    A_masked = A_masked_exp / (torch.sum(A_masked_exp, dim=-1, keepdim=True) + epsilon)

    # Step 5: Compute attention output
    attention_output = torch.matmul(A_masked, V)

    # Step 6: Relevance propagation - using more stable computation
    # Add small epsilon to prevent division by zero in edge cases
    attention_output_stable = stabilize(attention_output * 2, epsilon)
    relevance_norm_attn_out = R_out / attention_output_stable

    # Compute relevance for attention weights and values
    R_QK = torch.matmul(relevance_norm_attn_out, V.transpose(-2, -1)) * A
    R_V_raw = torch.matmul(A.transpose(-2, -1), relevance_norm_attn_out) * V

    # Apply conservation to R_V
    R_V = dlb_style_signed_conserve(R_V_raw, V)

    # Relevance calculation for Q and K with enhanced stability
    QK_output_stable = stabilize(QK_output * 2, epsilon)
    relevance_norm_QK_out = R_QK / QK_output_stable

    R_Q_raw = torch.matmul(relevance_norm_QK_out, K) * Q
    R_K_raw = torch.matmul(Q.transpose(-2, -1), relevance_norm_QK_out).transpose(-2, -1) * K

    # Apply conservation
    R_Q = dlb_style_signed_conserve(R_Q_raw, Q)
    R_K = dlb_style_signed_conserve(R_K_raw, K)

    # Relevance for masked_fill - exact computation
    delta_A = A - A_masked
    R_blocked_per_qk = torch.einsum('bhqk,bhkd->bhqk', delta_A, V)
    R_masked_fill = R_blocked_per_qk.sum(dim=1, keepdim=True)

    return R_Q, R_K, R_V, R_masked_fill