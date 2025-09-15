import numpy as np


# Toggle debug prints
DEBUG = False  # True
def log(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def dlb_style_nonneg_conserve(wts, inp, eps=1e-12):
    """
    Non-negative, mass-preserving DLB on |wts| (conserves L1).
    Keeps your current behavior: sum(out) == sum(abs(wts)) per (B,H).
    """
    assert wts.shape == inp.shape
    pos = inp > 0
    neg = inp < 0

    p_sum = np.sum(np.where(pos, inp, 0.0), axis=(-2,-1), keepdims=True)
    n_sum = -np.sum(np.where(neg, inp, 0.0), axis=(-2,-1), keepdims=True)
    denom = p_sum + n_sum + eps

    p_share = np.where(p_sum > 0, p_sum/denom, 0.0)
    n_share = np.where(n_sum > 0, n_sum/denom, 0.0)

    M = np.sum(np.abs(wts), axis=(-2,-1), keepdims=True)  # mass to conserve

    p_div = np.where(p_sum == 0, 1.0, p_sum)
    n_div = np.where(n_sum == 0, 1.0, n_sum)

    out = np.zeros_like(inp)
    out += np.where(pos, (inp / p_div) * (p_share * M), 0.0)
    out += np.where(neg, (inp / n_div) * (n_share * M) * (-1.0), 0.0)
    return out

def dlb_style_signed_conserve(wts, inp, eps=1e-12):
    """
    Signed, mass-preserving DLB:
    For each (B,H), sum over (T,D) of out == sum over (T,D) of wts (signed).
    Entries may be negative (as they should be if wts has negatives).
    """
    Rp = np.maximum(wts, 0.0)
    Rn = np.maximum(-wts, 0.0)  # magnitude of negative part

    P = dlb_style_nonneg_conserve(Rp, inp, eps)  # ≥0, sums to sum(Rp)
    N = dlb_style_nonneg_conserve(Rn, inp, eps)  # ≥0, sums to sum(Rn)

    return P - N  # signed result; per-(B,H) sums match sum(wts)

def stabilize(matrix, epsilon=1e-6):
    # If abs(val) < epsilon, set to epsilon (keeping original sign or + for zeros)
    return np.where(np.abs(matrix) < epsilon,
                    epsilon * np.sign(matrix + (matrix == 0)),
                    matrix)

def calculate_wt_self_attention(R_out, Q, K, V, masked_fill=None, scale=None, epsilon=1e-9):
    """
    Handles multi-head attention relevance backtrace.
    Args:
        R_out: [B, H, T_q, D]
        Q, K, V: [B, H, T_q, D] or [B, H, T_k, D]
        masked_fill: Optional additive mask of shape [B, 1, T_q, T_k] or [B, H, T_q, T_k]
        scale: Optional scaling factor (default: sqrt(D))
        epsilon: Small constant for numerical stability
    Returns:
        R_Q, R_K, R_V, R_masked_fill: same shape as Q, K, V, masked_fill
    """
    B, H, T_q, D = Q.shape
    T_k = K.shape[2]  # number of key tokens
    scale = scale or np.sqrt(D)

    # Step 1: Raw attention logits
    QK_output = np.matmul(Q, K.transpose(0, 1, 3, 2))  # [B, H, T_q, T_k]
    logits_unmasked = QK_output / scale 

    # Step 2: Softmax over unmasked logits (for debugging or interpretability)
    A = np.exp(logits_unmasked - np.max(logits_unmasked, axis=-1, keepdims=True))
    A = A / (np.sum(A, axis=-1, keepdims=True) + epsilon) 
    log(f"A (unmasked attention weights): {A.shape}")

    # Step 3: Apply additive attention mask (optional)
    masked_fill = None 
    if masked_fill is not None:
        logits_masked = logits_unmasked + masked_fill  # [B, H, T, T] + [B, 1, T, T]
        log(f"masked_fill--- minimum: {np.min(masked_fill)}, maximum: {np.max(masked_fill)}") 
    else:
        logits_masked = logits_unmasked.copy()

    # Step 4: Softmax over masked logits
    A_masked = np.exp(logits_masked - np.max(logits_masked, axis=-1, keepdims=True))
    A_masked = A_masked / (np.sum(A_masked, axis=-1, keepdims=True) + epsilon) 
    log(f"A_masked (masked attention weights): {A_masked.shape}")

    # Step 5: Compute attention output using masked weights
    attention_output = np.matmul(A_masked, V)  # [B, H, T_q, D] 
    log(f"attention_output (using A_masked): {attention_output.shape}")

    # Step 6: Relevance propagation to V and attention weights
    relevance_norm_attn_out = R_out / stabilize(attention_output * 2, epsilon)
    log(f"relevance_norm_attn_out---  value: {np.sum(relevance_norm_attn_out):.2f},  shape: {relevance_norm_attn_out.shape}")
    log(f"V: {V.shape}, A_masked: {A_masked.shape}")
    log(f"type(A_masked): {type(A_masked)}, type(V): {type(V)}, type(relevance_norm_attn_out): {type(relevance_norm_attn_out)}")

    R_QK = np.matmul(relevance_norm_attn_out, np.transpose(V, (0, 1, 3, 2))) * A
    R_V = np.matmul(np.transpose(A, (0, 1, 3, 2)), relevance_norm_attn_out) * V
    log(f"updated R_QK--- rel: {np.sum(R_QK):.2f}, shape: {R_QK.shape}")
    log(f"updated R_V--- rel: {np.sum(R_V):.2f}, shape: {R_V.shape}") 

    if (R_V >= 0).any():
        log(f"Before:  Negative value found in R_V")
    else:
        log(f"Before: No negative value found in R_V")

    R_V = dlb_style_signed_conserve(R_V, V)

    if (R_V < 0).any():
        log(f"After:  Negative value found in R_V")
    else:
        log(f"After: No negative value found in R_V")

    # Relevance Calculation for K and Q
    relevance_norm_QK_out = R_QK / stabilize(QK_output *2, epsilon)
    log(f"relevance_norm_QK_out---  value: {np.sum(relevance_norm_QK_out):.2f},  shape: {relevance_norm_QK_out.shape}")
    log(f"Q: {Q.shape},  K: {K.shape}")

    R_Q = np.matmul(relevance_norm_QK_out, K) * Q
    R_K = np.transpose(np.matmul(np.transpose(Q, (0, 1, 3, 2)), relevance_norm_QK_out), (0, 1, 3, 2)) * K

    log(f"updated R_Q--- rel: {np.sum(R_Q):.2f}, shape: {R_Q.shape}")
    log(f"updated R_K--- rel: {np.sum(R_K):.2f}, shape: {R_K.shape}")

    if (R_Q < 0).any():
        log(f"Before:  Negative value found in R_Q")
    else:
        log(f"Before: No negative value found in R_Q")

    if (R_K < 0).any():
        log(f"Before:  Negative value found in R_K")
    else:
        log(f"Before: No negative value found in R_K")

    R_Q = dlb_style_signed_conserve(R_Q, Q)
    R_K = dlb_style_signed_conserve(R_K, K)

    if (R_Q < 0).any():
        log(f"After:  Negative value found in R_Q")
    else:
        log(f"After: No negative value found in R_Q")

    if (R_K < 0).any():
        log(f"After:  Negative value found in R_K")
    else:
        log(f"After: No negative value found in R_K")

    # Relevance `masked_fill`
    delta_A = A - A_masked
    R_blocked_per_qk = np.einsum('bhqk,bhkd->bhqk', delta_A, V)
    R_masked_fill = R_blocked_per_qk.sum(axis=1, keepdims=True)
    log(f"R_masked_fill: {R_masked_fill.shape}")

    total_out = np.sum(R_out)
    total_prop = np.sum(R_Q) + np.sum(R_K) + np.sum(R_V) + np.sum(R_masked_fill)

    log(f"R_out sum:         {total_out:.4f}")
    log(f"Propagated sum:    {total_prop:.4f}")
    log(f"Difference (leak): {total_out - total_prop:.4f}")

    return R_Q, R_K, R_V, R_masked_fill
