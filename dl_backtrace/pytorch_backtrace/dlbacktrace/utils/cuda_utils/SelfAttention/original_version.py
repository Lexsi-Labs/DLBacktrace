import numpy as np

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
    B : batch size (Can change)
    H : number of heads (Fixed according to the model architecture, might go as high as 96)
    T_q : query length (depends on query sequence length)
    T_k : key length (depends on key sequence length)
    D : dimension of the query and key (Fixed according to the model architecture, might go as high as 128)
    
    Args:
        R_out: Relevance tensor of shape [B, H, T_q, D] 
        Q: Query tensor of shape [B, H, T_q, D]         
        K: Key tensor of shape [B, H, T_k, D]          
        V: Value tensor of shape [B, H, T_k, D]         
        mask: Optional additive mask of shape [B, 1, T_q, T_k] or [B, H, T_q, T_k] 
        scale: Optional scaling factor (default: sqrt(D))
        
    Returns:
        Tuple containing:
            - R_Q: Query relevance tensor, same shape as Q
            - R_K: Key relevance tensor, same shape as K
            - R_V: Value relevance tensor, same shape as V
            - R_mask: Mask relevance tensor, same shape as mask (or zeros)
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

    # Step 3: Apply additive attention mask (optional)
    masked_fill = None
    if masked_fill is not None:
        logits_masked = logits_unmasked + masked_fill  # [B, H, T, T] + [B, 1, T, T]
        # Step 4: Softmax over masked logits
        A_masked = np.exp(logits_masked - np.max(logits_masked, axis=-1, keepdims=True))
        A_masked = A_masked / (np.sum(A_masked, axis=-1, keepdims=True) + epsilon)
    else:
        # No mask applied - reuse A to avoid redundant computation and ensure delta_A = 0
        A_masked = A

    # Step 5: Compute attention output using masked weights
    attention_output = np.matmul(A_masked, V)  # [B, H, T_q, D]

    # Step 6: Relevance propagation to V and attention weights
    relevance_norm_attn_out = R_out / stabilize(attention_output * 2, epsilon)

    R_QK = np.matmul(relevance_norm_attn_out, np.transpose(V, (0, 1, 3, 2))) * A
    R_V = np.matmul(np.transpose(A, (0, 1, 3, 2)), relevance_norm_attn_out) * V

    R_V = dlb_style_signed_conserve(R_V, V)

    # Relevance Calculation for K and Q
    relevance_norm_QK_out = R_QK / stabilize(QK_output *2, epsilon)

    R_Q = np.matmul(relevance_norm_QK_out, K) * Q
    R_K = np.transpose(np.matmul(np.transpose(Q, (0, 1, 3, 2)), relevance_norm_QK_out), (0, 1, 3, 2)) * K

    R_Q = dlb_style_signed_conserve(R_Q, Q)
    R_K = dlb_style_signed_conserve(R_K, K)

    # Relevance `masked_fill`
    delta_A = A - A_masked
    R_blocked_per_qk = np.einsum('bhqk,bhkd->bhqk', delta_A, V)
    R_masked_fill = R_blocked_per_qk.sum(axis=1, keepdims=True)

    return R_Q, R_K, R_V, R_masked_fill
