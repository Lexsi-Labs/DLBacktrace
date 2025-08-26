import numpy as np

def stabilize(matrix, epsilon=1e-6):
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
    else:
        logits_masked = logits_unmasked.copy()

    # Step 4: Softmax over masked logits
    A_masked = np.exp(logits_masked - np.max(logits_masked, axis=-1, keepdims=True))
    A_masked = A_masked / (np.sum(A_masked, axis=-1, keepdims=True) + epsilon) 

    # Step 5: Compute attention output using masked weights
    attention_output = np.matmul(A_masked, V)  # [B, H, T_q, D] 

    # Step 6: Relevance propagation to V and attention weights
    relevance_norm_attn_out = R_out / stabilize(attention_output * 2, epsilon)

    R_QK = np.matmul(relevance_norm_attn_out, np.transpose(V, (0, 1, 3, 2))) * A
    R_V = np.matmul(np.transpose(A, (0, 1, 3, 2)), relevance_norm_attn_out) * V

    # Relevance Calculation for K and Q
    relevance_norm_QK_out = R_QK / stabilize(QK_output *2, epsilon)

    R_Q = np.matmul(relevance_norm_QK_out, K) * Q
    R_K = np.transpose(np.matmul(np.transpose(Q, (0, 1, 3, 2)), relevance_norm_QK_out), (0, 1, 3, 2)) * K

    # Relevance `masked_fill`
    delta_A = A - A_masked
    R_blocked_per_qk = np.einsum('bhqk,bhkd->bhqk', delta_A, V)
    R_masked_fill = R_blocked_per_qk.sum(axis=1, keepdims=True)

    return R_Q, R_K, R_V, R_masked_fill