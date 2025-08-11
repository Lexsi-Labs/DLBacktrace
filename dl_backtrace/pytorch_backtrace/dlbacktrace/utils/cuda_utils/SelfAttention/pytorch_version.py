import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def stabilize(matrix: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    # If abs(val) < epsilon, set to epsilon (keeping original sign or + for zeros)
    return torch.where(torch.abs(matrix) < epsilon, 
                      epsilon * torch.sign(matrix + (matrix == 0)), 
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Handles multi-head attention relevance backtrace using PyTorch tensors.
    
    This function performs relevance propagation through multi-head attention
    mechanism, computing relevance scores for Query, Key, Value matrices and
    optional attention mask.
    
    Args:
        R_out: Relevance tensor of shape [B, H, T_q, D]
        Q: Query tensor of shape [B, H, T_q, D]
        K: Key tensor of shape [B, H, T_k, D]
        V: Value tensor of shape [B, H, T_k, D]
        masked_fill: Optional additive mask of shape [B, 1, T_q, T_k] or [B, H, T_q, T_k]
        scale: Optional scaling factor (default: sqrt(D))
        epsilon: Small constant for numerical stability
        
    Returns:
        Tuple containing:
            - R_Q: Query relevance tensor, same shape as Q
            - R_K: Key relevance tensor, same shape as K
            - R_V: Value relevance tensor, same shape as V
            - R_masked_fill: Mask relevance tensor, same shape as masked_fill (or None)
    """
    B, H, T_q, D = Q.shape
    
    # Use provided scale or compute default (sqrt of dimension)
    if scale is None:
        scale = torch.sqrt(torch.tensor(D, dtype=Q.dtype))
    
    # Step 1: Raw attention logits using batch matrix multiplication
    # Q: [B, H, T_q, D] @ K^T: [B, H, D, T_k] -> [B, H, T_q, T_k]
    QK_output = torch.matmul(Q, K.transpose(-2, -1))
    logits_unmasked = QK_output / scale
    
    # Step 2: Softmax over unmasked logits (for debugging or interpretability)
    # Following original manual softmax implementation
    A = torch.exp(logits_unmasked - torch.max(logits_unmasked, dim=-1, keepdim=True)[0])
    A = A / (torch.sum(A, dim=-1, keepdim=True) + epsilon)
    
    # Step 3: Apply additive attention mask (optional)
    if masked_fill is not None:
        logits_masked = logits_unmasked + masked_fill
    else:
        logits_masked = logits_unmasked.clone()
    
    # Step 4: Softmax over masked logits
    # Following original manual softmax implementation
    A_masked = torch.exp(logits_masked - torch.max(logits_masked, dim=-1, keepdim=True)[0])
    A_masked = A_masked / (torch.sum(A_masked, dim=-1, keepdim=True) + epsilon)
    
    # Step 5: Compute attention output using masked weights
    # A_masked: [B, H, T_q, T_k] @ V: [B, H, T_k, D] -> [B, H, T_q, D]
    attention_output = torch.matmul(A_masked, V)
    
    # Step 6: Relevance propagation to V and attention weights
    # Stabilize attention output for division
    stabilized_attention = stabilize(attention_output * 2, epsilon)
    relevance_norm_attn_out = R_out / stabilized_attention
    
    # Compute relevance for attention weights and V
    # relevance_norm_attn_out: [B, H, T_q, D] @ V^T: [B, H, D, T_k] -> [B, H, T_q, T_k]
    R_QK = torch.matmul(relevance_norm_attn_out, V.transpose(-2, -1)) * A
    
    # A^T: [B, H, T_k, T_q] @ relevance_norm_attn_out: [B, H, T_q, D] -> [B, H, T_k, D]
    R_V = torch.matmul(A.transpose(-2, -1), relevance_norm_attn_out) * V
    
    # Relevance calculation for K and Q
    stabilized_QK = stabilize(QK_output * 2, epsilon)
    relevance_norm_QK_out = R_QK / stabilized_QK
    
    # Compute Q relevance: relevance_norm_QK_out: [B, H, T_q, T_k] @ K: [B, H, T_k, D] -> [B, H, T_q, D]
    R_Q = torch.matmul(relevance_norm_QK_out, K) * Q
    
    # Compute K relevance: Q^T: [B, H, D, T_q] @ relevance_norm_QK_out: [B, H, T_q, T_k] -> [B, H, D, T_k]
    # Then transpose back to [B, H, T_k, D]
    R_K = torch.matmul(Q.transpose(-2, -1), relevance_norm_QK_out).transpose(-2, -1) * K
    
    # Relevance for masked_fill
    # Compute difference in attention weights
    delta_A = A - A_masked
        
    # Use einsum for efficient computation: delta_A @ V -> relevance per position
    # delta_A: [B, H, T_q, T_k], V: [B, H, T_k, D] -> [B, H, T_q, T_k]
    R_blocked_per_qk = torch.einsum('bhqk,bhkd->bhqk', delta_A, V)
        
    # Sum over heads and keep dimension for broadcasting compatibility
    R_masked_fill = R_blocked_per_qk.sum(dim=1, keepdim=True)
    
    return R_Q, R_K, R_V, R_masked_fill
