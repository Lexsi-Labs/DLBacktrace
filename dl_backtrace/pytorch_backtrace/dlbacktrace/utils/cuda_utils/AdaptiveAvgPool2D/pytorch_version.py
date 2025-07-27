import torch
import torch.nn.functional as F
from typing import Tuple

@torch.compile
def calculate_wt_gavgpool(relevance_y: torch.Tensor, input_array: torch.Tensor) -> torch.Tensor:
    """
    Calculate weighted global average pooling with positive/negative weight aggregation.
    
    This function processes input tensors by separating positive and negative values,
    computing aggregated weights based on their sums, and applying weighted transformations
    across all channels and batch samples. Optimized for PyTorch with autograd support.
    
    Args:
        relevance_y (torch.Tensor): Relevance weights tensor of shape (batch_size, channels, height, width).
                                   Should be on the same device as input_array.
        input_array (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
                                    Should be on the same device as relevance_y.
    
    Returns:
        torch.Tensor: Weighted relevance tensor of shape (batch_size, channels, height, width).
                     Maintains gradient information for backpropagation.
    
    Note:
        - Replicates original logic for handling division by zero (sets sum to 1.0 when sum is 0.0)
        - Maintains exact numerical behavior including potential numerical instabilities
        - Optimized with torch.compile for improved execution speed
    """
    
    # Transpose tensors once for vectorized operations (batch, height, width, channels)
    input_transposed = input_array.permute(0, 2, 3, 1)  # (bs, h, w, c)
    weights_transposed = relevance_y.permute(0, 2, 3, 1)  # (bs, h, w, c)
    
    # Vectorized separation using boolean masking - more memory efficient
    positive_mask = input_transposed >= 0  # (bs, h, w, c)
    
    # Use torch.where for conditional selection - autograd compatible
    p_mat = torch.where(positive_mask, input_transposed, torch.zeros_like(input_transposed))
    n_mat = torch.where(positive_mask, torch.zeros_like(input_transposed), input_transposed)
    
    # Compute sums across spatial dimensions (h, w) for each batch and channel
    p_sum = torch.sum(p_mat, dim=(1, 2))  # (bs, c)
    n_sum = torch.sum(n_mat, dim=(1, 2)) * -1.0  # (bs, c) - make positive
    
    # Calculate aggregate weights - vectorized across all batches and channels
    total_sum = p_sum + n_sum  # (bs, c)
    
    # Replicate original logic: compute aggregate weights with division by zero handling
    # Use torch.where to avoid explicit boolean indexing
    p_agg_wt = torch.where(total_sum > 0.0, p_sum / total_sum, torch.zeros_like(p_sum))
    n_agg_wt = torch.where(total_sum > 0.0, n_sum / total_sum, torch.zeros_like(n_sum))
    
    # Handle division by zero: set sum to 1.0 when sum is 0.0 (replicating original logic)
    p_sum_safe = torch.where(p_sum == 0.0, torch.ones_like(p_sum), p_sum)
    n_sum_safe = torch.where(n_sum == 0.0, torch.ones_like(n_sum), n_sum)
    
    # Expand dimensions for broadcasting: (bs, 1, 1, c)
    p_sum_safe = p_sum_safe.unsqueeze(1).unsqueeze(1)
    n_sum_safe = n_sum_safe.unsqueeze(1).unsqueeze(1)
    p_agg_wt = p_agg_wt.unsqueeze(1).unsqueeze(1)
    n_agg_wt = n_agg_wt.unsqueeze(1).unsqueeze(1)
    
    # Vectorized weight calculation using broadcasting
    # Positive contribution: (p_mat / p_sum) * weight * p_agg_wt
    positive_contribution = (p_mat / p_sum_safe) * weights_transposed * p_agg_wt
    
    # Negative contribution: (n_mat / n_sum) * weight * n_agg_wt * -1.0
    negative_contribution = (n_mat / n_sum_safe) * weights_transposed * n_agg_wt * -1.0
    
    # Combine contributions efficiently
    wt_mat = positive_contribution + negative_contribution
    
    # Transpose back to original shape (batch_size, channels, height, width)
    return wt_mat.permute(0, 3, 1, 2)
