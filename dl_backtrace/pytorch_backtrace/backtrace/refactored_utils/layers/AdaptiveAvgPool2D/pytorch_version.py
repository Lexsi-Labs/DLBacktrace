import torch
from torch import Tensor

def calculate_wt_gavgpool(wts: Tensor, inp: Tensor) -> Tensor:
    """
    Calculate weighted global average pooling with separate handling of positive and negative values.
    
    This function performs weighted global average pooling by:
    1. Separating positive and negative parts of the input
    2. Computing aggregate weights based on the proportion of positive/negative sums
    3. Applying weighted averaging separately to positive and negative components
    
    Args:
        wts (Tensor): Weight tensor that becomes (channels,) after transpose
        inp (Tensor): Input tensor that becomes (..., channels) after transpose
        
    Returns:
        Tensor: Weighted pooling result with same shape as transposed input
    """
    # Transpose inputs to match original function behavior
    wts_t = wts.T
    inp_t = inp.T
    
    # After transpose, the last dimension is the channel dimension
    channels = inp_t.shape[-1]
    
    # Vectorized separation of positive and negative parts across all channels
    p_mat = torch.clamp(inp_t, min=0.0)
    n_mat = torch.clamp(inp_t, max=0.0)
    
    # Sum over all spatial/batch dimensions (all except the last one)
    if inp_t.ndim > 1:
        spatial_axes = tuple(range(inp_t.ndim - 1))
        p_sums = torch.sum(p_mat, dim=spatial_axes)
        n_sums = torch.sum(n_mat, dim=spatial_axes) * -1.0
    else:
        p_sums = p_mat
        n_sums = n_mat * -1.0
    
    # Compute aggregate weights vectorized across all channels
    total_sums = p_sums + n_sums
    
    # Only compute aggregate weights where total_sums > 0 (replicating original condition exactly)
    valid_mask = total_sums > 0.0
    
    # Initialize aggregate weights with zeros
    p_agg_wts = torch.zeros_like(total_sums)
    n_agg_wts = torch.zeros_like(total_sums)
    
    # Use torch.where for vectorized conditional assignment
    p_agg_wts = torch.where(valid_mask, p_sums / total_sums, p_agg_wts)
    n_agg_wts = torch.where(valid_mask, n_sums / total_sums, n_agg_wts)
    
    # Handle division by zero cases (replicating original behavior exactly)
    # Use torch.where for conditional replacement with dtype-preserving constants
    p_sums_normalized = torch.where(p_sums == 0.0, 
                                   torch.ones_like(p_sums), 
                                   p_sums)
    n_sums_normalized = torch.where(n_sums == 0.0, 
                                   torch.ones_like(n_sums), 
                                   n_sums)
    
    # Reshape sums and aggregate weights for broadcasting
    if inp_t.ndim > 1:
        broadcast_shape = (1,) * (inp_t.ndim - 1) + (channels,)
        p_sums_broadcast = p_sums_normalized.reshape(broadcast_shape)
        n_sums_broadcast = n_sums_normalized.reshape(broadcast_shape)
        p_agg_wts_broadcast = p_agg_wts.reshape(broadcast_shape)
        n_agg_wts_broadcast = n_agg_wts.reshape(broadcast_shape)
    else:
        p_sums_broadcast = p_sums_normalized
        n_sums_broadcast = n_sums_normalized
        p_agg_wts_broadcast = p_agg_wts
        n_agg_wts_broadcast = n_agg_wts
    
    # Corrected vectorized operation: Multiply normalized input by the *transpose* of the transposed wts
    positive_contribution = (p_mat / p_sums_broadcast) * wts_t.T * p_agg_wts_broadcast
    negative_contribution = (n_mat / n_sums_broadcast) * wts_t.T * n_agg_wts_broadcast * -1.0
    
    wt_mat = positive_contribution + negative_contribution
    
    return wt_mat

# Compile the function for optimized execution
calculate_wt_gavgpool_compiled = torch.compile(calculate_wt_gavgpool, mode='max-autotune')
