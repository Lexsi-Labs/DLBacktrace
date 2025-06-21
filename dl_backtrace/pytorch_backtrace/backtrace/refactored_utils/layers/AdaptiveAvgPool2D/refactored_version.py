import numpy as np
from typing import Tuple

def calculate_wt_gavgpool(wts: np.ndarray, inp: np.ndarray) -> np.ndarray:
    """
    Calculate weighted global average pooling with separate handling of positive and negative values.
    
    This function performs weighted global average pooling by:
    1. Separating positive and negative parts of the input
    2. Computing aggregate weights based on the proportion of positive/negative sums
    3. Applying weighted averaging separately to positive and negative components
    
    Args:
        wts (np.ndarray): Weight array that becomes (channels,) after transpose
        inp (np.ndarray): Input array that becomes (..., channels) after transpose
        
    Returns:
        np.ndarray: Weighted pooling result with same shape as transposed input
    """
    # Transpose inputs to match original function behavior
    wts_t = wts.T
    inp_t = inp.T
    
    # After transpose, the last dimension is the channel dimension
    channels = inp_t.shape[-1]
    
    # Vectorized separation of positive and negative parts across all channels
    p_mat = np.maximum(inp_t, 0)
    n_mat = np.minimum(inp_t, 0)
    
    # Sum over all spatial/batch dimensions (all except the last one)
    if inp_t.ndim > 1:
        spatial_axes = tuple(range(inp_t.ndim - 1))
        p_sums = np.sum(p_mat, axis=spatial_axes)
        n_sums = np.sum(n_mat, axis=spatial_axes) * -1
    else:
        p_sums = p_mat
        n_sums = n_mat * -1
    
    # Compute aggregate weights vectorized across all channels
    total_sums = p_sums + n_sums
    
    # Initialize aggregate weights
    p_agg_wts = np.zeros(channels, dtype=inp_t.dtype)
    n_agg_wts = np.zeros(channels, dtype=inp_t.dtype)
    
    # Only compute aggregate weights where total_sums > 0 (replicating original condition exactly)
    valid_mask = total_sums > 0.0
    p_agg_wts[valid_mask] = p_sums[valid_mask] / total_sums[valid_mask]
    n_agg_wts[valid_mask] = n_sums[valid_mask] / total_sums[valid_mask]
    
    # Handle division by zero cases (replicating original behavior exactly)
    # Use dtype-preserving constants
    p_sums_normalized = np.where(p_sums == 0.0, np.array(1.0, dtype=inp.dtype), p_sums)
    n_sums_normalized = np.where(n_sums == 0.0, np.array(1.0, dtype=inp.dtype), n_sums)
    
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
    # which is the original wts. (wts_t.T = (wts.T).T = wts)
    positive_contribution = (p_mat / p_sums_broadcast) * wts_t.T * p_agg_wts_broadcast
    negative_contribution = (n_mat / n_sums_broadcast) * wts_t.T * n_agg_wts_broadcast * -1.0
    
    wt_mat = positive_contribution + negative_contribution
    
    return wt_mat
