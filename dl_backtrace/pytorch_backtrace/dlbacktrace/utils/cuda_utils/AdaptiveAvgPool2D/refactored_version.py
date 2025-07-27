import numpy as np
from typing import Tuple

def calculate_wt_gavgpool(relevance_y: np.ndarray, input_array: np.ndarray) -> np.ndarray:
    """
    Calculate weighted global average pooling with positive/negative weight aggregation.
    
    This function processes input arrays by separating positive and negative values,
    computing aggregated weights based on their sums, and applying weighted transformations
    across all channels and batch samples.
    
    Args:
        relevance_y (np.ndarray): Relevance weights array of shape (batch_size, channels, height, width)
        input_array (np.ndarray): Input array of shape (batch_size, channels, height, width)
    
    Returns:
        np.ndarray: Weighted relevance array of shape (batch_size, channels, height, width)
    
    Note:
        - Replicates original logic for handling division by zero (sets sum to 1.0 when sum is 0.0)
        - Maintains exact numerical behavior including potential numerical instabilities
    """
    
    # Transpose all arrays once for vectorized operations (batch, height, width, channels)
    input_transposed = input_array.transpose(0, 2, 3, 1)  # (bs, h, w, c)
    weights_transposed = relevance_y.transpose(0, 2, 3, 1)  # (bs, h, w, c)
    
    # Initialize output array
    wt_mat = np.zeros_like(input_transposed)
    
    # Vectorized separation of positive and negative values across all batches and channels
    positive_mask = input_transposed >= 0  # (bs, h, w, c)
    negative_mask = input_transposed < 0   # (bs, h, w, c)
    
    # Create positive and negative matrices using broadcasting
    p_mat = input_transposed * positive_mask  # Zeros out negative values
    n_mat = input_transposed * negative_mask  # Zeros out positive values
    
    # Compute sums across spatial dimensions (h, w) for each batch and channel
    p_sum = np.sum(p_mat, axis=(1, 2))  # (bs, c)
    n_sum = np.sum(n_mat, axis=(1, 2)) * -1.0  # (bs, c) - make positive
    
    # Calculate aggregate weights - vectorized across all batches and channels
    total_sum = p_sum + n_sum  # (bs, c)
    
    # Replicate original logic: only compute aggregate weights when total_sum > 0
    valid_sum_mask = total_sum > 0.0
    p_agg_wt = np.zeros_like(total_sum)
    n_agg_wt = np.zeros_like(total_sum)
    
    # Use boolean indexing to apply original conditional logic
    p_agg_wt[valid_sum_mask] = p_sum[valid_sum_mask] / total_sum[valid_sum_mask]
    n_agg_wt[valid_sum_mask] = n_sum[valid_sum_mask] / total_sum[valid_sum_mask]
    
    # Handle division by zero: set sum to 1.0 when sum is 0.0 (replicating original logic)
    p_sum_safe = np.where(p_sum == 0.0, 1.0, p_sum)
    n_sum_safe = np.where(n_sum == 0.0, 1.0, n_sum)
    
    # Expand dimensions for broadcasting: (bs, 1, 1, c)
    p_sum_safe = p_sum_safe[:, np.newaxis, np.newaxis, :]
    n_sum_safe = n_sum_safe[:, np.newaxis, np.newaxis, :]
    p_agg_wt = p_agg_wt[:, np.newaxis, np.newaxis, :]
    n_agg_wt = n_agg_wt[:, np.newaxis, np.newaxis, :]
    
    # Vectorized weight calculation using broadcasting
    # Positive contribution: (p_mat / p_sum) * weight * p_agg_wt
    positive_contribution = (p_mat / p_sum_safe) * weights_transposed * p_agg_wt
    
    # Negative contribution: (n_mat / n_sum) * weight * n_agg_wt * -1.0
    negative_contribution = (n_mat / n_sum_safe) * weights_transposed * n_agg_wt * -1.0
    
    # Combine contributions
    wt_mat = positive_contribution + negative_contribution
    
    # Transpose back to original shape (batch_size, channels, height, width)
    return wt_mat.transpose(0, 3, 1, 2)
