import numpy as np
from typing import Tuple

def calculate_wt_max_unit(patch: np.ndarray, wts: np.ndarray, pool_size: int) -> np.ndarray:
    """
    Calculate weighted max unit values by distributing weights among maximum values in each channel.
    
    This function identifies the maximum values across the spatial dimensions (height and width)
    for each channel, then distributes the corresponding weights equally among all positions
    that achieve this maximum value within each channel.
    
    Args:
        patch (np.ndarray): Input 3D array of shape (height, width, channels) containing
                           the patch data to process.
        wts (np.ndarray): 1D array of shape (channels,) containing weights for each channel.
        pool_size (int): Pool size parameter (note: not used in current implementation).
    
    Returns:
        np.ndarray: 3D array of same shape as patch, where each position contains either
                   zero or the channel weight divided by the number of maximum positions
                   in that channel.
    
    Examples:
        >>> patch = np.array([[[1, 3], [2, 1]], [[2, 2], [1, 3]]])  # 2x2x2
        >>> wts = np.array([0.5, 1.0])
        >>> result = calculate_wt_max_unit(patch, wts, 2)
        # Returns weights distributed among maximum positions per channel
    """
    # Find maximum values across spatial dimensions (height, width) for each channel
    # Shape: (channels,)
    channel_max_values = np.max(patch, axis=(0, 1))
    
    # Create boolean mask identifying positions equal to maximum in each channel
    # Broadcasting: patch (H,W,C) == channel_max_values (C,) -> (H,W,C)
    max_positions_mask = (patch == channel_max_values)
    
    # Convert boolean mask to float32 (replicating original dtype choice)
    max_positions_float = max_positions_mask.astype(np.float32)
    
    # Count number of maximum positions per channel
    # Shape: (channels,)
    max_count_per_channel = np.sum(max_positions_float, axis=(0, 1))
    
    # Calculate normalization factors (1/count) for each channel
    # Replicating original's potential division by zero behavior
    normalization_factors = 1.0 / max_count_per_channel
    
    # Distribute weights among maximum positions
    # Broadcasting: max_positions_float (H,W,C) * normalization_factors (C,) -> (H,W,C)
    normalized_max_positions = max_positions_float * normalization_factors
    
    # Apply channel weights to get final output
    # Broadcasting: normalized_max_positions (H,W,C) * wts (C,) -> (H,W,C)
    weighted_output = normalized_max_positions * wts
    
    return weighted_output
