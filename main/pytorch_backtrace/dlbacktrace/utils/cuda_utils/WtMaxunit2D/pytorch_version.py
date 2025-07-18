import torch

def calculate_wt_max_unit_optimized(patch: torch.Tensor, wts: torch.Tensor, pool_size: int) -> torch.Tensor:
    """
    Optimized version with fused operations for better performance.
    
    Args:
        patch (torch.Tensor): Input 3D tensor of shape (height, width, channels).
        wts (torch.Tensor): 1D tensor of shape (channels,) containing weights.
        pool_size (int): Pool size parameter (unused).
    
    Returns:
        torch.Tensor: 3D tensor with distributed weights at maximum positions.
    """
    # Find maximum values across spatial dimensions (height, width) for each channel
    # Shape: (channels,)
    channel_max_values = torch.amax(patch, dim=(0, 1))
    
    # Create boolean mask identifying positions equal to maximum in each channel
    # Broadcasting: patch (H,W,C) == channel_max_values (C,) -> (H,W,C)
    max_positions_mask = torch.eq(patch, channel_max_values)
    
    # Convert boolean mask to float32 (replicating original dtype choice)
    max_positions_float = max_positions_mask.to(dtype=torch.float32)
    
    # Count number of maximum positions per channel
    # Shape: (channels,)
    max_count_per_channel = torch.sum(max_positions_float, dim=(0, 1))
    
    # Calculate normalization factors (1/count) for each channel
    # Replicating original's potential division by zero behavior
    normalization_factors = torch.reciprocal(max_count_per_channel)
    
    # Distribute weights among maximum positions
    # Broadcasting: max_positions_float (H,W,C) * normalization_factors (C,) -> (H,W,C)
    normalized_max_positions = max_positions_float * normalization_factors
    
    # Apply channel weights to get final output
    # Broadcasting: normalized_max_positions (H,W,C) * wts (C,) -> (H,W,C)
    weighted_output = normalized_max_positions * wts
    
    return weighted_output

# calculate_wt_max_unit_pytorch = torch.compile(calculate_wt_max_unit_optimized)
calculate_wt_max_unit_pytorch = calculate_wt_max_unit_optimized
