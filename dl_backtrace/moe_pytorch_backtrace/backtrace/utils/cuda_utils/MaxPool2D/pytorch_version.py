import torch
import torch.nn.functional as F
from typing import Tuple, Union
from ..WtMaxunit2D.pytorch_version import calculate_wt_max_unit_optimized as calculate_wt_max_unit
from ..Padding.pytorch import calculate_padding

def calculate_wt_maxpool(
    relevance_y: torch.Tensor,
    input_array: torch.Tensor,
    pool_size: Tuple[int, int],
    pad: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]]
) -> torch.Tensor:
    """
    Calculate weighted max pooling with relevance propagation using PyTorch.
    
    This function performs weighted max pooling on input tensors using relevance weights,
    propagating relevance values backward through the pooling operation. Optimized for
    GPU acceleration and vectorized operations.
    
    Args:
        relevance_y: Relevance weights tensor of shape (batch_size, channels, height, width)
        input_array: Input tensor of shape (batch_size, channels, height, width)
        pool_size: Tuple of (pool_height, pool_width) for pooling window size
        pad: Padding size, either int for symmetric padding or tuple of (pad_h, pad_w)
        stride: Stride size, either int for symmetric stride or tuple of (stride_h, stride_w)
    
    Returns:
        torch.Tensor: Relevance propagated tensor with same shape as input_array
    """
    batch_size, channels, input_h, input_w = input_array.shape
    device = input_array.device
    dtype = input_array.dtype
    
    # Normalize stride and padding to tuples for consistency
    strides = stride if isinstance(stride, tuple) else (stride, stride)
    padding = pad if isinstance(pad, tuple) else (pad, pad)
    
    # Pre-allocate result tensor for better memory efficiency
    relevance_x = torch.zeros_like(input_array)
    
    # Convert to NHWC format to match original NumPy logic: (B, C, H, W) -> (B, H, W, C)
    input_nhwc = input_array.permute(0, 2, 3, 1)
    relevance_nhwc = relevance_y.permute(0, 2, 3, 1)
    
    # Process each batch item (maintaining original structure for now)
    for batch_idx in range(batch_size):
        # Transpose for processing (original behavior preserved)
        # From (H, W, C) to (C, H, W) to match original .T operation
        weights_transposed = relevance_nhwc[batch_idx].permute(2, 0, 1)  # (C, H, W)
        input_transposed = input_nhwc[batch_idx].permute(2, 0, 1)        # (C, H, W)
        
        # Calculate padding using the existing function (maintains original logic)
        input_padded, paddings = calculate_padding(
            pool_size, input_transposed, padding, strides
        )
        
        # Initialize output with same shape as padded input
        output_downsampled = torch.zeros_like(input_padded)
        
        # Get output dimensions from relevance tensor
        output_height, output_width = weights_transposed.shape[1:3]
        
        # Vectorized processing where possible, but maintaining exact original logic
        for row_idx in range(output_height):
            for col_idx in range(output_width):
                # Calculate patch indices (preserving original indexing logic)
                row_start = row_idx * strides[0]
                row_end = row_start + pool_size[0]
                col_start = col_idx * strides[1]
                col_end = col_start + pool_size[1]
                
                # Extract patch using tensor slicing (maintains original behavior)
                patch = input_padded[:, row_start:row_end, col_start:col_end]
                
                # Calculate weighted max unit updates using existing function
                weight_slice = weights_transposed[:, row_idx, col_idx]  # Shape: (C,)
                updates = calculate_wt_max_unit(patch, weight_slice, pool_size)
                
                # Accumulate updates in-place (preserves original += operation)
                output_downsampled[:, row_start:row_end, col_start:col_end] += updates
        
        # Remove padding to restore original dimensions (exact original logic)
        pad_h_start, pad_h_end = paddings[0]
        pad_w_start, pad_w_end = paddings[1]
        
        unpadded_output = output_downsampled[
            :,
            pad_h_start:(pad_h_start + input_transposed.shape[1]),
            pad_w_start:(pad_w_start + input_transposed.shape[2])
        ]
        
        # Transpose back and store result (preserves original .T operation)
        # From (C, H, W) back to (H, W, C)
        relevance_nhwc_result = unpadded_output.permute(1, 2, 0)
        relevance_x[batch_idx] = relevance_nhwc_result.permute(2, 0, 1)  # Convert back to (C, H, W)
    
    return relevance_x
