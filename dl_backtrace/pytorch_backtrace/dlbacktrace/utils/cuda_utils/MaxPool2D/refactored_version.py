import numpy as np
from typing import Tuple, Union
from ..WtMaxunit2D.refactored_version import calculate_wt_max_unit
from ..Padding.original import calculate_padding

def calculate_wt_maxpool(
    relevance_y: np.ndarray, 
    input_array: np.ndarray, 
    pool_size: Tuple[int, int], 
    pad: Union[int, Tuple[int, int]], 
    stride: Union[int, Tuple[int, int]]
) -> np.ndarray:
    """
    Calculate weighted max pooling with relevance propagation.
    
    This function performs weighted max pooling on input arrays using relevance weights,
    propagating relevance values backward through the pooling operation.
    
    Args:
        relevance_y: Relevance weights array of shape (batch_size, height, width, channels)
        input_array: Input array of shape (batch_size, height, width, channels)
        pool_size: Tuple of (pool_height, pool_width) for pooling window size
        pad: Padding size, either int for symmetric padding or tuple of (pad_h, pad_w)
        stride: Stride size, either int for symmetric stride or tuple of (stride_h, stride_w)
    
    Returns:
        np.ndarray: Relevance propagated array with same shape as input_array
    """
    batch_size, _, _, _ = input_array.shape
    
    # Normalize stride and padding to tuples for consistency
    strides = stride if isinstance(stride, tuple) else (stride, stride)
    padding = pad if isinstance(pad, tuple) else (pad, pad)
    
    # Pre-allocate result array for better memory efficiency
    relevance_x = np.zeros_like(input_array)
    
    # Process each batch item
    for batch_idx in range(batch_size):
        # Transpose for processing (original behavior preserved)
        weights_transposed = relevance_y[batch_idx].T
        input_transposed = input_array[batch_idx].T
        
        # Calculate padding using the existing function (maintains original logic)
        input_padded, paddings = calculate_padding(
            pool_size, input_transposed, padding, strides
        )
        
        # Initialize output with same shape as padded input
        output_downsampled = np.zeros_like(input_padded)
        
        # Vectorized processing where possible, but maintaining exact original logic
        output_height, output_width = weights_transposed.shape[:2]
        
        for row_idx in range(output_height):
            for col_idx in range(output_width):
                # Calculate patch indices (preserving original indexing logic)
                row_indices = np.arange(
                    row_idx * strides[0], 
                    row_idx * strides[0] + pool_size[0]
                )
                col_indices = np.arange(
                    col_idx * strides[1], 
                    col_idx * strides[1] + pool_size[1]
                )
                
                # Extract patch using advanced indexing (maintains original behavior)
                patch = input_padded[np.ix_(row_indices, col_indices)]
                
                # Calculate weighted max unit updates using existing function
                weight_slice = weights_transposed[row_idx, col_idx, :]
                updates = calculate_wt_max_unit(patch, weight_slice, pool_size)
                
                # Accumulate updates in-place (preserves original += operation)
                output_downsampled[np.ix_(row_indices, col_indices)] += updates
        
        # Remove padding to restore original dimensions (exact original logic)
        unpadded_output = output_downsampled[
            paddings[0][0]:(paddings[0][0] + input_transposed.shape[0]),
            paddings[1][0]:(paddings[1][0] + input_transposed.shape[1]),
            :
        ]
        
        # Transpose back and store result (preserves original .T operation)
        relevance_x[batch_idx] = unpadded_output.T
    
    return relevance_x
