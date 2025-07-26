import numpy as np
from typing import Tuple, Union
from ..WtMaxunit2D.refactored_version import calculate_wt_max_unit
from ..Conv2D.refactored_version import calculate_padding

def calculate_wt_maxpool(wts: np.ndarray, inp: np.ndarray, pool_size: Union[int, Tuple[int, int]], 
                        padding: Union[int, Tuple[int, int]], strides: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Perform weighted max pooling operation on input tensor.
    
    This function applies weighted max pooling where weights are distributed among the maximum
    values within each pooling window. The operation slides a pooling window across the input
    with specified strides, and for each window, identifies maximum values per channel and
    distributes the corresponding weights among these maximum positions.
    
    Args:
        wts (np.ndarray): Weights tensor of shape (channels, out_height, out_width) containing
                         the weights to be applied at each output position for each channel.
        inp (np.ndarray): Input tensor of shape (channels, in_height, in_width) to be pooled.
        pool_size (Union[int, Tuple[int, int]]): Size of the pooling window. If int, same size
                                               is used for both height and width dimensions.
        padding (Union[int, Tuple[int, int]]): Padding to be applied. If int, same padding
                                             is used for both height and width dimensions.
        strides (Union[int, Tuple[int, int]]): Stride of the pooling operation. If int, same
                                             stride is used for both height and width dimensions.
    
    Returns:
        np.ndarray: Output tensor of same shape as input, where weighted max pooling has been
                   applied. Values represent the distributed weights at positions achieving
                   maximum values within their respective pooling windows.
    
    Notes:
        - Input tensors are transposed at the beginning and the result maintains original orientation
        - Padding is applied with -inf values to ensure they don't interfere with max operations
        - Overlapping pooling windows accumulate their contributions additively
    """
    # Transpose inputs to work with internal representation
    # This replicates the original's tensor orientation handling
    wts_transposed = wts.T
    inp_transposed = inp.T
    
    # Normalize stride and padding parameters to tuples
    strides_tuple = (strides, strides) if isinstance(strides, int) else strides
    padding_tuple = (padding, padding) if isinstance(padding, int) else padding
    
    # Normalize pool_size to tuple for consistent handling
    if isinstance(pool_size, int):
        pool_size_tuple = (pool_size, pool_size)
    else:
        pool_size_tuple = pool_size
    
    # Apply padding to input with -inf values (replicating original behavior)
    input_padded, paddings = calculate_padding(pool_size_tuple, inp_transposed, 
                                             padding_tuple, strides_tuple, -np.inf)
    
    # Initialize output array with zeros, same shape as padded input
    output_accumulated = np.zeros_like(input_padded)
    
    # Get output dimensions from weights tensor
    out_height, out_width = wts_transposed.shape[:2]
    
    # Iterate through each output position
    for output_row in range(out_height):
        for output_col in range(out_width):
            # Calculate the receptive field indices for current output position
            row_start = output_row * strides_tuple[0]
            row_end = row_start + pool_size_tuple[0]
            col_start = output_col * strides_tuple[1]
            col_end = col_start + pool_size_tuple[1]
            
            # Extract patch from padded input using advanced indexing
            row_indices = np.arange(row_start, row_end)
            col_indices = np.arange(col_start, col_end)
            current_patch = input_padded[np.ix_(row_indices, col_indices)]
            
            # Get weights for current output position across all channels
            current_weights = wts_transposed[output_row, output_col, :]
            
            # Calculate weighted max unit values for current patch
            weighted_updates = calculate_wt_max_unit(current_patch, current_weights, pool_size_tuple)
            
            # Accumulate updates into output array at corresponding positions
            # This handles overlapping pooling windows by summing contributions
            output_accumulated[np.ix_(row_indices, col_indices)] += weighted_updates
    
    # Remove padding to get final output with original input dimensions
    # Extract the region corresponding to the original input size
    final_output = output_accumulated[
        paddings[0][0]:(paddings[0][0] + inp_transposed.shape[0]),
        paddings[1][0]:(paddings[1][0] + inp_transposed.shape[1]),
        :
    ]
    
    return final_output
