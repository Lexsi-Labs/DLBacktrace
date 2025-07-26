import numpy as np
from typing import Tuple, Union, List, Callable, Optional, Dict, Any

def calculate_wt_conv_unit(
    patch: np.ndarray, 
    wts: np.ndarray, 
    w: np.ndarray, 
    b: Optional[np.ndarray], 
    act: Dict[str, Any]
) -> np.ndarray:
    """
    Calculate weighted convolution unit with activation function handling.
    
    This function computes convolution weights based on patch data, kernel weights,
    bias terms, and activation function parameters. It handles both monotonic and
    non-monotonic activation functions with optional range constraints.
    
    Args:
        patch: Input patch data of shape (i, j, k)
        wts: Weight values to be applied 
        w: Convolution kernel weights of shape (i, j, k, l)
        b: Optional bias array. If None, no bias is applied
        act: Dictionary containing activation function parameters with keys:
            - "type": str, either "mono" or "non_mono"
            - "range": dict with optional "l" (lower) and "u" (upper) bounds
            - "func": callable activation function (required for "non_mono" type)
    
    Returns:
        np.ndarray: Computed weight matrix of shape (i, j, k) after summing over
                   the last dimension
    """
    
    # Compute convolution output once
    conv_out = np.einsum("ijkl,ijk->ijkl", w, patch)
    
    # Extract positive and negative parts using vectorized operations
    p_ind = np.maximum(0, conv_out)  # Positive parts (replaces conv_out * (conv_out > 0))
    n_ind = np.minimum(0, conv_out)  # Negative parts (replaces conv_out * (conv_out < 0))
    
    # Sum over spatial dimensions (i, j, k) to get per-channel sums
    p_sum = np.sum(p_ind, axis=(0, 1, 2))  # Shape: (l,)
    n_sum = -np.sum(n_ind, axis=(0, 1, 2))  # Shape: (l,) - negative of negative parts
    t_sum = p_sum + n_sum
    
    # Handle bias terms if present
    if b is not None:
        bias_pos = np.maximum(0, b)  # Positive bias parts
        bias_neg = np.maximum(0, -b)  # Negative bias parts (made positive)
        denom_bias_term = bias_pos + bias_neg
    else:
        # Create zero arrays for bias terms when bias is None
        bias_pos = np.zeros_like(p_sum)
        bias_neg = np.zeros_like(n_sum)
        denom_bias_term = 0.0
    
    # Initialize saturation indicators
    p_saturate = p_sum > 0
    n_saturate = n_sum > 0
    
    # Handle activation function logic
    if act["type"] == 'mono':
        # Monotonic activation function
        if act["range"].get("l") is not None:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = temp_ind
        if act["range"].get("u") is not None:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = temp_ind
    
    elif act["type"] == 'non_mono':
        # Non-monotonic activation function
        t_act = act["func"](t_sum)
        p_act = act["func"](p_sum + bias_pos)
        n_act = act["func"](-(n_sum + bias_neg))
        
        # Apply range constraints if specified
        if act["range"].get("l") is not None:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = p_saturate * temp_ind
        if act["range"].get("u") is not None:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = n_saturate * temp_ind
        
        # Apply activation function difference thresholding (exactly replicating original logic)
        temp_ind = np.abs(t_act - p_act) > 1e-5
        n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - n_act) > 1e-5
        p_saturate = p_saturate * temp_ind
    
    # Calculate denominator with numerical stabilization (exactly replicating original method)
    denom = p_sum + n_sum + denom_bias_term
    denom = np.where(denom == 0, 1e-12, denom)  # Prevent division by zero using original epsilon
    
    # Calculate aggregated weights
    inv_denom = 1.0 / denom
    p_agg_wt = inv_denom * wts * p_saturate
    n_agg_wt = inv_denom * wts * n_saturate
    
    # Compute final weight matrix using broadcasting
    # p_ind and n_ind have shape (i,j,k,l), p_agg_wt and n_agg_wt have shape (l,)
    wt_mat = p_ind * p_agg_wt - n_ind * n_agg_wt  # Broadcasting handles shape alignment
    
    # Sum over the last dimension to get final result
    return np.sum(wt_mat, axis=-1)

def calculate_padding(
    kernel_size: Tuple[int, int], 
    inp: np.ndarray, 
    padding: Union[str, Tuple[Union[int, None], Union[int, None]]], 
    strides: Tuple[int, int], 
    const_val: float = 0.0
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Calculate and apply padding to input array for convolution operations.
    
    This function supports 'valid', 'same', and custom padding modes. For 'same' padding,
    it calculates the required padding to maintain output size. For custom padding,
    it applies symmetric padding based on provided values.
    
    Args:
        kernel_size: Tuple of (height, width) representing the kernel dimensions
        inp: Input array of shape (height, width, channels) to be padded
        padding: Padding mode - 'valid' for no padding, 'same' for size-preserving 
                padding, or tuple of (pad_h, pad_v) for custom padding
        strides: Tuple of (stride_height, stride_width) for convolution strides
        const_val: Constant value used for padding (default: 0.0)
    
    Returns:
        Tuple containing:
            - Padded input array (same as input if no padding applied)
            - List of padding values [[pad_h_before, pad_h_after], 
              [pad_v_before, pad_v_after], [0, 0]] for each dimension
    
    Examples:
        >>> inp = np.ones((4, 4, 3))
        >>> kernel_size = (3, 3)
        >>> strides = (1, 1)
        >>> padded_inp, paddings = calculate_padding(kernel_size, inp, 'same', strides)
    """
    # Handle 'valid' padding - no padding applied
    if padding == 'valid':
        return inp, [[0, 0], [0, 0], [0, 0]]
    
    # Handle 'same' padding - calculate padding to preserve output size
    elif padding == 'same':
        # Calculate required padding for height dimension
        height_remainder = inp.shape[0] % strides[0]
        if height_remainder == 0:
            pad_h = max(0, kernel_size[0] - strides[0])
        else:
            pad_h = max(0, kernel_size[0] - height_remainder)
        
        # Calculate required padding for width dimension  
        width_remainder = inp.shape[1] % strides[1]
        if width_remainder == 0:
            pad_v = max(0, kernel_size[1] - strides[1])
        else:
            pad_v = max(0, kernel_size[1] - width_remainder)
        
        # Calculate asymmetric padding (before, after) for each dimension
        # Replicating original floor division behavior exactly
        pad_h_before = int(np.floor(pad_h / 2.0))
        pad_h_after = int(np.floor((pad_h + 1) / 2.0))
        pad_v_before = int(np.floor(pad_v / 2.0))
        pad_v_after = int(np.floor((pad_v + 1) / 2.0))
        
        paddings = [
            [pad_h_before, pad_h_after],
            [pad_v_before, pad_v_after], 
            [0, 0]  # No padding for channel dimension
        ]
        
        # Apply padding using NumPy's optimized pad function
        inp_padded = np.pad(inp, paddings, mode='constant', constant_values=const_val)
        return inp_padded, paddings
    
    # Handle custom padding (tuple) or fallback cases
    else:
        # Check for valid custom padding tuple
        if isinstance(padding, tuple) and padding != (None, None):
            pad_h, pad_v = padding
            
            # Apply symmetric padding - same amount before and after
            paddings = [
                [int(np.floor(pad_h)), int(np.floor(pad_h))],
                [int(np.floor(pad_v)), int(np.floor(pad_v))],
                [0, 0]  # No padding for channel dimension
            ]
            
            # Apply padding using NumPy's optimized pad function
            inp_padded = np.pad(inp, paddings, mode='constant', constant_values=const_val)
            return inp_padded, paddings
        
        # Default case - no padding applied
        else:
            return inp, [[0, 0], [0, 0], [0, 0]]

def calculate_wt_conv(
    relevance_y: np.ndarray,
    input_array: np.ndarray, 
    w: np.ndarray,
    b: np.ndarray,
    padding: Union[str, Tuple[Union[int, None], Union[int, None]]],
    strides: Tuple[int, int],
    act: Callable
) -> np.ndarray:
    """
    Calculate weighted convolution for relevance propagation in neural networks.
    
    This function performs a weighted convolution operation that's commonly used in
    relevance propagation methods like Layer-wise Relevance Propagation (LRP). It
    processes each sample in the batch independently, applying convolution patches
    and accumulating relevance scores.
    
    Args:
        relevance_y: Relevance scores from the next layer, shape (batch_size, channels, height, width)
        input_array: Input activations, shape (batch_size, channels, height, width)  
        w: Convolution weights, shape (out_channels, in_channels, kernel_h, kernel_w)
        b: Bias terms, shape (out_channels,)
        padding: Padding mode - 'valid', 'same', or tuple of (pad_h, pad_v)
        strides: Convolution strides as (stride_h, stride_w)
        act: Activation function to apply during relevance calculation
        
    Returns:
        Relevance scores propagated to input layer, shape (batch_size, channels, height, width)
        
    Examples:
        >>> relevance_y = np.random.rand(2, 64, 8, 8)
        >>> input_array = np.random.rand(2, 32, 16, 16)
        >>> w = np.random.rand(64, 32, 3, 3)
        >>> b = np.random.rand(64)
        >>> relevance_x = calculate_wt_conv(relevance_y, input_array, w, b, 'same', (2, 2), relu)
    """
    batch_size = input_array.shape[0]
    
    # Transpose weight matrix to match original behavior
    w_transposed = w.T
    
    # Pre-allocate output array for better memory efficiency
    relevance_x = np.empty_like(input_array)
    
    # Process each sample in the batch
    for batch_idx in range(batch_size):
        # Extract current batch data and transpose to match original layout
        current_relevance = relevance_y[batch_idx].T  # Shape: (h, w, channels)
        current_input = input_array[batch_idx].T      # Shape: (h, w, channels)
        
        # Apply padding using the optimized calculate_padding function
        input_padded, paddings = calculate_padding(
            w_transposed.shape, current_input, padding, strides
        )
        
        # Initialize output tensor for accumulated updates
        output_accumulated = np.zeros_like(input_padded)
        
        # Get output spatial dimensions for iteration
        output_height, output_width = current_relevance.shape[0], current_relevance.shape[1]
        
        # Vectorized index calculation for better performance
        stride_h, stride_w = strides
        kernel_h, kernel_w = w_transposed.shape[0], w_transposed.shape[1]
        
        # Process each spatial location in the output
        for out_h in range(output_height):
            for out_w in range(output_width):
                # Calculate input patch indices using vectorized operations
                h_start = out_h * stride_h
                h_end = h_start + kernel_h
                w_start = out_w * stride_w  
                w_end = w_start + kernel_w
                
                # Extract input patch efficiently using advanced indexing
                input_patch = input_padded[h_start:h_end, w_start:w_end, :]
                
                # Get relevance weight for current output location
                relevance_weight = current_relevance[out_h, out_w, :]
                
                # Calculate weighted convolution updates for this patch
                patch_updates = calculate_wt_conv_unit(
                    input_patch, relevance_weight, w_transposed, b, act
                )
                
                # Accumulate updates into output tensor using in-place addition
                output_accumulated[h_start:h_end, w_start:w_end, :] += patch_updates
        
        # Remove padding to get final output, preserving original slice behavior
        pad_h_before, pad_w_before = paddings[0][0], paddings[1][0]
        original_h, original_w = current_input.shape[0], current_input.shape[1]
        
        output_unpadded = output_accumulated[
            pad_h_before:pad_h_before + original_h,
            pad_w_before:pad_w_before + original_w,
            :
        ]
        
        # Transpose back to original format and store in pre-allocated array
        relevance_x[batch_idx] = output_unpadded.T
    
    return relevance_x
