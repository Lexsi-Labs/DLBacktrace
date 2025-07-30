import torch
from typing import Tuple, Union, Callable, Dict, Any, Optional, List
import torch.nn.functional as F

def convert_to_pytorch_format(
    relevance_y,
    input_array, 
    w, 
    b,
    padding,
    strides
) -> torch.Tensor:
    """
    Convert the input tensors to the PyTorch format.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    relevance_y = torch.tensor(relevance_y, dtype=torch.float32, device=device)
    input_array = torch.tensor(input_array, dtype=torch.float32, device=device)
    w = torch.tensor(w, dtype=torch.float32, device=device)
    b = torch.tensor(b, dtype=torch.float32, device=device)
    strides = torch.tensor(strides, dtype=torch.int32, device=device)
    
    if padding != 'valid' and padding != 'same':
        padding = torch.tensor(padding, dtype=torch.int32, device=device)
        
    return relevance_y, input_array, w, b, padding, strides
    
def calculate_wt_conv_unit(
    patch: torch.Tensor, 
    wts: torch.Tensor, 
    w: torch.Tensor, 
    b: Optional[torch.Tensor], 
    act: Dict[str, Any]
) -> torch.Tensor:
    """
    Calculate weighted convolution unit with activation function handling.
    
    This function computes convolution weights based on patch data, kernel weights,
    bias terms, and activation function parameters. It handles both monotonic and
    non-monotonic activation functions with optional range constraints.
    
    Args:
        patch: Input patch data of shape (i, j, k)
        wts: Weight values to be applied 
        w: Convolution kernel weights of shape (i, j, k, l)
        b: Optional bias tensor. If None, no bias is applied
        act: Dictionary containing activation function parameters with keys:
            - "type": str, either "mono" or "non_mono"
            - "range": dict with optional "l" (lower) and "u" (upper) bounds
            - "func": callable activation function (required for "non_mono" type)
    
    Returns:
        torch.Tensor: Computed weight matrix of shape (i, j, k) after summing over
                     the last dimension
    """
    
    # Compute convolution output once using torch.einsum
    conv_out = torch.einsum("ijkl,ijk->ijkl", w, patch)
    
    # Extract positive and negative parts using vectorized operations
    p_ind = torch.clamp_min(conv_out, 0)  # Positive parts
    n_ind = torch.clamp_max(conv_out, 0)  # Negative parts
    
    # Sum over spatial dimensions (i, j, k) to get per-channel sums
    p_sum = torch.sum(p_ind, dim=(0, 1, 2))  # Shape: (l,)
    n_sum = -torch.sum(n_ind, dim=(0, 1, 2))  # Shape: (l,) - negative of negative parts
    t_sum = p_sum + n_sum
    
    # Handle bias terms if present
    if b is not None:
        bias_pos = torch.clamp_min(b, 0)  # Positive bias parts
        bias_neg = torch.clamp_min(-b, 0)  # Negative bias parts (made positive)
        denom_bias_term = bias_pos + bias_neg
    else:
        # Create zero tensors for bias terms when bias is None
        bias_pos = torch.zeros_like(p_sum)
        bias_neg = torch.zeros_like(n_sum)
        denom_bias_term = 0.0
    
    # Initialize saturation indicators
    p_saturate = (p_sum > 0).float()
    n_saturate = (n_sum > 0).float()
    
    # Handle activation function logic
    if act["type"] == 'mono':
        # Monotonic activation function
        if act["range"].get("l") is not None:
            temp_ind = (t_sum > act["range"]["l"]).float()
            p_saturate = temp_ind
        if act["range"].get("u") is not None:
            temp_ind = (t_sum < act["range"]["u"]).float()
            n_saturate = temp_ind
    
    elif act["type"] == 'non_mono':
        # Non-monotonic activation function
        t_act = act["func"](t_sum)
        p_act = act["func"](p_sum + bias_pos)
        n_act = act["func"](-(n_sum + bias_neg))
        
        # Apply range constraints if specified
        if act["range"].get("l") is not None:
            temp_ind = (t_sum > act["range"]["l"]).float()
            p_saturate = p_saturate * temp_ind
        if act["range"].get("u") is not None:
            temp_ind = (t_sum < act["range"]["u"]).float()
            n_saturate = n_saturate * temp_ind
        
        # Apply activation function difference thresholding
        temp_ind = (torch.abs(t_act - p_act) > 1e-5).float()
        n_saturate = n_saturate * temp_ind
        temp_ind = (torch.abs(t_act - n_act) > 1e-5).float()
        p_saturate = p_saturate * temp_ind
    
    # Calculate denominator with numerical stabilization
    denom = p_sum + n_sum + denom_bias_term
    denom = torch.where(denom == 0, torch.tensor(1e-12, device=denom.device, dtype=denom.dtype), denom)
    
    # Calculate aggregated weights
    inv_denom = 1.0 / denom
    p_agg_wt = inv_denom * wts * p_saturate
    n_agg_wt = inv_denom * wts * n_saturate
    
    # Compute final weight matrix using broadcasting
    # p_ind and n_ind have shape (i,j,k,l), p_agg_wt and n_agg_wt have shape (l,)
    wt_mat = p_ind * p_agg_wt - n_ind * n_agg_wt  # Broadcasting handles shape alignment
    
    # Sum over the last dimension to get final result
    return torch.sum(wt_mat, dim=-1)

def calculate_padding(
    kernel_size: Tuple[int, int], 
    inp: torch.Tensor, 
    padding: Union[str, Tuple[Union[int, None], Union[int, None]]], 
    strides: Tuple[int, int], 
    const_val: float = 0.0
) -> Tuple[torch.Tensor, List[List[int]]]:
    """
    Calculate and apply padding to input tensor for convolution operations.
    
    This function supports 'valid', 'same', and custom padding modes. For 'same' padding,
    it calculates the required padding to maintain output size. For custom padding,
    it applies symmetric padding based on provided values.
    
    Args:
        kernel_size: Tuple of (height, width) representing the kernel dimensions
        inp: Input tensor of shape (height, width, channels, batch_size) to be padded
        padding: Padding mode - 'valid' for no padding, 'same' for size-preserving 
                padding, or tuple of (pad_h, pad_v) for custom padding
        strides: Tuple of (stride_height, stride_width) for convolution strides
        const_val: Constant value used for padding (default: 0.0)
    
    Returns:
        Tuple containing:
            - Padded input tensor (same as input if no padding applied)
            - List of padding values [[pad_h_before, pad_h_after], 
              [pad_v_before, pad_v_after], [0, 0]] for each dimension
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
        
        # Calculate asymmetric padding exactly like the original version
        paddings = [
            torch.floor(torch.tensor([pad_h/2.0, (pad_h+1)/2.0])).to(torch.int32),
            torch.floor(torch.tensor([pad_v/2.0, (pad_v+1)/2.0])).to(torch.int32),
            torch.zeros(2, dtype=torch.int32)  # No padding for channel dimension
        ]
        
        # Apply padding using PyTorch's pad function
        # Note: F.pad expects (left, right, top, bottom) for 2D padding
        pad_values = (paddings[1][0].item(), paddings[1][1].item(), paddings[0][0].item(), paddings[0][1].item())
        inp = inp.permute(2, 0, 1)
        inp_padded = F.pad(inp, pad_values, mode='constant', value=const_val)
        inp_padded = inp_padded.permute(1, 2, 0)
        return inp_padded, paddings 
    
    # Handle custom padding (tuple) or fallback cases
    else:
        # Check for valid custom padding tuple
        if isinstance(padding, tuple) and padding != (None, None):
            pad_h, pad_v = padding
            
            # Apply symmetric padding exactly like the original version
            paddings = [
                torch.floor(torch.tensor([pad_h, pad_h])).to(torch.int32),
                torch.floor(torch.tensor([pad_v, pad_v])).to(torch.int32),
                torch.zeros(2, dtype=torch.int32)  # No padding for channel dimension
            ]
            
            # Apply padding using PyTorch's pad function
            pad_values = (paddings[1][0].item(), paddings[1][1].item(), paddings[0][0].item(), paddings[0][1].item())
            inp = inp.permute(2, 0, 1)
            inp_padded = F.pad(inp, pad_values, mode='constant', value=const_val)
            inp_padded = inp_padded.permute(1, 2, 0)
            return inp_padded, paddings
        
        # Default case - no padding applied
        else:
            return inp, [[0, 0], [0, 0], [0, 0]]

def calculate_wt_conv(
    relevance_y,
    input_array, 
    w,
    b,
    padding: Union[str, Tuple[Union[int, None], Union[int, None]]],
    strides: Tuple[int, int],
    act: Dict[str, Any]
) -> torch.Tensor:
    """
    Calculate weighted convolution for relevance propagation in neural networks.
    
    This function performs a weighted convolution operation that's commonly used in
    relevance propagation methods like Layer-wise Relevance Propagation (LRP). It
    processes each sample in the batch independently, applying convolution patches
    and accumulating relevance scores.
    
    Args:
        relevance_y: Relevance scores from the next layer, shape (batch_size, out_channels, height, width)
        input_array: Input activations, shape (batch_size, in_channels, height, width)  
        w: Convolution weights, shape (out_channels, in_channels, kernel_h, kernel_w)
        b: Bias terms, shape (out_channels,)
        padding: Padding mode - 'valid', 'same', or tuple of (pad_h, pad_v)
        strides: Convolution strides as (stride_h, stride_w)
        act: Dictionary with activation function parameters
        
    Returns:
        Relevance scores propagated to input layer, shape (batch_size, in_channels, height, width)
    """
    relevance_y, input_array, w, b, padding, strides = convert_to_pytorch_format(relevance_y, input_array, w, b, padding, strides)
    batch_size = input_array.shape[0]
    
    # Transpose weight matrix to match original behavior (out_channels, in_channels, h, w) -> (h, w, in_channels, out_channels)
    w_transposed = w.permute(2, 3, 1, 0)
    
    # Pre-allocate output tensor for better memory efficiency
    relevance_x = torch.zeros_like(input_array)
    
    # Process each sample in the batch
    for batch_idx in range(batch_size):
        # Extract current batch data and transpose to match original layout (batch, channels, h, w) -> (h, w, channels, batch)
        current_relevance = relevance_y[batch_idx].permute(2, 1, 0)  # Shape: (h, w, out_channels)
        current_input = input_array[batch_idx].permute(2, 1, 0)      # Shape: (h, w, in_channels)
        
        # Apply padding using kernel shape like the original version
        input_padded, paddings = calculate_padding(
            (kernel_h, kernel_w), current_input, padding, strides
        )
        
        # Initialize output tensor for accumulated updates
        output_accumulated = torch.zeros_like(input_padded)
        
        # Get output spatial dimensions for iteration
        output_height, output_width = current_relevance.shape[0], current_relevance.shape[1]
        
        # Vectorized index calculation for better performance
        stride_h, stride_w = strides
        kernel_h, kernel_w = w_transposed.shape[0], w_transposed.shape[1]
        
        # Process each spatial location in the output (matching original's loop structure)
        for out_h in range(output_height):
            for out_w in range(output_width):
                # Calculate index ranges exactly like the original version
                h_indices = torch.arange(out_h * stride_h, out_h * stride_h + kernel_h, device=input_padded.device)
                w_indices = torch.arange(out_w * stride_w, out_w * stride_w + kernel_w, device=input_padded.device)
                
                # Extract patch using advanced indexing (equivalent to np.ix_)
                # This guarantees exact kernel dimensions
                input_patch = input_padded[h_indices.unsqueeze(1), w_indices.unsqueeze(0), :]
                
                # Get relevance weight for current output location
                relevance_weight = current_relevance[out_h, out_w, :]
                
                # Calculate weighted convolution updates for this patch
                patch_updates = calculate_wt_conv_unit(
                    input_patch, relevance_weight, w_transposed, b, act
                )
                
                # Accumulate updates using the same indexing pattern
                output_accumulated[h_indices.unsqueeze(1), w_indices.unsqueeze(0), :] += patch_updates
        
        # Remove padding to get final output, preserving original slice behavior
        pad_h_before, pad_w_before = paddings[0][0], paddings[1][0]
        original_h, original_w = current_input.shape[0], current_input.shape[1]
        
        output_unpadded = output_accumulated[
            pad_h_before:pad_h_before + original_h,
            pad_w_before:pad_w_before + original_w,
            :
        ]
        
        # Transpose back to original format (h, w, channels) -> (channels, h, w) and store
        relevance_x[batch_idx] = output_unpadded.permute(2, 0, 1)
    
    return relevance_x.cpu().numpy()
