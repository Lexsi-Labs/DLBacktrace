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
    
    # Handle case where inputs are already PyTorch tensors to avoid double conversion
    if not isinstance(relevance_y, torch.Tensor):
        relevance_y = torch.tensor(relevance_y, dtype=torch.float32, device=device)
    else:
        relevance_y = relevance_y.to(device=device, dtype=torch.float32)
        
    if not isinstance(input_array, torch.Tensor):
        input_array = torch.tensor(input_array, dtype=torch.float32, device=device)
    else:
        input_array = input_array.to(device=device, dtype=torch.float32)
        
    if not isinstance(w, torch.Tensor):
        w = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        w = w.to(device=device, dtype=torch.float32)
        
    if b is not None:
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float32, device=device)
        else:
            b = b.to(device=device, dtype=torch.float32)
    
    if strides is not None:
        if not isinstance(strides, torch.Tensor):
            strides = torch.tensor(strides, dtype=torch.int32, device=device)
        else:
            strides = strides.to(device=device, dtype=torch.int32)
    
    if padding != 'valid' and padding != 'same':
        if not isinstance(padding, torch.Tensor):
            padding = torch.tensor(padding, dtype=torch.int32, device=device)
        else:
            padding = padding.to(device=device, dtype=torch.int32)
        
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
    
    # Extract positive and negative parts exactly like the original
    p_ind = conv_out * (conv_out > 0).float()  # Positive parts
    n_ind = conv_out * (conv_out < 0).float()  # Negative parts (remain negative)
    
    # Sum over spatial dimensions (i, j, k) to get per-channel sums
    p_sum = torch.sum(p_ind, dim=(0, 1, 2))  # Shape: (l,)
    n_sum = torch.sum(n_ind, dim=(0, 1, 2)) * -1.0  # Shape: (l,) - make negative parts positive
    t_sum = p_sum + n_sum
    
    # Initialize saturation indicators
    p_saturate = (p_sum > 0).float()
    n_saturate = (n_sum > 0).float()
    
    # Handle bias terms if present
    if b is not None:
        b_ind = (b > 0).float()
        bias_pos = b * b_ind
        b_ind = (b < 0).float()
        bias_neg = b * b_ind * -1.0
        denom_bias_term = bias_pos + bias_neg
    else:
        # Create zero tensors for bias terms when bias is None
        bias_pos = torch.zeros_like(p_sum)
        bias_neg = torch.zeros_like(n_sum)
        denom_bias_term = 0.0
    
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
    denom = torch.where(denom == 0, torch.tensor(1e-12, device=denom.device), denom)
    
    # Calculate aggregated weights
    p_agg_wt = (1.0 / denom) * wts * p_saturate
    n_agg_wt = (1.0 / denom) * wts * n_saturate
    
    # Calculate weighted matrix
    wt_mat = p_ind * p_agg_wt + n_ind * n_agg_wt * -1.0
    
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
        inp: Input tensor of shape (height, width, channels) to be padded
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
        h = inp.shape[0] % strides[0]
        if h == 0:
            pad_h = max(0, kernel_size[0] - strides[0])
        else:
            pad_h = max(0, kernel_size[0] - h)
        
        # Calculate required padding for width dimension  
        v = inp.shape[1] % strides[1]
        if v == 0:
            pad_v = max(0, kernel_size[1] - strides[1])
        else:
            pad_v = max(0, kernel_size[1] - v)
        
        # Calculate asymmetric padding exactly like the original version
        paddings = [
            [int(pad_h // 2), int((pad_h + 1) // 2)],
            [int(pad_v // 2), int((pad_v + 1) // 2)],
            [0, 0]  # No padding for channel dimension
        ]
        
        # Apply padding - convert list format to tuple for F.pad
        # Note: PyTorch pad expects (left, right, top, bottom, front, back) for 3D
        pad_tuple = (0, 0,  # No padding for channels (last dim)
                    paddings[1][0], paddings[1][1],  # Width padding
                    paddings[0][0], paddings[0][1])   # Height padding
        
        inp_padded = F.pad(inp, pad_tuple, mode='constant', value=const_val)
        return inp_padded, paddings
    
    # Handle custom padding (tuple) or fallback cases
    else:
        # Check for valid custom padding tuple
        if isinstance(padding, tuple) and padding != (None, None):
            pad_h, pad_v = padding
            
            # Apply symmetric padding exactly like the original version
            paddings = [
                [int(pad_h), int(pad_h)],
                [int(pad_v), int(pad_v)],
                [0, 0]  # No padding for channel dimension
            ]
            
            # Apply padding
            pad_tuple = (0, 0,  # No padding for channels
                        paddings[1][0], paddings[1][1],  # Width padding
                        paddings[0][0], paddings[0][1])   # Height padding
            
            inp_padded = F.pad(inp, pad_tuple, mode='constant', value=const_val)
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
    act: Dict[str, Any],
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
        b: Bias terms, shape (out_channels,) or None
        padding: Padding mode - 'valid', 'same', or tuple of (pad_h, pad_v)
        strides: Convolution strides as (stride_h, stride_w)
        act: Dictionary with activation function parameters
        
    Returns:
        Relevance scores propagated to input layer as numpy array, shape (batch_size, in_channels, height, width)
    """
    # Convert inputs to PyTorch format if needed
    relevance_y, input_array, w, b, padding, strides = convert_to_pytorch_format(
        relevance_y, input_array, w, b, padding, strides
    )
    
    batch_size = input_array.shape[0]
    
    # Transpose weight matrix to match NumPy behavior
    # NumPy does w = w.T which reverses all dimensions
    # For 4D tensor (out_channels, in_channels, h, w) -> (w, h, in_channels, out_channels)
    w_transposed = w.permute(3, 2, 1, 0)  # This matches w.T in NumPy for 4D arrays
    
    # Pre-allocate output tensor
    relevance_x = []
    
    # Process each sample in the batch
    for batch_idx in range(batch_size):
        # Extract current batch data and transpose to match NumPy layout
        # NumPy uses .T which reverses all dimensions
        current_relevance = relevance_y[batch_idx]  # Shape: (out_channels, height, width)
        current_input = input_array[batch_idx]      # Shape: (in_channels, height, width)
        
        # Apply .T transpose to match NumPy (reverses all dimensions)
        current_relevance = current_relevance.permute(2, 1, 0)  # (width, height, out_channels)
        current_input = current_input.permute(2, 1, 0)          # (width, height, in_channels)
        
        kernel_h, kernel_w = w_transposed.shape[0], w_transposed.shape[1]
        
        # Apply padding
        input_padded, paddings = calculate_padding(
            (kernel_h, kernel_w), current_input, padding, strides, const_val=0.0
        )
        
        # Initialize output tensor for accumulated updates
        output_accumulated = torch.zeros_like(input_padded)
        
        # Get output spatial dimensions for iteration
        output_height, output_width = current_relevance.shape[0], current_relevance.shape[1]
        
        stride_h, stride_w = strides
        
        # Process each spatial location in the output
        for out_h in range(output_height):
            for out_w in range(output_width):
                # Calculate indices for the patch
                h_start = out_h * stride_h
                h_end = h_start + kernel_h
                w_start = out_w * stride_w
                w_end = w_start + kernel_w
                
                # Ensure indices don't exceed padded input dimensions
                h_end = min(h_end, input_padded.shape[0])
                w_end = min(w_end, input_padded.shape[1])
                
                # Extract patch using slicing
                input_patch = input_padded[h_start:h_end, w_start:w_end, :]
                
                # Handle edge cases where patch might be smaller than kernel
                if input_patch.shape[0] < kernel_h or input_patch.shape[1] < kernel_w:
                    # Pad the patch to match kernel size
                    pad_h = kernel_h - input_patch.shape[0]
                    pad_w = kernel_w - input_patch.shape[1]
                    
                    if pad_h > 0 or pad_w > 0:
                        pad_tuple = (0, 0, 0, pad_w, 0, pad_h)  # (channels, width, height)
                        input_patch = F.pad(input_patch, pad_tuple, mode='constant', value=0.0)
                
                # Get relevance weight for current output location
                relevance_weight = current_relevance[out_h, out_w, :]
                
                # Calculate weighted convolution updates for this patch
                patch_updates = calculate_wt_conv_unit(
                    input_patch, relevance_weight, w_transposed, b, act
                )
                
                # Accumulate updates (handle potential size mismatch at boundaries)
                update_h = min(patch_updates.shape[0], h_end - h_start)
                update_w = min(patch_updates.shape[1], w_end - w_start)
                
                output_accumulated[h_start:h_start + update_h, 
                                 w_start:w_start + update_w, :] += \
                    patch_updates[:update_h, :update_w, :]
        
        # Remove padding to get final output
        pad_h_before = paddings[0][0]
        pad_w_before = paddings[1][0]
        original_h = current_input.shape[0]
        original_w = current_input.shape[1]
        
        output_unpadded = output_accumulated[
            pad_h_before:pad_h_before + original_h,
            pad_w_before:pad_w_before + original_w,
            :
        ]
        
        # Transpose back to match NumPy output format (apply .T)
        output_unpadded = output_unpadded.permute(2, 1, 0)  # Reverse dimensions again
        relevance_x.append(output_unpadded)
    
    # Stack results and convert to numpy
    relevance_x = torch.stack(relevance_x, dim=0)
    
    # Move to CPU and convert to numpy
    return relevance_x.cpu().numpy()