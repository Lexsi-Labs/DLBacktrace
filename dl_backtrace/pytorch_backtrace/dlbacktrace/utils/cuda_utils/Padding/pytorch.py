import torch
import torch.nn.functional as F
from typing import Union, Tuple, List

def calculate_padding(
    kernel_size: Tuple[int, int],
    inp: torch.Tensor,
    padding: Union[str, Tuple[int, int]],
    strides: Tuple[int, int],
    const_val: float = 0.0
) -> Tuple[torch.Tensor, List[List[int]]]:
    """
    Calculate and apply padding to input tensor for convolution operations.
    
    This function supports 'valid', 'same', and explicit padding modes, maintaining
    gradient flow for autograd compatibility.
    
    Args:
        kernel_size: Tuple of (height, width) kernel dimensions
        inp: Input tensor of shape (H, W, C) or (H, W)
        padding: Padding mode - 'valid', 'same', or tuple of (pad_h, pad_v)
        strides: Tuple of (stride_h, stride_v) for convolution
        const_val: Constant value for padding (default: 0.0)
    
    Returns:
        Tuple containing:
            - Padded tensor with same dtype as input
            - List of padding values [[pad_h_before, pad_h_after], 
                                    [pad_v_before, pad_v_after], 
                                    [0, 0]]
    """
    device = inp.device
    dtype = inp.dtype
    
    if padding == 'valid':
        zero_padding = [[0, 0], [0, 0], [0, 0]]
        return inp, zero_padding
    
    elif padding == 'same':
        # Cache shape access for efficiency
        h_size, w_size = inp.shape[0], inp.shape[1]
        
        # Vectorized computation for padding calculation
        h_remainder = h_size % strides[0]
        w_remainder = w_size % strides[1]
        
        # Use torch.max for consistency and autograd compatibility
        pad_h = torch.max(
            torch.tensor(0, device=device),
            torch.tensor(kernel_size[0] - strides[0] if h_remainder == 0 else kernel_size[0] - h_remainder, device=device)
        ).item()
        
        pad_w = torch.max(
            torch.tensor(0, device=device),
            torch.tensor(kernel_size[1] - strides[1] if w_remainder == 0 else kernel_size[1] - w_remainder, device=device)
        ).item()
        
        # Efficient padding calculation using floor division
        pad_h_before = pad_h // 2
        pad_h_after = (pad_h + 1) // 2
        pad_w_before = pad_w // 2
        pad_w_after = (pad_w + 1) // 2
        
        # Create padding tuple for F.pad (format: [left, right, top, bottom])
        # F.pad expects padding in reverse order: [dim_n-1, dim_n-1, dim_n-2, dim_n-2, ...]
        pad_tuple = (pad_w_before, pad_w_after, pad_h_before, pad_h_after)
        
        # Apply padding using F.pad for better performance and autograd support
        inp_pad = F.pad(inp, pad_tuple, mode='constant', value=const_val)
        
        # Return padding in original format for consistency
        paddings = [[pad_h_before, pad_h_after], 
                   [pad_w_before, pad_w_after], 
                   [0, 0]]
        
        return inp_pad, paddings
    
    else:
        # Handle explicit padding tuple
        if isinstance(padding, tuple) and padding != (None, None):
            pad_h, pad_w = padding
            
            # Create symmetric padding
            pad_tuple = (pad_w, pad_w, pad_h, pad_h)
            inp_pad = F.pad(inp, pad_tuple, mode='constant', value=const_val)
            
            paddings = [[pad_h, pad_h], 
                       [pad_w, pad_w], 
                       [0, 0]]
            
            return inp_pad, paddings
        else:
            zero_padding = [[0, 0], [0, 0], [0, 0]]
            return inp, zero_padding
