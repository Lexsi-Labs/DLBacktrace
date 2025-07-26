import torch
import torch.nn.functional as F
from typing import Union, Tuple, List

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
        
        # Calculate asymmetric padding (before, after) for each dimension
        pad_h_before = int(pad_h // 2)
        pad_h_after = int((pad_h + 1) // 2)
        pad_v_before = int(pad_v // 2)
        pad_v_after = int((pad_v + 1) // 2)
        
        paddings = [
            torch.floor(torch.tensor([pad_h_before, pad_h_after])).to(torch.int32),
            torch.floor(torch.tensor([pad_v_before, pad_v_after])).to(torch.int32), 
            torch.tensor([0, 0]).to(torch.int32)  # No padding for channel dimension
        ]
        
        # Apply padding using PyTorch's pad function
        # Note: F.pad expects (left, right, top, bottom) for 2D padding
        pad_values = (pad_v_before, pad_v_after, pad_h_before, pad_h_after)
        inp= inp.permute(2, 0, 1)
        inp_padded = F.pad(inp, pad_values, mode='constant', value=const_val)
        inp_padded = inp_padded.permute(1, 2, 0)
        return inp_padded, paddings 
    
    # Handle custom padding (tuple) or fallback cases
    else:
        # Check for valid custom padding tuple
        if isinstance(padding, tuple) and padding != (None, None):
            pad_h, pad_v = padding
            
            # Apply symmetric padding - same amount before and after
            pad_h = int(pad_h)
            pad_v = int(pad_v)
            paddings = [
                torch.floor(torch.tensor([pad_h, pad_h])).to(torch.int32),
                torch.floor(torch.tensor([pad_v, pad_v])).to(torch.int32),
                torch.tensor([0, 0]).to(torch.int32)  # No padding for channel dimension
            ]
            
            # Apply padding using PyTorch's pad function
            pad_values = (pad_v, pad_v, pad_h, pad_h)
            inp= inp.permute(2, 0, 1)
            inp_padded = F.pad(inp, pad_values, mode='constant', value=const_val)
            inp_padded = inp_padded.permute(1, 2, 0)
            return inp_padded, paddings
        
        # Default case - no padding applied
        else:
            return inp, [[0, 0], [0, 0], [0, 0]]
