import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Union, Any # Any is used for tuple elements to match original's broad acceptance

# Helper type alias for the padding_details part of the return type
PaddingDetails = Union[List[List[int]], List[np.ndarray]]

def calculate_padding_pytorch(
    kernel_size: Tuple[int, int],
    input_tensor: torch.Tensor,
    padding_mode: Union[str, Tuple[Any, Any]],
    strides: Tuple[int, int],
    const_val: float = 0.0
) -> Tuple[torch.Tensor, PaddingDetails]:
    """
    Calculates padding for a multi-dimensional tensor, typically for operations
    like convolution, replicating the logic of the original NumPy-based function.

    The padding configuration is primarily defined for three dimensions. If input_tensor.ndim < 2,
    an IndexError will occur. If input_tensor.ndim == 2 for 'same' or tuple padding_mode,
    a ValueError will be raised, consistent with the original np.pad behavior when
    a 3-dimension padding spec is applied to a 2D array.

    Args:
        kernel_size: A tuple of two integers `(kernel_h, kernel_w)`
                     representing the height and width of the kernel.
        input_tensor: The input PyTorch tensor. Padding is calculated primarily based
                      on its first two dimensions (assumed to be height and width).
        padding_mode: The padding mode. Can be:
            - 'valid': No padding is applied.
            - 'same': TensorFlow-style 'SAME' padding. Calculates padding
                      such that the output of a convolution (with the given
                      kernel_size and strides) would have spatial dimensions
                      approximately the same as the input_tensor. Padding is
                      applied to the first two dimensions, and zero padding
                      to subsequent dimensions if input_tensor.ndim >= 3,
                      or an error is raised if input_tensor.ndim < 3 (IndexError for <2D,
                      ValueError for 2D).
            - Tuple[Any, Any]: Explicit padding amounts for the first two
                dimensions, e.g., `(pad_val_dim0, pad_val_dim1)`.
                Each `pad_val_dimX` specifies the padding to be applied
                symmetrically (both before and after) to dimension X of input_tensor.
                These values are floored before being applied. For example,
                `(2, 3)` implies 2 units of padding before and 2 after
                dimension 0, and 3 units before and 3 after dimension 1.
                Dimension 2 (the third dimension, if present) receives zero padding.
                Subsequent dimensions also receive zero padding.
                This mode is activated if `padding_mode` is a tuple and
                not equal to `(None, None)`. If tuple elements are not
                numerical or the tuple length is insufficient, errors will occur.
                Raises ValueError if input_tensor.ndim == 2, or IndexError if <2D.
        strides: A tuple of two integers `(stride_h, stride_w)` representing
                 the strides for height and width.
        const_val: The constant value to use for padding. Defaults to 0.0.

    Returns:
        A tuple `(padded_tensor, padding_details)`:
        - `padded_tensor`: The input tensor, possibly padded.
        - `padding_details`: Information about the padding applied. The structure
          and type of this element strictly match the original function:
            - For 'valid' or fallback modes:
              `[[0,0],[0,0],[0,0]]` (a list of lists of Python integers).
            - For 'same' or tuple modes: A list of three 1D NumPy arrays,
              `[pad_dim0_arr, pad_dim1_arr, pad_dim2_arr]`. Each array
              is of the form `np.array([pad_before, pad_after], dtype=np.int32)`.
              These correspond to padding for dimensions 0, 1, and 2 of
              the input_tensor, respectively.
    """
    # Default padding details for 'valid' or fallback cases.
    default_padding_details_valid_fallback: List[List[int]] = [[0, 0], [0, 0], [0, 0]]

    if padding_mode == 'valid':
        return input_tensor, default_padding_details_valid_fallback

    elif padding_mode == 'same':
        # IndexError if input_tensor.ndim < 2, consistent with original
        input_h = input_tensor.shape[0]
        input_w = input_tensor.shape[1]

        kernel_h, kernel_w = kernel_size[0], kernel_size[1]
        stride_h, stride_w = strides[0], strides[1]
        
        # Calculate total padding for height (dimension 0)
        h_rem = input_h % stride_h
        if h_rem == 0:
            pad_h_total = max(0, kernel_h - stride_h)
        else:
            pad_h_total = max(0, kernel_h - h_rem)

        # Calculate total padding for width (dimension 1)
        w_rem = input_w % stride_w
        if w_rem == 0:
            pad_w_total = max(0, kernel_w - stride_w)
        else:
            pad_w_total = max(0, kernel_w - w_rem)

        # Distribute padding: [floor(total/2.0), floor((total+1)/2.0)]
        # Convert to int32 tensor
        pad_dim0_arr_torch = torch.floor(torch.tensor([pad_h_total / 2.0, (pad_h_total + 1) / 2.0])).to(torch.int32)
        pad_dim1_arr_torch = torch.floor(torch.tensor([pad_w_total / 2.0, (pad_w_total + 1) / 2.0])).to(torch.int32)
        pad_dim2_arr_torch = torch.tensor([0, 0], dtype=torch.int32) # Dim 2 is zero-padded

    elif isinstance(padding_mode, tuple) and padding_mode != (None, None):

        pad_val_dim0 = padding_mode[0]
        pad_val_dim1 = padding_mode[1]

        pad_dim0_arr_torch = torch.floor(torch.tensor([pad_val_dim0, pad_val_dim0])).to(torch.int32)
        pad_dim1_arr_torch = torch.floor(torch.tensor([pad_val_dim1, pad_val_dim1])).to(torch.int32)
        pad_dim2_arr_torch = torch.tensor([0, 0], dtype=torch.int32) # Dim 2 is zero-padded
    
    else: # Fallback for unrecognized string, (None, None), or other types
        return input_tensor, default_padding_details_valid_fallback

    if input_tensor.ndim == 2:
        # Original np.pad would receive a 3-element pad_width spec for a 2D array, causing ValueError.
        # We replicate this error explicitly.
        raise ValueError(
            f"For 'padding_mode' of '{padding_mode}', input_tensor.ndim must be at least 3. "
            f"Received input_tensor.ndim = 2. This mirrors np.pad's behavior "
            f"when applying a 3-dimensional padding specification to a 2D array."
        )
    
    pad_config_for_f_pad_pairs = []
    if input_tensor.ndim >= 1:
        pad_config_for_f_pad_pairs.append(pad_dim0_arr_torch)
    if input_tensor.ndim >= 2:
        pad_config_for_f_pad_pairs.append(pad_dim1_arr_torch)
    if input_tensor.ndim >= 3:
        pad_config_for_f_pad_pairs.append(pad_dim2_arr_torch)
    
    # Add zero padding for any additional dimensions beyond the first 3
    if input_tensor.ndim > 3:
        for _ in range(input_tensor.ndim - 3):
            pad_config_for_f_pad_pairs.append(torch.tensor([0, 0], dtype=torch.int32))

    # Convert to the flat list format required by F.pad, reversing the order of dimensions
    f_pad_input_tuple = []
    for i in range(len(pad_config_for_f_pad_pairs) - 1, -1, -1):
        pad_pair = pad_config_for_f_pad_pairs[i]
        f_pad_input_tuple.extend([pad_pair[0].item(), pad_pair[1].item()])
    
    padded_tensor = F.pad(input_tensor, tuple(f_pad_input_tuple), mode='constant', value=const_val)

    # Returned padding_details should be the 3-element list of np.ndarray, as per original.
    returned_padding_details: List[np.ndarray] = [
        pad_dim0_arr_torch,
        pad_dim1_arr_torch,
        pad_dim2_arr_torch
    ]
    return padded_tensor, returned_padding_details

# calculate_padding = torch.compile(calculate_padding_pytorch)
calculate_padding = calculate_padding_pytorch
