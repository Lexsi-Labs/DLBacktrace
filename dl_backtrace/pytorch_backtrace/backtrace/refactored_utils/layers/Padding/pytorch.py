import torch
import torch.nn.functional as F
from typing import Tuple, List, Union, Any 

TorchPaddingDetails = List[torch.Tensor]
PaddingModeType = Union[str, Tuple[Any, Any]] 

@torch.jit.script
def calculate_padding(
    kernel_size: Tuple[int, int],
    input_tensor: torch.Tensor,
    padding_mode: PaddingModeType, 
    strides: Tuple[int, int],
    const_val: float = 0.0
) -> Tuple[torch.Tensor, TorchPaddingDetails]:
    """
    Calculates padding for a multi-dimensional tensor using PyTorch, replicating
    the logic of the original NumPy-based function.

    Args:
        kernel_size: Tuple `(kernel_h, kernel_w)`.
        input_tensor: The input PyTorch tensor.
        padding_mode: 'valid', 'same', or Tuple[Any, Any] for explicit padding.
                      Tuple `(None, None)` or non-numeric tuple elements lead to fallback.
        strides: Tuple `(stride_h, stride_w)`.
        const_val: Value for constant padding.

    Returns:
        Tuple `(padded_tensor, padding_details)`.
        `padding_details` is `List[List[int]]` or `List[torch.Tensor]`.
    """
    input_h = input_tensor.shape[0]
    input_w = input_tensor.shape[1] 

    zero = torch.tensor([0.0, 0.0], dtype=torch.float32, device=input_tensor.device)
    default_padding_details_valid_fallback: List[torch.Tensor] = [zero, zero, zero]

    # padding_details_tensors_list: List[torch.Tensor] = [] 
    padding_details_tensors_list: List[torch.Tensor] = [] 

    if isinstance(padding_mode, str):
        if padding_mode == 'valid':
            ret_details: TorchPaddingDetails = default_padding_details_valid_fallback
            return input_tensor, ret_details
        elif padding_mode == 'same':
            kernel_h, kernel_w = kernel_size[0], kernel_size[1]
            stride_h, stride_w = strides[0], strides[1]

            h_rem = input_h % stride_h
            pad_h_total: int = 0 
            if h_rem == 0:
                pad_h_total = max(0, kernel_h - stride_h)
            else:
                pad_h_total = max(0, kernel_h - h_rem)

            w_rem = input_w % stride_w
            pad_w_total: int = 0 
            if w_rem == 0:
                pad_w_total = max(0, kernel_w - stride_w)
            else:
                pad_w_total = max(0, kernel_w - w_rem)
            
            pad_dim0_arr = torch.tensor(
                [float(pad_h_total) / 2.0, (float(pad_h_total) + 1.0) / 2.0],
                dtype=torch.float32, device=input_tensor.device
            ).floor()
            pad_dim1_arr = torch.tensor(
                [float(pad_w_total) / 2.0, (float(pad_w_total) + 1.0) / 2.0],
                dtype=torch.float32, device=input_tensor.device
            ).floor()
            pad_dim2_arr = torch.tensor([0.0, 0.0], dtype=torch.float32, device=input_tensor.device)
            
            padding_details_tensors_list = [pad_dim0_arr, pad_dim1_arr, pad_dim2_arr]
        else: 
            ret_details: TorchPaddingDetails = default_padding_details_valid_fallback
            return input_tensor, ret_details

    elif isinstance(padding_mode, tuple):
        if len(padding_mode) == 2:
            elem0 = padding_mode[0]
            elem1 = padding_mode[1]

            is_none_tuple = (elem0 is None) and (elem1 is None)
            if is_none_tuple:
                ret_details: TorchPaddingDetails = default_padding_details_valid_fallback
                return input_tensor, ret_details

            temp_val0: float = 0.0
            elem0_is_valid_numeric = False
            if isinstance(elem0, int):
                temp_val0 = float(elem0)
                elem0_is_valid_numeric = True
            elif isinstance(elem0, float):
                temp_val0 = elem0
                elem0_is_valid_numeric = True

            temp_val1: float = 0.0
            elem1_is_valid_numeric = False
            if isinstance(elem1, int):
                temp_val1 = float(elem1)
                elem1_is_valid_numeric = True
            elif isinstance(elem1, float):
                temp_val1 = elem1
                elem1_is_valid_numeric = True
            
            if elem0_is_valid_numeric and elem1_is_valid_numeric:
                pad_dim0_arr = torch.tensor(
                    [temp_val0, temp_val0], 
                    dtype=torch.float32, device=input_tensor.device
                ).floor()
                pad_dim1_arr = torch.tensor(
                    [temp_val1, temp_val1],
                    dtype=torch.float32, device=input_tensor.device
                ).floor()
                pad_dim2_arr = torch.tensor([0.0, 0.0], dtype=torch.float32, device=input_tensor.device)
                
                padding_details_tensors_list = [pad_dim0_arr, pad_dim1_arr, pad_dim2_arr]
            else:
                ret_details: TorchPaddingDetails = default_padding_details_valid_fallback
                return input_tensor, ret_details
        else: 
            ret_details: TorchPaddingDetails = default_padding_details_valid_fallback
            return input_tensor, ret_details
    else: 
        ret_details: TorchPaddingDetails = default_padding_details_valid_fallback
        return input_tensor, ret_details

    if not padding_details_tensors_list:
        ret_details: TorchPaddingDetails = default_padding_details_valid_fallback
        return input_tensor, ret_details

    p0_int: List[int] = padding_details_tensors_list[0].long().tolist()
    p1_int: List[int] = padding_details_tensors_list[1].long().tolist() 
    p2_int: List[int] = padding_details_tensors_list[2].long().tolist()

    ndim = input_tensor.dim()
    padded_tensor: torch.Tensor = torch.empty(0, dtype=input_tensor.dtype, device=input_tensor.device) 

    # F.pad expects padding for (last_dim, second_to_last, ..., first_dim)
    if ndim == 2: # e.g., (H, W)
        # F.pad wants (pad_W_L, pad_W_R, pad_H_L, pad_H_R)
        # Our p0 is for H (dim 0), p1 for W (dim 1)
        pad_tuple = (p1_int[0], p1_int[1],    # Dim 1 (W)
                     p0_int[0], p0_int[1])    # Dim 0 (H)
        padded_tensor = F.pad(input_tensor, pad_tuple, mode='constant', value=const_val)
    elif ndim == 3: # e.g., (H, W, C) or (D, H, W)
        # F.pad wants (pad_C_L, pad_C_R, pad_W_L, pad_W_R, pad_H_L, pad_H_R)
        # Our p0 is H (dim 0), p1 is W (dim 1), p2 is C/D (dim 2)
        pad_tuple = (p2_int[0], p2_int[1],    # Dim 2 (conceptual C/D)
                     p1_int[0], p1_int[1],    # Dim 1 (W)
                     p0_int[0], p0_int[1])    # Dim 0 (H)
        padded_tensor = F.pad(input_tensor, pad_tuple, mode='constant', value=const_val)
    elif ndim == 4: # e.g., (N, C, H, W) -> F.pad wants (W, H, C, N)
                    # or (D, H, W, C) -> F.pad wants (C, W, H, D)
        # Our p0 for dim 0, p1 for dim 1, p2 for dim 2. Dim 3 is 0-padded.
        # F.pad: (pad_dim3_L/R, pad_dim2_L/R, pad_dim1_L/R, pad_dim0_L/R)
        pad_tuple = (0, 0,                   # Dim 3 (0-padded)
                     p2_int[0], p2_int[1],   # Dim 2
                     p1_int[0], p1_int[1],   # Dim 1
                     p0_int[0], p0_int[1])   # Dim 0
        padded_tensor = F.pad(input_tensor, pad_tuple, mode='constant', value=const_val)
    elif ndim == 5: # e.g. (N, C, D, H, W) -> F.pad wants (W,H,D,C,N)
        # Our p0 for dim 0, p1 for dim 1, p2 for dim 2. Dims 3,4 are 0-padded.
        # F.pad: (pad_dim4, pad_dim3, pad_dim2, pad_dim1, pad_dim0)
        pad_tuple = (0, 0,                   # Dim 4 (0-padded)
                     0, 0,                   # Dim 3 (0-padded)
                     p2_int[0], p2_int[1],   # Dim 2
                     p1_int[0], p1_int[1],   # Dim 1
                     p0_int[0], p0_int[1])   # Dim 0
        padded_tensor = F.pad(input_tensor, pad_tuple, mode='constant', value=const_val)
    elif ndim < 2:
        raise ValueError(f"Input tensor must have at least 2 dimensions, got {ndim}")
    
    final_ret_details: TorchPaddingDetails = padding_details_tensors_list
    return padded_tensor, final_ret_details
