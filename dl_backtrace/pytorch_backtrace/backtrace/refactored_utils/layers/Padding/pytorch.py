import torch
import torch.nn.functional as F
from typing import Tuple, List, Union, cast

# Helper type alias for the padding_details part of the return type (PyTorch version)
TorchPaddingDetails = Union[List[List[int]], List[torch.Tensor]]

# More specific type for padding_mode tuple elements, original was Tuple[Any, Any]
PaddingModeTupleElementType = Union[int, float]
PaddingModeTuple = Tuple[PaddingModeTupleElementType, PaddingModeTupleElementType]


@torch.jit.script
def calculate_padding(
    kernel_size: Tuple[int, int],
    input_tensor: torch.Tensor,
    padding_mode: Union[str, PaddingModeTuple],
    strides: Tuple[int, int],
    const_val: float = 0.0
) -> Tuple[torch.Tensor, TorchPaddingDetails]:
    """
    Calculates padding for a multi-dimensional tensor using PyTorch, replicating
    the logic of the original NumPy-based function.

    Padding calculation logic is primarily for the first two dimensions (assumed
    to be height and width). The third dimension receives zero padding as per
    the original logic. If the input_tensor has more than 3 dimensions,
    dimensions beyond the third are also zero-padded by F.pad.

    If input_tensor.ndim < 2, indexing input_tensor.shape[1] will raise an
    IndexError, consistent with original behavior.

    Args:
        kernel_size: A tuple of two integers `(kernel_h, kernel_w)`
                     representing the height and width of the kernel.
        input_tensor: The input PyTorch tensor.
        padding_mode: The padding mode. Can be:
            - 'valid': No padding is applied.
            - 'same': TensorFlow-style 'SAME' padding. Calculates padding
                      such that the output of a convolution (with the given
                      kernel_size and strides) would have spatial dimensions
                      approximately the same as the input_tensor. Padding is
                      calculated for the first two dimensions, and zero padding
                      for the third dimension.
            - Tuple[Union[int, float], Union[int, float]]: Explicit padding
                amounts for the first two dimensions, e.g., `(pad_val_dim0, pad_val_dim1)`.
                Each `pad_val_dimX` specifies symmetrical padding for dimension X.
                Values are floored. Dimension 2 (third dimension) gets zero padding.
                Activated if `padding_mode` is a tuple and not `(None, None)`.
                Note: `(None,None)` itself is not a valid `PaddingModeTuple` due to type hints.
                      If (None,None) is passed, it will fall to the 'else' fallback.
        strides: A tuple of two integers `(stride_h, stride_w)` representing
                 the strides for height and width.
        const_val: The constant value to use for padding with `F.pad`.
                   Defaults to 0.0.

    Returns:
        A tuple `(padded_tensor, padding_details)`:
        - `padded_tensor`: The input tensor, possibly padded.
        - `padding_details`: Information about the padding applied.
            - For 'valid' or fallback modes:
              `[[0,0],[0,0],[0,0]]` (List of lists of Python integers).
            - For 'same' or tuple modes: A list of three 1D PyTorch tensors,
              `[pad_dim0_arr, pad_dim1_arr, pad_dim2_arr]`. Each tensor
              is of shape `(2,)`, dtype `torch.float32`, representing
              `[pad_before, pad_after]`.
    """
    # This will raise IndexError if input_tensor.ndim < 2, matching original
    input_h = input_tensor.shape[0]
    input_w = input_tensor.shape[1] # Fails if ndim < 2

    # Default padding details for 'valid' or fallback cases.
    default_padding_details_valid_fallback: List[List[int]] = [[0, 0], [0, 0], [0, 0]]

    if padding_mode == 'valid':
        # Use cast to help JIT understand the Union type from different branches
        return input_tensor, cast(TorchPaddingDetails, default_padding_details_valid_fallback)

    elif padding_mode == 'same':
        kernel_h, kernel_w = kernel_size[0], kernel_size[1]
        stride_h, stride_w = strides[0], strides[1]

        h_rem = input_h % stride_h
        if h_rem == 0:
            pad_h_total = max(0, kernel_h - stride_h)
        else:
            pad_h_total = max(0, kernel_h - h_rem)

        w_rem = input_w % stride_w
        if w_rem == 0:
            pad_w_total = max(0, kernel_w - stride_w)
        else:
            pad_w_total = max(0, kernel_w - w_rem)
        
        # Calculations are scalar, convert to tensor with floor and specific dtype
        pad_dim0_arr = torch.tensor(
            [pad_h_total / 2.0, (pad_h_total + 1) / 2.0],
            dtype=torch.float32, device=input_tensor.device
        ).floor()
        pad_dim1_arr = torch.tensor(
            [pad_w_total / 2.0, (pad_w_total + 1) / 2.0],
            dtype=torch.float32, device=input_tensor.device
        ).floor()
        pad_dim2_arr = torch.tensor([0.0, 0.0], dtype=torch.float32, device=input_tensor.device)

        padding_details_tensors: List[torch.Tensor] = [pad_dim0_arr, pad_dim1_arr, pad_dim2_arr]

    # Handles tuple padding_mode or fallback for unrecognized strings / (None,None)
    else:
        # Check for tuple padding_mode.
        # Original condition: isinstance(padding_mode, tuple) and padding_mode != (None, None)
        # For JIT, direct comparison to (None,None) with a typed tuple might be tricky.
        # The type hint PaddingModeTuple already excludes None for elements.
        # If padding_mode is indeed (None,None) it won't match PaddingModeTuple.
        # isinstance check is robust.
        if isinstance(padding_mode, tuple):
            # Ensure it's not (None, None) if that's a possible input not caught by type hints
            # This specific check for (None,None) tuple is from the original logic
            is_none_tuple = True
            if len(padding_mode) == 2: # pyright: ignore[reportUnnecessaryComparison]
                # This check is to satisfy JIT about tuple structure before element access
                # And to be absolutely sure about the (None,None) case
                if padding_mode[0] is not None and padding_mode[1] is not None: # pyright: ignore[reportUnnecessaryComparison]
                    is_none_tuple = False
            
            if not is_none_tuple:
                # At this point, padding_mode should conform to PaddingModeTuple
                # We can cast it for type checking / JIT, though direct access is often fine.
                # However, to be safe with JIT and potential mixed types if `Any` was used:
                # We assume padding_mode[0] and padding_mode[1] are numeric (int or float)
                # as per PaddingModeTupleElementType.
                
                # pad_val_dim0 = cast(PaddingModeTupleElementType, padding_mode[0]) # JIT might not need cast here
                # pad_val_dim1 = cast(PaddingModeTupleElementType, padding_mode[1])
                pad_val_dim0 = padding_mode[0] # type: ignore # mypy might complain if padding_mode is just `object`
                pad_val_dim1 = padding_mode[1] # type: ignore

                pad_dim0_arr = torch.tensor(
                    [float(pad_val_dim0), float(pad_val_dim0)], # Ensure float for division/floor consistency
                    dtype=torch.float32, device=input_tensor.device
                ).floor()
                pad_dim1_arr = torch.tensor(
                    [float(pad_val_dim1), float(pad_val_dim1)],
                    dtype=torch.float32, device=input_tensor.device
                ).floor()
                pad_dim2_arr = torch.tensor([0.0, 0.0], dtype=torch.float32, device=input_tensor.device)

                padding_details_tensors: List[torch.Tensor] = [pad_dim0_arr, pad_dim1_arr, pad_dim2_arr]
            else: # Fallback for (None,None) or other non-conforming tuples
                return input_tensor, cast(TorchPaddingDetails, default_padding_details_valid_fallback)
        else: # Fallback for unrecognized strings or other types
            return input_tensor, cast(TorchPaddingDetails, default_padding_details_valid_fallback)

    # Common padding application for 'same' and valid tuple modes
    # F.pad expects padding amounts as integers, in reverse order of dimensions
    # (pad_last_dim_left, pad_last_dim_right, pad_prev_dim_left, pad_prev_dim_right, ...)
    
    # Convert float padding tensors to integer lists for F.pad
    # P0, P1, P2 correspond to padding for dimensions 0, 1, 2 of input_tensor
    p0_int = pad_dim0_arr.long().tolist()
    p1_int = pad_dim1_arr.long().tolist()
    p2_int = pad_dim2_arr.long().tolist()

    ndim = input_tensor.dim()
    torch_pad_tuple_flat: List[int] = []

    if ndim == 2: # Assuming (H, W)
        # Pad W (dim 1), then H (dim 0)
        torch_pad_tuple_flat = [p1_int[0], p1_int[1], p0_int[0], p0_int[1]]
    elif ndim == 3: # Assuming (H, W, D2)
        # Pad D2 (dim 2), then W (dim 1), then H (dim 0)
        torch_pad_tuple_flat = [p2_int[0], p2_int[1], p1_int[0], p1_int[1], p0_int[0], p0_int[1]]
    elif ndim > 3:
        # Pad first 3 dimensions as specified (0, 1, 2), zero pad higher dimensions
        # F.pad order: (pad_dimN-1_L, pad_dimN-1_R, ..., pad_dim0_L, pad_dim0_R)
        num_higher_dims = ndim - 3
        for _ in range(num_higher_dims):
            torch_pad_tuple_flat.extend([0, 0]) # Zero padding for dimensions 3, 4, ...
        torch_pad_tuple_flat.extend([p2_int[0], p2_int[1]]) # Dim 2
        torch_pad_tuple_flat.extend([p1_int[0], p1_int[1]]) # Dim 1
        torch_pad_tuple_flat.extend([p0_int[0], p0_int[1]]) # Dim 0
    elif ndim < 2:
        # This case should have been caught by input_h/input_w access earlier if ndim < 1 or < 2.
        # If ndim is 1 (e.g. (H,)), original code fails at input_w access.
        # If somehow reached here with ndim = 1, one might pad only dim 0:
        # torch_pad_tuple_flat = [p0_int[0], p0_int[1]]
        # However, current logic relies on ndim >= 2 due to input_w.
        # For robustness, if code gets here implying ndim==0 or 1 without error (unlikely),
        # we can return unpadded tensor as a safe default, though it signals an issue.
        # Given the problem, it's better to rely on IndexError from shape access for ndim < 2.
        # If ndim == 0, input_tensor.shape[0] would error.
        # So this 'elif ndim < 2' block is practically unreachable if earlier checks are active.
        # It's included for logical completeness of ndim handling here if we were to bypass shape access.
         pass # Should not be reached due to shape access error for ndim < 2.

    if not torch_pad_tuple_flat: # If ndim < 2 (e.g. 0 or 1, and somehow no error from shape access) or tuple was empty.
        # This path indicates an issue or an unhandled low-dimension case for padding.
        # The original code would error on shape access for ndim < 2.
        # If ndim is 0 or 1, padding is ill-defined by the original's H,W logic.
        # Returning unpadded tensor and default details for such edge cases not covered by original errors.
        # However, the IndexError on shape access is the expected behavior for ndim < 2.
        # This 'if not torch_pad_tuple_flat' mostly acts as a safeguard if input_tensor.dim() is unexpectedly low.
        # For ndim = 0 or 1, original logic fails before F.pad call.
        # For this refactoring, we assume ndim >= 2 if this point is reached.
        # If torch_pad_tuple_flat is empty, it means ndim was < 2 and it was not caught by shape access (which is unlikely).
        # In a robust scenario, one might raise an error here explicitly for ndim < 2.
        # Let's assume if we are here, ndim >= 2.
        if ndim < 2: # Should be caught by shape access. If not, problem.
             raise ValueError(f"Input tensor must have at least 2 dimensions for this padding logic, got {ndim}")


    padded_tensor = F.pad(input_tensor, tuple(torch_pad_tuple_flat), mode='constant', value=const_val)
    return padded_tensor, cast(TorchPaddingDetails, padding_details_tensors)
