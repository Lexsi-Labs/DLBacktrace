import numpy as np
from typing import Tuple, List, Union, Any # Any is used for tuple elements to match original's broad acceptance

# Helper type alias for the padding_details part of the return type
PaddingDetails = Union[List[List[int]], List[np.ndarray]]

def calculate_padding(
    kernel_size: Tuple[int, ...],
    input_array: np.ndarray,
    padding_mode: Union[str, Tuple[Any, Any]],
    strides: Tuple[int, ...],
    const_val: float = 0.0
) -> Tuple[np.ndarray, PaddingDetails]:
    """
    Calculates padding for a multi-dimensional array, typically for operations
    like convolution, replicating the logic of the original function.

    The padding configuration is defined for three dimensions. If input_array.ndim < 3,
    np.pad may raise an error in 'same' or tuple padding_mode, which is
    consistent with the original behavior.

    Args:
        kernel_size: A list of two integers `[kernel_h, kernel_w]`
                     representing the height and width of the kernel.
        input_array: The input NumPy array. Padding is calculated primarily based
                     on its first two dimensions (assumed to be height and width).
        padding_mode: The padding mode. Can be:
            - 'valid': No padding is applied.
            - 'same': TensorFlow-style 'SAME' padding. Calculates padding
                      such that the output of a convolution (with the given
                      kernel_size and strides) would have spatial dimensions
                      approximately the same as the input_array. Padding is
                      applied to the first two dimensions, and zero padding
                      to the third dimension of the input_array.
            - Tuple[Any, Any]: Explicit padding amounts for the first two
                dimensions, e.g., `(pad_val_dim0, pad_val_dim1)`.
                Each `pad_val_dimX` specifies the padding to be applied
                symmetrically (both before and after) to dimension X of input_array.
                These values are floored before being applied. For example,
                `(2, 3)` implies 2 units of padding before and 2 after
                dimension 0, and 3 units before and 3 after dimension 1.
                Dimension 2 (the third dimension) receives zero padding.
                This mode is activated if `padding_mode` is a tuple and
                not equal to `(None, None)`. If tuple elements are not
                numerical or the tuple length is insufficient, `np.floor`
                or indexing will raise an error, matching original behavior.
        strides: A list of two integers `[stride_h, stride_w]` representing
                 the strides for height and width.
        const_val: The constant value to use for padding when `np.pad` is called.
                   Defaults to 0.0.

    Returns:
        A tuple `(padded_array, padding_details)`:
        - `padded_array`: The input array, possibly padded.
        - `padding_details`: Information about the padding applied. The structure
          and type of this element strictly match the original function:
            - For 'valid' or fallback modes (e.g., unrecognized string for
              padding_mode, or if padding_mode is (None,None)):
              `[[0,0],[0,0],[0,0]]` (a list of lists of Python integers).
            - For 'same' or tuple modes: A list of three 1D NumPy arrays,
              `[pad_dim0_arr, pad_dim1_arr, pad_dim2_arr]`. Each array
              is of the form `np.array([pad_before, pad_after], dtype=np.int32)`.
              These correspond to padding for dimensions 0, 1, and 2 of
              the input_array, respectively.
    """
    input_h, input_w = input_array.shape[0], input_array.shape[1]

    # Default padding details for 'valid' or fallback cases.
    # This specific type (List[List[int]]) matches the original function's output for these cases.
    default_padding_details_valid_fallback: List[List[int]] = [[0, 0], [0, 0], [0, 0]]

    if padding_mode == 'valid':
        return input_array, default_padding_details_valid_fallback

    elif padding_mode == 'same':
        kernel_h, kernel_w = kernel_size[0], kernel_size[1]
        stride_h, stride_w = strides[0], strides[1]

        # Calculate total padding for height (dimension 0 of input_array)
        # This logic is directly preserved from the original function to ensure identical behavior.
        h_rem = input_h % stride_h
        if h_rem == 0:
            # If input height is perfectly divisible by stride.
            pad_h_total = np.maximum(0, kernel_h - stride_h)
        else:
            # If there's a remainder from input height / stride.
            pad_h_total = np.maximum(0, kernel_h - h_rem)

        # Calculate total padding for width (dimension 1 of input_array)
        # This logic is also directly preserved.
        w_rem = input_w % stride_w
        if w_rem == 0:
            pad_w_total = np.maximum(0, kernel_w - stride_w)
        else:
            pad_w_total = np.maximum(0, kernel_w - w_rem)

        # Distribute padding using the original method: np.floor([total/2.0, (total+1)/2.0])
        # This ensures pad_before <= pad_after, e.g.:
        # total_pad = 3 -> [floor(1.5), floor(2.0)] -> np.array([1, 2])
        # total_pad = 4 -> [floor(2.0), floor(2.5)] -> np.array([2, 2])
        # The result is converted to np.int32, matching the original.
        pad_dim0_arr = np.floor([pad_h_total / 2.0, (pad_h_total + 1) / 2.0]).astype(np.int32)
        pad_dim1_arr = np.floor([pad_w_total / 2.0, (pad_w_total + 1) / 2.0]).astype(np.int32)
        
        # Dimension 2 (e.g., channels, if input_array is HWC) receives zero padding.
        # Original used np.zeros((2)).astype("int32").
        pad_dim2_arr = np.array([0, 0], dtype=np.int32)

        # The padding configuration for np.pad, as a list of np.ndarray.
        # This matches the original function's output type for 'same' mode.
        padding_config_for_np_pad: List[np.ndarray] = [pad_dim0_arr, pad_dim1_arr, pad_dim2_arr]
        
        if input_array.ndim > 3:
            num_extra_dims = input_array.ndim - 3
            padding_config_for_np_pad.extend([np.array([0, 0], dtype=np.int32)] * num_extra_dims)
            
        elif input_array.ndim < 3 and input_array.ndim > 0:
            pass
        
        padded_array = np.pad(input_array, padding_config_for_np_pad, mode='constant', constant_values=const_val)
        
        # The returned padding_details should always be the 3-element list, as per original.
        # So we return the original 3-element version, not the extended one for N-D.
        returned_padding_details: List[np.ndarray] = [pad_dim0_arr, pad_dim1_arr, pad_dim2_arr]
        return padded_array, returned_padding_details

    # This 'else' block handles cases where padding_mode is not 'valid' or 'same'.
    # It then checks if padding_mode is a tuple (and not (None, None)).
    else:
        # Check for tuple padding_mode, excluding (None, None) specifically, as in original.
        if isinstance(padding_mode, tuple) and padding_mode != (None, None):
            # Original code directly accesses padding_mode[0] and padding_mode[1].
            # This assumes padding_mode is a sequence of at least two elements.
            # If not, an IndexError would occur (preserved behavior).
            # If elements are not numeric, np.floor will raise TypeError (preserved behavior).
            pad_val_dim0 = padding_mode[0]
            pad_val_dim1 = padding_mode[1]

            # Original uses np.floor([val, val]).astype("int32").
            # This means `pad_val_dim0` is applied symmetrically (before and after) to dim 0.
            # `np.floor` handles potential float inputs by truncating towards negative infinity.
            # The result is converted to np.int32.
            pad_dim0_arr = np.floor([pad_val_dim0, pad_val_dim0]).astype(np.int32)
            pad_dim1_arr = np.floor([pad_val_dim1, pad_val_dim1]).astype(np.int32)
            
            # Dimension 2 receives zero padding, as in original.
            pad_dim2_arr = np.array([0, 0], dtype=np.int32)

            # The padding configuration for np.pad, as a list of np.ndarray.
            # This matches the original function's output type for tuple mode.
            padding_config_for_np_pad: List[np.ndarray] = [pad_dim0_arr, pad_dim1_arr, pad_dim2_arr]

            # Similar N-D handling as in 'same' mode for np.pad compatibility
            if input_array.ndim > 3:
                num_extra_dims = input_array.ndim - 3
                padding_config_for_np_pad.extend([np.array([0, 0], dtype=np.int32)] * num_extra_dims)
            elif input_array.ndim < 3 and input_array.ndim > 0:
                pass # Let np.pad handle it, replicating original's potential error

            padded_array = np.pad(input_array, padding_config_for_np_pad, mode='constant', constant_values=const_val)
            
            # Return the 3-element padding details, consistent with original.
            returned_padding_details: List[np.ndarray] = [pad_dim0_arr, pad_dim1_arr, pad_dim2_arr]
            return padded_array, returned_padding_details
        else:
            # Fallback: if padding_mode is an unrecognized string, or (None, None),
            # or any other type not handled by the 'if isinstance(padding_mode, tuple)...' condition.
            # Behaves like 'valid' mode, returning List[List[int]] for padding_details.
            return input_array, default_padding_details_valid_fallback
