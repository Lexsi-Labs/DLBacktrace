import numpy as np
from typing import Dict, Any, Union, Tuple
from ..ConvUnit2D.refactored_version import calculate_wt_conv_unit
from ..Padding.refactored import calculate_padding

def calculate_wt_conv(
    grad_output_scales: np.ndarray,
    input_activations: np.ndarray,
    kernel_weights_orig_shape: np.ndarray,
    bias: np.ndarray,
    padding_mode: Union[str, Tuple[Any, Any]],
    strides: Tuple[int, int],
    activation_params: Dict[str, Any]
) -> np.ndarray:
    """
    Calculates the gradient with respect to the input activations of a convolutional
    layer, often part of the backpropagation process. This function iterates
    over spatial dimensions of the output gradient, applying a weighted
    convolution-like unit operation and accumulating results.

    The input arrays `grad_output_scales`, `input_activations`, and
    `kernel_weights_orig_shape` are transposed internally as per the original
    function's logic. Their expected input shapes should account for this.

    Args:
        grad_output_scales (np.ndarray): Gradient scales corresponding to the
            output features of a convolution. Expected shape before transpose:
            (F, O_w, O_h), where F is number of output features/filters,
            O_w is output width, O_h is output height. After transpose, it becomes
            (O_h, O_w, F).
        input_activations (np.ndarray): The input activations to the convolutional
            layer. Expected shape before transpose: (C_in, W_in_orig, H_in_orig),
            where C_in is input channels. After transpose, it becomes
            (H_in_orig, W_in_orig, C_in). The first two dimensions of this
            transposed array (H_in_orig, W_in_orig) are used as height and width
            for padding calculations.
        kernel_weights_orig_shape (np.ndarray): The original kernel weights.
            Expected shape before transpose, (F, C_in, K_w, K_h), where K_w, K_h
            are kernel width and height. After transpose, it becomes
            (K_h, K_w, C_in, F), which is the shape expected by
            `calculate_wt_conv_unit`.
        bias (np.ndarray): Bias terms for the convolutional layer, shape (F,).
        padding_mode (Union[str, Tuple[Any, Any]]): Padding mode for the
            operation. See `calculate_padding` function for details.
        strides (Tuple[int, int]): Strides for the convolution, (stride_h, stride_w).
        activation_params (Dict[str, Any]): Parameters for the activation
            function used in `calculate_wt_conv_unit`. See its docstring.

    Returns:
        np.ndarray: The calculated gradient with respect to the input activations.
            Its shape will be (H_in_orig, W_in_orig, C_in), matching the spatial
            dimensions and channel layout of `input_activations` after its
            internal transpose.
    """
    # Transpose inputs as per original function's logic
    # These transposed versions are used throughout the function
    grad_output_scales_T: np.ndarray = grad_output_scales.T
    input_activations_T: np.ndarray = input_activations.T
    kernel_weights_T: np.ndarray = kernel_weights_orig_shape.T

    # Decompose shapes for clarity (using dimensions of transposed arrays)
    # output_height_loop, output_width_loop, num_filters = grad_output_scales_T.shape
    # input_h_for_padding, input_w_for_padding, _ = input_activations_T.shape # C_in is last
    kernel_h, kernel_w, _, _ = kernel_weights_T.shape # C_in, F are 3rd, 4th

    # 1. Calculate padding for the (transposed) input activations
    # `kernel_weights_T.shape[:2]` provides (kernel_h, kernel_w) for padding calculation.
    # `input_activations_T` is the array to be padded.
    input_padded, padding_config = calculate_padding(
        kernel_size=kernel_weights_T.shape[:2], # Original used w.shape (transposed kernel)
        input_array=input_activations_T,
        padding_mode=padding_mode,
        strides=strides
    )

    # 2. Initialize the output array for accumulating gradients
    # Dtype will be inherited from input_padded. If input_activations_T is float32
    # and padding is e.g. 'same', input_padded (and thus grad_input_padded)
    # will be float32. If input_activations_T is int and const_val for padding is float (0.0 default),
    # then input_padded becomes float64. This matches original behavior.
    grad_input_padded: np.ndarray = np.zeros_like(input_padded)

    stride_h, stride_w = strides

    # 3. Iterate over "output" spatial dimensions (from grad_output_scales_T)
    # This loop structure is preserved from the original function.
    for out_h_idx in range(grad_output_scales_T.shape[0]): # Loop O_h times
        for out_w_idx in range(grad_output_scales_T.shape[1]): # Loop O_w times
            # 3.a. Define the slice region in the padded input
            start_h = out_h_idx * stride_h
            end_h = start_h + kernel_h
            start_w = out_w_idx * stride_w
            end_w = start_w + kernel_w

            # 3.b. Extract the current input patch
            # Slicing creates a view, which is efficient.
            # The third dimension (channels) is fully included.
            current_input_patch: np.ndarray = input_padded[start_h:end_h, start_w:end_w, :]

            # 3.c. Get the specific gradient scale for this output location
            current_grad_scale_slice: np.ndarray = grad_output_scales_T[out_h_idx, out_w_idx, :]

            # 3.d. Calculate updates using the provided helper function
            # `kernel_weights_T` is (K_h, K_w, C_in, F)
            # `current_grad_scale_slice` is (F,)
            # `current_input_patch` is (K_h, K_w, C_in)
            # `bias` is (F,)
            # `updates` will have shape (K_h, K_w, C_in)
            updates: np.ndarray = calculate_wt_conv_unit(
                patch=current_input_patch,
                wts=current_grad_scale_slice,
                kernel_weights=kernel_weights_T,
                bias=bias,
                activation_params=activation_params
            )

            # 3.e. Accumulate the calculated updates into the corresponding region
            # of the padded input gradient map. This is an in-place addition.
            grad_input_padded[start_h:end_h, start_w:end_w, :] += updates

    # 4. Remove padding to get the final gradient w.r.t. input_activations
    # `padding_config` elements can be float (from 'same' mode in calculate_padding).
    # Slicing indices are implicitly converted to int.
    pad_h_before: float = padding_config[0][0]
    pad_w_before: float = padding_config[1][0]
    # The third dimension's padding is [0,0] as per calculate_padding logic,
    # so direct slicing `:, :` is appropriate.

    # `input_activations_T.shape[0]` is H_in_orig (height of input_activations after transpose)
    # `input_activations_T.shape[1]` is W_in_orig (width of input_activations after transpose)
    # The slicing extracts the central part corresponding to the original unpadded dimensions.
    unpadded_height_end = int(pad_h_before + input_activations_T.shape[0])
    unpadded_width_end = int(pad_w_before + input_activations_T.shape[1])

    grad_input_unpadded: np.ndarray = grad_input_padded[
        int(pad_h_before):unpadded_height_end,
        int(pad_w_before):unpadded_width_end,
        : # All channels
    ]

    return grad_input_unpadded
