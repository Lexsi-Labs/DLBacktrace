import numpy as np
from typing import Dict, Any, Callable, Union, Tuple

# Type alias for the activation function callable
ActivationFunctionType = Callable[[np.ndarray], np.ndarray]

def calculate_wt_conv_unit(
    patch: np.ndarray,
    wts: Union[float, np.ndarray],
    kernel_weights: np.ndarray,
    bias: np.ndarray,
    activation_params: Dict[str, Any]
) -> np.ndarray:
    """
    Calculates weighted convolutional unit contributions based on input patch,
    kernel weights, biases, and activation parameters.

    The function processes inputs through a series of steps:
    1. Separates positive and absolute negative components of the bias.
    2. Performs an element-wise multiplication of kernel weights with the
       (broadcasted) input patch to get 'convolution output'.
    3. Separates positive and negative parts of this convolution output.
    4. Sums these parts along specified axes (H, W, C_in dimensions).
    5. Applies activation logic, potentially adjusting saturation masks based
       on 'mono' or 'non_mono' activation types and their ranges. This step
       includes a numerical comparison using a fixed epsilon (1e-5) for
       'non_mono' type.
    6. Calculates aggregated weights using a denominator derived from sums of
       convolution outputs and biases. Division by zero is handled by NumPy's
       default behavior (producing inf/nan), replicating the original function's
       numerical characteristics.
    7. Combines the processed convolution output parts with these aggregated
       weights.
    8. Sums the result along the last axis (feature dimension) to produce the final output.

    Args:
        patch (np.ndarray): Input data, expected to be a 3D array
            (e.g., dimensions H, W, C_in).
        wts (Union[float, np.ndarray]): Scaling weights. Can be a scalar or a 1D
            NumPy array (e.g., dimension F corresponding to output features).
        kernel_weights (np.ndarray): Kernel weights, expected to be a 4D array
            (e.g., dimensions H, W, C_in, F), where the first three dimensions
            match `patch` and the last dimension F corresponds to output features.
        bias (np.ndarray): Bias terms, expected to be a 1D NumPy array
            (e.g., dimension F).
        activation_params (Dict[str, Any]): Dictionary specifying activation behavior.
            It must contain:
            - "type" (str): Activation type, either 'mono' or 'non_mono'.
            - "range" (Dict[str, Union[float, None]]): Specifies saturation ranges.
                - "l" (Optional[float]): Lower bound for saturation.
                - "u" (Optional[float]): Upper bound for saturation.
            - "func" (ActivationFunctionType): The activation function
              (e.g., a Python callable like np.sigmoid), used if "type" is 'non_mono'.

    Returns:
        np.ndarray: The calculated weight matrix, summed over the feature
            dimension ('F'), resulting in a 3D array with dimensions
            matching the first three of `kernel_weights` (e.g., H, W, C_in).
    """

    # 1. Bias processing
    # Original logic separated positive and negative parts of bias.
    # np.maximum(0, bias) isolates positive parts.
    # np.maximum(0, -bias) isolates absolute values of negative parts.
    # These operations are vectorized.
    positive_bias: np.ndarray = np.maximum(0, bias)  # Shape (F,)
    abs_negative_bias: np.ndarray = np.maximum(0, -bias)  # Shape (F,)

    # 2. Convolution-like operation (element-wise multiplication using broadcasting)
    # Original: conv_out = np.einsum("ijkl,ijk->ijkl", kernel_weights, patch)
    # This is an element-wise product where `patch` is broadcast across the last dimension of `kernel_weights`.
    # kernel_weights shape: (H, W, C_in, F), patch shape: (H, W, C_in)
    # patch[..., np.newaxis] reshapes patch to (H, W, C_in, 1) for broadcasting.
    conv_out: np.ndarray = kernel_weights * patch[..., np.newaxis]  # Shape (H,W,C_in,F)

    # 3. Separate positive and negative parts of conv_out (vectorized)
    # np.maximum(0, conv_out) gets parts >= 0.
    # np.minimum(0, conv_out) gets parts <= 0 (retaining their negative sign).
    positive_conv_out_parts: np.ndarray = np.maximum(0, conv_out) # Shape (H,W,C_in,F)
    negative_conv_out_parts: np.ndarray = np.minimum(0, conv_out) # Shape (H,W,C_in,F), contains negative values or zero

    # 4. Sum positive and absolute negative parts along (H, W, C_in) axes
    # Original used np.einsum("ijkl->l", ...). np.sum over specified axes is equivalent and often clearer.
    sum_axes: Tuple[int, int, int] = (0, 1, 2) # Sum over H, W, C_in dimensions
    sum_positive_conv_out: np.ndarray = np.sum(positive_conv_out_parts, axis=sum_axes)  # Shape (F,)
    # `negative_conv_out_parts` are <= 0. Summing them yields a non-positive result.
    # Multiplying by -1 gives the sum of their absolute values, matching original `n_sum`.
    sum_abs_negative_conv_out: np.ndarray = -np.sum(negative_conv_out_parts, axis=sum_axes)  # Shape (F,)

    # 5. Calculate total sum of absolute activations (t_sum in original)
    sum_abs_total_conv_out: np.ndarray = sum_positive_conv_out + sum_abs_negative_conv_out  # Shape (F,)

    # 6. Initialize saturation masks based on summed parts
    # These boolean masks (shape F,) indicate if there's any positive/negative contribution initially.
    positive_saturation_mask: np.ndarray = sum_positive_conv_out > 0
    negative_saturation_mask: np.ndarray = sum_abs_negative_conv_out > 0

    # 7. Activation logic application
    # This section modifies `positive_saturation_mask` and `negative_saturation_mask`
    # based on `activation_params`, precisely replicating the original's conditional logic.
    act_type: str = activation_params["type"]
    act_range_l: Union[float, None] = activation_params["range"]["l"]
    act_range_u: Union[float, None] = activation_params["range"]["u"]

    if act_type == 'mono':
        if act_range_l:
            # `positive_saturation_mask` is entirely replaced, as per original logic.
            positive_saturation_mask = sum_abs_total_conv_out > act_range_l
        if act_range_u:
            # `negative_saturation_mask` is entirely replaced, as per original logic.
            negative_saturation_mask = sum_abs_total_conv_out < act_range_u
    elif act_type == 'non_mono':
        activation_func: ActivationFunctionType = activation_params["func"]
        
        activated_total_sum: np.ndarray = activation_func(sum_abs_total_conv_out)
        activated_positive_sum_plus_bias: np.ndarray = activation_func(sum_positive_conv_out + positive_bias)
        # Original multiplies (sum_abs_negative_conv_out + abs_negative_bias) by -1 before activation.
        activated_neg_sum_plus_bias: np.ndarray = activation_func(
            -1 * (sum_abs_negative_conv_out + abs_negative_bias)
        )

        if act_range_l:
            # Mask is updated with a logical AND (&=), equivalent to original `mask = mask * condition`.
            saturation_lower_bound_mask: np.ndarray = sum_abs_total_conv_out > act_range_l
            positive_saturation_mask &= saturation_lower_bound_mask
        if act_range_u:
            # Mask is updated with a logical AND.
            saturation_upper_bound_mask: np.ndarray = sum_abs_total_conv_out < act_range_u
            negative_saturation_mask &= saturation_upper_bound_mask

        # Epsilon comparison (1e-5) for floating-point numbers is critical and preserved.
        non_mono_neg_saturation_check: np.ndarray = \
            np.abs(activated_total_sum - activated_positive_sum_plus_bias) > 1e-5 # Preserving 1e-5
        negative_saturation_mask &= non_mono_neg_saturation_check

        non_mono_pos_saturation_check: np.ndarray = \
            np.abs(activated_total_sum - activated_neg_sum_plus_bias) > 1e-5 # Preserving 1e-5
        positive_saturation_mask &= non_mono_pos_saturation_check

    # 8. Calculate aggregated weights
    # Denominator composition: sum_abs_total_conv_out + positive_bias + abs_negative_bias
    # is arithmetically equivalent to original (p_sum + n_sum + bias_pos + bias_neg).
    # Note: (positive_bias + abs_negative_bias) is equivalent to np.abs(bias).
    denominator: np.ndarray = sum_abs_total_conv_out + positive_bias + abs_negative_bias  # Shape (F,)

    # CRITICAL: Replicating original division behavior without adding stabilization epsilon.
    # If `denominator` contains zeros, `1.0 / denominator` will produce `inf` (and a NumPy RuntimeWarning).
    # Subsequent multiplications (e.g., `inf * 0`) might then produce `nan`.
    # This matches the numerical behavior of the original code precisely.
    inv_denominator_values: np.ndarray = 1.0 / denominator # This step may produce inf/nan.

    # `wts` (scalar or 1D array of shape F) broadcasts with `inv_denominator_values` (shape F)
    # and `*_saturation_mask` (shape F, boolean, casts to 0.0/1.0 on multiplication).
    positive_aggregated_weights: np.ndarray = \
        inv_denominator_values * wts * positive_saturation_mask # Shape (F,)
    negative_aggregated_weights: np.ndarray = \
        inv_denominator_values * wts * negative_saturation_mask # Shape (F,)

    # 9. Construct the final weighted contributions matrix
    # `positive_aggregated_weights` (F,) broadcasts with `positive_conv_out_parts` (H,W,C_in,F).
    term_positive: np.ndarray = positive_conv_out_parts * positive_aggregated_weights
    
    # `negative_conv_out_parts` are <= 0. Original multiplies this entire term by -1.0.
    # Example: if a part is -X (X>0), its contribution is (-X) * W_neg * (-1.0) = X * W_neg.
    # This correctly applies `negative_aggregated_weights` to the absolute value of negative parts.
    term_negative: np.ndarray = \
        negative_conv_out_parts * negative_aggregated_weights * -1.0
    
    # Original initialized wt_mat = np.zeros_like(kernel_weights) then added terms.
    # Summing the terms directly is equivalent.
    weighted_contributions: np.ndarray = term_positive + term_negative # Shape (H,W,C_in,F)

    # 10. Sum contributions over the feature axis (last axis, F)
    # Original: wt_mat = np.sum(wt_mat, axis=-1)
    final_output_weights: np.ndarray = np.sum(weighted_contributions, axis=-1) # Result shape (H,W,C_in)

    return final_output_weights

