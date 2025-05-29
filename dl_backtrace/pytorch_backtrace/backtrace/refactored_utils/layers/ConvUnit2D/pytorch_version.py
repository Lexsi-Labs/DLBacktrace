import torch
import torch.jit
from typing import Dict, Any, Union, Tuple

def calculate_wt_conv_unit_pytorch(
    patch: torch.Tensor,
    wts: Union[float, torch.Tensor],
    kernel_weights: torch.Tensor,
    bias: torch.Tensor,
    activation_params: Dict[str, Any]
) -> torch.Tensor:
    """
    Calculates weighted convolutional unit contributions in PyTorch.

    This function is a PyTorch equivalent of the NumPy-based
    `refactored_calculate_wt_conv_unit`. It processes inputs through a series of
    vectorized tensor operations:
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
       convolution outputs and biases. Division by zero produces inf/nan,
       replicating the original function's numerical characteristics.
    7. Combines the processed convolution output parts with these aggregated
       weights.
    8. Sums the result along the last axis (feature dimension) to produce the final output.

    Args:
        patch (torch.Tensor): Input data, expected to be a 3D tensor
            (e.g., dimensions H, W, C_in).
        wts (Union[float, torch.Tensor]): Scaling weights. Can be a Python float
            or a PyTorch tensor (scalar or 1D, e.g., dimension F corresponding
            to output features).
        kernel_weights (torch.Tensor): Kernel weights, expected to be a 4D tensor
            (e.g., dimensions H, W, C_in, F).
        bias (torch.Tensor): Bias terms, expected to be a 1D PyTorch tensor
            (e.g., dimension F).
        activation_params (Dict[str, Any]): Dictionary specifying activation behavior.
            It must contain:
            - "type" (str): Activation type, either 'mono' or 'non_mono'.
            - "range" (Dict[str, Optional[float]]): Specifies saturation ranges.
                - "l" (Optional[float]): Lower bound for saturation.
                - "u" (Optional[float]): Upper bound for saturation.
            - "func" (ActivationFunctionTypeTorch): The PyTorch-compatible
              activation function (e.g., `torch.sigmoid`), used if "type" is
              'non_mono'.

    Returns:
        torch.Tensor: The calculated weight matrix, summed over the feature
            dimension ('F'), resulting in a 3D tensor with dimensions
            matching the first three of `kernel_weights` (e.g., H, W, C_in).
    """

    ref_device: torch.device = kernel_weights.device
    ref_dtype: torch.dtype = kernel_weights.dtype

    # Ensure wts is a tensor
    if isinstance(wts, float):
        wts_tensor = torch.tensor(wts, device=ref_device, dtype=ref_dtype)
    else:
        wts_tensor = wts.to(device=ref_device, dtype=ref_dtype)

    # 1. Bias processing
    positive_bias = torch.relu(bias)  # Shape (F,)
    abs_negative_bias = torch.relu(-bias)  # Shape (F,)

    # 2. Convolution-like operation (element-wise multiplication using broadcasting)
    conv_out = kernel_weights * patch.unsqueeze(-1)  # Shape (H,W,C_in,F)

    # 3. Separate positive and negative parts of conv_out
    positive_conv_out_parts = torch.relu(conv_out) # Shape (H,W,C_in,F)
    negative_conv_out_parts = torch.clamp(conv_out, max=0.0) # Shape (H,W,C_in,F)

    # 4. Sum positive and absolute negative parts along (H, W, C_in) axes
    sum_positive_conv_out = torch.sum(positive_conv_out_parts, dim=(0, 1, 2))  # Shape (F,)
    sum_abs_negative_conv_out = -torch.sum(negative_conv_out_parts, dim=(0, 1, 2))  # Shape (F,)

    # 5. Calculate total sum of absolute activations
    sum_abs_total_conv_out = sum_positive_conv_out + sum_abs_negative_conv_out  # Shape (F,)

    # 6. Initialize saturation masks based on summed parts (boolean tensors)
    positive_saturation_mask = sum_positive_conv_out > 0.0
    negative_saturation_mask = sum_abs_negative_conv_out > 0.0

    # 7. Activation logic application
    act_type = activation_params["type"]
    act_range_config = activation_params["range"]
    act_range_l = act_range_config["l"]
    act_range_u = act_range_config["u"]

    if act_type == 'mono':
        if act_range_l:
            positive_saturation_mask = sum_abs_total_conv_out > act_range_l
        if act_range_u:
            negative_saturation_mask = sum_abs_total_conv_out < act_range_u
            
    elif act_type == 'non_mono':
        activation_func = activation_params["func"]
        
        activated_total_sum = activation_func(sum_abs_total_conv_out)
        activated_positive_sum_plus_bias = activation_func(sum_positive_conv_out + positive_bias)
        activated_neg_sum_plus_bias = activation_func(-1.0 * (sum_abs_negative_conv_out + abs_negative_bias))

        if act_range_l:
            saturation_lower_bound_mask = sum_abs_total_conv_out > act_range_l
            positive_saturation_mask = positive_saturation_mask & saturation_lower_bound_mask
        if act_range_u:
            saturation_upper_bound_mask = sum_abs_total_conv_out < act_range_u
            negative_saturation_mask = negative_saturation_mask & saturation_upper_bound_mask

        # Epsilon comparison (1e-5)
        epsilon: float = 1e-5
        
        non_mono_neg_saturation_check = torch.abs(activated_total_sum - activated_positive_sum_plus_bias) > epsilon
        negative_saturation_mask = negative_saturation_mask & non_mono_neg_saturation_check

        non_mono_pos_saturation_check = torch.abs(activated_total_sum - activated_neg_sum_plus_bias) > epsilon
        positive_saturation_mask = positive_saturation_mask & non_mono_pos_saturation_check

    # 8. Calculate aggregated weights
    denominator = sum_abs_total_conv_out + positive_bias + abs_negative_bias # Shape (F,)

    # Replicating original division behavior: produces inf for 0 in denominator.
    inv_denominator_values = 1.0 / denominator

    # Boolean masks (positive_saturation_mask, negative_saturation_mask) cast to float (0.0/1.0)
    # during multiplication.
    positive_aggregated_weights = inv_denominator_values * wts_tensor * positive_saturation_mask # Shape (F,)
    negative_aggregated_weights = inv_denominator_values * wts_tensor * negative_saturation_mask # Shape (F,)

    # 9. Construct the final weighted contributions matrix
    term_positive = positive_conv_out_parts * positive_aggregated_weights
    term_negative = negative_conv_out_parts * negative_aggregated_weights * -1.0
    
    weighted_contributions = term_positive + term_negative # Shape (H,W,C_in,F)

    # 10. Sum contributions over the feature axis (last axis, F)
    final_output_weights = torch.sum(weighted_contributions, dim = -1) # Result shape (H,W,C_in)

    return final_output_weights

calculate_wt_conv_unit = torch.compile(calculate_wt_conv_unit_pytorch)
