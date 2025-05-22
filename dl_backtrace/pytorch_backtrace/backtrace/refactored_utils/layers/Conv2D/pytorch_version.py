import torch
import torch.nn.functional as F
from typing import Dict, Any, Union, Tuple, List, Optional
from backtrace.refactored_utils.layers.ConvUnit2D.pytorch_version import calculate_wt_conv_unit
from backtrace.refactored_utils.layers.Padding.pytorch import calculate_padding

PaddingModeTupleElementType = Union[int, float]
PaddingModeTuple = Tuple[PaddingModeTupleElementType, PaddingModeTupleElementType]

@torch.jit.script
def pytorch_calculate_wt_conv(
    grad_output_scales: torch.Tensor,
    input_activations: torch.Tensor,
    kernel_weights_orig_shape: torch.Tensor,
    bias: torch.Tensor,
    padding_mode: Union[str, Tuple[PaddingModeTupleElementType, PaddingModeTupleElementType]],
    strides: Tuple[int, int],
    activation_params: Dict[str, Any]
) -> torch.Tensor:
    """
    Calculates the gradient with respect to the input activations of a
    convolutional-like operation using PyTorch, vectorized for performance.

    This function mirrors the logic of the original NumPy-based
    `refactored_calculate_wt_conv` but replaces explicit loops with
    vectorized PyTorch operations (`unfold` and `fold`) and batched computations.

    Input tensors are expected in specific shapes and are transposed internally
    to match the conventions used throughout the calculation.

    Args:
        grad_output_scales (torch.Tensor): Gradient scales for output features.
            Expected shape (F, O_w, O_h). Transposed to (O_h, O_w, F).
        input_activations (torch.Tensor): Input activations.
            Expected shape (C_in, W_in_orig, H_in_orig). Transposed to
            (H_in_orig, W_in_orig, C_in).
        kernel_weights_orig_shape (torch.Tensor): Original kernel weights.
            Expected shape (F, C_in, K_w, K_h). Transposed to (K_h, K_w, C_in, F).
        bias (torch.Tensor): Bias terms, shape (F,).
        padding_mode (Union[str, Tuple[Union[int,float], Union[int,float]]]):
            Padding mode for `calculate_padding_pytorch`.
        strides (Tuple[int, int]): Strides (stride_h, stride_w).
        activation_params (Dict[str, Any]): Parameters for activation logic,
            passed to the batched version of `torch_calculate_wt_conv_unit` logic.
            Requires "type", "range", and "func" keys.

    Returns:
        torch.Tensor: The calculated gradient w.r.t. input activations.
            Shape (H_in_orig, W_in_orig, C_in).
    """
    # Transpose inputs to match original function's internal layout
    # (F, O_w, O_h) -> (O_h, O_w, F)
    grad_output_scales_T = grad_output_scales.permute(2, 1, 0)
    # (C_in, W_in_orig, H_in_orig) -> (H_in_orig, W_in_orig, C_in)
    input_activations_T = input_activations.permute(2, 1, 0)
    # (F, C_in, K_w, K_h) -> (K_h, K_w, C_in, F)
    kernel_weights_T = kernel_weights_orig_shape.permute(3, 2, 1, 0)

    # Decomposed shapes (from transposed tensors)
    O_h, O_w, F_dim = grad_output_scales_T.shape
    H_in_orig, W_in_orig, C_in_dim = input_activations_T.shape
    K_h, K_w, _, _ = kernel_weights_T.shape # C_in, F_dim are inner dims

    kernel_size_tuple = (int(K_h), int(K_w))

    # 1. Calculate padding using the provided helper
    input_padded, padding_config = calculate_padding(
        kernel_size=kernel_size_tuple,
        input_tensor=input_activations_T,
        padding_mode=padding_mode,
        strides=strides
    )
    H_pad, W_pad, _ = input_padded.shape

    # 2. Prepare for unfold: input_padded needs (N, C, H, W)
    # (H_pad, W_pad, C_in_dim) -> (1, C_in_dim, H_pad, W_pad)
    input_padded_nchw = input_padded.permute(2, 0, 1).unsqueeze(0)

    # Unfold to extract all patches: (N, C_in_dim * K_h * K_w, L)
    # L = O_h * O_w (number of patches)
    patches_unfolded = F.unfold(
        input_padded_nchw,
        kernel_size=kernel_size_tuple,
        stride=strides,
        padding=0 # Input is already padded
    )
    L_patches = patches_unfolded.shape[2] # Number of patches

    # Reshape patches for batched processing: (L, K_h, K_w, C_in_dim)
    # (1, C_in_dim*K_h*K_w, L) -> (1, C_in_dim, K_h, K_w, L)
    patches_reshaped = patches_unfolded.view(1, C_in_dim, K_h, K_w, L_patches)
    # (1, C_in_dim, K_h, K_w, L) -> (L, K_h, K_w, C_in_dim)
    all_patches = patches_reshaped.permute(0, 4, 2, 3, 1).squeeze(0)

    # Prepare grad_output_scales for batched processing: (L, F_dim)
    all_grad_scale_slices = grad_output_scales_T.reshape(L_patches, F_dim)

    # 3. Batched computation of updates (inlined and adapted torch_calculate_wt_conv_unit logic)
    # This section replicates the logic of torch_calculate_wt_conv_unit but for batched inputs.
    # Inputs: all_patches (L,K_h,K_w,C_in), all_grad_scale_slices (L,F), kernel_weights_T (K_h,K_w,C_in,F), bias (F)
    ref_device = kernel_weights_T.device
    ref_dtype = kernel_weights_T.dtype

    wts_tensor_batched = all_grad_scale_slices.to(device=ref_device, dtype=ref_dtype) # (L, F)

    positive_bias = torch.relu(bias)  # (F,)
    abs_negative_bias = torch.relu(-bias) # (F,)

    # kernel_weights_T (K_h,K_w,C_in,F) -> kernel_weights_b (1,K_h,K_w,C_in,F) for broadcasting
    kernel_weights_b = kernel_weights_T.unsqueeze(0)
    # all_patches (L,K_h,K_w,C_in) -> all_patches_exp (L,K_h,K_w,C_in,1)
    all_patches_exp = all_patches.unsqueeze(-1)

    # conv_out: (L, K_h, K_w, C_in_dim, F_dim)
    conv_out = kernel_weights_b * all_patches_exp

    positive_conv_out_parts = torch.relu(conv_out)
    negative_conv_out_parts = torch.clamp(conv_out, max=0.0)

    # Sum over K_h, K_w, C_in_dim axes (1,2,3 for batched input)
    sum_axes_batched: List[int] = [1, 2, 3]
    sum_positive_conv_out = torch.sum(positive_conv_out_parts, dim=sum_axes_batched) # (L, F)
    sum_abs_negative_conv_out = -torch.sum(negative_conv_out_parts, dim=sum_axes_batched) # (L, F)

    sum_abs_total_conv_out = sum_positive_conv_out + sum_abs_negative_conv_out # (L, F)

    # Saturation masks, initialized based on summed parts
    current_positive_saturation_mask = sum_positive_conv_out > 0.0 # (L, F)
    current_negative_saturation_mask = sum_abs_negative_conv_out > 0.0 # (L, F)

    # Activation logic (adapted from torch_calculate_wt_conv_unit)
    act_type = str(activation_params["type"])
    act_range_config = activation_params["range"] # type: Dict[str, Optional[float]]

    _act_range_l_any = act_range_config.get("l") # Use .get for safety if key might be missing
    _act_range_u_any = act_range_config.get("u")
    
    act_range_l: Optional[float] = None
    if isinstance(_act_range_l_any, (int, float)): act_range_l = float(_act_range_l_any)
    
    act_range_u: Optional[float] = None
    if isinstance(_act_range_u_any, (int, float)): act_range_u = float(_act_range_u_any)

    if act_type == 'mono':
        if act_range_l is not None:
            current_positive_saturation_mask = sum_abs_total_conv_out > act_range_l
        if act_range_u is not None:
            current_negative_saturation_mask = sum_abs_total_conv_out < act_range_u
    elif act_type == 'non_mono':
        # activation_func: ActivationFunctionTypeTorch
        activation_func_any = activation_params["func"] 
        # JIT requires careful handling here. Assuming activation_func_any is a ScriptFunction or JIT-able.
        activation_func = activation_func_any # type: ignore 

        activated_total_sum = activation_func(sum_abs_total_conv_out)
        activated_positive_sum_plus_bias = activation_func(sum_positive_conv_out + positive_bias)
        activated_neg_sum_plus_bias = activation_func(
            -1.0 * (sum_abs_negative_conv_out + abs_negative_bias)
        )

        if act_range_l is not None:
            saturation_lower_bound_mask = sum_abs_total_conv_out > act_range_l
            current_positive_saturation_mask = current_positive_saturation_mask & saturation_lower_bound_mask
        if act_range_u is not None:
            saturation_upper_bound_mask = sum_abs_total_conv_out < act_range_u
            current_negative_saturation_mask = current_negative_saturation_mask & saturation_upper_bound_mask
        
        epsilon = 1e-5 # Preserved epsilon
        non_mono_neg_saturation_check = \
            torch.abs(activated_total_sum - activated_positive_sum_plus_bias) > epsilon
        current_negative_saturation_mask = current_negative_saturation_mask & non_mono_neg_saturation_check

        non_mono_pos_saturation_check = \
            torch.abs(activated_total_sum - activated_neg_sum_plus_bias) > epsilon
        current_positive_saturation_mask = current_positive_saturation_mask & non_mono_pos_saturation_check

    denominator = sum_abs_total_conv_out + positive_bias + abs_negative_bias # (L, F)
    inv_denominator_values = 1.0 / denominator # (L, F), handles div by zero -> inf

    # Ensure masks are float for multiplication
    positive_mask_float = current_positive_saturation_mask.to(dtype=ref_dtype)
    negative_mask_float = current_negative_saturation_mask.to(dtype=ref_dtype)

    positive_aggregated_weights = inv_denominator_values * wts_tensor_batched * positive_mask_float # (L,F)
    negative_aggregated_weights = inv_denominator_values * wts_tensor_batched * negative_mask_float # (L,F)

    # Expand aggregated weights for broadcasting with conv_out_parts
    # (L,F) -> (L, 1, 1, 1, F)
    pos_agg_w_exp = positive_aggregated_weights.view(L_patches, 1, 1, 1, F_dim)
    neg_agg_w_exp = negative_aggregated_weights.view(L_patches, 1, 1, 1, F_dim)

    term_positive = positive_conv_out_parts * pos_agg_w_exp
    term_negative = negative_conv_out_parts * neg_agg_w_exp * -1.0
    
    weighted_contributions = term_positive + term_negative # (L, K_h, K_w, C_in, F)
    all_updates = torch.sum(weighted_contributions, dim=-1) # Sum over F_dim -> (L, K_h, K_w, C_in)
    # --- End of inlined/batched logic ---

    # 4. Fold `all_updates` back to form `grad_input_padded`
    # `all_updates` is (L, K_h, K_w, C_in_dim)
    # `F.fold` expects input (N, C_in_dim * K_h * K_w, L)
    # (L, K_h, K_w, C_in_dim) -> permute to (L, C_in_dim, K_h, K_w)
    all_updates_lchw = all_updates.permute(0, 3, 1, 2)
    # (L, C_in_dim, K_h, K_w) -> reshape to (L, C_in_dim * K_h * K_w)
    all_updates_l_ckhw = all_updates_lchw.reshape(L_patches, C_in_dim * K_h * K_w)
    # (L, C_in_dim*K_h*K_w) -> transpose to (C_in_dim*K_h*K_w, L)
    all_updates_ckhw_l = all_updates_l_ckhw.transpose(0, 1)
    # Add batch dim N=1: (1, C_in_dim*K_h*K_w, L)
    all_updates_for_fold = all_updates_ckhw_l.unsqueeze(0)
    
    grad_input_padded_nchw = F.fold(
        all_updates_for_fold,
        output_size=(H_pad, W_pad),
        kernel_size=kernel_size_tuple,
        stride=strides
    ) # Output: (1, C_in_dim, H_pad, W_pad)

    # Convert back to (H_pad, W_pad, C_in_dim)
    grad_input_padded = grad_input_padded_nchw.squeeze(0).permute(1, 2, 0)

    # 5. Remove padding to get final gradient
    pad_h_before_val: float
    pad_w_before_val: float

    # padding_config is TorchPaddingDetails = Union[List[List[int]], List[torch.Tensor]]
    # Accessing padding_config[0] (for height)
    # It's either List[int] or torch.Tensor
    # JIT requires specific type checks
    padding_dim0 = padding_config[0]
    padding_dim1 = padding_config[1]

    if isinstance(padding_dim0, torch.Tensor) and isinstance(padding_dim1, torch.Tensor):
        pad_h_before_val = padding_dim0[0].item()
        pad_w_before_val = padding_dim1[0].item()
    elif isinstance(padding_dim0, list) and isinstance(padding_dim1, list):
        # Assuming List[int] if not Tensor
        # Check elements are int for robustness, though types imply it
        if isinstance(padding_dim0[0], int) and isinstance(padding_dim1[0], int):
             pad_h_before_val = float(padding_dim0[0])
             pad_w_before_val = float(padding_dim1[0])
        else:
            # This case should not be reached if padding_config conforms to TorchPaddingDetails
            # For JIT, an error or default might be needed if this state is possible
            # Defaulting to 0 or raising error. Let's assume valid structure.
            # Fallback for safety, though ideally an error if types are wrong.
            pad_h_before_val = 0.0 
            pad_w_before_val = 0.0
            # Or raise RuntimeError("Unexpected padding_config structure")

    else:
        # Fallback or error for mixed/unexpected types in padding_config
        pad_h_before_val = 0.0 
        pad_w_before_val = 0.0
        # Or raise RuntimeError("Unexpected padding_config structure")


    start_h_slice = int(pad_h_before_val)
    start_w_slice = int(pad_w_before_val)
    
    end_h_slice = start_h_slice + H_in_orig
    end_w_slice = start_w_slice + W_in_orig

    grad_input_unpadded = grad_input_padded[
        start_h_slice:end_h_slice,
        start_w_slice:end_w_slice,
        : # All channels
    ]

    return grad_input_unpadded
