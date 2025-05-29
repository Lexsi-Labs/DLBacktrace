import torch
import torch.nn.functional as F
from typing import Dict, Any, Union, Tuple
from ..Padding.pytorch import calculate_padding

PaddingModeType = Union[str, Tuple[Any, Any]] 
        
def calculate_wt_conv_pytorch_optimized(
    grad_output_scales: torch.Tensor,
    input_activations: torch.Tensor,
    kernel_weights_orig_shape: torch.Tensor,
    bias: torch.Tensor,
    padding_mode: PaddingModeType,
    strides: Tuple[int, int],
    activation_params: Dict[str, Any]
) -> torch.Tensor:
    """
    Optimized version of calculate_wt_conv_pytorch with improved performance.
    
    Key optimizations:
    1. Reduced tensor permutations and reshaping operations
    2. More efficient memory layout handling
    3. Simplified activation logic branching
    4. Optimized padding extraction logic
    5. Better tensor contiguity management
    """
    device = grad_output_scales.device
    dtype = grad_output_scales.dtype
    
    # Get dimensions directly without unnecessary transposes
    F_dim, O_w, O_h = grad_output_scales.shape
    C_in_dim, W_in_orig, H_in_orig = input_activations.shape
    _, _, K_w, K_h = kernel_weights_orig_shape.shape
    
    kernel_size_tuple = (K_h, K_w)
    
    # Transpose only once and make contiguous
    input_activations_T = input_activations.permute(2, 1, 0).contiguous()  # (H, W, C)
    
    # Calculate padding
    input_padded, padding_config = calculate_padding(
        kernel_size=kernel_size_tuple,
        input_tensor=input_activations_T,
        padding_mode=padding_mode,
        strides=strides
    )
    H_pad, W_pad, _ = input_padded.shape
    
    # More efficient unfold preparation - avoid extra unsqueeze/squeeze
    input_padded_nchw = input_padded.permute(2, 0, 1)[None, ...]  # (1, C, H, W)
    
    # Unfold patches
    patches_unfolded = F.unfold(
        input_padded_nchw,
        kernel_size=kernel_size_tuple,
        stride=strides,
        padding=0
    )  # (1, C*K_h*K_w, L)
    
    L_patches = patches_unfolded.shape[2]
    
    # More efficient reshape - avoid multiple permutations
    patches = patches_unfolded.view(C_in_dim, K_h, K_w, L_patches).permute(3, 1, 2, 0)  # (L, K_h, K_w, C)
    
    # Prepare gradients - direct reshape instead of transpose then reshape
    grad_scales = grad_output_scales.permute(2, 1, 0).reshape(L_patches, F_dim)  # (L, F)
    
    # Prepare kernel weights - single permute operation
    kernel_weights = kernel_weights_orig_shape.permute(3, 2, 1, 0)[None, ...]  # (1, K_h, K_w, C, F)
    
    # Optimized convolution computation using einsum for better performance
    # patches: (L, K_h, K_w, C), kernel_weights: (1, K_h, K_w, C, F)
    conv_out = torch.einsum('lhwc,bhwcf->lhwcf', patches, kernel_weights)  # (L, K_h, K_w, C, F)
    
    # Separate positive and negative parts
    positive_conv = torch.clamp(conv_out, min=0.0)
    negative_conv = torch.clamp(conv_out, max=0.0)
    
    # Sum over spatial and channel dimensions more efficiently
    sum_positive = positive_conv.sum(dim=(1, 2, 3))  # (L, F)
    sum_abs_negative = (-negative_conv).sum(dim=(1, 2, 3))  # (L, F)
    sum_abs_total = sum_positive + sum_abs_negative  # (L, F)
    
    # Precompute bias terms
    positive_bias = torch.clamp(bias, min=0.0)  # (F,)
    abs_negative_bias = torch.clamp(-bias, min=0.0)  # (F,)
    
    # Initialize saturation masks
    pos_mask = sum_positive > 0.0  # (L, F)
    neg_mask = sum_abs_negative > 0.0  # (L, F)
    
    # Activation logic with reduced branching
    act_type = activation_params["type"]
    act_range = activation_params["range"]
    
    if act_type == 'mono':
        if act_range["l"]:
            pos_mask = sum_abs_total > act_range["l"]
        if act_range["u"]:
            neg_mask = sum_abs_total < act_range["u"]
    
    elif act_type == 'non_mono':
        activation_func = activation_params["func"]
        
        # Batch all activation computations
        activated_total = activation_func(sum_abs_total)
        activated_pos_bias = activation_func(sum_positive + positive_bias)
        activated_neg_bias = activation_func(-(sum_abs_negative + abs_negative_bias))
        
        if act_range["l"]:
            range_mask_l = sum_abs_total > act_range["l"]
            pos_mask = pos_mask & range_mask_l
        if act_range["u"]:
            range_mask_u = sum_abs_total < act_range["u"]
            neg_mask = neg_mask & range_mask_u
        
        # Vectorized epsilon comparisons
        epsilon = 1e-5
        neg_check = torch.abs(activated_total - activated_pos_bias) > epsilon
        pos_check = torch.abs(activated_total - activated_neg_bias) > epsilon
        
        neg_mask = neg_mask & neg_check
        pos_mask = pos_mask & pos_check
    
    # Compute weights more efficiently
    denominator = sum_abs_total + positive_bias + abs_negative_bias  # (L, F)
    inv_denom = torch.reciprocal(denominator)  # More efficient than 1.0 / denominator
    
    # Convert masks to float in-place for efficiency
    pos_weights = inv_denom * grad_scales * pos_mask.to(dtype)  # (L, F)
    neg_weights = inv_denom * grad_scales * neg_mask.to(dtype)  # (L, F)
    
    # Efficient broadcasting and computation
    # Reshape for broadcasting: (L, F) -> (L, 1, 1, 1, F)
    pos_weights_bc = pos_weights.view(L_patches, 1, 1, 1, F_dim)
    neg_weights_bc = neg_weights.view(L_patches, 1, 1, 1, F_dim)
    
    # Compute weighted contributions
    updates = (positive_conv * pos_weights_bc - negative_conv * neg_weights_bc).sum(dim=-1)  # (L, K_h, K_w, C)
    
    # Efficient fold operation - prepare tensor layout directly
    updates_fold = updates.permute(0, 3, 1, 2).reshape(L_patches, -1).t()[None, ...]  # (1, C*K_h*K_w, L)
    
    # Fold back to padded input space
    grad_input_padded_nchw = F.fold(
        updates_fold,
        output_size=(H_pad, W_pad),
        kernel_size=kernel_size_tuple,
        stride=strides
    )  # (1, C, H_pad, W_pad)
    
    # Convert back and extract unpadded region
    grad_input_padded = grad_input_padded_nchw[0].permute(1, 2, 0)  # (H_pad, W_pad, C)
    
    # Optimized padding extraction
    if isinstance(padding_config[0], list):
        pad_h_before = padding_config[0][0]
        pad_w_before = padding_config[1][0]
    elif isinstance(padding_config[0], torch.Tensor):
        pad_h_before = int(padding_config[0][0].item())
        pad_w_before = int(padding_config[1][0].item())
    else:
        raise RuntimeError(f"Unexpected padding config type: {type(padding_config[0])}")
    
    # Extract unpadded region
    return grad_input_padded[
        pad_h_before:pad_h_before + H_in_orig,
        pad_w_before:pad_w_before + W_in_orig,
        :
    ]

# Compiled version for even better performance
calculate_wt_conv = torch.compile(calculate_wt_conv_pytorch_optimized)
