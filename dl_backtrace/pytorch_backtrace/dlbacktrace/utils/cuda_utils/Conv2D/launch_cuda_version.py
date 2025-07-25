# import torch
# import torch.nn.functional as F
# from typing import Dict, Any, Union, Tuple
# from ..Padding.pytorch import calculate_padding
# from .cuda_version.wt_conv_ops import calculate_wt_conv2d_interface as calculate_wt_conv_cuda

# PaddingModeType = Union[str, Tuple[Any, Any]] 
        
# def calculate_wt_conv(
#     grad_output_scales: torch.Tensor,
#     input_activations: torch.Tensor,
#     kernel_weights_orig_shape: torch.Tensor,
#     bias: torch.Tensor,
#     padding_mode: PaddingModeType,
#     strides: Tuple[int, int],
#     activation_params: Dict[str, Any]
# ) -> torch.Tensor:
#     """
#     CUDA Optimized version of calculate_wt_conv_pytorch with improved performance.
    
#     Key optimizations:
#     1. Reduced tensor permutations and reshaping operations
#     2. More efficient memory layout handling
#     3. Simplified activation logic branching
#     4. Optimized padding extraction logic
#     5. Better tensor contiguity management
#     """
    
#     # Get dimensions directly without unnecessary transposes
#     F_dim, O_w, O_h = grad_output_scales.shape
#     C_in_dim, W_in_orig, H_in_orig = input_activations.shape
#     _, _, K_w, K_h = kernel_weights_orig_shape.shape
    
#     if activation_params["type"] == "mono":
#         act_type = 0
#     elif activation_params["type"] == "non_mono":
#         act_type = 1
#     else:
#         raise ValueError(f"Invalid activation type: {activation_params['type']}")
    
#     # Fix activation function mapping to match reference implementation
#     if activation_params["type"] == "non_mono":
#         if callable(activation_params["func"]):
#             # Check if it's a torch function by name or callable
#             func_name = activation_params["func"].__name__ if hasattr(activation_params["func"], '__name__') else str(activation_params["func"])
#             if 'relu' in func_name.lower():
#                 act_func = 1  # ReLU
#             elif 'sigmoid' in func_name.lower():
#                 act_func = 2  # Sigmoid
#             else:
#                 act_func = 0  # Identity
#         else:
#             act_func = 0  # Default to identity if not callable
#     else:
#         act_func = 0  # For mono type, function doesn't matter
    
#     # Handle range values properly - convert None to 0.0 and ensure they're floats
#     act_range_l = float(activation_params["range"]["l"]) if activation_params["range"]["l"] is not None else 0.0
#     act_range_u = float(activation_params["range"]["u"]) if activation_params["range"]["u"] is not None else 0.0
    
#     # Boolean flags should be based on whether range values are actually set (not None)
#     has_range_l = activation_params["range"]["l"] is not None
#     has_range_u = activation_params["range"]["u"] is not None
    
#     kernel_size_tuple = (K_h, K_w)
    
#     # Transpose only once and make contiguous
#     input_activations_T = input_activations.permute(2, 1, 0).contiguous()  # (H, W, C)
    
#     # Calculate padding
#     input_padded, padding_config = calculate_padding(
#         kernel_size=kernel_size_tuple,
#         input_tensor=input_activations_T,
#         padding_mode=padding_mode,
#         strides=strides
#     )
#     H_pad, W_pad, _ = input_padded.shape
    
#     # More efficient unfold preparation - avoid extra unsqueeze/squeeze
#     input_padded_nchw = input_padded.permute(2, 0, 1)[None, ...]  # (1, C, H, W)
    
#     # Unfold patches
#     patches_unfolded = F.unfold(
#         input_padded_nchw,
#         kernel_size=kernel_size_tuple,
#         stride=strides,
#         padding=0
#     )  # (1, C*K_h*K_w, L)
    
#     L_patches = patches_unfolded.shape[2]
    
#     # More efficient reshape - avoid multiple permutations
#     patches = patches_unfolded.view(C_in_dim, K_h, K_w, L_patches).permute(3, 1, 2, 0)  # (L, K_h, K_w, C)
    
#     # Prepare gradients - direct reshape instead of transpose then reshape
#     grad_scales = grad_output_scales.permute(2, 1, 0).reshape(L_patches, F_dim)  # (L, F)
    
#     # Prepare kernel weights - single permute operation
#     kernel_weights = kernel_weights_orig_shape.permute(3, 2, 1, 0)[None, ...]  # (1, K_h, K_w, C, F)
    
#     updates = torch.zeros_like(patches)
    
#     # Calculate updates
#     updates = calculate_wt_conv_cuda(
#         patches,
#         kernel_weights,
#         bias,
#         grad_scales,
#         L_patches, K_h, K_w, C_in_dim, F_dim,
#         act_type, act_func,
#         act_range_l, act_range_u,
#         has_range_l, has_range_u
#     )
#     torch.cuda.synchronize()
    
#     # Efficient fold operation - prepare tensor layout directly
#     updates_fold = updates.permute(0, 3, 1, 2).reshape(L_patches, -1).t()[None, ...]  # (1, C*K_h*K_w, L)
    
#     # Fold back to padded input space
#     grad_input_padded_nchw = F.fold(
#         updates_fold,
#         output_size=(H_pad, W_pad),
#         kernel_size=kernel_size_tuple,
#         stride=strides
#     )  # (1, C, H_pad, W_pad)
    
#     # Convert back and extract unpadded region
#     grad_input_padded = grad_input_padded_nchw[0].permute(1, 2, 0)  # (H_pad, W_pad, C)
    
#     # Optimized padding extraction
#     if isinstance(padding_config[0], list):
#         pad_h_before = padding_config[0][0]
#         pad_w_before = padding_config[1][0]
#     elif isinstance(padding_config[0], torch.Tensor):
#         pad_h_before = int(padding_config[0][0].item())
#         pad_w_before = int(padding_config[1][0].item())
#     else:
#         raise RuntimeError(f"Unexpected padding config type: {type(padding_config[0])}")
    
#     # Extract unpadded region
#     return grad_input_padded[
#         pad_h_before:pad_h_before + H_in_orig,
#         pad_w_before:pad_w_before + W_in_orig,
#         :
#     ]
