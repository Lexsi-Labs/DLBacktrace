# import torch
# from typing import Tuple, Union
# from ..WtMaxunit2D.cuda_version.wt_maxunit2d_ops import calculate_wt_maxunit2d_interface as calculate_wt_max_unit_cuda
# from ..Padding.pytorch import calculate_padding

# def calculate_wt_maxpool_cuda(
#     wts: torch.Tensor, 
#     inp: torch.Tensor, 
#     pool_size: Union[int, Tuple[int, int]], 
#     padding: Union[int, Tuple[int, int]], 
#     strides: Union[int, Tuple[int, int]]
# ) -> torch.Tensor:
#     """
#     Perform weighted max pooling operation on input tensor using optimized PyTorch operations.
    
#     This function applies weighted max pooling where weights are distributed among the maximum
#     values within each pooling window. The operation slides a pooling window across the input
#     with specified strides, and for each window, identifies maximum values per channel and
#     distributes the corresponding weights among these maximum positions.
    
#     Args:
#         wts (torch.Tensor): Weights tensor of shape (channels, out_height, out_width) containing
#                            the weights to be applied at each output position for each channel.
#         inp (torch.Tensor): Input tensor of shape (channels, in_height, in_width) to be pooled.
#         pool_size (Union[int, Tuple[int, int]]): Size of the pooling window. If int, same size
#                                                is used for both height and width dimensions.
#         padding (Union[int, Tuple[int, int]]): Padding to be applied. If int, same padding
#                                              is used for both height and width dimensions.
#         strides (Union[int, Tuple[int, int]]): Stride of the pooling operation. If int, same
#                                              stride is used for both height and width dimensions.
    
#     Returns:
#         torch.Tensor: Output tensor of same shape as input, where weighted max pooling has been
#                      applied. Values represent the distributed weights at positions achieving
#                      maximum values within their respective pooling windows.
    
#     Notes:
#         - Input tensors are transposed at the beginning and the result maintains original orientation
#         - Padding is applied with -inf values to ensure they don't interfere with max operations
#         - Overlapping pooling windows accumulate their contributions additively
#         - This function is optimized for autograd compatibility and GPU acceleration
#     """
    
#     # Transpose inputs to work with internal representation (replicating original behavior)
#     wts_transposed = wts.transpose(0, 2).transpose(0, 1)  # Equivalent to .T for 3D
#     inp_transposed = inp.transpose(0, 2).transpose(0, 1)  # Equivalent to .T for 3D
    
#     # Normalize pool_size
#     actual_pool_size = pool_size
#     if isinstance(pool_size, torch.Tensor):
#         if pool_size.ndim == 0:  # scalar tensor
#             actual_pool_size = pool_size.item()
#         else:  # assuming 1D tensor for tuple-like e.g. tensor([2,2])
#             actual_pool_size = tuple(pool_size.tolist())
#     pool_size_tuple = (actual_pool_size, actual_pool_size) if isinstance(actual_pool_size, int) else actual_pool_size

#     # Normalize padding
#     actual_padding = padding
#     if isinstance(padding, torch.Tensor):
#         if padding.ndim == 0:  # scalar tensor
#             actual_padding = padding.item()
#         else:  # assuming 1D tensor for tuple-like
#             actual_padding = tuple(padding.tolist())
#     padding_tuple = (actual_padding, actual_padding) if isinstance(actual_padding, int) else actual_padding

#     # Normalize strides
#     actual_strides = strides
#     if isinstance(strides, torch.Tensor):
#         if strides.ndim == 0:  # scalar tensor
#             actual_strides = strides.item()
#         else:  # assuming 1D tensor for tuple-like
#             actual_strides = tuple(strides.tolist())
#     strides_tuple = (actual_strides, actual_strides) if isinstance(actual_strides, int) else actual_strides
    
#     # Apply padding with -inf values (replicating original behavior)
#     input_padded, paddings = calculate_padding(
#         pool_size_tuple, inp_transposed, padding_tuple, strides_tuple, -float('inf')
#     )
    
#     # Initialize output array with zeros, same shape as padded input
#     output_accumulated = torch.zeros_like(input_padded)
    
#     # Get output dimensions from weights tensor
#     out_height, out_width = wts_transposed.shape[:2]
    
#     # Vectorized approach using unfold for efficient patch extraction
#     # This replaces the nested loops with vectorized operations
#     patches_h = input_padded.unfold(0, pool_size_tuple[0], strides_tuple[0])
#     patches_hw = patches_h.unfold(1, pool_size_tuple[1], strides_tuple[1])
    
#     # Reshape patches for batch processing: (out_h, out_w, channels, pool_h, pool_w)
#     patches_reshaped = patches_hw.permute(0, 1, 3, 4, 2)
    
#     # Process all patches in a vectorized manner
#     for output_row in range(out_height):
#         for output_col in range(out_width):
#             # Get current patch and weights
#             current_patch = patches_reshaped[output_row, output_col]
#             current_weights = wts_transposed[output_row, output_col, :]
            
#             # Ensure pool_size_tuple is appropriate for the CUDA kernel
#             if isinstance(pool_size_tuple, tuple):
#                 assert pool_size_tuple[0] == pool_size_tuple[1], \
#                     "CUDA kernel currently expects square pooling dimensions."
#                 kernel_pool_size = pool_size_tuple[0]
#             else: # it's an int
#                 kernel_pool_size = pool_size_tuple

#             # Calculate weighted updates using the compiled function
#             weighted_updates = calculate_wt_max_unit_cuda(
#                 current_patch, current_weights, kernel_pool_size
#             )
            
#             # Calculate indices for accumulation
#             row_start = output_row * strides_tuple[0]
#             row_end = row_start + pool_size_tuple[0]
#             col_start = output_col * strides_tuple[1]
#             col_end = col_start + pool_size_tuple[1]
            
#             # Accumulate updates (handling overlapping windows)
#             output_accumulated[row_start:row_end, col_start:col_end] += weighted_updates
    
#     # Remove padding to get final output with original input dimensions
#     if isinstance(paddings, list) and len(paddings) > 0 and hasattr(paddings[0], '__len__'):
#         # Handle the case where paddings is a list of arrays/lists
#         pad_h_before = int(paddings[0][0]) if hasattr(paddings[0], '__getitem__') else paddings[0][0]
#         pad_w_before = int(paddings[1][0]) if hasattr(paddings[1], '__getitem__') else paddings[1][0]
        
#         final_output = output_accumulated[
#             pad_h_before:(pad_h_before + inp_transposed.shape[0]),
#             pad_w_before:(pad_w_before + inp_transposed.shape[1]),
#             :
#         ]
#     else:
#         # Fallback for simple padding format
#         final_output = output_accumulated[
#             :inp_transposed.shape[0],
#             :inp_transposed.shape[1],
#             :
#         ]
    
#     return final_output
