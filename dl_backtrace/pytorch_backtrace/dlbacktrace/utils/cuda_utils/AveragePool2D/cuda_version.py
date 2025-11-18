import torch
from typing import Tuple
import torch.nn.functional as F
from typing import Tuple, Union

from cuda_wt_avg_unit import calculate_wt_avg_unit_cuda
from cuda_padding import calculate_padding_cuda

def calculate_wt_avg(
    relevance_y: torch.Tensor,
    input_array: torch.Tensor, 
    pool_size: Tuple[int, int],
    padding: Union[str, Tuple[Union[int, None], Union[int, None]]],
    strides: Tuple[int, int],
) -> torch.Tensor:

    batch_size = input_array.shape[0]

    relevance_x = []

    for batch_idx in range(batch_size):
        current_relevance = relevance_y[batch_idx]
        current_input = input_array[batch_idx]

        current_relevance = current_relevance.permute(2, 1, 0)
        current_input = current_input.permute(2, 1, 0)

        strides_h, strides_w = strides
        padding_h, padding_w = padding

        input_padded, paddings = calculate_padding_cuda(
            (pool_size[0], pool_size[1]), current_input, (padding_h, padding_w), (strides_h, strides_w), float('-inf'))

        output_accumulated = torch.zeros_like(input_padded)

        for out_h in range(current_relevance.shape[0]):
            for out_w in range(current_relevance.shape[1]):
                h_start = out_h * strides_h
                h_end = h_start + pool_size[0]
                w_start = out_w * strides_w
                w_end = w_start + pool_size[1]

                # Ensure indices don't exceed padded input dimensions
                h_end = min(h_end, input_padded.shape[0])
                w_end = min(w_end, input_padded.shape[1])
                
                # Extract patch using slicing
                input_patch = input_padded[h_start:h_end, w_start:w_end, :]
                
                # Handle edge cases where patch might be smaller than kernel
                if input_patch.shape[0] < pool_size[0] or input_patch.shape[1] < pool_size[1]:
                    # Pad the patch to match kernel size
                    pad_h = pool_size[0] - input_patch.shape[0]
                    pad_w = pool_size[1] - input_patch.shape[1]
                    
                    if pad_h > 0 or pad_w > 0:
                        pad_tuple = (0, 0, 0, pad_w, 0, pad_h)  # (channels, width, height)
                        input_patch = F.pad(input_patch, pad_tuple, mode='constant', value=0.0)
                
                # Get relevance weight for current output location
                relevance_weight = current_relevance[out_h, out_w, :]

                patch_updates = calculate_wt_avg_unit_cuda(input_patch, relevance_weight)

                update_h = min(patch_updates.shape[0], h_end - h_start)
                update_w = min(patch_updates.shape[1], w_end - w_start)

                output_accumulated[h_start:h_start + update_h, w_start:w_start + update_w, :] += patch_updates[:update_h, :update_w, :]

        pad_h_before = paddings[0][0]
        pad_w_before = paddings[1][0]
        original_h = current_input.shape[0]
        original_w = current_input.shape[1]

        output_unpadded = output_accumulated[pad_h_before:pad_h_before + original_h, pad_w_before:pad_w_before + original_w, :]
        relevance_x.append(output_unpadded)

    relevance_x = torch.stack(relevance_x, dim=0)

    return relevance_x