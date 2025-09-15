from typing import Tuple
import torch

@torch.compile
def calculate_wt_mul_gpu(
    R: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ultra GPU-optimized version using in-place operations (use with caution for autograd).
    
    This version uses in-place operations for maximum GPU performance but may not be
    compatible with autograd if R requires gradients.
    
    Args:
        R (torch.Tensor): Relevance from the output. Modified in-place.
        device (Optional[torch.device]): Target device. If None, uses CUDA if available.
        non_blocking (bool): If True, uses non-blocking transfer for CPU->GPU moves.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the same modified tensor twice.
        
    Warning:
        This function modifies the input tensor in-place. Do not use if R requires gradients
        or if you need to preserve the original values.
    """

    R.mul_(0.5)
    
    return R, R