from typing import Tuple
import torch

@torch.compile
def calculate_wt_mul_gpu(
    R: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    R.mul_(0.5)
    
    return R, R