import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union, Callable

@torch.compile
def calculate_wt_conv_unit(
    patch: torch.Tensor,
    wts: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    act: Dict[str, Union[str, Dict[str, Optional[float]], Callable]]
) -> torch.Tensor:
    """
    Calculate weighted convolution unit with activation-aware processing.
    
    This function computes convolution weights by analyzing positive and negative
    components of the convolution output, applying activation-specific logic,
    and aggregating weights accordingly.
    
    Args:
        patch (torch.Tensor): Input patch tensor of shape (i, j, k)
        wts (torch.Tensor): Weight tensor for aggregation
        w (torch.Tensor): Convolution kernel weights of shape (i, j, k, l)
        b (torch.Tensor, optional): Bias tensor of shape (l,). If None, no bias is applied
        act (Dict): Activation configuration containing:
            - 'type' (str): Either 'mono' or 'non_mono'
            - 'range' (Dict): Contains 'l' (lower) and 'u' (upper) bounds
            - 'func' (Callable, optional): Activation function for 'non_mono' type
    
    Returns:
        torch.Tensor: Processed weight matrix of shape (i, j, k)
    """
    device = patch.device
    dtype = patch.dtype
    
    # Ensure all tensors are on the same device and dtype
    k = w.to(device=device, dtype=dtype)
    wts = wts.to(device=device, dtype=dtype)
    
    # Compute convolution output using einsum
    conv_out = torch.einsum("ijkl,ijk->ijkl", k, patch)
    
    # Vectorized positive/negative component extraction
    p_ind = torch.where(conv_out > 0, conv_out, torch.zeros_like(conv_out))
    n_ind = torch.where(conv_out < 0, -conv_out, torch.zeros_like(conv_out))
    
    # Sum across spatial dimensions
    p_sum = torch.sum(p_ind, dim=(0, 1, 2))  # Shape: (l,)
    n_sum = torch.sum(n_ind, dim=(0, 1, 2))  # Shape: (l,)
    t_sum = p_sum + n_sum
    
    # Handle bias processing
    if b is not None:
        bias = b.to(device=device, dtype=dtype)
        bias_pos = torch.where(bias > 0, bias, torch.zeros_like(bias))
        bias_neg = torch.where(bias < 0, -bias, torch.zeros_like(bias))
    else:
        bias_pos = torch.zeros_like(p_sum)
        bias_neg = torch.zeros_like(n_sum)
    
    # Initialize saturation masks
    p_saturate = (p_sum > 0).to(dtype=dtype)
    n_saturate = (n_sum > 0).to(dtype=dtype)
    
    # Process activation logic
    if act["type"] == 'mono':
        if act["range"]["l"] is not None:
            p_saturate = (t_sum > act["range"]["l"]).to(dtype=dtype)
        if act["range"]["u"] is not None:
            n_saturate = (t_sum < act["range"]["u"]).to(dtype=dtype)
            
    elif act["type"] == 'non_mono':
        t_act = act["func"](t_sum)
        p_act = act["func"](p_sum + bias_pos)
        n_act = act["func"](-(n_sum + bias_neg))
        
        if act["range"]["l"] is not None:
            range_l_mask = (t_sum > act["range"]["l"]).to(dtype=dtype)
            p_saturate = p_saturate * range_l_mask
            
        if act["range"]["u"] is not None:
            range_u_mask = (t_sum < act["range"]["u"]).to(dtype=dtype)
            n_saturate = n_saturate * range_u_mask
            
        # Apply activation difference thresholds
        t_p_diff_mask = (torch.abs(t_act - p_act) > 1e-5).to(dtype=dtype)
        n_saturate = n_saturate * t_p_diff_mask
        
        t_n_diff_mask = (torch.abs(t_act - n_act) > 1e-5).to(dtype=dtype)
        p_saturate = p_saturate * t_n_diff_mask
    
    # Compute denominators with numerical stability
    denom = p_sum + n_sum + bias_pos + bias_neg
    denom = torch.where(denom == 0, torch.tensor(1e-12, device=device, dtype=dtype), denom)
    
    # Compute aggregated weights
    inv_denom = 1.0 / denom
    p_agg_wt = inv_denom * wts * p_saturate
    n_agg_wt = inv_denom * wts * n_saturate
    
    # Expand weights for broadcasting and compute final weight matrix
    p_agg_wt_expanded = p_agg_wt.view(1, 1, 1, -1)  # Shape: (1, 1, 1, l)
    n_agg_wt_expanded = n_agg_wt.view(1, 1, 1, -1)  # Shape: (1, 1, 1, l)
    
    # Apply weights to positive and negative components
    wt_mat = p_ind * p_agg_wt_expanded - n_ind * n_agg_wt_expanded
    
    # Sum across the last dimension
    wt_mat = torch.sum(wt_mat, dim=-1)
    
    return wt_mat
