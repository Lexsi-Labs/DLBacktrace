import torch
from typing import Dict, Optional, Union, Callable, Any

@torch.compile
def calculate_wt_fc(
    relevance_y: torch.Tensor,
    input_array: torch.Tensor, 
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    act: Dict[str, Any]
) -> torch.Tensor:
    """
    DL Backtrace-style relevance tracing for fully connected (linear) layers.
    
    Optimized PyTorch implementation that vectorizes the relevance propagation
    computation across batch and neuron dimensions for improved performance.
    
    Args:
        relevance_y: Relevance at the output, same shape as linear output
        input_array: Input to the linear layer, shape [..., input_dim]
        w: Weight matrix of the linear layer, shape [output_dim, input_dim]
        b: Bias vector, shape [output_dim], or None
        act: Dictionary containing activation info with keys:
            - "type": str, either "mono" or "non_mono"
            - "range": dict with "l" (lower) and "u" (upper) bounds or None
            - "func": callable activation function (for non_mono type)
    
    Returns:
        torch.Tensor: Relevance at the input, same shape as input_array
    """
    device = input_array.device
    dtype = input_array.dtype
    
    # Preserve original shape for reshaping at the end
    original_shape = input_array.shape
    batch_dims = original_shape[:-1]
    feature_dim = original_shape[-1]
    
    # Flatten inputs for vectorized processing
    input_flat = input_array.view(-1, feature_dim)  # [batch_size, input_dim]
    relevance_flat = relevance_y.view(-1, relevance_y.shape[-1])  # [batch_size, output_dim]
    
    batch_size, input_dim = input_flat.shape
    output_dim = relevance_flat.shape[1]
    
    # Vectorized contribution matrix computation
    # input_flat: [batch_size, input_dim] -> [batch_size, 1, input_dim]
    # w.T: [input_dim, output_dim] -> [1, input_dim, output_dim] 
    # Result: [batch_size, output_dim, input_dim]
    input_expanded = input_flat.unsqueeze(1)  # [batch_size, 1, input_dim]
    w_expanded = w.T.unsqueeze(0)  # [1, input_dim, output_dim]
    mul_mat = torch.einsum('bni,nio->boi', input_expanded, w_expanded)  # [batch_size, output_dim, input_dim]
    
    # Initialize weight matrix
    wt_mat = torch.zeros_like(mul_mat)  # [batch_size, output_dim, input_dim]
    
    # Vectorized positive/negative contribution separation
    p_mask = mul_mat > 0  # [batch_size, output_dim, input_dim]
    n_mask = mul_mat < 0
    
    # Compute positive and negative sums
    p_contribs = torch.where(p_mask, mul_mat, torch.zeros_like(mul_mat))
    n_contribs = torch.where(n_mask, mul_mat, torch.zeros_like(mul_mat))
    
    p_sum = p_contribs.sum(dim=-1)  # [batch_size, output_dim]
    n_sum = -n_contribs.sum(dim=-1)  # [batch_size, output_dim]
    
    # Handle bias vectorized
    if b is not None:
        bias_expanded = b.unsqueeze(0).expand(batch_size, -1)  # [batch_size, output_dim]
        pbias = torch.clamp(bias_expanded, min=0)
        nbias = -torch.clamp(bias_expanded, max=0)
    else:
        pbias = torch.zeros(batch_size, output_dim, device=device, dtype=dtype)
        nbias = torch.zeros(batch_size, output_dim, device=device, dtype=dtype)
    
    t_sum = p_sum + pbias - n_sum - nbias
    
    # Activation-aware handling (vectorized)
    if act["type"] == "mono":
        if act["range"]["l"] is not None:
            p_sum = torch.where(t_sum < act["range"]["l"], torch.zeros_like(p_sum), p_sum)
        if act["range"]["u"] is not None:
            n_sum = torch.where(t_sum > act["range"]["u"], torch.zeros_like(n_sum), n_sum)
            
    elif act["type"] == "non_mono":
        t_act = act["func"](t_sum)
        p_act = act["func"](p_sum + pbias)
        n_act = act["func"](-1 * (n_sum + nbias))
        
        if act["range"]["l"] is not None:
            p_sum = torch.where(t_sum < act["range"]["l"], torch.zeros_like(p_sum), p_sum)
        if act["range"]["u"] is not None:
            n_sum = torch.where(t_sum > act["range"]["u"], torch.zeros_like(n_sum), n_sum)
        
        # Vectorized comparison logic
        both_nonzero = (p_sum > 0) & (n_sum > 0)
        t_eq_p = torch.isclose(t_act, p_act) & both_nonzero
        t_eq_n = torch.isclose(t_act, n_act) & both_nonzero
        
        n_sum = torch.where(t_eq_p, torch.zeros_like(n_sum), n_sum)
        p_sum = torch.where(t_eq_n, torch.zeros_like(p_sum), p_sum)
    
    # Avoid divide by zero (vectorized)
    p_sum = torch.where(p_sum == 0, torch.ones_like(p_sum), p_sum)
    n_sum = torch.where(n_sum == 0, torch.ones_like(n_sum), n_sum)
    
    # Compute aggregated weights (vectorized)
    total_sum = p_sum + n_sum + pbias + nbias
    
    # Handle edge case where total_sum might be zero
    total_sum = torch.where(total_sum == 0, torch.ones_like(total_sum), total_sum)
    
    p_agg_wt = torch.where(
        p_sum > 0,
        ((p_sum + pbias) / total_sum) * (p_sum / (p_sum + pbias)),
        torch.zeros_like(p_sum)
    )
    
    n_agg_wt = torch.where(
        n_sum > 0,
        ((n_sum + nbias) / total_sum) * (n_sum / (n_sum + nbias)),
        torch.zeros_like(n_sum)
    )
    
    # Expand for broadcasting: [batch_size, output_dim] -> [batch_size, output_dim, 1]
    relevance_expanded = relevance_flat.unsqueeze(-1)  # [batch_size, output_dim, 1]
    p_agg_wt_expanded = p_agg_wt.unsqueeze(-1)
    n_agg_wt_expanded = n_agg_wt.unsqueeze(-1)
    p_sum_expanded = p_sum.unsqueeze(-1)
    n_sum_expanded = n_sum.unsqueeze(-1)
    
    # Vectorized relevance redistribution
    p_weights = torch.where(
        p_mask,
        (p_contribs / p_sum_expanded) * relevance_expanded * p_agg_wt_expanded,
        torch.zeros_like(mul_mat)
    )
    
    n_weights = torch.where(
        n_mask,
        (n_contribs / n_sum_expanded) * relevance_expanded * n_agg_wt_expanded * -1.0,
        torch.zeros_like(mul_mat)
    )
    
    wt_mat = p_weights + n_weights
    
    # Sum across output dimension to get final relevance
    relevance_x_flat = wt_mat.sum(dim=1)  # [batch_size, input_dim]
    
    # Reshape back to original input shape
    relevance_x = relevance_x_flat.view(*batch_dims, feature_dim)
    
    return relevance_x
