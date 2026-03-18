import torch
from typing import Dict, Any, Tuple, Optional

def torch_swish(x: torch.Tensor, beta: float = 0.75) -> torch.Tensor:
    x_beta = (beta * x)
    x_beta_clipped = torch.clamp(x_beta, -500.0, 500.0)
    sigmoid = torch.sigmoid(x_beta_clipped)
    return x * sigmoid

def torch_topk(arr: torch.Tensor, k: int, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    if axis != -1:
        raise NotImplementedError("Only axis=-1 supported for topk")
    return torch.topk(arr, k, dim=axis, sorted=True)

def stabilize(matrix: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    abs_matrix = torch.abs(matrix)
    sign_matrix = torch.where(matrix == 0, 
                             torch.ones_like(matrix), 
                             torch.sign(matrix))
    
    return torch.where(abs_matrix < epsilon, 
                      epsilon * sign_matrix, 
                      matrix)

@torch.compile
def calculate_wt_lm_head_vectorized(
    wts: torch.Tensor,
    inp: torch.Tensor,
    w: Dict[str, torch.Tensor]
) -> torch.Tensor:

    # Compute contribution matrix: (vocab_size, hidden_dim) × (batch, seq, hidden) -> (batch, seq, vocab, hidden)
    contribution_matrix = torch.einsum('vh,bsh->bsvh', w['W_lm_head'], inp)
    
    # Create masks for positive and negative contributions
    positive_mask = contribution_matrix > 0
    negative_mask = contribution_matrix < 0
    
    # Separate positive and negative contributions
    positive_contrib = torch.where(positive_mask, contribution_matrix, torch.zeros_like(contribution_matrix))
    negative_contrib = torch.where(negative_mask, contribution_matrix, torch.zeros_like(contribution_matrix))
    
    # Sum across hidden dimension
    positive_sum = positive_contrib.sum(dim=3)  # (batch, seq, vocab)
    negative_sum = negative_contrib.sum(dim=3) * -1  # (batch, seq, vocab)
    
    # Calculate total sum for normalization
    total_sum = positive_sum + negative_sum
    
    # Calculate aggregate weights for positive and negative contributions
    positive_agg_wt = torch.where(positive_sum > 0, positive_sum / total_sum, torch.zeros_like(positive_sum))
    negative_agg_wt = torch.where(negative_sum > 0, negative_sum / total_sum, torch.zeros_like(negative_sum))
    
    # Create safe denominators
    positive_sum_safe = torch.where(positive_sum != 0, positive_sum, torch.ones_like(positive_sum))
    negative_sum_safe = torch.where(negative_sum != 0, negative_sum, torch.ones_like(negative_sum))
    
    # Expand weights for broadcasting
    wts_expanded = wts.unsqueeze(3)  # (batch, seq, vocab, 1)
    
    # Calculate normalized contributions
    positive_normalized = (
        positive_contrib / positive_sum_safe.unsqueeze(3)
    ) * wts_expanded * positive_agg_wt.unsqueeze(3)
    
    negative_normalized = (
        negative_contrib / negative_sum_safe.unsqueeze(3)
    ) * wts_expanded * negative_agg_wt.unsqueeze(3) * -1.0
    
    # Combine weighted contributions
    weighted_contributions = positive_normalized + negative_normalized
    
    # Sum across vocabulary dimension to get relevance for input
    relevance_input = weighted_contributions.sum(dim=2)  # (batch, seq, hidden)
    
    return relevance_input

def calculate_relevance_proj(wts: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

    device = output.device
    
    # Create masks for positive and negative values
    p_mask = output > 0
    n_mask = output < 0
    
    # Extract positive and negative values
    p_vals = output[p_mask]
    n_vals = output[n_mask]
    
    # Compute sums
    p_sum = torch.sum(p_vals)
    n_sum = torch.sum(n_vals).abs()
    
    # Compute total sum
    total_sum = p_sum + n_sum
    
    p_agg_wt = torch.where(
        (total_sum > 0) & (p_sum > 0),
        p_sum / total_sum,
        torch.tensor(0.0, device=device)
    )
    n_agg_wt = torch.where(
        (total_sum > 0) & (n_sum > 0),
        n_sum / total_sum,
        torch.tensor(0.0, device=device)
    )
    
    # Safe denominators
    p_sum_safe = torch.where(p_sum != 0, p_sum, torch.tensor(1.0, device=device))
    n_sum_safe = torch.where(n_sum != 0, n_sum, torch.tensor(1.0, device=device))
    
    # Compute total weight
    total_wt = torch.sum(wts)
    
    # Initialize output tensor
    wt_mat_total = torch.zeros_like(output)
    
    if p_mask.any():
        wt_mat_total[p_mask] = torch.where(
            p_agg_wt > 0,
            (p_vals / p_sum_safe) * total_wt * p_agg_wt,
            torch.tensor(0.0, device=device)
        )
    
    if n_mask.any():
        wt_mat_total[n_mask] = torch.where(
            n_agg_wt > 0,
            (n_vals / n_sum_safe) * total_wt * n_agg_wt * -1.0,
            torch.tensor(0.0, device=device)
        )
    
    return wt_mat_total    

def calculate_relevance_gated_proj(
    wts: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:

    device = output.device
    
    # Create masks for positive and negative values
    pos_mask = output > 0
    neg_mask = output < 0
    
    # Extract positive and negative values
    pos_values = output[pos_mask]
    neg_values = output[neg_mask]
    
    # Calculate base sums
    pos_sum_base = torch.sum(pos_values)
    neg_sum_base = torch.sum(neg_values) * -1
    t_sum_base = pos_sum_base - neg_sum_base
    
    # Calculate activations once
    t_act = torch_swish(t_sum_base)
    p_act = torch_swish(pos_sum_base)
    n_act = torch_swish(-1 * neg_sum_base)
    
    # Determine conditions that apply to all weights
    threshold_condition = t_sum_base < -6
    both_positive = (pos_sum_base > 0) & (neg_sum_base > 0)
    
    # Apply conditional logic to determine effective sums
    # Initialize with base values
    pos_sum = pos_sum_base
    neg_sum = neg_sum_base
    
    if threshold_condition:
        pos_sum = torch.tensor(0.0, device=device)
    
    if both_positive:
        if t_act == p_act:
            neg_sum = torch.tensor(0.0, device=device)
        elif t_act == n_act:
            pos_sum = torch.tensor(0.0, device=device)
    
    # Calculate aggregation weights
    denominator = pos_sum + neg_sum
    pos_agg_wt = torch.where(pos_sum > 0, pos_sum / denominator, torch.tensor(0.0, device=device))
    neg_agg_wt = torch.where(neg_sum > 0, neg_sum / denominator, torch.tensor(0.0, device=device))
    
    # Normalization denominators (avoid division by zero)
    pos_sum_norm = torch.where(pos_sum != 0, pos_sum, torch.tensor(1.0, device=device))
    neg_sum_norm = torch.where(neg_sum != 0, neg_sum, torch.tensor(1.0, device=device))
    
    total_weight = torch.sum(wts)
    
    # Initialize output accumulator
    wt_mat_total = torch.zeros_like(output)
    
    # Compute contributions
    if pos_agg_wt != 0:
        pos_contrib = (pos_values / pos_sum_norm) * pos_agg_wt * total_weight
        wt_mat_total[pos_mask] = pos_contrib
    
    if neg_agg_wt != 0:
        neg_contrib = (neg_values / neg_sum_norm) * neg_agg_wt * total_weight * -1.0
        wt_mat_total[neg_mask] = neg_contrib
    
    return wt_mat_total

def calculate_wt_router_logits(
    wts: torch.Tensor,
    inp: torch.Tensor,
    W_router: torch.Tensor
) -> torch.Tensor:
    
    # Constants
    act_range_lower = -1.0
    act_range_upper = 2.0
    
    contribution_matrix = inp.unsqueeze(2) * W_router.unsqueeze(0)
    
    # Create masks for positive and negative contributions
    p_mask = contribution_matrix > 0
    n_mask = contribution_matrix < 0
    
    # Sum positive and negative contributions along last axis
    # (n_samples, n_features, n_features) -> (n_samples, n_features)
    p_sums = torch.sum(contribution_matrix * p_mask, dim=2)
    n_sums = torch.sum(contribution_matrix * n_mask, dim=2) * -1.0
    t_sums = p_sums - n_sums
    
    # Apply activation range filtering
    p_sums = torch.where(t_sums < act_range_lower, torch.zeros_like(p_sums), p_sums)
    n_sums = torch.where(t_sums > act_range_upper, torch.zeros_like(n_sums), n_sums)
    
    # Calculate aggregation weights
    total_sums = p_sums + n_sums
    p_agg_wts = torch.where(p_sums > 0, p_sums / total_sums, torch.zeros_like(p_sums))
    n_agg_wts = torch.where(n_sums > 0, n_sums / total_sums, torch.zeros_like(n_sums))
    
    # Normalize sums (avoid division by zero)
    p_sums_normalized = torch.where(p_sums != 0, p_sums, torch.ones_like(p_sums))
    n_sums_normalized = torch.where(n_sums != 0, n_sums, torch.ones_like(n_sums))
    
    # Calculate weights: (n_samples, n_features) -> (n_samples, n_features, 1)
    p_weights = (wts * p_agg_wts / p_sums_normalized).unsqueeze(2)
    n_weights = (wts * n_agg_wts / n_sums_normalized).unsqueeze(2)
    
    # Apply weights to contribution matrices
    wt_mat_p = contribution_matrix * p_mask * p_weights
    wt_mat_n = contribution_matrix * n_mask * n_weights * -1.0
    
    # Combine and sum along feature dimension
    wt_mat = wt_mat_p + wt_mat_n
    wt_mat_total = wt_mat.sum(dim=1)  # (n_samples, n_features)
    
    return wt_mat_total

def qwen_moe_mlp_forward(
    hidden_states: torch.Tensor,
    w: Dict[str, Any],
    config: Any
) -> Dict[str, Any]:

    top_k = int(config.num_experts_per_tok)
    norm_topk_prob = bool(config.norm_topk_prob)
    
    B, T, H = hidden_states.shape
    tokens = B * T
    
    hs = hidden_states.reshape(tokens, H)
    
    # Router computation
    W_gate = w['W_gate']
    router_logits = torch.einsum('th,eh->te', hs, W_gate)  # (tokens, E)
    
    # Softmax with numerical stability
    x_shifted = router_logits - router_logits.max(dim=1, keepdim=True).values
    exp_x = torch.exp(torch.clamp(x_shifted, -50.0, 50.0))
    denom = exp_x.sum(dim=1, keepdim=True)
    routing_weights_full = exp_x / torch.clamp(denom, min=1e-30)  # (tokens, E)
    
    # Top-k selection
    routing_topk_raw, selected_experts = torch_topk(
        routing_weights_full, top_k, axis=-1
    )  # both (tokens, k)
    
    # Normalize top-k weights if required
    if norm_topk_prob:
        denom_topk = torch.clamp(
            routing_topk_raw.sum(dim=-1, keepdim=True),
            min=1e-30
        )
        routing_topk = routing_topk_raw / denom_topk
    else:
        routing_topk = routing_topk_raw
    
    # Initialize intermediates dictionary
    intermediates = {
        'router_logits': router_logits,
        'softmax_routing_weights': routing_weights_full,
        'selected_experts': selected_experts,
        'routing_weights': routing_topk_raw,
        'norm_routing_weights': routing_topk,
    }
    
    # Get unique experts
    experts_used = torch.unique(selected_experts)
    intermediates['expert_hit'] = experts_used
    
    # Initialize contribution tensor
    contrib_full = torch.zeros((tokens, top_k, H), device=hs.device)
    
    # Process each expert
    for e_tensor in experts_used:
        
        # Create mask for this expert
        mask = (selected_experts == e_tensor)  # (tokens, k)
        
        if not mask.any():
            continue
        
        # Get token and slot indices
        tok_idx, slot_idx = torch.nonzero(mask, as_tuple=True)  # (M,), (M,)
        
        # Gather input tokens for this expert
        x = hs[tok_idx]  # (M, H)
        
        # Load expert weights
        Wg = w[f'{e_tensor}']['W_gate_proj'].to(torch.float32)   # (I, H)
        Wu = w[f'{e_tensor}']['W_up_proj'].to(torch.float32)     # (I, H)
        Wd = w[f'{e_tensor}']['W_down_proj'].to(torch.float32)   # (H, I)
        
        # Expert computation: gate projection
        gate = torch.einsum('mh,ih->mi', x, Wg)  # (M, I)
        
        # Expert computation: up projection
        up = torch.einsum('mh,ih->mi', x, Wu)    # (M, I)
        
        # Custom Swish activation with beta=0.75
        inter = torch_swish(gate, beta=0.75) * up  # (M, I)
        
        # Expert computation: down projection
        down = torch.einsum('mi,hi->mh', inter, Wd)  # (M, H)
        
        # Apply routing weights
        r = routing_topk[tok_idx, slot_idx].unsqueeze(-1)  # (M, 1)
        contrib = down * r  # (M, H)
        
        # Scatter contributions back
        contrib_full[tok_idx, slot_idx, :] = contrib
        
        # Store expert-specific intermediates
        intermediates[f'expert_{e_tensor}'] = {
            'tok_idx': tok_idx,
            'slot_idx': slot_idx,
            'current_state': x,
            'gate_proj_output': gate,
            'up_proj_output': up,
            'intermediate_output': inter,
            'down_proj_output': down,
            'contrib': contrib,
            'routing_scalar': r.squeeze(-1),  # (M,)
        }
    
    intermediates['contrib_full'] = contrib_full  # (tokens, k, H)
    
    return intermediates

@torch.compile
def calculate_wt_feed_forward(
    wts: torch.Tensor,
    inp: torch.Tensor,
    w: Dict[str, Any],
    config: Any
) -> Tuple[torch.Tensor, torch.Tensor]:

    num_experts = int(config.num_experts)
    B, T, H = inp.shape
    tokens = B * T
    eps = 1e-30

    # Forward pass through MoE
    inter = qwen_moe_mlp_forward(inp, w, config)

    # Reshape to token-level: (tokens, H)
    hidden_states = inp.reshape(tokens, H)
    R_out = wts.reshape(tokens, H)

    selected_experts = inter['selected_experts']  # (tokens, k)
    routing_topk = inter.get('norm_routing_weights', inter['routing_weights'])
    
    # Normalize routing weights
    routing_topk = routing_topk / torch.clamp(routing_topk.sum(dim=-1, keepdim=True), min=eps)

    tokens_chk, k = selected_experts.shape
    assert tokens_chk == tokens, f"Token count mismatch: {tokens_chk} vs {tokens}"

    contrib_full = inter['contrib_full']  # (tokens, k, H)
    z_abs = torch.abs(contrib_full)
    sum_abs = z_abs.sum(dim=1)  # (tokens, H)

    # Compute allocation using absolute contributions
    alloc = z_abs / torch.clamp(sum_abs.unsqueeze(1), min=eps)  # (tokens, k, H)

    # Handle fallback: where sum_abs is near zero, use routing weights
    fallback_mask = (sum_abs <= eps)  # (tokens, H)
    
    if fallback_mask.any():
        rt_expanded = routing_topk.unsqueeze(2).expand(-1, -1, H)  # (tokens, k, H)
        rt_expanded = rt_expanded / torch.clamp(rt_expanded.sum(dim=1, keepdim=True), min=eps)
        
        # Apply fallback using mask: broadcast (tokens, 1, H) to (tokens, k, H)
        mask3 = fallback_mask.unsqueeze(1)  # (tokens, 1, H)
        alloc = torch.where(mask3, rt_expanded, alloc)

    # Distribute output relevance to slots
    R_slot = alloc * R_out.unsqueeze(1)  # (tokens, k, H)

    # Initialize output tensors
    final_relevance_input = torch.zeros_like(hidden_states)  # (tokens, H)
    relevance_expert = torch.zeros(num_experts, device=hidden_states.device)
    expert_hit = inter['expert_hit']

    # Vectorized expert processing
    # Collect all expert data for batch processing
    all_tok_idx = []
    all_slot_idx = []
    all_expert_ids = []
    
    for e in expert_hit:
        ed = inter[f'expert_{e}']
        tok_idx = ed['tok_idx']
        slot_idx = ed['slot_idx']
        
        if tok_idx.numel() == 0:
            continue
            
        all_tok_idx.append(tok_idx)
        all_slot_idx.append(slot_idx)
        all_expert_ids.append(torch.full_like(tok_idx, e))
    
    if all_tok_idx:
        # Concatenate all indices for batch processing
        all_tok_idx = torch.cat(all_tok_idx)
        all_slot_idx = torch.cat(all_slot_idx)
        all_expert_ids = torch.cat(all_expert_ids)
        
        # Batch gather all R_out_expert values
        R_out_expert_all = R_slot[all_tok_idx, all_slot_idx, :]  # (total_M, H)
        
        # Process each expert's contribution (still need loop for expert-specific data)
        offset = 0
        for e in expert_hit:
            ed = inter[f'expert_{e}']
            tok_idx = ed['tok_idx']
            slot_idx = ed['slot_idx']
            
            if tok_idx.numel() == 0:
                continue
            
            M = tok_idx.numel()
            R_out_expert = R_out_expert_all[offset:offset + M]
            
            inter_out = ed['intermediate_output']  # (M, I)
            current_state = ed['current_state']    # (M, H)

            # Relevance calculations
            R_intermediate = calculate_relevance_proj(R_out_expert, inter_out)  # (M, I)
            R_gate = 0.5 * R_intermediate
            R_up = 0.5 * R_intermediate

            R_in_gate = calculate_relevance_gated_proj(R_gate, current_state)  # (M, H)
            R_in_up = calculate_relevance_proj(R_up, current_state)
            
            R_curr = R_in_gate + R_in_up  # (M, H)

            # Scatter add to final relevance
            final_relevance_input.index_add_(0, tok_idx, R_curr)
            
            # Accumulate expert relevance
            relevance_expert[e] += R_curr.sum()
            
            offset += M

    router_fraction = 0.0
    
    if router_fraction > 0.0:
        # Compute per-slot scalar relevance
        per_slot_scalar = R_slot.sum(dim=-1) * router_fraction  # (tokens, k)
        
        # Vectorized routing mass computation
        routing_mass_te = torch.zeros((tokens, num_experts), device=hidden_states.device)
        
        # Flatten and use advanced indexing for scatter
        tok_indices = torch.arange(tokens, device=hidden_states.device).unsqueeze(1).expand(-1, k).reshape(-1)
        expert_indices = selected_experts.reshape(-1)
        values = per_slot_scalar.reshape(-1)
        
        nz_mask = values != 0
        if nz_mask.any():
            routing_mass_te.index_put_(
                (tok_indices[nz_mask], expert_indices[nz_mask]),
                values[nz_mask],
                accumulate=True
            )
        
        R_router_inp = calculate_wt_router_logits(
            routing_mass_te, hidden_states, w['W_gate']
        )
        final_relevance_input += R_router_inp

    final_relevance_input = final_relevance_input.reshape(B, T, H)
    
    return final_relevance_input, relevance_expert

def calculate_relevance_single(
    wts: torch.Tensor, 
    inp: torch.Tensor, 
    w: torch.Tensor
) -> torch.Tensor:

    batch_size, seq_len, output_features = wts.shape
    _, _, input_features = inp.shape
    device = wts.device
    
    # Vectorized contribution matrix computation
    # w: (output_features, input_features)
    # inp: (batch_size, seq_len, input_features)
    # contribution_matrix: (batch_size, seq_len, output_features, input_features)
    contribution_matrix = w.unsqueeze(0).unsqueeze(0) * inp.unsqueeze(2)
    
    # Create positive and negative masks
    p_mask = contribution_matrix > 0
    n_mask = contribution_matrix < 0
    
    # Compute positive and negative sums efficiently
    p_contributions = contribution_matrix * p_mask
    n_contributions = contribution_matrix * n_mask
    
    # Sum over input_features dimension
    p_sums = p_contributions.sum(dim=3)  # (batch_size, seq_len, output_features)
    n_sums = -n_contributions.sum(dim=3)  # (batch_size, seq_len, output_features)
    
    # Calculate aggregation weights
    total_sums = p_sums + n_sums
    p_agg_wts = torch.where(p_sums > 0, p_sums / total_sums, torch.zeros_like(p_sums))
    n_agg_wts = torch.where(n_sums > 0, n_sums / total_sums, torch.zeros_like(n_sums))
    
    # Safe division denominators
    p_sums_safe = torch.where(p_sums != 0, p_sums, torch.ones_like(p_sums))
    n_sums_safe = torch.where(n_sums != 0, n_sums, torch.ones_like(n_sums))
    
    # Expand dimensions for broadcasting
    R_expanded = wts.unsqueeze(3)  # (batch_size, seq_len, output_features, 1)
    p_agg_wts_expanded = p_agg_wts.unsqueeze(3)  # (batch_size, seq_len, output_features, 1)
    n_agg_wts_expanded = n_agg_wts.unsqueeze(3)  # (batch_size, seq_len, output_features, 1)
    p_sums_safe_expanded = p_sums_safe.unsqueeze(3)  # (batch_size, seq_len, output_features, 1)
    n_sums_safe_expanded = n_sums_safe.unsqueeze(3)  # (batch_size, seq_len, output_features, 1)
    
    # Calculate positive relevance contributions
    p_relevance = torch.where(
        p_mask,
        (contribution_matrix / p_sums_safe_expanded) * R_expanded * p_agg_wts_expanded,
        torch.zeros_like(contribution_matrix)
    )
    
    # Calculate negative relevance contributions
    n_relevance = torch.where(
        n_mask,
        -(contribution_matrix / n_sums_safe_expanded) * R_expanded * n_agg_wts_expanded,
        torch.zeros_like(contribution_matrix)
    )
    
    # Sum positive and negative contributions and aggregate over output_features dimension
    relevance_input = (p_relevance + n_relevance).sum(dim=2)  # (batch_size, seq_len, input_features)
    
    return relevance_input

def dlb_style_nonneg_conserve(wts: torch.Tensor, inp: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:

    assert wts.shape == inp.shape
    
    # Use boolean masks for efficiency
    pos = inp > 0
    neg = inp < 0

    # Vectorized sum operations with keepdims
    p_sum = torch.sum(torch.where(pos, inp, 0.0), dim=(-2, -1), keepdim=True)
    n_sum = -torch.sum(torch.where(neg, inp, 0.0), dim=(-2, -1), keepdim=True)
    denom = p_sum + n_sum + eps

    # Conditional shares using torch.where
    p_share = torch.where(p_sum > 0, p_sum / denom, 0.0)
    n_share = torch.where(n_sum > 0, n_sum / denom, 0.0)

    # Mass to conserve
    M = torch.sum(torch.abs(wts), dim=(-2, -1), keepdim=True)

    # Division stabilization
    p_div = torch.where(p_sum == 0, 1.0, p_sum)
    n_div = torch.where(n_sum == 0, 1.0, n_sum)

    # Vectorized output computation
    out = torch.zeros_like(inp)
    out = out + torch.where(pos, (inp / p_div) * (p_share * M), 0.0)
    out = out + torch.where(neg, (inp / n_div) * (n_share * M) * (-1.0), 0.0)
    
    return out


def dlb_style_signed_conserve(wts: torch.Tensor, inp: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:

    Rp = torch.clamp(wts, min=0.0)
    Rn = torch.clamp(-wts, min=0.0)

    P = dlb_style_nonneg_conserve(Rp, inp, eps)
    N = dlb_style_nonneg_conserve(Rn, inp, eps)

    return P - N

def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.reciprocal(torch.sqrt(variance + eps))
    return (x_normed * weight).to(x.dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    
    return q_embed, k_embed

def build_rope_cos_sin(
    B: int,
    T: int, 
    D: int,
    theta: float = 1_000_000.0,
    dtype: torch.dtype = torch.float32,
    position_ids: Optional[torch.Tensor] = None,
    start: int = 0,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:

    if device is None:
        device = torch.device('cuda')
    
    if position_ids is None:
        pos = start + torch.arange(T, dtype=torch.int32, device=device).unsqueeze(0).expand(B, -1)
    else:
        pos = position_ids
        if pos.dim() == 1:
            pos = pos.unsqueeze(0).expand(B, -1)
        if pos.shape != (B, T):
            raise ValueError(f"position_ids must be (B,T) or (T,), got {pos.shape}, expected ({B},{T})")
    
    half_dim = D // 2
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=dtype, device=device) / max(1, half_dim)))
    
    # Compute position embeddings
    freqs = torch.einsum('bt,d->btd', pos.to(dtype), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    
    return emb.cos().to(dtype), emb.sin().to(dtype)

def qwen_gqa_forward(
    hidden_states: torch.Tensor,
    w: Dict[str, torch.Tensor],
    config: Any,
    causal: bool = True,
    eps: float = 1e-6,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    start: int = 0
) -> Dict[str, torch.Tensor]:

    B, T, hidden = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype
    
    # Extract config parameters
    H = int(getattr(config, "num_attention_heads"))
    Kv = int(getattr(config, "num_key_value_heads"))
    D = config.hidden_size // config.num_attention_heads
    theta = float(getattr(config, "rope_theta", 1_000_000.0))
    
    # Project to Q, K, V using optimized einsum
    q_flat = torch.einsum('bts,os->bto', hidden_states, w['W_q'])
    k_flat = torch.einsum('bts,os->bto', hidden_states, w['W_k'])
    v_flat = torch.einsum('bts,os->bto', hidden_states, w['W_v'])
    
    # Verify dimensions
    D = q_flat.shape[-1] // H
    assert q_flat.shape[-1] == H * D and k_flat.shape[-1] == Kv * D and v_flat.shape[-1] == Kv * D
    
    # Reshape and apply RMSNorm to Q and K
    q = rmsnorm(q_flat.view(B, T, H, D), w['q_norm'], eps=eps)
    k = rmsnorm(k_flat.view(B, T, Kv, D), w['k_norm'], eps=eps)
    v = v_flat.view(B, T, Kv, D)
    
    # Transpose for attention computation: (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Build or use provided RoPE embeddings
    if cos is None or sin is None:
        cos, sin = build_rope_cos_sin(
            B, T, D, theta=theta, dtype=dtype,
            position_ids=position_ids, start=start, device=device
        )
    
    # Apply RoPE
    q, k = apply_rope(q, k, cos, sin)
    
    # Handle grouped query attention (repeat K, V for each group)
    groups = H // Kv
    if groups * Kv != H:
        raise ValueError(f"H ({H}) must be divisible by Kv ({Kv})")
    
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    
    # Scaled dot-product attention
    scale = 1.0 / torch.sqrt(torch.tensor(D, dtype=torch.float32, device=q.device))
    
    # Compute attention scores
    QK_output = torch.einsum('bhtd,bhsd->bhts', q, k)
    logits_unmasked = QK_output * scale
    
    # Apply softmax
    x_shifted = logits_unmasked - logits_unmasked.max(dim=1, keepdim=True).values
    exp_x = torch.exp(torch.clamp(x_shifted, -50.0, 50.0))
    denom = exp_x.sum(dim=1, keepdim=True)
    A = exp_x / torch.clamp(denom, min=1e-30) 
    
    # Apply attention to values
    out_heads = torch.einsum('bhts,bhsd->bhtd', A, v)
    
    # Reshape and concatenate heads
    out = out_heads.transpose(1, 2).contiguous().view(B, T, H * D)
    
    # Final output projection
    attn_out = torch.einsum('btm,hm->bth', out, w['W_d'])
    
    # Return intermediates for debugging/analysis
    intermediates = {
        'q_flat': q_flat,
        'k_flat': k_flat,
        'v_flat': v_flat,
        'q': q,
        'k': k,
        'v': v,
        'QK_output': QK_output,
        'logits_unmasked': logits_unmasked,
        'A': A,
        'out_heads': out_heads,
        'out': out,
        'attn_out': attn_out,
    }
    
    return intermediates

def collapse_to_kv_heads(
    R_bhtd: torch.Tensor,
    num_kv_heads: int,
    groups: int
) -> torch.Tensor:
    B, H, T, D = R_bhtd.shape
    assert H == num_kv_heads * groups, f"Expected H={num_kv_heads * groups}, got H={H}"
    R_grouped = R_bhtd.reshape(B, groups, num_kv_heads, T, D)
    R_kv = R_grouped.sum(dim=1)
    return R_kv.permute(0, 2, 1, 3).reshape(B, T, num_kv_heads * D)

@torch.compile
def calculate_wt_self_attention(
    wts: torch.Tensor,
    inp: torch.Tensor,
    w: Dict[str, torch.Tensor],
    config: Any,
    eps: float = 1e-9
) -> torch.Tensor:

    B, T, H = inp.shape
    
    # Forward pass through GQA attention
    inter = qwen_gqa_forward(inp, w, config)
    
    # Calculate relevance for attention output
    wt_mat_attn = calculate_relevance_single(wts, inter['out'], w['W_d'])
    
    # Reshape to head dimension
    wt_mat_out_heads = wt_mat_attn.reshape(inter['out_heads'].shape)
    
    # Normalize relevance by stabilized output heads
    relevance_norm_out_heads = wt_mat_out_heads / stabilize(inter['out_heads'] * 2, eps)
    
    # Propagate relevance through attention mechanism
    # R_QK: relevance from attention weights to QK product
    R_QK = torch.matmul(
        relevance_norm_out_heads,
        inter['v'].transpose(-2, -1)
    ) * inter['A']
    
    # R_V: relevance to value vectors
    R_V = torch.matmul(
        inter['A'].transpose(-2, -1),
        relevance_norm_out_heads
    ) * inter['v']
    
    # Apply signed conservation to value relevance
    R_V = dlb_style_signed_conserve(R_V, inter['v'])
    
    # Normalize relevance for QK output
    relevance_norm_QK_out = R_QK / stabilize(inter['QK_output'] * 2, eps)
    
    # Propagate relevance to query and key
    R_Q = torch.matmul(relevance_norm_QK_out, inter['k']) * inter['q']
    R_K = torch.matmul(
        inter['q'].transpose(-2, -1),
        relevance_norm_QK_out
    ).transpose(-2, -1) * inter['k']
    
    # Apply signed conservation
    R_Q = dlb_style_signed_conserve(R_Q, inter['q'])
    R_K = dlb_style_signed_conserve(R_K, inter['k'])
    
    # Get attention configuration
    num_heads = int(getattr(config, "num_attention_heads"))
    num_kv_heads = int(getattr(config, "num_key_value_heads"))
    groups = num_heads // num_kv_heads
    
    # Collapse query heads (multi-head) to KV heads (GQA)
    R_K_kv = collapse_to_kv_heads(R_K, num_kv_heads, groups)
    R_V_kv = collapse_to_kv_heads(R_V, num_kv_heads, groups)
    
    # Reshape query relevance from (B, H, T, D) to (B, T, H*D)
    R_Q = R_Q.transpose(1, 2).reshape(B, T, -1)
    
    # Calculate input relevance from each component
    input_relevance_from_Q = calculate_relevance_single(R_Q, inp, w['W_q'])
    input_relevance_from_K = calculate_relevance_single(R_K_kv, inp, w['W_k'])
    input_relevance_from_V = calculate_relevance_single(R_V_kv, inp, w['W_v'])
    
    # Aggregate all relevance contributions
    input_relevance = input_relevance_from_Q + input_relevance_from_K + input_relevance_from_V
    
    return input_relevance


