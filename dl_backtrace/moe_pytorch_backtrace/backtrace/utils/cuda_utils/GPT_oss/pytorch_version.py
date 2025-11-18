import torch
from typing import Dict, Any, Tuple, Optional

def torch_swish(x: torch.Tensor, beta: float = 0.75) -> torch.Tensor:
    x_beta = (beta * x)
    x_beta_clipped = torch.clamp(x_beta, -500.0, 500.0)
    sigmoid = torch.sigmoid(x_beta_clipped)
    return x * sigmoid

def stabilize(matrix: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    abs_matrix = torch.abs(matrix)
    sign_matrix = torch.where(matrix == 0, 
                             torch.ones_like(matrix), 
                             torch.sign(matrix))
    
    return torch.where(abs_matrix < epsilon, 
                      epsilon * sign_matrix, 
                      matrix)

def torch_topk(arr: torch.Tensor, k: int, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    if axis != -1:
        raise NotImplementedError("Only axis=-1 supported for topk")
    return torch.topk(arr, k, dim=axis, sorted=True)

def torch_softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    x_shifted = x - x.max(dim=axis, keepdim=True).values
    exp_x = torch.exp(torch.clamp(x_shifted, -50.0, 50.0))
    denom = exp_x.sum(dim=axis, keepdim=True)
    out = exp_x / torch.clamp(denom, min=1e-30)
    return out

def calculate_wt_lm_head(
    wts: torch.Tensor,
    inp: torch.Tensor,
    w: Dict[str, torch.Tensor],
    chunk_rows: int = 4096,
    eps: float = 1e-12
) -> torch.Tensor:

    # Convert to float32 for computation if needed (handles BFloat16)
    compute_dtype = torch.float32
    wts = wts.to(compute_dtype)
    inp = inp.to(compute_dtype)
    w = {k: v.to(compute_dtype) if isinstance(v, torch.Tensor) else v for k, v in w.items()}
    
    W = w['W_lm_head']  # (V, D)
    B, T, V = wts.shape
    _, _, D = inp.shape
    assert W.shape == (V, D), f"Weight shape mismatch: {W.shape} vs expected {(V, D)}"
    
    # Flatten batch and time dimensions for vectorized processing
    wts_flat = wts.reshape(B * T, V)  # (B*T, V)
    inp_flat = inp.reshape(B * T, D)  # (B*T, D)
    
    # Pre-allocate output buffer
    relevance_flat = torch.zeros(B * T, D, dtype=torch.float32, device=inp.device)
    
    # Process vocabulary in chunks to manage memory
    for start in range(0, V, chunk_rows):
        end = min(start + chunk_rows, V)
        
        # Extract chunk of weights and relevance scores
        W_chunk = W[start:end, :]  # (C, D) where C = chunk size
        R_chunk = wts_flat[:, start:end]  # (B*T, C)
        
        # Compute element-wise products via broadcasting: (B*T, C, D)
        L = inp_flat.unsqueeze(1) * W_chunk.unsqueeze(0)
        
        # Separate positive and negative contributions
        L_pos = torch.clamp(L, min=0.0)  # (B*T, C, D)
        L_neg = torch.clamp(L, max=0.0)  # (B*T, C, D)
        
        # Sum across hidden dimension with keepdim for broadcasting
        p_sum = L_pos.sum(dim=2, keepdim=True)  # (B*T, C, 1)
        n_sum = (-L_neg).sum(dim=2, keepdim=True)  # (B*T, C, 1)
        
        # Denominator for normalization
        denom = p_sum + n_sum + eps  # (B*T, C, 1)
        
        # Aggregate weights for positive and negative parts
        p_agg = torch.where(p_sum > 0, p_sum / denom, torch.tensor(0.0, device=inp.device))
        n_agg = torch.where(n_sum > 0, n_sum / denom, torch.tensor(0.0, device=inp.device))
        
        # Normalize positive and negative contributions
        p_norm = torch.where(p_sum > 0, L_pos / (p_sum + eps), torch.tensor(0.0, device=inp.device))
        n_norm = torch.where(n_sum > 0, L_neg / (n_sum + eps), torch.tensor(0.0, device=inp.device))
        
        # Combine contributions weighted by relevance scores
        R_expanded = R_chunk.unsqueeze(2)  # (B*T, C, 1)
        contrib = p_norm * (R_expanded * p_agg) + (-n_norm) * (R_expanded * n_agg)
        
        # Accumulate contributions across vocabulary chunk
        relevance_flat.add_(contrib.sum(dim=1))  # (B*T, D)
    
    # Reshape back to original dimensions
    relevance_input = relevance_flat.reshape(B, T, D)
    
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

def gpt_oss_moe_mlp_forward(
    hidden_states: torch.Tensor,
    w: Dict[str, torch.Tensor],
    config: Any
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    B, S, H = hidden_states.shape
    top_k = config.num_experts_per_tok
    hidden_dim = config.hidden_size
    
    # Convert to float32 for computation if needed (handles BFloat16)
    compute_dtype = torch.float32
    hidden_states = hidden_states.to(compute_dtype)
    w = {k: v.to(compute_dtype) if isinstance(v, torch.Tensor) else v for k, v in w.items()}
    
    hidden_states_flat = hidden_states.reshape(-1, hidden_dim)
    
    # Router
    router_logits = torch.einsum('th,eh->te', hidden_states_flat, w['W_router']) + w['b_router']
    router_top_value_raw, router_indices = torch.topk(router_logits, k=top_k, dim=-1)
    
    # Softmax with numerical stability
    routing_weights_full = torch_softmax(router_top_value_raw, axis=1)

    # Scatter routing scores
    router_scores = torch.zeros_like(router_logits)
    rows = torch.arange(router_logits.shape[0], device=router_logits.device).unsqueeze(1)
    router_scores[rows, router_indices] = routing_weights_full
    
    # Expert computations
    alpha = 1.702
    limit = 7.0
    
    gate_up = torch.einsum('nh,ehd->end', hidden_states_flat, w['W_gate_up_proj']) + w['b_gate_up_proj'].unsqueeze(1)
    
    gate = gate_up[..., 0::2]
    up = gate_up[..., 1::2]
    
    gate = torch.clamp(gate, max=limit)
    up = torch.clamp(up, min=-limit, max=limit)
    
    glu = gate * torch.sigmoid(gate * alpha)
    pre = (up + 1.0) * glu
    
    next_state = torch.einsum('end,edh->enh', pre, w['W_down_proj']) + w['b_down_proj'].unsqueeze(1)
    
    # Mix experts
    wts = router_scores.T.unsqueeze(2)
    y = (next_state * wts).sum(dim=0)
    routed_out = y.reshape(B, S, H)
    
    intermediates = {
        'router_logits': router_logits,
        'router_top_value_raw': router_top_value_raw,
        'router_top_value': routing_weights_full,
        'router_indices': router_indices,
        'router_scores': router_scores,
        'gate_up': gate_up,
        'gate': gate,
        'up': up,
        'glu': glu,
        'pre': pre,
        'next_state': next_state,
        'wts': wts,
        'y': y,
        'routed_out': routed_out,
    }
    
    return intermediates

@torch.compile(mode="default", fullgraph=False)
def calculate_wt_gpt_oss_feed_forward_parallel(
    wts: torch.Tensor,
    inp: torch.Tensor,
    w: Dict[str, torch.Tensor],
    config: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
 
    num_experts = int(config.num_local_experts)
    B, T, H = inp.shape
    tokens = B * T
    device = inp.device
    
    # Convert to float32 for computation if needed (handles BFloat16)
    original_dtype = wts.dtype
    compute_dtype = torch.float32
    wts = wts.to(compute_dtype)
    inp = inp.to(compute_dtype)
    w = {k: v.to(compute_dtype) if isinstance(v, torch.Tensor) else v for k, v in w.items()}
    
    # Forward pass
    inter = gpt_oss_moe_mlp_forward(inp, w, config)
    
    selected_experts = inter['router_indices']
    routing_topk = inter['router_top_value']
    next_state_enh = inter['next_state']
    pre_e_nd = inter['pre']
    
    # Flatten relevance and inputs
    R_out = wts.reshape(tokens, H)
    hs = inp.reshape(tokens, H)
    
    # Gather expert outputs for selected experts
    t_idx = torch.arange(tokens, device=device).unsqueeze(1)
    e_idx = selected_experts
    z = next_state_enh[e_idx, t_idx, :] * routing_topk.unsqueeze(2)
    
    # Allocation using absolute value rule
    z_abs = torch.abs(z)
    sum_abs = z_abs.sum(dim=1)
    eps = 1e-30
    alloc = z_abs / torch.clamp(sum_abs.unsqueeze(1), min=eps)
    
    # Fallback to routing weights where sum_abs is zero
    fallback_mask = sum_abs <= eps
    if fallback_mask.any():
        rt_expanded = routing_topk.unsqueeze(2).expand(-1, -1, H)
        rt_normalized = rt_expanded / torch.clamp(rt_expanded.sum(dim=1, keepdim=True), min=eps)
        alloc = torch.where(fallback_mask.unsqueeze(1), rt_normalized, alloc)
    
    # Per-slot relevance
    R_slot = alloc * R_out.unsqueeze(1)
    
    # Initialize outputs
    final_relevance_input = torch.zeros((tokens, H), dtype=torch.float32, device=device)
    relevance_expert = torch.zeros(num_experts, dtype=torch.float32, device=device)
    
    # Get unique experts
    expert_hit = torch.unique(selected_experts)
    
    # Process each expert
    for e in expert_hit:
        mask = selected_experts == e
        if not mask.any():
            continue
            
        tok_idx, slot_idx = torch.where(mask)
        M = tok_idx.shape[0]
        if M == 0:
            continue
        
        # Relevance for this expert path
        R_out_expert = R_slot[tok_idx, slot_idx, :]
        inter_out_D = pre_e_nd[e, tok_idx, :]
        
        # Backprop through down projection
        R_intermediate = calculate_relevance_proj(R_out_expert, inter_out_D)
        
        # Split to gate and up branches
        R_gate = 0.5 * R_intermediate
        R_up = 0.5 * R_intermediate
        
        # Backprop to input
        current_state = hs[tok_idx, :]
        R_in_gate = calculate_relevance_gated_proj(R_gate, current_state)
        R_in_up = calculate_relevance_proj(R_up, current_state)
        
        R_curr = R_in_gate + R_in_up
        
        # Accumulate
        final_relevance_input.index_add_(0, tok_idx, R_curr)
        relevance_expert[e] += R_curr.sum()
    
    final_relevance_input = final_relevance_input.reshape(B, T, H)
    
    # Keep output in float32 for compatibility with numpy conversion
    return final_relevance_input.to(torch.float32), relevance_expert.to(torch.float32)

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

def gpt_oss_gqa_forward(
    hidden_states: torch.Tensor,
    w: Dict[str, torch.Tensor],
    config: Any,
    causal: bool = True,
    eps: float = 1e-6,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    start: int = 0,
    attn_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    B, T, hidden = hidden_states.shape
    device = hidden_states.device
    
    # Convert to float32 for computation if needed (handles BFloat16)
    compute_dtype = torch.float32
    hidden_states = hidden_states.to(compute_dtype)
    w = {k: v.to(compute_dtype) if isinstance(v, torch.Tensor) else v for k, v in w.items()}
    dtype = compute_dtype
    
    # Extract config parameters
    H = int(getattr(config, "num_attention_heads"))
    Kv = int(getattr(config, "num_key_value_heads"))
    theta = float(getattr(config, "rope_theta", 1_000_000.0))
    
    # Q, K, V Projections - weights are (out, in) so transpose them
    q_flat = hidden_states @ w['W_q'].T  # (B, T, H*D)
    if 'b_q' in w:
        q_flat = q_flat + w['b_q']
    
    k_flat = hidden_states @ w['W_k'].T  # (B, T, Kv*D)
    if 'b_k' in w:
        k_flat = k_flat + w['b_k']
    
    v_flat = hidden_states @ w['W_v'].T  # (B, T, Kv*D)
    if 'b_v' in w:
        v_flat = v_flat + w['b_v']
    
    # Infer head dimension from Q projection
    D = q_flat.shape[-1] // H
    assert q_flat.shape[-1] == H * D and k_flat.shape[-1] == Kv * D and v_flat.shape[-1] == Kv * D
    
    # Reshape and apply optional per-head RMS normalization
    q = q_flat.view(B, T, H, D)
    if 'q_norm' in w:
        q = rmsnorm(q, w['q_norm'], eps=eps)
    
    k = k_flat.view(B, T, Kv, D)
    if 'k_norm' in w:
        k = rmsnorm(k, w['k_norm'], eps=eps)
    
    v = v_flat.view(B, T, Kv, D)
    
    # Transpose to (B, heads, T, D) for attention computation
    q = q.transpose(1, 2)  # (B, H, T, D)
    k = k.transpose(1, 2)  # (B, Kv, T, D)
    v = v.transpose(1, 2)  # (B, Kv, T, D)
    
    # Apply RoPE
    if cos is None or sin is None:
        cos, sin = build_rope_cos_sin(
            B, T, D, theta=theta, dtype=dtype, device=device,
            position_ids=position_ids, start=start
        )
    q, k = apply_rope(q, k, cos, sin)
    
    # GQA: Expand K, V from Kv heads to H heads
    groups = H // Kv
    if groups * Kv != H:
        raise ValueError(f"num_attention_heads ({H}) must be divisible by num_key_value_heads ({Kv})")
    
    # Use repeat_interleave for memory-efficient expansion
    k = k.repeat_interleave(groups, dim=1)  # (B, H, T, D)
    v = v.repeat_interleave(groups, dim=1)  # (B, H, T, D)
    
    # Compute attention scores: Q @ K^T
    scale = 1.0 / torch.sqrt(torch.tensor(D, dtype=dtype, device=device))
    QK_output = torch.einsum('bhtd,bhsd->bhts', q, k)  # (B, H, T, T)
    logits_unmasked = QK_output * scale
    
    # Apply attention mask if provided
    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        logits = logits_unmasked + attn_mask
    else:
        logits = logits_unmasked
    
    # Attention sinks mechanism (GPT-OSS trick)
    sink_prob = None
    alpha = None
    if 'W_sinks' in w:
        sinks = w['W_sinks'].view(1, H, 1, 1).to(dtype)
        # Concatenate sink logits
        combined = torch.cat([logits, sinks.expand(B, H, T, 1)], dim=-1)  # (B, H, T, T+1)
        probs_all = torch_softmax(combined, axis=-1)
        sink_prob = probs_all[..., -1]  # (B, H, T)
        A = probs_all[..., :-1]  # (B, H, T, T)
        alpha = 1.0 - sink_prob  # (B, H, T)
    else:
        A = torch_softmax(logits, axis=-1)  # (B, H, T, T)
    
    # Apply attention to values
    out_heads = torch.einsum('bhts,bhsd->bhtd', A, v)  # (B, H, T, D)
    
    # Merge heads back to (B, T, H*D)
    out = out_heads.transpose(1, 2).reshape(B, T, H * D)
    
    # Output projection
    attn_out = out @ w['W_d'].T  # (B, T, hidden_size)
    if 'b_d' in w:
        attn_out = attn_out + w['b_d']
    
    # Store intermediates for debugging/analysis
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
    
    if sink_prob is not None:
        intermediates['sink_prob'] = sink_prob
        intermediates['alpha'] = alpha
    
    return attn_out, intermediates

def _causal_mask_torch(T: int, dtype: torch.dtype = torch.float32, device: torch.device = None) -> torch.Tensor:

    m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    mask = torch.zeros(T, T, dtype=dtype, device=device)
    mask.masked_fill_(m, float('-inf'))
    return mask


def _sliding_causal_mask_torch(T: int, W: int, dtype: torch.dtype = torch.float32, 
                                device: torch.device = None) -> torch.Tensor:

    j = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    i = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
    allow = (j <= i) & (j >= (i - W + 1))
    mask = torch.full((T, T), float('-inf'), dtype=dtype, device=device)
    mask.masked_fill_(allow, 0.0)
    return mask


def collapse_to_kv_heads(R_bhtd: torch.Tensor, Kv: int, groups: int) -> torch.Tensor:

    B, H, T, D = R_bhtd.shape
    assert H == Kv * groups, f"H={H} must equal Kv*groups={Kv*groups}"
    
    # Reshape to (B, groups, Kv, T, D) and sum over groups dimension
    R_kv = R_bhtd.view(B, groups, Kv, T, D).sum(dim=1)  # (B, Kv, T, D)
    
    # Transpose and reshape to (B, T, Kv*D)
    return R_kv.permute(0, 2, 1, 3).reshape(B, T, Kv * D)


@torch.compile(mode="default", fullgraph=False)
def calculate_wt_self_attention_parallel_torch(
    wts: torch.Tensor,
    inp: torch.Tensor,
    w: Dict[str, torch.Tensor],
    config,
    attn_type: str = "full",
    sliding_window: Optional[int] = None,
    with_sink_return: bool = False
) -> torch.Tensor:

    B, T, H = inp.shape
    device = inp.device
    
    # Convert to float32 for computation if needed (handles BFloat16)
    compute_dtype = torch.float32
    wts = wts.to(compute_dtype)
    inp = inp.to(compute_dtype)
    w = {k: v.to(compute_dtype) if isinstance(v, torch.Tensor) else v for k, v in w.items()}
    
    # Build attention mask (small, O(T^2))
    if attn_type == "sliding" and sliding_window is not None:
        attn_mask = _sliding_causal_mask_torch(T, sliding_window, dtype=torch.float32, device=device)
    else:
        attn_mask = _causal_mask_torch(T, dtype=torch.float32, device=device)
    
    # Forward pass to get routing & expert caches
    with torch.no_grad():
        out, inter = gpt_oss_gqa_forward(inp, w, config, attn_mask=attn_mask)
    
    # -----------  Relevance Calculation ------------------
    # 1. attn_output -> out
    wt_mat_attn = calculate_relevance_single(wts, inter['out'], w['W_d'])
    wt_mat_out_heads = wt_mat_attn.reshape(inter['out_heads'].shape)
    
    # 2. Convert to float32 for numerical precision
    A = inter['A'].to(torch.float32)                    # (B,H,T,S)
    out_heads = inter['out_heads'].to(torch.float32)    # (B,H,T,D)
    W = wt_mat_out_heads.to(torch.float32)              # (B,H,T,D)
    
    # 3. Sink bookkeeping
    has_sink = 'alpha' in inter
    if has_sink:
        alpha = inter['alpha'].to(torch.float32)        # (B,H,T)
        alpha_bhtd = alpha.unsqueeze(-1)                # (B,H,T,1)
        R_sink = (1.0 - alpha_bhtd) * W                 # (B,H,T,D)
        wt_eff = alpha_bhtd * W                         # (B,H,T,D)
    else:
        R_sink = 0.0
        wt_eff = W
    

    eps: float = 1e-9
    
    # 4. Relevance calculation of R_QK and R_V
    relevance_norm_out_heads = wt_eff / stabilize(out_heads * 2, eps)
    
    # Convert inter tensors to float32
    v_64 = inter['v'].to(torch.float32)
    q_64 = inter['q'].to(torch.float32)
    k_64 = inter['k'].to(torch.float32)
    
    # R_QK: (B,H,T,S) - using torch.matmul for clarity
    R_QK = torch.matmul(relevance_norm_out_heads, v_64.transpose(-2, -1)) * A
    
    # R_V: (B,H,S,D)
    R_V = torch.matmul(A.transpose(-2, -1), relevance_norm_out_heads) * v_64
    
    # 5. Fold sink mass into QK (no double counting)
    if has_sink:
        # No-sink probabilities per row
        S = A / torch.clamp(alpha.unsqueeze(-1), min=1e-12)  # (B,H,T,S)
        
        # Signed sink mass per row (sum over D)
        m_sink = R_sink.sum(dim=-1)  # (B,H,T)
        
        # Base addition
        add = m_sink.unsqueeze(-1) * S  # (B,H,T,S)
        
        # Tiny per-row correction
        err = m_sink.unsqueeze(-1) - add.sum(dim=-1, keepdim=True)
        
        # Spread correction uniformly across nonzero S entries
        nz = (S > 0).to(torch.float32)
        nz_cnt = nz.sum(dim=-1, keepdim=True)
        corr = torch.where(nz_cnt > 0, err / torch.clamp(nz_cnt, min=1.0), 0.0) * nz
        
        R_QK = R_QK + add + corr
    
    # Optional sanity check
    if has_sink:
        _ = (add + corr).sum().item()  # Compute but don't print
    
    # 6. Signed conservation on V
    R_V = dlb_style_signed_conserve(R_V, v_64)
    
    # 7. Relevance calculation of R_Q and R_K
    QK_output_64 = inter['QK_output'].to(torch.float32)
    relevance_norm_QK_out = R_QK / stabilize(QK_output_64 * 2, eps)
    
    R_Q = torch.matmul(relevance_norm_QK_out, k_64) * q_64
    R_K = torch.matmul(q_64.transpose(-2, -1), relevance_norm_QK_out).transpose(-2, -1) * k_64
    
    # Apply signed conservation
    R_Q = dlb_style_signed_conserve(R_Q, q_64)
    R_K = dlb_style_signed_conserve(R_K, k_64)
    
    # 8. Collapse to kv_heads
    H_total = int(getattr(config, "num_attention_heads"))
    Kv = int(getattr(config, "num_key_value_heads"))
    groups = H_total // Kv
    
    R_K_kv = collapse_to_kv_heads(R_K, Kv, groups)  # (B,T,Kv*D)
    R_V_kv = collapse_to_kv_heads(R_V, Kv, groups)  # (B,T,Kv*D)
    
    # 9. Reshape R_Q and calculate input relevance
    R_Q = R_Q.permute(0, 2, 1, 3).reshape(B, T, -1)
    
    input_relevance_from_Q = calculate_relevance_single(R_Q, inp, w['W_q'])
    input_relevance_from_K = calculate_relevance_single(R_K_kv, inp, w['W_k'])
    input_relevance_from_V = calculate_relevance_single(R_V_kv, inp, w['W_v'])
    
    input_relevance = input_relevance_from_Q + input_relevance_from_K + input_relevance_from_V
    
    # Keep output in float32 for compatibility with numpy conversion
    return input_relevance.to(torch.float32)