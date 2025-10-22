import numpy as np
from typing import Dict, Any, Tuple, Optional

def np_sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -500, 500)
    return 1 / (1 + np.exp(-x))

def np_swish(x, beta=0.75):
    return x * np_sigmoid(beta * x)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(np.clip(x_shifted, -50.0, 50.0))
    denom = np.sum(exp_x, axis=axis, keepdims=True)
    out = exp_x / np.maximum(denom, 1e-30)
    return out

def stabilize(matrix, epsilon=1e-6):
    return np.where(np.abs(matrix) < epsilon,
                    epsilon * np.sign(matrix + (matrix == 0)),
                    matrix)

def calculate_wt_lm_head_vectorized(
    wts: np.ndarray, 
    inp: np.ndarray, 
    w: Dict[str, np.ndarray]
) -> np.ndarray:

    contribution_matrix = np.einsum('vh,bsh->bsvh', w['W_lm_head'], inp)

    # Shape: (batch_size, seq_len, vocab_size, hidden_dim)
    positive_mask = contribution_matrix > 0
    negative_mask = contribution_matrix < 0
    
    positive_contrib = np.where(positive_mask, contribution_matrix, 0)
    negative_contrib = np.where(negative_mask, contribution_matrix, 0)
    
    positive_sum = np.sum(positive_contrib, axis=3)
    negative_sum = np.sum(negative_contrib, axis=3) * -1 
    
    total_sum = positive_sum + negative_sum
    
    positive_agg_wt = np.where(positive_sum > 0, positive_sum / total_sum, 0)
    negative_agg_wt = np.where(negative_sum > 0, negative_sum / total_sum, 0)
    
    positive_sum_safe = np.where(positive_sum != 0, positive_sum, 1)
    negative_sum_safe = np.where(negative_sum != 0, negative_sum, 1)
    
    wts_expanded = wts[:, :, :, None]
    
    positive_normalized = (
        positive_contrib / positive_sum_safe[:, :, :, None]
    ) * wts_expanded * positive_agg_wt[:, :, :, None]
    
    negative_normalized = (
        negative_contrib / negative_sum_safe[:, :, :, None]
    ) * wts_expanded * negative_agg_wt[:, :, :, None] * -1.0
    
    weighted_contributions = positive_normalized + negative_normalized
    
    relevance_input = np.sum(weighted_contributions, axis=2)
    
    return relevance_input

def calculate_relevance_proj(wts: np.ndarray, output: np.ndarray) -> np.ndarray:
    p_mask = output > 0
    n_mask = output < 0
    
    p_vals = output[p_mask]
    n_vals = output[n_mask]
    
    p_sum = np.sum(p_vals)
    n_sum = np.sum(n_vals) * -1
    
    total_sum = p_sum + n_sum
    if total_sum > 0:
        p_agg_wt = np.where(p_sum > 0, p_sum / total_sum, 0)
        n_agg_wt = np.where(n_sum > 0, n_sum / total_sum, 0)
    else:
        p_agg_wt = 0
        n_agg_wt = 0
    
    p_sum_safe = np.where(p_sum != 0, p_sum, 1)
    n_sum_safe = np.where(n_sum != 0, n_sum, 1)
    
    total_wt = np.sum(wts)
    
    wt_mat_total = np.zeros(output.shape)
    
    if p_agg_wt > 0:
        wt_mat_total[p_mask] = (p_vals / p_sum_safe) * total_wt * p_agg_wt
    
    if n_agg_wt > 0:
        wt_mat_total[n_mask] = (n_vals / n_sum_safe) * total_wt * n_agg_wt * -1.0
    
    return wt_mat_total

def calculate_relevance_gated_proj(
    wts: np.ndarray,
    output: np.ndarray,
) -> np.ndarray:

    wt_mat_total = np.zeros(output.shape)
    
    pos_mask = output > 0
    neg_mask = output < 0
    
    pos_values = output[pos_mask]
    neg_values = output[neg_mask]
    
    pos_sum_base = np.sum(pos_values)
    neg_sum_base = np.sum(neg_values) * -1
    t_sum_base = pos_sum_base - neg_sum_base
    
    n_i, n_j = wts.shape
    for i in range(n_i):
        wt_mat = np.zeros(output.shape)
        
        for j in range(n_j):
            wt = wts[i, j]
            
            pos_sum = pos_sum_base
            neg_sum = neg_sum_base
            t_sum = t_sum_base
            
            t_act = np_swish(t_sum)
            p_act = np_swish(pos_sum)
            n_act = np_swish(-1 * neg_sum)
            
            if t_sum < -6:
                pos_sum = 0
            
            if pos_sum > 0 and neg_sum > 0:
                if t_act == p_act:
                    neg_sum = 0
                elif t_act == n_act:
                    pos_sum = 0
            
            denominator = pos_sum + neg_sum
            pos_agg_wt = np.where(pos_sum > 0, pos_sum / denominator, 0)
            neg_agg_wt = np.where(neg_sum > 0, neg_sum / denominator, 0)
            
            pos_sum_norm = np.where(pos_sum != 0, pos_sum, 1)
            neg_sum_norm = np.where(neg_sum != 0, neg_sum, 1)
            
            if pos_agg_wt != 0:
                wt_mat[pos_mask] += (pos_values / pos_sum_norm) * wt * pos_agg_wt
            
            if neg_agg_wt != 0:
                wt_mat[neg_mask] += (neg_values / neg_sum_norm) * wt * neg_agg_wt * -1.0

        wt_mat_total += wt_mat
    
    return wt_mat_total    


def calculate_wt_router_logits(
    wts: np.ndarray, 
    inp: np.ndarray, 
    W_router: np.ndarray
) -> np.ndarray:

    n_samples, n_features = inp.shape
    wt_mat_total = np.zeros_like(inp)
    
    act_range_lower = -1
    act_range_upper = 2
    
    for i in range(n_samples):
        contribution_matrix = W_router * inp[i]  # Shape: (n_features, n_features)
        
        R = wts[i]  # Shape: (n_features,)
        
        p_mask = contribution_matrix > 0
        n_mask = contribution_matrix < 0
        
        p_sums = np.sum(contribution_matrix * p_mask, axis=1)
        n_sums = np.sum(contribution_matrix * n_mask, axis=1) * -1  
        t_sums = p_sums - n_sums
        
        p_sums = np.where(t_sums < act_range_lower, 0, p_sums)
        n_sums = np.where(t_sums > act_range_upper, 0, n_sums)
        
        total_sums = p_sums + n_sums
        p_agg_wts = np.where(p_sums > 0, p_sums / total_sums, 0)
        n_agg_wts = np.where(n_sums > 0, n_sums / total_sums, 0)
        
        p_sums_normalized = np.where(p_sums != 0, p_sums, 1)
        n_sums_normalized = np.where(n_sums != 0, n_sums, 1)

        p_weights = (R * p_agg_wts / p_sums_normalized)[:, np.newaxis]
        wt_mat_p = contribution_matrix * p_mask * p_weights
        
        n_weights = (R * n_agg_wts / n_sums_normalized)[:, np.newaxis]
        wt_mat_n = contribution_matrix * n_mask * n_weights * -1.0
        
        wt_mat = wt_mat_p + wt_mat_n
        
        wt_mat_total[i] = wt_mat.sum(axis=0)
    
    return wt_mat_total

def topk(arr: np.ndarray, k: int, axis: int = -1):
    if axis != -1:
        raise NotImplementedError("Only axis=-1 supported for topk")
    idx = np.argpartition(arr, -k, axis=axis)[..., -k:]
    top_vals = np.take_along_axis(arr, idx, axis=axis)
    order = np.argsort(top_vals, axis=axis)[..., ::-1]
    sorted_idx = np.take_along_axis(idx, order, axis=axis)
    sorted_vals = np.take_along_axis(arr, sorted_idx, axis=axis)
    return sorted_vals, sorted_idx

def qwen_moe_mlp_forward(
    hidden_states: np.ndarray,
    w: Dict[str, Any],
    config: Any
) -> Dict[str, Any]:

    top_k = int(config.num_experts_per_tok)
    norm_topk_prob = bool(config.norm_topk_prob)
    
    B, T, H = hidden_states.shape
    tokens = B * T
    
    hs = hidden_states.reshape(tokens, H).astype(np.float32, copy=False)
    
    W_gate = w['W_gate'].astype(np.float32, copy=False)  # (E, H)
    router_logits = np.einsum('th,eh->te', hs, W_gate)   # (tokens, E)
    
    routing_weights_full = softmax(router_logits, axis=1)  # (tokens, E)
    
    routing_topk_raw, selected_experts = topk(
        routing_weights_full, top_k, axis=-1
    )  # both (tokens, k)
    
    if norm_topk_prob:
        denom = np.maximum(
            routing_topk_raw.sum(axis=-1, keepdims=True),
            1e-30
        )
        routing_topk = routing_topk_raw / denom
    else:
        routing_topk = routing_topk_raw
    
    intermediates = {
        'router_logits': router_logits,
        'softmax_routing_weights': routing_weights_full,
        'selected_experts': selected_experts,
        'routing_weights': routing_topk_raw,
        'norm_routing_weights': routing_topk,
    }
    
    experts_used = np.unique(selected_experts)
    intermediates['expert_hit'] = experts_used
    
    contrib_full = np.zeros((tokens, top_k, H), dtype=np.float32)
    
    for e in experts_used:
        mask = (selected_experts == e)  # (tokens, k)
        
        if not mask.any():
            continue
        
        tok_idx, slot_idx = np.nonzero(mask)  # (M,), (M,)
        
        x = hs[tok_idx]  # (M, H)
        
        Wg = w[f'{e}']['W_gate_proj'].astype(np.float32, copy=False)   # (I, H)
        Wu = w[f'{e}']['W_up_proj'].astype(np.float32, copy=False)     # (I, H)
        Wd = w[f'{e}']['W_down_proj'].astype(np.float32, copy=False)   # (H, I)
        
        gate = np.einsum('mh,ih->mi', x, Wg)  # (M, I)
        
        up = np.einsum('mh,ih->mi', x, Wu)    # (M, I)
        
        inter = np_swish(gate) * up           # (M, I)
        
        down = np.einsum('mi,hi->mh', inter, Wd)  # (M, H)
        
        r = routing_topk[tok_idx, slot_idx][:, None]  # (M, 1)
        contrib = down * r  # (M, H)
        
        contrib_full[tok_idx, slot_idx, :] = contrib
        
        intermediates[f'expert_{e}'] = {
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

def calculate_wt_feed_forward(
    wts: np.ndarray, 
    inp: np.ndarray, 
    w: Dict[str, Any], 
    config: Any
) -> Tuple[np.ndarray, np.ndarray]:

    num_experts = int(config.num_experts)
    B, T, H = inp.shape
    tokens = B * T
    eps = 1e-30  

    inter = qwen_moe_mlp_forward(inp, w, config)

    hidden_states = inp.reshape(tokens, H).astype(np.float32, copy=False)
    R_out = wts.reshape(tokens, H).astype(np.float32, copy=False)

    selected_experts = inter['selected_experts']  # (tokens, k)
    routing_topk = inter.get('norm_routing_weights', inter['routing_weights'])
    
    routing_topk = routing_topk / np.maximum(routing_topk.sum(axis=-1, keepdims=True), eps)

    tokens_chk, k = selected_experts.shape
    assert tokens_chk == tokens, f"Token count mismatch: {tokens_chk} vs {tokens}"

    contrib_full = inter['contrib_full']  # (tokens, k, H)
    z_abs = np.abs(contrib_full)  # Absolute values for allocation
    sum_abs = z_abs.sum(axis=1)  # (tokens, H) - sum across expert slots

    alloc = z_abs / np.maximum(sum_abs[:, None, :], eps)  # (tokens, k, H)

    fallback_mask = (sum_abs <= eps)  # (tokens, H)
    
    if np.any(fallback_mask):
        rt_expanded = np.repeat(routing_topk[:, :, None], H, axis=2)
        rt_expanded = rt_expanded / np.maximum(rt_expanded.sum(axis=1, keepdims=True), eps)
        
        alloc = alloc.copy()
        
        mask3 = fallback_mask[:, None, :]  # (tokens, 1, H)
        alloc[mask3] = rt_expanded[mask3]

    R_slot = alloc * R_out[:, None, :]  # (tokens, k, H)

    final_relevance_input = np.zeros_like(hidden_states)  # (tokens, H)
    relevance_expert = np.zeros(num_experts, dtype=np.float32)
    expert_hit = inter['expert_hit']

    for e in expert_hit:
        ed = inter[f'expert_{e}']
        tok_idx = ed['tok_idx']    # (M,) - token indices processed by this expert
        slot_idx = ed['slot_idx']  # (M,) - slot indices for these tokens

        if tok_idx.size == 0:
            continue

        R_out_expert = R_slot[tok_idx, slot_idx, :]  # (M, H)

        inter_out = ed['intermediate_output']  # (M, I)
        current_state = ed['current_state']    # (M, H)

        R_intermediate = calculate_relevance_proj(R_out_expert, inter_out)  # (M, I)

        R_gate = 0.5 * R_intermediate
        R_up = 0.5 * R_intermediate

        R_in_gate = calculate_relevance_gated_proj(R_gate, current_state)  # (M, H)
        R_in_up = calculate_relevance_proj(R_up, current_state)            # (M, H)
        
        R_curr = R_in_gate + R_in_up  # (M, H)

        np.add.at(final_relevance_input, tok_idx, R_curr)

        relevance_expert[e] += float(R_curr.sum())

    router_fraction = 0.0
    
    if router_fraction > 0.0:
        per_slot_scalar = R_slot.sum(axis=-1) * router_fraction  # (tokens, k)
        
        routing_mass_te = np.zeros((tokens, num_experts), dtype=np.float32)
        
        for s in range(k):
            e_ids = selected_experts[:, s]
            nz = per_slot_scalar[:, s] != 0
            nz_indices = np.where(nz)[0]
            np.add.at(routing_mass_te, (nz_indices, e_ids[nz]), per_slot_scalar[nz, s])
        
        R_router_inp = calculate_wt_router_logits(
            routing_mass_te, hidden_states, w['W_gate']
        )
        final_relevance_input += R_router_inp

    final_relevance_input = final_relevance_input.reshape(B, T, H)
    
    return final_relevance_input, relevance_expert    


def calculate_relevance_single(wts: np.ndarray, inp: np.ndarray, w: np.ndarray) -> np.ndarray:
    batch_size, seq_len, output_features = wts.shape
    _, _, input_features = inp.shape
    
    relevance_input = np.zeros((batch_size, seq_len, input_features), dtype=wts.dtype)
    
    for i in range(batch_size):
        for j in range(seq_len):
            # contribution_matrix: (output_features, input_features)
            contribution_matrix = w * inp[i, j]  # Broadcasting: (output_features, input_features) * (input_features,)
            
            p_mask = contribution_matrix > 0
            n_mask = contribution_matrix < 0
            
            p_sums = np.sum(contribution_matrix * p_mask, axis=1)  # (output_features,)
            n_sums = np.sum(contribution_matrix * n_mask, axis=1) * -1  # (output_features,)
            
            total_sums = p_sums + n_sums
            p_agg_wts = np.where(p_sums > 0, p_sums / total_sums, 0)
            n_agg_wts = np.where(n_sums > 0, n_sums / total_sums, 0)
            
            p_sums_safe = np.where(p_sums != 0, p_sums, 1)
            n_sums_safe = np.where(n_sums != 0, n_sums, 1)
            
            R = wts[i, j, :]  # (output_features,)
            
            # Expand for broadcasting
            p_agg_wts_exp = p_agg_wts[:, np.newaxis]  # (output_features, 1)
            n_agg_wts_exp = n_agg_wts[:, np.newaxis]  # (output_features, 1)
            p_sums_safe_exp = p_sums_safe[:, np.newaxis]  # (output_features, 1)
            n_sums_safe_exp = n_sums_safe[:, np.newaxis]  # (output_features, 1)
            R_exp = R[:, np.newaxis]  # (output_features, 1)
            
            p_contributions = np.where(
                p_mask,
                (contribution_matrix / p_sums_safe_exp) * R_exp * p_agg_wts_exp,
                0
            )
            
            n_contributions = np.where(
                n_mask,
                (contribution_matrix / n_sums_safe_exp) * R_exp * n_agg_wts_exp * -1.0,
                0
            )
            
            relevance_input[i, j] = (p_contributions + n_contributions).sum(axis=0)
    
    return relevance_input

def dlb_style_nonneg_conserve(
    wts: np.ndarray, 
    inp: np.ndarray, 
    eps: float = 1e-12
) -> np.ndarray:

    assert wts.shape == inp.shape, "wts and inp must have the same shape"
    
    pos_mask = inp > 0
    neg_mask = inp < 0
    
    p_sum = np.sum(inp * pos_mask, axis=(-2, -1), keepdims=True)
    n_sum = -np.sum(inp * neg_mask, axis=(-2, -1), keepdims=True)
    
    denom = p_sum + n_sum + eps
    
    p_share = np.where(p_sum > 0, p_sum / denom, 0.0)
    n_share = np.where(n_sum > 0, n_sum / denom, 0.0)
    
    M = np.sum(np.abs(wts), axis=(-2, -1), keepdims=True)
    
    p_div = np.where(p_sum == 0, 1.0, p_sum)
    n_div = np.where(n_sum == 0, 1.0, n_sum)
    
    pos_contrib = np.where(pos_mask, (inp / p_div) * (p_share * M), 0.0)
    
    neg_contrib = np.where(neg_mask, (inp / n_div) * (n_share * M) * (-1.0), 0.0)
    
    out = pos_contrib + neg_contrib
    
    return out

def dlb_style_signed_conserve(
    wts: np.ndarray, 
    inp: np.ndarray, 
    eps: float = 1e-12
) -> np.ndarray:

    Rp = np.maximum(wts, 0.0)
    Rn = np.maximum(-wts, 0.0)
    
    P = dlb_style_nonneg_conserve(Rp, inp, eps)
    N = dlb_style_nonneg_conserve(Rn, inp, eps)
    
    return P - N

def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:

    x32 = x.astype(np.float32, copy=False)
    var = np.mean(np.square(x32), axis=-1, keepdims=True)
    xhat = x32 * np.reciprocal(np.sqrt(var + eps))
    return (xhat * weight).astype(x.dtype, copy=False)

def rotate_half(x: np.ndarray) -> np.ndarray:

    d = x.shape[-1]
    d2 = d // 2

    x1, x2 = x[..., :d2], x[..., d2:]

    return np.concatenate([-x2, x1], axis=-1)

def _apply_rope(
    q: np.ndarray, 
    k: np.ndarray, 
    cos: np.ndarray, 
    sin: np.ndarray, 
    unsqueeze_dim: int = 1
) -> Tuple[np.ndarray, np.ndarray]:

    cos = np.expand_dims(cos, axis=unsqueeze_dim)
    sin = np.expand_dims(sin, axis=unsqueeze_dim)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

    return q_embed, k_embed

def _build_rope_cos_sin(
    B: int,
    T: int, 
    D: int,
    theta: float = 1_000_000.0,
    dtype: np.dtype = np.float32,
    position_ids: Optional[np.ndarray] = None,
    start: int = 0
) -> Tuple[np.ndarray, np.ndarray]:

    if position_ids is None:
        pos = start + np.arange(T, dtype=np.int32)[None, :].repeat(B, axis=0)
    else:
        pos = np.asarray(position_ids)
        if pos.ndim == 1:
            pos = pos[None, :].repeat(B, axis=0)
        if pos.shape != (B, T):
            raise ValueError(f"position_ids must be (B,T) or (T,), got {pos.shape}, expected ({B},{T})")
    half = D // 2
    inv_freq = (theta ** (np.arange(0, half, dtype=dtype) / max(1, half))) ** -1
    freqs = np.einsum("bt,d->btd", pos.astype(dtype), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(dtype), np.sin(emb).astype(dtype)

def qwen_gqa_forward(
    hidden_states: np.ndarray,
    w: Dict[str, np.ndarray],
    config: Any,
    causal: bool = True,
    eps: float = 1e-6,
    cos: Optional[np.ndarray] = None,
    sin: Optional[np.ndarray] = None,
    position_ids: Optional[np.ndarray] = None,
    start: int = 0
) -> Dict[str, np.ndarray]:

    B, T, hidden = hidden_states.shape
    
    H = int(getattr(config, "num_attention_heads"))
    Kv = int(getattr(config, "num_key_value_heads"))
    D = config.hidden_size // config.num_attention_heads
    theta = float(getattr(config, "rope_theta", 1_000_000.0))

    q_flat = np.einsum('bts,os->bto', hidden_states, w['W_q'], optimize=True)
    k_flat = np.einsum('bts,os->bto', hidden_states, w['W_k'], optimize=True)
    v_flat = np.einsum('bts,os->bto', hidden_states, w['W_v'], optimize=True)
    
    D = q_flat.shape[-1] // H
    assert q_flat.shape[-1] == H * D and k_flat.shape[-1] == Kv * D and v_flat.shape[-1] == Kv * D
    
    q = _rmsnorm(q_flat.reshape(B, T, H, D), w['q_norm'], eps=eps)
    k = _rmsnorm(k_flat.reshape(B, T, Kv, D), w['k_norm'], eps=eps)
    v = v_flat.reshape(B, T, Kv, D)
    
    q = np.transpose(q, (0, 2, 1, 3))
    k = np.transpose(k, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))
    
    if cos is None or sin is None:
        cos, sin = _build_rope_cos_sin(
            B, T, D, theta=theta, dtype=hidden_states.dtype,
            position_ids=position_ids, start=start
        )

    q, k = _apply_rope(q, k, cos, sin)

    groups = H // Kv
    if groups * Kv != H:
        raise ValueError(f"H ({H}) must be divisible by Kv ({Kv})")
    
    k = np.repeat(k, repeats=groups, axis=1)
    v = np.repeat(v, repeats=groups, axis=1)
    
    scale = 1.0 / np.sqrt(D)
    QK_output = np.einsum('bhtd,bhsd->bhts', q, k, optimize=True)
    logits_unmasked = QK_output * scale

    A = softmax(logits_unmasked, axis=-1)

    out_heads = np.einsum('bhts,bhsd->bhtd', A, v, optimize=True)

    out = np.transpose(out_heads, (0, 2, 1, 3)).reshape(B, T, H * D)

    attn_out = np.einsum('btm,hm->bth', out, w['W_d'], optimize=True)
    
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

def calculate_wt_self_attention(wts: np.ndarray, inp: np.ndarray, w: dict, config, eps=1e-9):
    B, T, H = inp.shape

    inter = qwen_gqa_forward(inp, w, config)

    wt_mat_attn = calculate_relevance_single(wts, inter['out'], w['W_d'])

    wt_mat_out_heads = wt_mat_attn.reshape(inter['out_heads'].shape)

    relevance_norm_out_heads = wt_mat_out_heads / stabilize(inter['out_heads'] * 2, eps)

    R_QK = np.matmul(relevance_norm_out_heads, np.transpose(inter['v'], (0, 1, 3, 2))) * inter['A']
    R_V = np.matmul(np.transpose(inter['A'], (0, 1, 3, 2)), relevance_norm_out_heads) * inter['v']

    R_V = dlb_style_signed_conserve(R_V, inter['v'])

    relevance_norm_QK_out = R_QK / stabilize(inter['QK_output'] * 2, eps)

    R_Q = np.matmul(relevance_norm_QK_out, inter['k']) * inter['q']
    R_K = np.transpose(np.matmul(np.transpose(inter['q'], (0, 1, 3, 2)), relevance_norm_QK_out), (0, 1, 3, 2)) * inter['k']

    R_Q = dlb_style_signed_conserve(R_Q, inter['q'])
    R_K = dlb_style_signed_conserve(R_K, inter['k'])

    H  = int(getattr(config, "num_attention_heads"))
    Kv = int(getattr(config, "num_key_value_heads"))
    groups = H // Kv

    def collapse_to_kv_heads(R_bhtd, Kv, groups):
        B, H, T, D = R_bhtd.shape
        assert H == Kv * groups
        R_kv = R_bhtd.reshape(B, groups, Kv, T, D).sum(axis=1)
        return np.transpose(R_kv, (0, 2, 1, 3)).reshape(B, T, Kv * D)

    R_K_kv = collapse_to_kv_heads(R_K, Kv, groups)  # (B,T,512)
    R_V_kv = collapse_to_kv_heads(R_V, Kv, groups)  # (B,T,512)

    R_Q = np.transpose(R_Q, (0, 2, 1, 3)).reshape(B, T, -1)

    input_relevance_from_Q = calculate_relevance_single(R_Q, inp, w['W_q'])
    input_relevance_from_K = calculate_relevance_single(R_K_kv, inp, w['W_k'])
    input_relevance_from_V = calculate_relevance_single(R_V_kv, inp, w['W_v'])

    input_relevance = input_relevance_from_Q + input_relevance_from_K + input_relevance_from_V

    return input_relevance