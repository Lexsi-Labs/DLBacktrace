import numpy as np

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

def topk(arr: np.ndarray, k: int, axis: int = -1):
    if axis != -1:
        raise NotImplementedError("Only axis=-1 supported for topk")
    idx = np.argpartition(arr, -k, axis=axis)[..., -k:]
    top_vals = np.take_along_axis(arr, idx, axis=axis)
    order = np.argsort(top_vals, axis=axis)[..., ::-1]
    sorted_idx = np.take_along_axis(idx, order, axis=axis)
    sorted_vals = np.take_along_axis(arr, sorted_idx, axis=axis)
    return sorted_vals, sorted_idx

def stabilize(matrix, epsilon=1e-6):
    return np.where(np.abs(matrix) < epsilon,
                    epsilon * np.sign(matrix + (matrix == 0)),
                    matrix)

def calculate_wt_lm_head(wts, inp, w):
    relevance_input = np.zeros(inp.shape)

    # Loop over batch
    for i in range(wts.shape[0]):
        wts_b = wts[i]
        inp_b = inp[i]

        # Loop over rows
        for j in range(wts_b.shape[0]):
            R = wts_b[j]
            contribution_matrix = np.einsum('ij,j->ij', w['W_lm_head'], inp_b[j])
            wt_mat = np.zeros(contribution_matrix.shape)

            # Loop over contribution rows
            for k in range(contribution_matrix.shape[0]):
                l1_ind1 = contribution_matrix[k]
                wt = R[k]

                p_ind = l1_ind1 > 0
                n_ind = l1_ind1 < 0

                p_sum = np.sum(l1_ind1[p_ind])
                n_sum = np.sum(l1_ind1[n_ind]) * -1

                p_agg_wt = p_sum / (p_sum + n_sum) if p_sum > 0 else 0
                n_agg_wt = n_sum / (p_sum + n_sum) if n_sum > 0 else 0

                p_sum = p_sum if p_sum != 0 else 1
                n_sum = n_sum if n_sum != 0 else 1

                wt_mat[k][p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
                wt_mat[k][n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

            relevance_input[i, j] = wt_mat.sum(axis=0)

    return relevance_input


def calculate_relevance_proj(wts, output):

    wt_mat_total = np.zeros(output.shape)

    # Loop over the first dimension of wts
    for i in range(wts.shape[0]):
        wt_mat = np.zeros(output.shape)

        # Loop over the second dimension of wts
        for j in range(wts.shape[1]):
            l1_ind1 = output
            wt = wts[i, j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0

            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1

            p_agg_wt = p_sum / (p_sum + n_sum) if p_sum > 0 else 0
            n_agg_wt = n_sum / (p_sum + n_sum) if n_sum > 0 else 0

            p_sum = p_sum if p_sum != 0 else 1
            n_sum = n_sum if n_sum != 0 else 1

            wt_mat[p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_mat[n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        wt_mat_total += wt_mat

    return wt_mat_total


def calculate_relevance_gated_proj(wts, output):

    wt_mat_total = np.zeros(output.shape)

    # Loop over the first dimension of wts
    for i in range(wts.shape[0]):
        wt_mat = np.zeros(output.shape)

        # Loop over the second dimension
        for j in range(wts.shape[1]):
            l1_ind1 = output
            wt = wts[i, j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0

            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1
            t_sum = p_sum - n_sum

            act = {
                'name': 'swish',
                'range': {'l': -6, 'u': None},
                'type': 'non_mono',
                'func': np_swish
            }

            if act["type"] == "mono":
                if act["range"]["l"] and t_sum < act["range"]["l"]:
                    p_sum = 0
                if act["range"]["u"] and t_sum > act["range"]["u"]:
                    n_sum = 0
            elif act["type"] == "non_mono":
                t_act = act["func"](t_sum)
                p_act = act["func"](p_sum)
                n_act = act["func"](-1 * n_sum)

                if act["range"]["l"] and t_sum < act["range"]["l"]:
                    p_sum = 0
                if act["range"]["u"] and t_sum > act["range"]["u"]:
                    n_sum = 0

                if p_sum > 0 and n_sum > 0:
                    if t_act == p_act:
                        n_sum = 0
                    elif t_act == n_act:
                        p_sum = 0

            p_agg_wt = p_sum / (p_sum + n_sum) if p_sum > 0 else 0
            n_agg_wt = n_sum / (p_sum + n_sum) if n_sum > 0 else 0

            p_sum = p_sum if p_sum != 0 else 1
            n_sum = n_sum if n_sum != 0 else 1

            wt_mat[p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_mat[n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        wt_mat_total += wt_mat

    return wt_mat_total

def calculate_wt_router_logits(wts, inp, W_router):

    wt_mat_total = np.zeros(inp.shape)

    # Loop over the first dimension of wts
    for i in range(wts.shape[0]):
        R = wts[i]
        contribution_matrix = W_router * inp[i]
        wt_mat = np.zeros(contribution_matrix.shape)

        # Loop through contribution rows
        for j in range(contribution_matrix.shape[0]):
            l1_ind1 = contribution_matrix[j]
            wt = R[j]

            # Positive/negative separation
            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0

            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1
            t_sum = p_sum - n_sum

            # Activation configuration (softmax layer)
            act = {
                "name": "softmax",
                "range": {"l": -1, "u": 2},
                "type": "mono",
                "func": None,
            }

            # Activation logic
            if act["type"] == "mono":
                if act["range"]["l"] and t_sum < act["range"]["l"]:
                    p_sum = 0
                if act["range"]["u"] and t_sum > act["range"]["u"]:
                    n_sum = 0

            # Aggregate weight computation
            p_agg_wt = p_sum / (p_sum + n_sum) if p_sum > 0 else 0
            n_agg_wt = n_sum / (p_sum + n_sum) if n_sum > 0 else 0

            p_sum = p_sum if p_sum != 0 else 1
            n_sum = n_sum if n_sum != 0 else 1

            # Weighted relevance distribution
            wt_mat[j][p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_mat[j][n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        relevance_input = wt_mat.sum(axis=0)
        wt_mat_total[i] = relevance_input

    return wt_mat_total

# ---------- forward (Qwen MoE MLP) ----------
def qwen_moe_mlp_forward(hidden_states: np.ndarray, w: dict, config):
    """
    hidden_states: (B, T, H)
    w: {
        'W_gate': (E, H),
        f'{e}': {
            'W_gate_proj': (I, H),
            'W_up_proj'  : (I, H),
            'W_down_proj': (H, I),
        }
    }
    config: has num_experts_per_tok (top_k), norm_topk_prob (bool), num_experts (E)
    returns: intermediate_outputs
    """
    top_k = int(config.num_experts_per_tok)
    norm_topk_prob = bool(config.norm_topk_prob)

    B, T, H = hidden_states.shape
    tokens = B * T
    hs = hidden_states.reshape(tokens, H).astype(np.float32, copy=False)

    # Router logits (tokens, E)
    W_gate = w['W_gate'].astype(np.float32, copy=False)          # (E, H)
    router_logits = np.einsum('th,eh->te', hs, W_gate)

    # Routing softmax over experts, take top-k per token
    routing_weights_full = softmax(router_logits, axis=1)        # (tokens, E)

    routing_topk_raw, selected_experts = topk(routing_weights_full, top_k, axis=-1)  # both (tokens, k)

    # Optional renorm over chosen k (recommended for MoE)
    if norm_topk_prob:
        denom = np.maximum(routing_topk_raw.sum(axis=-1, keepdims=True), 1e-30)
        routing_topk = routing_topk_raw / denom
    else:
        routing_topk = routing_topk_raw

    # Prepare intermediates
    intermediates = {
        'router_logits': router_logits,                 # (tokens, E)
        'softmax_routing_weights': routing_weights_full,# (tokens, E)
        'selected_experts': selected_experts,           # (tokens, k)
        'routing_weights': routing_topk_raw,            # (tokens, k) pre-renorm
        'norm_routing_weights': routing_topk,           # (tokens, k) sums ~1 per token
    }

    # Process only experts that were actually used
    experts_used = np.unique(selected_experts.reshape(-1))
    intermediates['expert_hit'] = experts_used

    # We'll also assemble a (tokens, k, H) "contrib" tensor lazily from the per-expert passes
    contrib_full = np.zeros((tokens, routing_topk.shape[1], H), dtype=np.float32)

    for e in experts_used:
        # mask over (tokens, k) where this expert is chosen
        mask = (selected_experts == e)
        if not mask.any():
            continue

        tok_idx, slot_idx = np.nonzero(mask)           # (M,), (M,)
        x = hs[tok_idx]                                 # (M, H)

        # Expert weights
        Wg = w[f'{e}']['W_gate_proj'].astype(np.float32, copy=False)   # (I, H)
        Wu = w[f'{e}']['W_up_proj'].astype(np.float32, copy=False)     # (I, H)
        Wd = w[f'{e}']['W_down_proj'].astype(np.float32, copy=False)   # (H, I)

        # SwiGLU
        gate = np.einsum('mh,ih->mi', x, Wg)           # (M, I)
        up   = np.einsum('mh,ih->mi', x, Wu)           # (M, I)
        inter = np_swish(gate) * up                    # (M, I)
        down  = np.einsum('mi,hi->mh', inter, Wd)      # (M, H)

        # Weight by routing prob for the specific (token,slot)
        r = routing_topk[tok_idx, slot_idx][:, None]   # (M, 1)
        contrib = down * r                              # (M, H)

        # Scatter to the per-slot tensor for DL-Backtrace split at combine
        contrib_full[tok_idx, slot_idx, :] = contrib

        # Cache per-expert intermediates needed for relevance
        intermediates[f'expert_{e}'] = {
            'tok_idx': tok_idx,
            'slot_idx': slot_idx,
            'current_state': x,                 # input to expert projections
            'gate_proj_output': gate,
            'up_proj_output': up,
            'intermediate_output': inter,
            'down_proj_output': down,
            'contrib': contrib,          # store z_{t,s,:}
            'routing_scalar': r.squeeze(-1),  # (M,)
        }

    intermediates['contrib_full'] = contrib_full  # (tokens,k,H)    
    return intermediates    

def calculate_wt_feed_forward(wts: np.ndarray, inp: np.ndarray, w: dict, config):
    """
    DL-Backtrace relevance calculation for Qwen-MoE feed-forward
    Args:
        wts: relevance at FF output, shape (B, T, H)
        inp: FF input, shape (B, T, H)
        w:   weights dict (see forward)
        config: has num_experts, num_experts_per_tok, norm_topk_prob
    Returns:
        final_relevance_input: (B, T, H)
        relevance_expert: (E,) total relevance attributed to each expert
    """
    num_experts = int(config.num_experts)
    B, T, H = inp.shape
    tokens = B * T

    # Forward to get routing & expert caches
    inter = qwen_moe_mlp_forward(inp, w, config)

    # Flatten input/relevance
    hidden_states = inp.reshape(tokens, H).astype(np.float32, copy=False)
    R_out = wts.reshape(tokens, H).astype(np.float32, copy=False)

    selected_experts = inter['selected_experts']                  # (tokens, k)
    routing_topk = inter.get('norm_routing_weights', inter['routing_weights'])
    # Ensure per-token normalization over chosen k
    routing_topk = routing_topk / np.maximum(routing_topk.sum(-1, keepdims=True), 1e-30)

    tokens_chk, k = selected_experts.shape
    assert tokens_chk == tokens

    # Build z = r * y_e (already cached as 'contrib') into a dense (tokens,k,H) table
    contrib_full = inter['contrib_full']  # (tokens,k,H)
    z_abs = np.abs(contrib_full)          # (tokens,k,H)
    sum_abs = z_abs.sum(axis=1)           # (tokens,H)

    eps = 1e-30
    # Allocation factors across slots per token,d: |z_{t,s,d}| / sum_s |z_{t,s,d}|
    alloc = z_abs / np.maximum(sum_abs[:, None, :], eps)  # (tokens,k,H)

    # Fallback where sum_abs==0: use routing weights (same across H) to distribute
    fallback_mask = (sum_abs <= eps)  # (tokens,H)
    if np.any(fallback_mask):
        # Expand routing_topk to (tokens,k,H) then mask insert
        rt_expanded = np.repeat(routing_topk[:, :, None], H, axis=2)  # (tokens,k,H)
        # Normalize along k (already 1.0) but keep safe
        rt_expanded = rt_expanded / np.maximum(rt_expanded.sum(axis=1, keepdims=True), eps)
        # Zero out alloc where fallback used, then replace
        alloc = alloc.copy()
        # Broadcast mask to (tokens,1,H)
        mask3 = fallback_mask[:, None, :]
        alloc[mask3] = rt_expanded[mask3]

    # Per-slot relevance vectors R_slot = alloc * R_out[:,None,:]
    R_slot = alloc * R_out[:, None, :]  # (tokens,k,H)

    # Buffers
    final_relevance_input = np.zeros_like(hidden_states)               # (tokens,H)
    relevance_expert = np.zeros((num_experts,), dtype=np.float32)
    expert_hit = inter['expert_hit']

    # ---- propagate *within* each expert using the per-slot relevance mass R_slot[t, slot_idx, :]
    for e in expert_hit:
        ed = inter[f'expert_{e}']
        tok_idx = ed['tok_idx']           # (M,)
        slot_idx = ed['slot_idx']         # (M,)

        if tok_idx.size == 0:
            continue

        # Gather the relevance vector assigned to that token-slot path
        R_out_expert = R_slot[tok_idx, slot_idx, :]  # (M,H)

        # 1) down proj relevance -> intermediate (your existing helper)
        inter_out = ed['intermediate_output']              # (M,I)
        current_state    = ed['current_state']             # (M,H)

        # R_intermediate from down projection
        R_intermediate = calculate_relevance_proj(R_out_expert, inter_out)  # (M,I)

        # Split evenly across SwiGLU branches (you can refine later if needed)
        R_gate = 0.5 * R_intermediate
        R_up   = 0.5 * R_intermediate

        # Back to current_state
        R_in_gate = calculate_relevance_gated_proj(R_gate, current_state)  # (M,H)
        R_in_up   = calculate_relevance_proj(R_up, current_state)          # (M,H)
        R_curr = R_in_gate + R_in_up                                                # (M,H)

        # Scatter-add to tokens
        np.add.at(final_relevance_input, tok_idx, R_curr)
        relevance_expert[e] += float(R_curr.sum())

    routing_mass_te = np.zeros((tokens, num_experts), dtype=np.float32)

    router_fraction = 0.0
    if router_fraction > 0.0:
        per_slot_scalar = R_slot.sum(-1) * router_fraction  # (tokens,k)
        for s in range(k):
            e_ids = selected_experts[:, s]
            nz = per_slot_scalar[:, s] != 0
            np.add.at(routing_mass_te, (np.where(nz)[0], e_ids[nz]), per_slot_scalar[nz, s])
        R_router_inp = calculate_wt_router_logits(routing_mass_te, hidden_states, w['W_gate'])
        final_relevance_input += R_router_inp

    final_relevance_input = final_relevance_input.reshape(B, T, H)
    return final_relevance_input, relevance_expert   

def calculate_relevance_single(wts, inp, w):

    relevance_input = np.zeros(inp.shape)

    for i in range(wts.shape[0]):
        wts_b = wts[i]
        inp_b = inp[i]
        
        for j in range(wts_b.shape[0]):
            R = wts_b[j]
            contribution_matrix = np.einsum('ij,j->ij', w, inp_b[j])
            wt_mat = np.zeros(contribution_matrix.shape)

            for k in range(contribution_matrix.shape[0]):
                l1_ind1 = contribution_matrix[k]
                wt = R[k]

                p_ind = l1_ind1 > 0
                n_ind = l1_ind1 < 0

                p_sum = np.sum(l1_ind1[p_ind])
                n_sum = np.sum(l1_ind1[n_ind]) * -1

                p_agg_wt = p_sum / (p_sum + n_sum) if p_sum > 0 else 0
                n_agg_wt = n_sum / (p_sum + n_sum) if n_sum > 0 else 0

                p_sum = p_sum if p_sum != 0 else 1
                n_sum = n_sum if n_sum != 0 else 1

                wt_mat[k][p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
                wt_mat[k][n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

            relevance_input[i, j] = wt_mat.sum(axis=0)

    return relevance_input

def dlb_style_nonneg_conserve(wts, inp, eps=1e-12):
    assert wts.shape == inp.shape
    pos = inp > 0
    neg = inp < 0

    p_sum = np.sum(np.where(pos, inp, 0.0), axis=(-2,-1), keepdims=True)
    n_sum = -np.sum(np.where(neg, inp, 0.0), axis=(-2,-1), keepdims=True)
    denom = p_sum + n_sum + eps

    p_share = np.where(p_sum > 0, p_sum/denom, 0.0)
    n_share = np.where(n_sum > 0, n_sum/denom, 0.0)

    M = np.sum(np.abs(wts), axis=(-2,-1), keepdims=True)

    p_div = np.where(p_sum == 0, 1.0, p_sum)
    n_div = np.where(n_sum == 0, 1.0, n_sum)

    out = np.zeros_like(inp)
    out += np.where(pos, (inp / p_div) * (p_share * M), 0.0)
    out += np.where(neg, (inp / n_div) * (n_share * M) * (-1.0), 0.0)
    return out

def dlb_style_signed_conserve(wts, inp, eps=1e-12):

    Rp = np.maximum(wts, 0.0)
    Rn = np.maximum(-wts, 0.0)

    P = dlb_style_nonneg_conserve(Rp, inp, eps)
    N = dlb_style_nonneg_conserve(Rn, inp, eps)

    return P - N

def _rmsnorm(x, weight, eps=1e-6):
    x32 = x.astype(np.float32, copy=False)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    xhat = x32 * np.reciprocal(np.sqrt(var + eps))
    return (xhat * weight).astype(x.dtype, copy=False)


def rotate_half(x):
    d = x.shape[-1]
    d2 = d // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return np.concatenate([-x2, x1], axis=-1)


def _apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos = np.expand_dims(cos, axis=unsqueeze_dim)  # (B, 1, T, D)
    sin = np.expand_dims(sin, axis=unsqueeze_dim)  # (B, 1, T, D)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed    

def _build_rope_cos_sin(B, T, D, theta=1_000_000.0, dtype=np.float32, position_ids=None, start=0):
    if position_ids is None:
        pos = start + np.arange(T, dtype=np.int32)[None, :].repeat(B, axis=0)
    else:
        pos = np.asarray(position_ids)
        if pos.ndim == 1:
            pos = pos[None, :].repeat(B, axis=0)
        if pos.shape != (B, T):
            raise ValueError(f"position_ids must be (B,T) or (T,), got {pos.shape}, expected ({B},{T})")
    half = D // 2
    inv_freq = (theta ** (np.arange(0, half, dtype=dtype) / max(1, half))) ** -1  # (D/2,)
    freqs = np.einsum("bt,d->btd", pos.astype(dtype), inv_freq)  # (B,T,D/2)
    emb = np.concatenate([freqs, freqs], axis=-1)                # (B,T,D)
    return np.cos(emb).astype(dtype), np.sin(emb).astype(dtype)  

def qwen_gqa_forward(hidden_states, w, config, causal=True, eps=1e-6, cos=None, sin=None, position_ids=None, start=0):

    B, T, hidden = hidden_states.shape

    H  = int(getattr(config, "num_attention_heads"))
    Kv = int(getattr(config, "num_key_value_heads"))
    D = config.hidden_size // config.num_attention_heads
    theta = float(getattr(config, "rope_theta", 1_000_000.0))

    q_flat = np.einsum('bts,os->bto', hidden_states, w['W_q'], optimize=True)  # (B,T,4096)
    k_flat = np.einsum('bts,os->bto', hidden_states, w['W_k'], optimize=True)  # (B,T, 512)
    v_flat = np.einsum('bts,os->bto', hidden_states, w['W_v'], optimize=True)  # (B,T, 512)

    D = q_flat.shape[-1] // H
    assert q_flat.shape[-1] == H*D and k_flat.shape[-1] == Kv*D and v_flat.shape[-1] == Kv*D

    q = _rmsnorm(q_flat.reshape(B, T, H, D), w['q_norm'], eps=eps)
    k = _rmsnorm(k_flat.reshape(B, T, Kv, D), w['k_norm'], eps=eps)
    v = v_flat.reshape(B, T, Kv, D)

    q = np.transpose(q, (0, 2, 1, 3))
    k = np.transpose(k, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))

    if cos is None or sin is None:
        cos, sin = _build_rope_cos_sin(B, T, D, theta=theta, dtype=hidden_states.dtype,
                                       position_ids=position_ids, start=start)
    q, k = _apply_rope(q, k, cos, sin)

    groups = H // Kv
    if groups * Kv != H:
        raise ValueError(f"H ({H}) must be divisible by Kv ({Kv})")
    k = np.repeat(k, repeats=groups, axis=1)
    v = np.repeat(v, repeats=groups, axis=1)

    scale = 1.0 / np.sqrt(D)
    QK_output = np.einsum('bhtd,bhsd->bhts', q, k, optimize=True)   # (B,H,T,T)
    logits_unmasked = QK_output * scale
    A = softmax(logits_unmasked, axis=-1)                                # (B,H,T,T)
    out_heads = np.einsum('bhts,bhsd->bhtd', A, v, optimize=True)  # (B,H,T,D)

    out = np.transpose(out_heads, (0, 2, 1, 3)).reshape(B, T, H * D)  # (B,T,4096)

    attn_out = np.einsum('btm,hm->bth', out, w['W_d'], optimize=True)  # (B,T,hidden=2048)

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

def calculate_wt_self_attention_parallel(wts: np.ndarray, inp: np.ndarray, w: dict, config, eps=1e-9):
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