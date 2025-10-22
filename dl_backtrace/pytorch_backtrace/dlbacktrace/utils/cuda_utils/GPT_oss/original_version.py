import numpy as np
import concurrent.futures

def np_sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -500, 500)
    return 1 / (1 + np.exp(-x))

def np_swish(x, beta=0.75):
    return x * np_sigmoid(beta * x)

def stabilize(matrix, epsilon=1e-6):
    return np.where(np.abs(matrix) < epsilon,
                    epsilon * np.sign(matrix + (matrix == 0)),
                    matrix)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(np.clip(x_shifted, -50.0, 50.0))
    denom = np.sum(exp_x, axis=axis, keepdims=True)
    out = exp_x / np.maximum(denom, 1e-30)
    return out

def calculate_wt_lm_head(wts, inp, w, chunk_rows=4096, eps=1e-12):
    """
    wts: (B, T, V)          # relevance at logits (per vocab row)
    inp: (B, T, D)          # hidden states going into LM head
    w:   dict with {'W_lm_head': (V, D)}  # weight matrix
    Returns: (B, T, D) relevance on inputs
    """

    W = w['W_lm_head']              # (V, D)
    B, T, V = wts.shape
    _, _, D = inp.shape
    assert W.shape == (V, D), (W.shape, (V, D))

    # ensure float32 to reduce memory & avoid bf16 issues
    W = W.astype(np.float32, copy=False)
    wts = wts.astype(np.float32, copy=False)
    inp = inp.astype(np.float32, copy=False)

    relevance_input = np.zeros((B, T, D), dtype=np.float32)

    for b in range(B):
        for t in range(T):
            R = wts[b, t]        # (V,)
            x = inp[b, t]        # (D,)
            r_in = np.zeros((D,), dtype=np.float32)

            # process rows of W in manageable chunks
            for start in range(0, V, chunk_rows):
                end = min(start + chunk_rows, V)
                Wc = W[start:end, :]                     # (C, D)
                Rc = R[start:end].reshape(-1, 1)         # (C, 1)

                # L = Wc * x (broadcast) but chunked
                L = Wc * x[None, :]                      # (C, D)

                # positive / negative parts
                L_pos = np.maximum(L, 0.0)
                L_neg = np.minimum(L, 0.0)

                p_sum = L_pos.sum(axis=1, keepdims=True)           # (C,1)
                n_sum = (-L_neg).sum(axis=1, keepdims=True)        # (C,1)
                denom = p_sum + n_sum + eps

                # aggregated weights
                p_agg = np.where(p_sum > 0, p_sum / denom, 0.0)    # (C,1)
                n_agg = np.where(n_sum > 0, n_sum / denom, 0.0)    # (C,1)

                # Avoid divide-by-zero
                p_norm = np.where(p_sum > 0, L_pos / (p_sum + eps), 0.0)  # (C,D)
                n_norm = np.where(n_sum > 0, L_neg / (n_sum + eps), 0.0)  # (C,D) (note: L_neg ≤ 0)

                # Combine (note the -1 on negatives to match your original)
                contrib = p_norm * (Rc * p_agg) + (-n_norm) * (Rc * n_agg)  # (C,D)

                r_in += contrib.sum(axis=0).astype(np.float32)

            relevance_input[b, t] = r_in

    return relevance_input

def topk(arr: np.ndarray, k: int, axis: int = -1):
    if axis != -1:
        raise NotImplementedError("Only axis=-1 supported for topk")
    idx = np.argpartition(arr, -k, axis=axis)[..., -k:]
    top_vals = np.take_along_axis(arr, idx, axis=axis)
    order = np.argsort(top_vals, axis=axis)[..., ::-1]
    sorted_idx = np.take_along_axis(idx, order, axis=axis)
    sorted_vals = np.take_along_axis(arr, sorted_idx, axis=axis)
    return sorted_vals, sorted_idx

def gpt_oss_moe_mlp_forward(hidden_states, w, config):
    B, S, H = hidden_states.shape

    # code here
    top_k = config.num_experts_per_tok
    num_experts = config.num_local_experts
    hidden_dim = config.hidden_size

    hidden_states = hidden_states.reshape(-1, hidden_dim)

    # ------------ Router --------------------
    router_logits = np.einsum('th,eh->te', hidden_states, w['W_router']) + w['b_router']

    router_top_value_raw, router_indices = topk(router_logits, k=top_k, axis=-1)

    # softmax over the chosen experts
    router_top_value = softmax(router_top_value_raw, axis=1)  # (tokens, E)

    # scatter into full (tokens, num_experts)
    router_scores = np.zeros_like(router_logits, dtype=router_logits.dtype)
    rows = np.arange(router_logits.shape[0])[:, None]  # (tokens, 1)
    router_scores[rows, router_indices] = router_top_value.astype(router_scores.dtype)

    intermediates = {
        'router_logits': router_logits,
        'router_top_value_raw': router_top_value_raw,
        'router_top_value': router_top_value,
        'router_indices': router_indices,
        'router_scores': router_scores,
    }

    # ------------ Experts --------------------
    alpha = 1.702
    limit = 7.0

    # First `per-expert` linear: (E, N, 2D) = (N, H) x (E, H, 2D)
    gate_up = np.einsum('nh,ehd->end', hidden_states, w['W_gate_up_proj']) + w['b_gate_up_proj'][:, None, :]    # (E, N, 2D)

    gate = gate_up[..., 0::2]    # (E, N, D)
    up   = gate_up[..., 1::2]    # (E, N, D)

    gate = np.minimum(gate, limit)
    up = np.clip(up, -limit, limit)

    glu = gate * np_sigmoid(gate * alpha)
    pre = (up + 1.0) * glu

    # Second `per-expert` linear: (E, N, H) = (E, N, D) x (E, D, H)
    next_state = np.einsum('end,edh->enh', pre, w['W_down_proj']) + w['b_down_proj'][:, None, :]    # (E, N, H)

    # Mix experts with routing weights: (N, E) -> (E, N, 1)

    wts = router_scores.T[:, :, None].astype(np.float32)    # (E, N, 1)
    y = (next_state * wts).sum(axis=0)    # (N, H)
    routed_out = y.reshape(B, S, H)    # (B, S, H)

    intermediates['gate_up'] = gate_up
    intermediates['gate'] = gate
    intermediates['up'] = up
    intermediates['glu'] = glu
    intermediates['pre'] = pre
    intermediates['next_state'] = next_state
    intermediates['wts'] = wts
    intermediates['y'] = y
    intermediates['routed_out'] = routed_out

    return routed_out, intermediates

def process_single_relevance_gated_proj(i, wts, output):
    wt_mat = np.zeros(output.shape)

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

        # Activation function processing (same as before)
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

        if p_sum > 0:
            p_agg_wt = p_sum / (p_sum + n_sum)
        else:
            p_agg_wt = 0
        if n_sum > 0:
            n_agg_wt = n_sum / (p_sum + n_sum)
        else:
            n_agg_wt = 0

        if p_sum == 0:
            p_sum = 1
        if n_sum == 0:
            n_sum = 1

        wt_mat[p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
        wt_mat[n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

    return wt_mat


# Optimized parallel function
def calculate_relevance_gated_proj_parallel(wts, output):
    wt_mat_total = np.zeros(output.shape)

    # Parallel processing using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_relevance_gated_proj, range(wts.shape[0]), [wts] * wts.shape[0], [output] * wts.shape[0]))

    # Combine the results into the final wt_mat matrix
    for result in results:
        wt_mat_total += result

    return wt_mat_total

def process_single_relevance_proj(i, wts, output):
    wt_mat = np.zeros(output.shape)
    for j in range(wts.shape[1]):
        l1_ind1 = output
        wt = wts[i, j]

        p_ind = l1_ind1 > 0
        n_ind = l1_ind1 < 0
        p_sum = np.sum(l1_ind1[p_ind])
        n_sum = np.sum(l1_ind1[n_ind]) * -1

        if p_sum > 0:
            p_agg_wt = p_sum / (p_sum + n_sum)
        else:
            p_agg_wt = 0
        if n_sum > 0:
            n_agg_wt = n_sum / (p_sum + n_sum)
        else:
            n_agg_wt = 0

        if p_sum == 0:
            p_sum = 1
        if n_sum == 0:
            n_sum = 1

        wt_mat[p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
        wt_mat[n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

    return wt_mat

# Optimized parallel function
def calculate_relevance_proj_parallel(wts, output):
    wt_mat_total = np.zeros(output.shape)

    # Parallel processing using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_relevance_proj, range(wts.shape[0]), [wts] * wts.shape[0], [output] * wts.shape[0]))

    # Combine the results into the final wt_mat matrix
    for result in results:
        wt_mat_total += result

    return wt_mat_total

def calculate_wt_gpt_oss_feed_forward_parallel(wts, inp, w, config):
    # code here
    num_experts = int(config.num_local_experts)
    B, T, H = inp.shape
    tokens = B * T

    # Forward to get routing & expert caches
    ff_output, inter = gpt_oss_moe_mlp_forward(inp, w, config)

    # --------- Relevance Calculation ---------------------
    # inter provides:
    #  - 'router_indices': (N,k)
    #  - 'router_top_value': (N,k) softmax over chosen k
    #  - 'router_scores': (N,E) sparse full
    #  - 'next_state': (E,N,H) expert outputs before mixing
    #  - 'pre': (E,N,D) = (up+1)*glu  (intermediate D before down-proj)

    selected_experts = inter['router_indices']        # (N, k)
    routing_topk     = inter['router_top_value']      # (N, k)
    next_state_enh   = inter['next_state']            # (E, N, H)
    pre_e_nd         = inter['pre']                   # (E, N, D)

    # ---- Flatten relevance at FF output ----
    R_out = wts.reshape(tokens, H).astype(np.float32, copy=False)   # (N,H)
    hs    = inp.reshape(tokens, H).astype(np.float32, copy=False)   # (N,H)

    # ---- Build dense per-slot contributions z_{t,s,:} = y_{e*,t,:} * r_{t,s}
    # Gather expert outputs for each (t,slot): next_state_enh[e, t, :]
    t_idx = np.arange(tokens)[:, None]                                # (N,1)
    e_idx = selected_experts                                     # (N,k)
    z = next_state_enh[e_idx, t_idx, :] * routing_topk[..., None]  # (N,k,H)

    # ---- Allocation across slots (|z| rule) with fallback to routing weights
    z_abs  = np.abs(z)                  # (N,k,H)
    sum_abs = z_abs.sum(axis=1)         # (N,H)
    eps = 1e-30
    alloc = z_abs / np.maximum(sum_abs[:, None, :], eps)     # (N,k,H)

    # Fallback where sum_abs==0 for some (t,d): use routing_topk (broadcast over H)
    fallback_mask = (sum_abs <= eps)                           # (N,H)
    if np.any(fallback_mask):
        rt_expanded = np.repeat(routing_topk[:, :, None], H, axis=2)   # (N,k,H)
        rt_expanded = rt_expanded / np.maximum(rt_expanded.sum(axis=1, keepdims=True), eps)
        alloc = alloc.copy()
        alloc[fallback_mask[:, None, :]] = rt_expanded[fallback_mask[:, None, :]]


    # Per-slot relevance at output H
    R_slot = alloc * R_out[:, None, :]       # (N,k,H)

    # ---- Backprop relevance within each expert path
    final_relevance_input = np.zeros((tokens, H), dtype=np.float32)
    relevance_expert = np.zeros((num_experts,), dtype=np.float32)

    # Experts used this pass
    expert_hit = np.unique(selected_experts.reshape(-1))

    for e in expert_hit:
        # find all (token,slot) where this expert was selected
        mask = (selected_experts == e)
        if not mask.any():
            continue
        tok_idx, slot_idx = np.nonzero(mask)        # (M,), (M,)
        M = tok_idx.size
        if M == 0:
            continue

        # Relevance assigned to this expert path at output H
        R_out_expert = R_slot[tok_idx, slot_idx, :]           # (M,H)

        # Intermediate D output for this expert at those tokens
        inter_out_D  = pre_e_nd[e, tok_idx, :]                # (M,D)

        # ---- D ⟵ H through down-proj (your proportional rule helper)
        # returns (M,D)
        R_intermediate = calculate_relevance_proj_parallel(R_out_expert, inter_out_D)

        # ---- Split evenly to the two SwiGLU branches (you can refine later)
        R_gate = 0.5 * R_intermediate
        R_up   = 0.5 * R_intermediate

        # ---- Back to input H for both branches
        current_state = hs[tok_idx, :]                         # (M,H)

        # gate branch uses your gated rule
        R_in_gate = calculate_relevance_gated_proj_parallel(R_gate, current_state)   # (M,H)

        # up branch uses linear rule
        R_in_up   = calculate_relevance_proj_parallel(R_up,   current_state)         # (M,H)

        R_curr = R_in_gate + R_in_up                          # (M,H)

        # Scatter-add to tokens and accumulate per-expert mass
        np.add.at(final_relevance_input, tok_idx, R_curr)
        relevance_expert[e] += float(R_curr.sum())

    # Reshape back to (B,T,H)
    final_relevance_input = final_relevance_input.reshape(B, T, H)
    return final_relevance_input, relevance_expert

def process_single_row_QKV_relevance(i, wts, inp, w):
    relevance_input_row = np.zeros(inp.shape[1])
    R = wts[i]
    contribution_matrix = np.einsum('ij,j->ij', w, inp[i])
    wt_mat = np.zeros(contribution_matrix.shape)

    for j in range(contribution_matrix.shape[0]):
        l1_ind1 = contribution_matrix[j]
        wt = R[j]

        p_ind = l1_ind1 > 0
        n_ind = l1_ind1 < 0

        p_sum = np.sum(l1_ind1[p_ind])
        n_sum = np.sum(l1_ind1[n_ind]) * -1

        p_agg_wt = p_sum / (p_sum + n_sum) if p_sum > 0 else 0
        n_agg_wt = n_sum / (p_sum + n_sum) if n_sum > 0 else 0

        p_sum = p_sum if p_sum != 0 else 1
        n_sum = n_sum if n_sum != 0 else 1

        wt_mat[j][p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
        wt_mat[j][n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

    relevance_input_row = wt_mat.sum(axis=0)
    return relevance_input_row

# Optimized parallel function
def calculate_QKV_relevance_parallel(wts, inp, w):
    relevance_input = np.zeros(inp.shape)

    for i in range(wts.shape[0]):
        wts_b = wts[i]
        inp_b = inp[i]

        # Parallel processing using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(process_single_row_QKV_relevance, range(wts_b.shape[0]), [wts_b]*wts_b.shape[0], [inp_b]*wts_b.shape[0], [w]*wts_b.shape[0]))

        # Combine the results into the final relevance_input matrix
        for j, result in enumerate(results):
            relevance_input[i, j] = result

    return relevance_input

def process_single_relevance_attention_output(i, wts, inp, w):
    relevance_input_row = np.zeros(inp.shape[1])
    R = wts[i]
    contribution_matrix = np.einsum('ij,j->ij', w['W_d'], inp[i])
    wt_mat = np.zeros(contribution_matrix.shape)

    for j in range(contribution_matrix.shape[0]):
        l1_ind1 = contribution_matrix[j]
        wt = R[j]

        p_ind = l1_ind1 > 0
        n_ind = l1_ind1 < 0

        p_sum = np.sum(l1_ind1[p_ind])
        n_sum = np.sum(l1_ind1[n_ind]) * -1

        p_agg_wt = p_sum / (p_sum + n_sum) if p_sum > 0 else 0
        n_agg_wt = n_sum / (p_sum + n_sum) if n_sum > 0 else 0

        p_sum = p_sum if p_sum != 0 else 1
        n_sum = n_sum if n_sum != 0 else 1

        wt_mat[j][p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
        wt_mat[j][n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

    relevance_input_row = wt_mat.sum(axis=0)
    return relevance_input_row

# Optimized parallel function
def calculate_wt_attention_output_parallel(wts, inp, w):
    relevance_input = np.zeros(inp.shape)

    for i in range(wts.shape[0]):
        wts_b = wts[i]
        inp_b = inp[i]

        # Parallel processing using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(process_single_relevance_attention_output, range(wts_b.shape[0]), [wts_b]*wts_b.shape[0], [inp_b]*wts_b.shape[0], [w]*wts_b.shape[0]))

        # Combine the results into the final relevance_input matrix
        for j, result in enumerate(results):
            relevance_input[i, j] = result

    return relevance_input

def dlb_style_nonneg_conserve(wts, inp, eps=1e-12):
    """
    Non-negative, mass-preserving DLB on |wts| (conserves L1).
    Keeps your current behavior: sum(out) == sum(abs(wts)) per (B,H).
    """
    assert wts.shape == inp.shape
    pos = inp > 0
    neg = inp < 0

    p_sum = np.sum(np.where(pos, inp, 0.0), axis=(-2,-1), keepdims=True)
    n_sum = -np.sum(np.where(neg, inp, 0.0), axis=(-2,-1), keepdims=True)
    denom = p_sum + n_sum + eps

    p_share = np.where(p_sum > 0, p_sum/denom, 0.0)
    n_share = np.where(n_sum > 0, n_sum/denom, 0.0)

    M = np.sum(np.abs(wts), axis=(-2,-1), keepdims=True)  # mass to conserve

    p_div = np.where(p_sum == 0, 1.0, p_sum)
    n_div = np.where(n_sum == 0, 1.0, n_sum)

    out = np.zeros_like(inp)
    out += np.where(pos, (inp / p_div) * (p_share * M), 0.0)
    out += np.where(neg, (inp / n_div) * (n_share * M) * (-1.0), 0.0)
    return out

def dlb_style_signed_conserve(wts, inp, eps=1e-12):
    """
    Signed, mass-preserving DLB:
    For each (B,H), sum over (T,D) of out == sum over (T,D) of wts (signed).
    Entries may be negative (as they should be if wts has negatives).
    """
    Rp = np.maximum(wts, 0.0)
    Rn = np.maximum(-wts, 0.0)  # magnitude of negative part

    P = dlb_style_nonneg_conserve(Rp, inp, eps)  # ≥0, sums to sum(Rp)
    N = dlb_style_nonneg_conserve(Rn, inp, eps)  # ≥0, sums to sum(Rn)

    return P - N  # signed result; per-(B,H) sums match sum(wts)

def _rmsnorm(x, weight, eps=1e-6):
    # x: (B, T, H or Kv, D), weight: (D,)
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
    # q: (B, H, T, D), k: (B, Kv, T, D); cos/sin: (B, T, D)
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

def _causal_mask_np(T, dtype=np.float32):
    m = np.triu(np.ones((T,T), dtype=bool), k=1)          # True above diag → disallow
    mask = np.zeros((T,T), dtype=dtype)
    mask[m] = -np.inf                                     # additive mask
    return mask

def _sliding_causal_mask_np(T, W, dtype=np.float32):
    j = np.arange(T)[None, :]
    i = np.arange(T)[:, None]
    allow = (j <= i) & (j >= (i - W + 1))
    mask = np.full((T,T), -np.inf, dtype=dtype)
    mask[allow] = 0.0
    return mask

def gpt_oss_gqa_forward(hidden_states, w, config, causal=True, eps=1e-6, cos=None, sin=None, position_ids=None, start=0, attn_mask=None):
    """
    Grouped-Query Attention with RoPE + causal mask (NumPy)
    """

    B, T, hidden = hidden_states.shape

    H  = int(getattr(config, "num_attention_heads"))
    Kv = int(getattr(config, "num_key_value_heads"))
    D = config.hidden_size // config.num_attention_heads
    theta = float(getattr(config, "rope_theta", 1_000_000.0))

    # Projections with (out, in): use 'os'
    q_flat = np.einsum('bts,os->bto', hidden_states, w['W_q'], optimize=True)  # (B,T,H*D)
    if 'b_q' in w: q_flat = q_flat + w['b_q']

    k_flat = np.einsum('bts,os->bto', hidden_states, w['W_k'], optimize=True)  # (B,T,Kv*D)
    if 'b_k' in w: k_flat = k_flat + w['b_k']

    v_flat = np.einsum('bts,os->bto', hidden_states, w['W_v'], optimize=True)  # (B,T,Kv*D)
    if 'b_v' in w: v_flat = v_flat + w['b_v']

    # Infer D from W_q
    D = q_flat.shape[-1] // H   # 4096//32 = 128
    assert q_flat.shape[-1] == H*D and k_flat.shape[-1] == Kv*D and v_flat.shape[-1] == Kv*D

    # Reshape and per-head RMSNorm
    if 'q_norm' in w:
        q = _rmsnorm(q_flat.reshape(B, T, H, D), w['q_norm'], eps=eps)
    else:
        q = q_flat.reshape(B, T, H, D)

    if 'k_norm' in w:
        k = _rmsnorm(k_flat.reshape(B, T, Kv, D), w['k_norm'], eps=eps)
    else:
        k = k_flat.reshape(B, T, Kv, D)

    v = v_flat.reshape(B, T, Kv, D)

    # (B, heads, T, D)
    q = np.transpose(q, (0, 2, 1, 3))
    k = np.transpose(k, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))

    # RoPE
    if cos is None or sin is None:
        cos, sin = _build_rope_cos_sin(B, T, D, theta=theta, dtype=hidden_states.dtype,
                                       position_ids=position_ids, start=start)
    q, k = _apply_rope(q, k, cos, sin)

    # GQA expand Kv -> H
    groups = H // Kv
    if groups * Kv != H:
        raise ValueError(f"H ({H}) must be divisible by Kv ({Kv})")
    k = np.repeat(k, repeats=groups, axis=1)  # (B,H,T,D)
    v = np.repeat(v, repeats=groups, axis=1)  # (B,H,T,D)

    # Attention
    scale = 1.0 / np.sqrt(D)
    QK_output = np.einsum('bhtd,bhsd->bhts', q, k, optimize=True)   # (B,H,T,T)
    logits_unmasked = QK_output * scale

    # NEW: apply mask if provided (broadcast to (B,H,T,T))
    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask[None, None, :, :]               # (1,1,T,T)
        logits = logits_unmasked + attn_mask
    else:
        logits = logits_unmasked

    # A = softmax(logits_unmasked, axis=-1)                                # (B,H,T,T)

    # Optional: implement the GPT-OSS "sinks" trick if provided
    sink_prob = None
    alpha = None
    if 'W_sinks' in w:
        sinks = w['W_sinks'].reshape(1, H, 1, 1).astype(logits.dtype)
        combined = np.concatenate([logits, np.broadcast_to(sinks, (B, H, T, 1))], axis=-1)
        probs_all = softmax(combined, axis=-1)
        sink_prob = probs_all[..., -1]                  # (B,H,T)
        A = probs_all[..., :-1]  # drop sink column     # (B,H,T,T)
        alpha = 1.0 - sink_prob                         # (B,H,T)
    else:
        A = softmax(logits, axis=-1)

    out_heads = np.einsum('bhts,bhsd->bhtd', A, v, optimize=True)  # (B,H,T,D)

    # Merge heads -> (B,T,H*D)
    out = np.transpose(out_heads, (0, 2, 1, 3)).reshape(B, T, H * D)  # (B,T,4096)

    # Output projection: W_d is (hidden, H*D), so use 'btm,hm->bth'
    attn_out = np.einsum('btm,hm->bth', out, w['W_d'], optimize=True)  # (B,T,hidden=2048)
    if 'b_d' in w: attn_out = attn_out + w['b_d']

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
        intermediates['sink_prob'] = sink_prob         # (B,H,T)
        intermediates['alpha'] = alpha                 # (B,H,T)

    return out, intermediates

def calculate_wt_self_attention_parallel(wts: np.ndarray, inp: np.ndarray, w: dict, config,
                                         eps: float = 1e-9, attn_type: str = "full", sliding_window: int | None = None,
                                         with_sink_return: bool = False):
    # code here
    B, T, H = inp.shape

    # build mask (small, O(T^2))
    if attn_type == "sliding" and sliding_window is not None:
        attn_mask = _sliding_causal_mask_np(T, sliding_window, dtype=np.float32)
    else:
        attn_mask = _causal_mask_np(T, dtype=np.float32)

    # Forward to get routing & expert caches
    gqa_output, inter = gpt_oss_gqa_forward(inp, w, config, attn_mask=attn_mask)


    # -----------  Relevance Calculation ------------------
    # 1. attn_output -> out
    out = inter['out']
    wt_mat_attn = calculate_wt_attention_output_parallel(wts, inter['out'], w)

    wt_mat_out_heads = wt_mat_attn.reshape(inter['out_heads'].shape)

    # 2. Sink bookkeeping (use the alpha computed in forward) ---
    A = inter['A'].astype(np.float64)                          # (B,H,T,S)
    out_heads = inter['out_heads'].astype(np.float64)          # (B,H,T,D)
    W = wt_mat_out_heads.astype(np.float64)                    # (B,H,T,D)

    has_sink = ('alpha' in inter)
    if has_sink:
        alpha = inter['alpha'].astype(np.float64)              # (B,H,T), from forward (no clipping)
        alpha_bhtd = alpha[..., None]                          # (B,H,T,1)
        R_sink = (1.0 - alpha_bhtd) * W                        # (B,H,T,D)
        wt_eff = alpha_bhtd * W                                # (B,H,T,D)
    else:
        R_sink = 0.0
        wt_eff = W

    # 3. Relevance calculation of `R_QK` and `R_V`

    relevance_norm_out_heads = wt_eff / stabilize(inter['out_heads'] * 2, eps)

    R_QK = np.matmul(relevance_norm_out_heads, np.transpose(inter['v'], (0, 1, 3, 2))) * inter['A']    # (B,H,T,S)
    R_V = np.matmul(np.transpose(inter['A'], (0, 1, 3, 2)), relevance_norm_out_heads) * inter['v']     # (B,H,S,D)

    # 3b. Fold the sink mass into QK (no double counting now)
    if has_sink:
        # no-sink probs per row (sum over S = 1)
        S = A / np.maximum(alpha[..., None], 1e-12)            # (B,H,T,S)

        # signed sink mass per row (sum over D)
        m_sink = np.sum(R_sink, axis=-1)                       # (B,H,T)

        # base add
        add = m_sink[..., None] * S                            # (B,H,T,S)

        # tiny per-row correction to make sum_S(add) == m_sink exactly
        # err shape: (B,H,T,1)
        err = m_sink[..., None] - np.sum(add, axis=-1, keepdims=True)

        # spread correction uniformly across nonzero S entries (avoid creating new support)
        nz = (S > 0).astype(np.float64)
        nz_cnt = np.sum(nz, axis=-1, keepdims=True)
        corr = np.where(nz_cnt > 0, err / np.maximum(nz_cnt, 1.0), 0.0) * nz

        R_QK = R_QK + add + corr

    if has_sink:
        print("[chk] sum(R_sink) =", float(np.sum(R_sink)))
        print("[chk] ΔR_QK (added) ≈ sum(R_sink):",
              float(np.sum(add + corr)))

    # 4. Signed conservation on V (unchanged)


    R_V = dlb_style_signed_conserve(R_V, inter['v'])



    # 5. Relevance calculation of `R_Q` and `R_K`
    relevance_norm_QK_out = R_QK / stabilize(inter['QK_output'] * 2, eps)

    R_Q = np.matmul(relevance_norm_QK_out, inter['k']) * inter['q']
    R_K = np.transpose(np.matmul(np.transpose(inter['q'], (0, 1, 3, 2)), relevance_norm_QK_out), (0, 1, 3, 2)) * inter['k']

    R_Q = dlb_style_signed_conserve(R_Q, inter['q'])
    R_K = dlb_style_signed_conserve(R_K, inter['k'])

    # 6. Calculate relevance leakage
    total_out = np.sum(wts)
    total_prop = np.sum(R_Q) + np.sum(R_K) + np.sum(R_V)


    # 7. Collapse to `kv_heads`
    H  = int(getattr(config, "num_attention_heads"))
    Kv = int(getattr(config, "num_key_value_heads"))
    D = config.hidden_size // config.num_attention_heads
    groups = H // Kv

    def collapse_to_kv_heads(R_bhtd, Kv, groups):
        # R_bhtd: (B, H, T, D) where H = Kv * groups
        B, H, T, D = R_bhtd.shape
        assert H == Kv * groups
        # group heads: (B, groups, Kv, T, D) -> sum across groups -> (B, Kv, T, D)
        R_kv = R_bhtd.reshape(B, groups, Kv, T, D).sum(axis=1)
        # to (B, T, Kv*D)
        return np.transpose(R_kv, (0, 2, 1, 3)).reshape(B, T, Kv * D)

    # Do this BEFORE your (B,T,-1) reshape used for Q
    R_K_kv = collapse_to_kv_heads(R_K, Kv, groups)  # (B,T,512)
    R_V_kv = collapse_to_kv_heads(R_V, Kv, groups)  # (B,T,512)

    # 8. Calculate `input_relevance`
    R_Q = np.transpose(R_Q, (0, 2, 1, 3)).reshape(B, T, -1)
    # R_K = np.transpose(R_K, (0, 2, 1, 3)).reshape(B, T, -1)
    # R_V = np.transpose(R_V, (0, 2, 1, 3)).reshape(B, T, -1)

    input_relevance_from_Q = calculate_QKV_relevance_parallel(R_Q, inp, w['W_q'])
    input_relevance_from_K = calculate_QKV_relevance_parallel(R_K_kv, inp, w['W_k'])
    input_relevance_from_V = calculate_QKV_relevance_parallel(R_V_kv, inp, w['W_v'])

    input_relevance = input_relevance_from_Q + input_relevance_from_K + input_relevance_from_V

    return input_relevance