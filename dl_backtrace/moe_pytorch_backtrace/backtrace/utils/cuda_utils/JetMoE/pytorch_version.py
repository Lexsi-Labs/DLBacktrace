import torch 
from typing import Dict, Any, Tuple, List

def torch_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid activation function."""
    return torch.sigmoid(x)

def torch_silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Swish) activation function."""
    return torch.nn.functional.silu(x)

def stabilize(matrix: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Stabilize matrix values to avoid division by zero."""
    abs_matrix = torch.abs(matrix)
    sign_matrix = torch.where(matrix == 0, 
                             torch.ones_like(matrix), 
                             torch.sign(matrix))
    
    return torch.where(abs_matrix < epsilon, 
                      epsilon * sign_matrix, 
                      matrix)

def calculate_expert_level_relevance(
    relevance_values: torch.Tensor, 
    expert_size: torch.Tensor
) -> List[float]:
    """
    Calculate expert-level relevance based on block sizes in expert_size.
    
    Args:
        relevance_values: Tensor of relevance values.
        expert_size: Tensor of block sizes for each expert.
    
    Returns:
        A list of expert-level relevance values.
    """
    start_idx = 0
    expert_level_relevance = []
    
    for size in expert_size:
        size = int(size.item())
        if size > 0:
            block_sum = torch.sum(relevance_values[start_idx:start_idx + size]).item()
            expert_level_relevance.append(block_sum)
            start_idx += size
        else:
            expert_level_relevance.append(0.0)
    
    return expert_level_relevance

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

def calculate_wt_moe_experts( # also called as `calculate_relevance_single`
    wts: torch.Tensor, 
    inp: torch.Tensor, 
    w: torch.Tensor
) -> torch.Tensor:

    batch_size, seq_len, output_features = wts.shape
    _, _, input_features = inp.shape
    
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

def calculate_jetmoe_topk_gating(
    w: Dict[str, torch.Tensor],
    inp: torch.Tensor,
    model: Any
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Calculate top-k gating for JetMoE.
    
    Returns:
        index_sorted_experts, batch_index, batch_gates, expert_size, router_logits, output_dict
    """
    output_dict = {}
    top_k = model.config.num_experts_per_tok
    device = inp.device
    
    # Router logits: (seq_len, num_experts)
    router_logits = torch.einsum('ij,kj->ki', w['W_router'], inp)
    output_dict['router_logits'] = router_logits
    
    # Top-k routing
    top_k_logits, top_k_indices = torch.topk(router_logits, top_k, dim=1)
    
    # Softmax over top-k
    top_k_gates = torch.softmax(top_k_logits, dim=1)
    output_dict['top_k_indices'] = top_k_indices
    output_dict['top_k_logits'] = top_k_logits
    output_dict['top_k_gates'] = top_k_gates
    
    # Gate assignments (binary)
    gates = torch.zeros_like(router_logits)
    gates.scatter_(1, top_k_indices, 1.0)
    output_dict['gates'] = gates
    
    # Compute expert sizes
    expert_size = gates.sum(dim=0)
    output_dict['expert_size'] = expert_size
    
    # Flatten and sort indices for top-k experts
    top_k_experts = top_k_indices.flatten()
    index_sorted_experts = torch.argsort(top_k_experts)
    batch_index = index_sorted_experts // top_k
    output_dict['index_sorted_experts'] = index_sorted_experts
    output_dict['batch_index'] = batch_index
    
    # Flatten and sort gates for grouped tokens
    top_k_gates_flat = top_k_gates.flatten()
    batch_gates = top_k_gates_flat[index_sorted_experts]
    output_dict['batch_gates'] = batch_gates
    
    return index_sorted_experts, batch_index, batch_gates, expert_size, router_logits, output_dict

def split_tensor_by_sizes(tensor: torch.Tensor, sizes: torch.Tensor) -> List[torch.Tensor]:
    """Split a tensor into chunks based on exact sizes."""
    result = []
    start_idx = 0
    for size in sizes:
        size = int(size.item())
        end_idx = start_idx + size
        result.append(tensor[start_idx:end_idx])
        start_idx = end_idx
    return result

def calculate_moe_parallel_experts(
    inp: torch.Tensor,
    w: List[torch.Tensor],
    expert_size: torch.Tensor,
    num_experts: int
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Process input through parallel MoE experts.
    
    Args:
        inp: Input tensor
        w: List of weight matrices for each expert
        expert_size: Size of each expert's input
        num_experts: Number of experts
    
    Returns:
        hidden_states: Concatenated output from all experts
        output_dict: Dictionary with intermediate states
    """
    output_dict = {}
    
    input_list = split_tensor_by_sizes(inp, expert_size)
    output_dict['input_list'] = input_list
    
    hidden_states_list = []
    for i in range(num_experts):
        if input_list[i].shape[0] > 0:
            hidden_state = torch.matmul(input_list[i], w[i].T)
            hidden_states_list.append(hidden_state)
        else:
            # Handle empty inputs
            hidden_states_list.append(torch.zeros(0, w[i].shape[0], device=inp.device, dtype=inp.dtype))
    
    output_dict['hidden_states_list'] = hidden_states_list
    
    hidden_states = torch.cat(hidden_states_list, dim=0)
    output_dict['hidden_states'] = hidden_states
    
    return hidden_states, output_dict

def merge_with_suffix(
    original_dict: Dict[str, Any],
    new_dict: Dict[str, Any],
    suffix: str
) -> Dict[str, Any]:
    """
    Merge new_dict into original_dict with a suffix appended to keys.
    """
    for k, v in new_dict.items():
        original_dict[f"{k}_{suffix}"] = v
    return original_dict

def calculate_jetmoemoe_output(
    w: Dict[str, Any],
    inp: torch.Tensor,
    model: Any
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Calculate the output of the JetMoeMoE layer.
    
    Args:
        w: Dictionary containing weights
        inp: Input tensor to the JetMoeMoE layer
        model: Model config
    
    Returns:
        layer_output_with_bias: Output tensor
        intermediate_states: Dictionary with intermediate computations
    """
    intermediate_states = {}
    
    num_experts = model.config.num_local_experts
    top_k = model.config.num_experts_per_tok
    device = inp.device
    
    batch_size_seq_len, input_size = inp.shape
    
    # Calculate gating
    _, batch_index, batch_gates, expert_size, router_logits, gating_output_dict = \
        calculate_jetmoe_topk_gating(w, inp, model)
    intermediate_states.update(gating_output_dict)
    
    # Gather expert inputs
    expert_inputs = inp[batch_index]
    intermediate_states['expert_inputs'] = expert_inputs
    
    # Calculate expert sizes
    expert_size = torch.round(expert_size).long()
    
    # Process inputs through W_in
    hidden_states, output_dict_input_experts = calculate_moe_parallel_experts(
        expert_inputs, w['W_in'], expert_size, num_experts
    )
    intermediate_states = merge_with_suffix(intermediate_states, output_dict_input_experts, 'in')
    
    # Apply chunking and activation
    chunk_size = hidden_states.shape[-1] // 2
    chunked_hidden_states = torch.split(hidden_states, chunk_size, dim=-1)
    activated_hidden = torch_silu(chunked_hidden_states[0]) * chunked_hidden_states[1]
    intermediate_states['chunked_hidden_states'] = chunked_hidden_states
    intermediate_states['activated_hidden'] = activated_hidden
    
    # Process outputs through W_out
    expert_outputs, output_dict_output_experts = calculate_moe_parallel_experts(
        activated_hidden, w['W_out'], expert_size, num_experts
    )
    intermediate_states = merge_with_suffix(intermediate_states, output_dict_output_experts, 'out')
    
    # Scale outputs by gates
    scaled_expert_outputs = expert_outputs * batch_gates.unsqueeze(1)
    intermediate_states['scaled_expert_outputs'] = scaled_expert_outputs
    
    # Accumulate outputs
    layer_output = torch.zeros((batch_size_seq_len, input_size), 
                               dtype=scaled_expert_outputs.dtype, 
                               device=device)
    layer_output.index_add_(0, batch_index, scaled_expert_outputs)
    intermediate_states['layer_output'] = layer_output
    
    # Add bias
    layer_output_with_bias = layer_output + w['bias']
    intermediate_states['layer_output_with_bias'] = layer_output_with_bias
    
    return layer_output_with_bias, intermediate_states

def calculate_wt_jetmoe_feed_forward(
    wts: torch.Tensor,
    inp: torch.Tensor,
    w: Dict[str, Any],
    model: Any
) -> Tuple[torch.Tensor, List[float]]:
    """
    Calculate relevance propagation through JetMoE feed-forward layer.
    
    Args:
        wts: Output relevance weights
        inp: Input tensor
        w: Weight dictionary
        model: Model config
    
    Returns:
        relevance_inp: Input relevance
        expert_relevance: Per-expert relevance scores
    """
    layer_output, intermediate_states = calculate_jetmoemoe_output(w, inp, model)
    
    # 1. Relevance from bias
    relevance_bias = wts * (w['bias'] / intermediate_states['layer_output_with_bias'])
    relevance_layer_output = wts - relevance_bias
    
    # 2. Relevance of layer_output propagated to scaled_expert_outputs
    gate_sums = torch.zeros_like(relevance_layer_output[:, 0])
    gate_sums.index_add_(0, intermediate_states['batch_index'], intermediate_states['batch_gates'])
    normalized_batch_gates = intermediate_states['batch_gates'] / gate_sums[intermediate_states['batch_index']]
    
    relevance_scaled_expert_outputs = torch.zeros_like(intermediate_states['scaled_expert_outputs'])
    for i, idx in enumerate(intermediate_states['batch_index']):
        relevance_scaled_expert_outputs[i] = relevance_layer_output[idx] * normalized_batch_gates[i]
    
    # 3. Relevance split between expert outputs and gates
    relevance_expert_outputs = 0.5 * relevance_scaled_expert_outputs
    relevance_batch_gates = 0.5 * relevance_scaled_expert_outputs
    
    # 4. Relevance through W_out
    expert_size = torch.round(intermediate_states['expert_size']).long()
    relevance_expert_outputs_list = split_tensor_by_sizes(relevance_expert_outputs, expert_size)
    relevance_hidden_states_list = []
    
    for i in range(model.config.num_local_experts):
        relevance_hidden_states = calculate_wt_moe_experts(
            relevance_expert_outputs_list[i],
            intermediate_states['input_list_out'][i],
            w['W_out'][i]
        )
        relevance_hidden_states_list.append(relevance_hidden_states)
    
    relevance_hidden_states = torch.cat(relevance_hidden_states_list, dim=0)
    
    # 5. Relevance through chunking
    relevance_chunked_hidden_states_0 = 0.5 * relevance_hidden_states
    relevance_chunked_hidden_states_1 = 0.5 * relevance_hidden_states
    
    relevance_hidden_states = torch.cat(
        [relevance_chunked_hidden_states_0, relevance_chunked_hidden_states_1], 
        dim=-1
    )
    
    # 6. Relevance through W_in
    relevance_hidden_states_list = split_tensor_by_sizes(relevance_hidden_states, expert_size)
    relevance_expert_inputs_list = []
    
    for i in range(model.config.num_local_experts):
        relevance_expert_inputs = calculate_wt_moe_experts(
            relevance_hidden_states_list[i],
            intermediate_states['input_list_in'][i],
            w['W_in'][i]
        )
        relevance_expert_inputs_list.append(relevance_expert_inputs)
    
    relevance_expert_inputs = torch.cat(relevance_expert_inputs_list, dim=0)
    
    # 7. Relevance of inp from expert_inputs
    relevance_inp = torch.zeros_like(inp)
    relevance_inp.index_add_(0, intermediate_states['batch_index'], relevance_expert_inputs)
    
    # 8. Relevance of top-k gating
    relevance_batch_gates = relevance_batch_gates.sum(dim=-1)
    
    # Relevance top_k_gates
    flattened_size = intermediate_states['top_k_gates'].numel()
    relevance_top_k_gates = torch.zeros(flattened_size, device=inp.device)
    relevance_top_k_gates.index_add_(0, intermediate_states['index_sorted_experts'], relevance_batch_gates)
    relevance_top_k_gates = relevance_top_k_gates.reshape(intermediate_states['top_k_gates'].shape)
    
    # Relevance router_logits
    relevance_router_logits = torch.zeros_like(intermediate_states['router_logits'])
    batch_indices = torch.arange(intermediate_states['top_k_indices'].shape[0], device=inp.device).unsqueeze(1)
    relevance_router_logits.index_put_(
        (batch_indices.expand_as(intermediate_states['top_k_indices']), 
         intermediate_states['top_k_indices']),
        relevance_top_k_gates,
        accumulate=True
    )
    
    # Relevance inp from router_logits
    relevance_inp_from_router_logits = calculate_wt_router_logits(
        relevance_router_logits, inp, w['W_router']
    )
    relevance_inp += relevance_inp_from_router_logits
    
    # Expert-level relevance
    expert_relevance = calculate_expert_level_relevance(relevance_expert_inputs, expert_size)
    
    return relevance_inp, expert_relevance

def moe_map(
    inp: torch.Tensor,
    w: Dict[str, Any],
    model: Any
) -> Tuple[torch.Tensor, torch.Tensor, Tuple, Dict[str, Any]]:
    """
    MoE map operation for JetMoE attention.
    
    Args:
        inp: Input tensor
        w: Weight dictionary
        model: Model config
    
    Returns:
        reshaped_layer_output: Output tensor reshaped for attention
        router_logits: Router logits
        topo_info: Topology information tuple
        moe_map_output: Dictionary with intermediate states
    """
    moe_map_output = {}
    num_experts = model.config.num_local_experts
    hidden_size = model.config.kv_channels * model.config.num_key_value_heads
    top_k = model.config.num_experts_per_tok
    device = inp.device
    
    length, emb_size = inp.shape
    
    # Compute gating topology
    index_sorted_experts, batch_index, batch_gates, expert_size, router_logits, gating_output_dict = \
        calculate_jetmoe_topk_gating(w, inp, model)
    moe_map_output.update(gating_output_dict)
    topo_info = (index_sorted_experts, batch_index, batch_gates, expert_size)
    
    # Group inputs according to topology
    expert_inputs = inp[batch_index]
    moe_map_output['map_expert_inputs'] = expert_inputs
    
    expert_size = torch.round(expert_size).long()
    expert_outputs, output_dict_input_experts = calculate_moe_parallel_experts(
        expert_inputs, w['W_in'], expert_size, num_experts
    )
    moe_map_output = merge_with_suffix(moe_map_output, output_dict_input_experts, 'in')
    moe_map_output.update(output_dict_input_experts)
    
    layer_output = torch.zeros((length * top_k, hidden_size), 
                               dtype=expert_outputs.dtype, 
                               device=device)
    layer_output.index_add_(0, index_sorted_experts, expert_outputs)
    moe_map_output['map_layer_output'] = layer_output
    
    reshaped_layer_output = layer_output.reshape(length, top_k, -1)
    moe_map_output['reshaped_map_layer_output'] = reshaped_layer_output
    
    return reshaped_layer_output, router_logits, topo_info, moe_map_output

def moe_reduce(
    inp: torch.Tensor,
    topo_info: Tuple,
    w: Dict[str, Any],
    config: Any
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    MoE reduce operation for JetMoE attention.
    
    Args:
        inp: Input tensor (length, k, hidden_size)
        topo_info: Topology information from moe_map
        w: Weight dictionary
        config: Model config
    
    Returns:
        layer_output_with_bias: Output tensor
        moe_reduce_output: Dictionary with intermediate states
    """
    moe_reduce_output = {}
    
    num_experts = config.num_local_experts
    input_size = config.hidden_size
    device = inp.device
    
    length, k, hidden_size = inp.shape
    layer_input = inp.reshape(-1, hidden_size)
    moe_reduce_output['reduce_layer_input'] = layer_input
    
    index_sorted_experts, batch_index, batch_gates, expert_size = topo_info
    
    expert_inputs = layer_input[index_sorted_experts]
    moe_reduce_output['reduce_expert_inputs'] = expert_inputs
    
    expert_size = torch.round(expert_size).long()
    expert_outputs, output_dict_output_experts = calculate_moe_parallel_experts(
        expert_inputs, w['W_out'], expert_size, num_experts
    )
    moe_reduce_output = merge_with_suffix(moe_reduce_output, output_dict_output_experts, 'out')
    moe_reduce_output.update(output_dict_output_experts)
    
    scaled_expert_outputs = expert_outputs * batch_gates.unsqueeze(1)
    moe_reduce_output['scaled_expert_outputs'] = scaled_expert_outputs
    
    layer_output = torch.zeros((length, input_size), 
                               dtype=expert_outputs.dtype, 
                               device=device)
    layer_output.index_add_(0, batch_index, scaled_expert_outputs)
    moe_reduce_output['reduce_layer_output'] = layer_output
    
    layer_output_with_bias = layer_output + w['bias']
    moe_reduce_output['reduce_layer_output_with_bias'] = layer_output_with_bias
    
    return layer_output_with_bias, moe_reduce_output

def calculate_moe_moa_output(
    inp: torch.Tensor,
    w: Dict[str, Any],
    model: Any
) -> Dict[str, Any]:
    """
    Calculate MoE Mixture of Attention (MoA) output.
    
    Args:
        inp: Input tensor
        w: Weight dictionary
        model: Model config
    
    Returns:
        intermediate_states: Dictionary with all intermediate computations
    """
    intermediate_states = {}
    device = inp.device
    
    q_len, _ = inp.shape
    kv_projection_size = model.config.kv_channels * model.config.num_key_value_heads
    
    # MoE map for query projection
    query_states, router_logits, topo_info, moe_map_output = moe_map(inp, w, model)
    intermediate_states.update(moe_map_output)
    
    # Key-Value projection
    projected = torch.matmul(inp, w['W_kv'].T)
    intermediate_states['projected'] = projected
    
    key_states, value_states = torch.chunk(projected, 2, dim=-1)
    
    num_heads = model.config.num_attention_heads
    top_k = model.config.num_experts_per_tok
    num_key_value_heads = getattr(model.config, 'num_key_value_heads', model.config.num_heads)
    head_dim = model.config.kv_channels
    
    # Reshape for attention computation
    query_states = query_states.reshape(q_len, num_heads, head_dim).transpose(0, 1)
    key_states = key_states.reshape(q_len, num_key_value_heads, head_dim).transpose(0, 1)
    value_states = value_states.reshape(q_len, num_key_value_heads, head_dim).transpose(0, 1)
    
    # Repeat key/value states for top-k experts
    key_states = key_states.repeat_interleave(top_k, dim=0)
    value_states = value_states.repeat_interleave(top_k, dim=0)
    
    intermediate_states['query_states'] = query_states
    intermediate_states['key_states'] = key_states
    intermediate_states['value_states'] = value_states
    
    # Attention computation
    QK_output = torch.einsum('hqd,hkd->hqk', query_states, key_states)
    intermediate_states['QK_output'] = QK_output
    attn_weights = QK_output / torch.sqrt(torch.tensor(head_dim, dtype=QK_output.dtype, device=device))
    
    # Softmax
    attn_weights = torch.softmax(attn_weights, dim=-1)
    intermediate_states['attn_weights'] = attn_weights
    
    # Weighted sum of values
    attn_output = torch.einsum('hqk,hkl->hql', attn_weights, value_states)
    intermediate_states['attn_output'] = attn_output
    
    # Reshape attention output
    attn_output = attn_output.permute(1, 0, 2)
    intermediate_states['attn_output'] = attn_output
    
    reshaped_attn_output = attn_output.reshape(q_len, top_k, kv_projection_size)
    intermediate_states['reshaped_attn_output'] = reshaped_attn_output
    
    # MoE reduce
    layer_output_with_bias, moe_reduce_output = moe_reduce(reshaped_attn_output, topo_info, w, model.config)
    intermediate_states.update(moe_reduce_output)
    
    return intermediate_states

def calculate_wt_jetmoe_self_attention_parallel(
    wts: torch.Tensor,
    inp: torch.Tensor,
    w: Dict[str, Any],
    model: Any
) -> Tuple[torch.Tensor, List[float]]:
    """
    Calculate relevance propagation through JetMoE self-attention layer.
    
    Args:
        wts: Output relevance weights
        inp: Input tensor
        w: Weight dictionary
        model: Model config
    
    Returns:
        relevance_inp: Input relevance
        expert_relevance: Per-expert relevance scores
    """
    q_len, _ = inp.shape
    device = inp.device
    
    # Forward pass
    intermediate_states = calculate_moe_moa_output(inp, w, model)
    
    # 1. Relevance from bias
    relevance_bias = wts * (w['bias'] / intermediate_states['reduce_layer_output_with_bias'])
    relevance_reduce_layer_output = wts - relevance_bias
    
    # 2. Relevance propagated to scaled_expert_outputs
    gate_sums = torch.zeros_like(relevance_reduce_layer_output[:, 0])
    gate_sums.index_add_(0, intermediate_states['batch_index'], intermediate_states['batch_gates'])
    normalized_batch_gates = intermediate_states['batch_gates'] / gate_sums[intermediate_states['batch_index']]
    
    relevance_scaled_expert_outputs = torch.zeros_like(intermediate_states['scaled_expert_outputs'])
    for i, idx in enumerate(intermediate_states['batch_index']):
        relevance_scaled_expert_outputs[i] = relevance_reduce_layer_output[idx] * normalized_batch_gates[i]
    
    # 3. Split relevance
    relevance_expert_outputs = 0.5 * relevance_scaled_expert_outputs
    relevance_batch_gates = 0.5 * relevance_scaled_expert_outputs
    
    # 4. Relevance through W_out
    expert_size = torch.round(intermediate_states['expert_size']).long()
    relevance_expert_outputs_list = split_tensor_by_sizes(relevance_expert_outputs, expert_size)
    relevance_hidden_states_list = []
    
    for i in range(model.config.num_local_experts):
        relevance_hidden_states = calculate_wt_moe_experts(
            relevance_expert_outputs_list[i],
            intermediate_states['input_list_out'][i],
            w['W_out'][i]
        )
        relevance_hidden_states_list.append(relevance_hidden_states)
    
    relevance_reduce_expert_inputs = torch.cat(relevance_hidden_states_list, dim=0)
    
    # 5. Relevance of layer_input from expert_inputs
    relevance_reduce_layer_input = torch.zeros_like(intermediate_states['reduce_layer_input'])
    relevance_reduce_layer_input.index_add_(0, intermediate_states['index_sorted_experts'], relevance_reduce_expert_inputs)
    
    relevance_reshaped_attn_output = relevance_reduce_layer_input.reshape(
        intermediate_states['reshaped_attn_output'].shape
    )
    
    # 6. Relevance through self-attention
    relevance_attn_output = relevance_reshaped_attn_output.reshape(
        intermediate_states['attn_output'].shape
    )
    
    stabilized_attn_output = stabilize(intermediate_states['attn_output'] * 2)
    norm_wt_mat_attn = relevance_attn_output / stabilized_attn_output
    norm_wt_mat_attn = norm_wt_mat_attn.permute(1, 0, 2)
    
    relevance_QK = torch.einsum('htd,hbd->htb', norm_wt_mat_attn, intermediate_states['value_states']) * \
                   intermediate_states['attn_weights']
    relevance_V = torch.einsum('hdt,hdb->htb', intermediate_states['attn_weights'], norm_wt_mat_attn) * \
                  intermediate_states['value_states']
    
    stabilized_QK_output = stabilize(intermediate_states['QK_output'] * 2)
    norm_wt_mat_QK = relevance_QK / stabilized_QK_output
    relevance_Q = torch.einsum('htd,hdb->htb', norm_wt_mat_QK, intermediate_states['key_states']) * \
                  intermediate_states['query_states']
    relevance_K = torch.einsum('htd,htb->hbd', intermediate_states['query_states'], norm_wt_mat_QK) * \
                  intermediate_states['key_states']
    
    relevance_Q = relevance_Q.permute(1, 0, 2).reshape(q_len, -1)
    relevance_K = relevance_K.permute(1, 0, 2).reshape(q_len, -1)
    relevance_V = relevance_V.permute(1, 0, 2).reshape(q_len, -1)
    
    relevance_projected = relevance_K + relevance_V
    
    # Calculate relevance from KV projection
    relevance_inp_from_kv = calculate_wt_moe_experts(
        relevance_projected.unsqueeze(0), 
        inp.unsqueeze(0), 
        w['W_kv']
    ).squeeze(0)
    
    # 7. Relevance through moe_map
    relevance_map_reshaped_layer_output = relevance_Q.reshape(
        intermediate_states['reshaped_map_layer_output'].shape
    )
    
    relevance_map_layer_output = relevance_map_reshaped_layer_output.reshape(
        intermediate_states['map_layer_output'].shape
    )
    
    # Calculate relevance of expert_outputs
    gate_sums = torch.zeros_like(relevance_map_layer_output[:, 0])
    gate_sums.index_add_(0, intermediate_states['index_sorted_experts'], intermediate_states['batch_gates'])
    normalized_batch_gates = intermediate_states['batch_gates'] / gate_sums[intermediate_states['index_sorted_experts']]
    
    relevance_expert_outputs = torch.zeros_like(intermediate_states['hidden_states_in'])
    for i, idx in enumerate(intermediate_states['index_sorted_experts']):
        relevance_expert_outputs[i] = relevance_map_layer_output[idx] * normalized_batch_gates[i]
    
    # Relevance through W_in
    relevance_hidden_states_list = split_tensor_by_sizes(relevance_expert_outputs, expert_size)
    relevance_expert_inputs_list = []
    
    for i in range(model.config.num_local_experts):
        relevance_hidden_states = calculate_wt_moe_experts(
            relevance_hidden_states_list[i],
            intermediate_states['input_list_in'][i],
            w['W_in'][i]
        )
        relevance_expert_inputs_list.append(relevance_hidden_states)
    
    relevance_map_expert_inputs = torch.cat(relevance_expert_inputs_list, dim=0)
    
    # Relevance for inp from expert_inputs
    relevance_inp_from_expert_inputs = torch.zeros_like(inp)
    relevance_inp_from_expert_inputs.index_add_(0, intermediate_states['batch_index'], relevance_map_expert_inputs)
    
    # 8. Relevance of top-k gating
    relevance_batch_gates = relevance_batch_gates.sum(dim=-1)
    
    # Relevance top_k_gates
    flattened_size = intermediate_states['top_k_gates'].numel()
    relevance_top_k_gates = torch.zeros(flattened_size, device=device)
    relevance_top_k_gates.index_add_(0, intermediate_states['index_sorted_experts'], relevance_batch_gates)
    relevance_top_k_gates = relevance_top_k_gates.reshape(intermediate_states['top_k_gates'].shape)
    
    # Relevance router_logits
    relevance_router_logits = torch.zeros_like(intermediate_states['router_logits'])
    batch_indices = torch.arange(intermediate_states['top_k_indices'].shape[0], device=device).unsqueeze(1)
    relevance_router_logits.index_put_(
        (batch_indices.expand_as(intermediate_states['top_k_indices']),
         intermediate_states['top_k_indices']),
        relevance_top_k_gates,
        accumulate=True
    )
    
    # Relevance inp from router_logits
    relevance_inp_from_router_logits = calculate_wt_router_logits(
        relevance_router_logits, inp, w['W_router']
    )
    
    # Total input relevance
    relevance_inp = relevance_inp_from_kv + relevance_inp_from_expert_inputs + relevance_inp_from_router_logits
    
    # Expert-level relevance
    expert_relevance = calculate_expert_level_relevance(relevance_map_expert_inputs, expert_size)
    
    return relevance_inp, expert_relevance