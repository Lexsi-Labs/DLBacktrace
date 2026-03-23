import numpy as np

def np_sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z

def np_silu(x):
    return x * np_sigmoid(x)

def stabilize(matrix, epsilon=1e-6):
    return np.where(np.abs(matrix) < epsilon,
                    epsilon * np.sign(matrix + (matrix == 0)),
                    matrix)

def calculate_expert_level_relevance(relevance_values, expert_size):
    """
    Calculate expert-level relevance based on block sizes in expert_size.

    Parameters:
    - relevance_values: List or numpy array of relevance values.
    - expert_size: List or numpy array of block sizes for each expert.

    Returns:
    - A list of expert-level relevance values.
    """
    start_idx = 0
    expert_level_relevance = []

    for size in expert_size:
        size = int(size)  # Convert size to integer
        if size > 0:
            block_sum = np.sum(relevance_values[start_idx:start_idx + size])
            expert_level_relevance.append(block_sum)
            start_idx += size
        else:
            expert_level_relevance.append(0)  # Append 0 for block size 0

    return expert_level_relevance


def calculate_jetmoe_topk_gating(w, inp, model):
    output_dict = {}
    top_k = model.config.num_experts_per_tok

    # Router logits
    router_logits = np.einsum('ij,kj->ki', w['W_router'], inp)
    output_dict['router_logits'] = router_logits

    # Top-k routing
    top_k_indices = np.argsort(router_logits, axis=1)[:, -top_k:]  # Indices of top-k experts
    top_k_logits = np.take_along_axis(router_logits, top_k_indices, axis=1)  # Top-k logits
    top_k_gates = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits), axis=1, keepdims=True)  # Softmax over top-k
    output_dict['top_k_indices'] = top_k_indices
    output_dict['top_k_logits'] = top_k_logits
    output_dict['top_k_gates'] = top_k_gates

    # Gate assignments (binary)
    gates = np.zeros_like(router_logits)
    np.put_along_axis(gates, top_k_indices, 1, axis=1)  # Shape: (sequence_length, num_experts)
    output_dict['gates'] = gates

    # Compute expert sizes
    expert_size = np.sum(gates, axis=0)
    output_dict['expert_size'] = expert_size

    # Flatten and sort indices for top-k experts
    top_k_experts = top_k_indices.flatten()    # Shape: (sequence_length * top_k,)
    index_sorted_experts = np.argsort(top_k_experts)
    batch_index = index_sorted_experts // top_k    # Repeated indices for top-k inputs
    output_dict['index_sorted_experts'] = index_sorted_experts
    output_dict['batch_index'] = batch_index

    # Flatten and sort gates for grouped tokens
    top_k_gates = top_k_gates.flatten()
    batch_gates = top_k_gates[index_sorted_experts]
    output_dict['batch_gates'] = batch_gates

    return index_sorted_experts, batch_index, batch_gates, expert_size, router_logits, output_dict


def split_array_by_sizes(array, sizes):
    """Split a NumPy array into chunks based on exact sizes."""
    result = []
    start_idx = 0
    for size in sizes:
        end_idx = start_idx + size
        result.append(array[start_idx:end_idx])
        start_idx = end_idx
    return result


def calculate_moe_parallel_experts(inp, w,  expert_size, num_experts):
    output_dict = {}

    input_list = split_array_by_sizes(inp, expert_size)
    output_dict['input_list'] = input_list

    hidden_states_list = []
    for i in range(num_experts):
        hidden_state = np.dot(w[i], input_list[i].T).T
        hidden_states_list.append(hidden_state)

    output_dict['hidden_states_list'] = hidden_states_list

    hidden_states = np.concatenate(hidden_states_list, axis=0)
    output_dict['hidden_states'] = hidden_states
    return hidden_states , output_dict


def merge_with_suffix(original_dict, new_dict, suffix):
    """
    Merge new_dict into original_dict with a suffix appended to keys to avoid overwriting.
    """
    for k, v in new_dict.items():
        original_dict[f"{k}_{suffix}"] = v
    return original_dict


def calculate_jetmoemoe_output(w, inp, model):
    """
    Calculates the output of the JetMoeMoE layer.

    Args:
        w (dict): Dictionary containing the weights of JetMoeMoE, JetMoeParallelExperts, and JetMoeTopKGating.
        inp (np.ndarray): Input tensor to the JetMoeMoE layer.

    Returns:
        torch.Tensor: Output tensor from the JetMoeMoE layer.
        torch.Tensor: Router logits.
    """
    intermediate_states = {}

    num_experts = model.config.num_local_experts
    top_k = model.config.num_experts_per_tok
    activation_function = model.config.activation_function    # silu activation

    batch_size_seq_len, input_size = inp.shape

    ########################### Calculate Experts ##############################
    # print(f"\n\n Calculate Experts...\n")
    _, batch_index, batch_gates, expert_size, router_logits, gating_output_dict = calculate_jetmoe_topk_gating(w, inp, model)
    intermediate_states = intermediate_states | gating_output_dict

    #############################  Call JetMoeParallelExperts  #################
    #### Gather expert inputs
    expert_inputs = inp[batch_index]
    intermediate_states['expert_inputs'] = expert_inputs

    #### Calculate expert sizes
    expert_size = np.round(expert_size).astype(int)

    # Process inputs for each expert using W_in
    hidden_states, output_dict_input_experts  = calculate_moe_parallel_experts(expert_inputs, w['W_in'],  expert_size, num_experts)
    intermediate_states = merge_with_suffix(intermediate_states, output_dict_input_experts, 'in')

    num_chunks = 2
    chunk_size = hidden_states.shape[-1] // 2
    chunked_hidden_states = np.split(hidden_states, [chunk_size], axis=-1)
    activated_hidden = np_silu(chunked_hidden_states[0]) * chunked_hidden_states[1]
    intermediate_states['chunked_hidden_states'] = chunked_hidden_states
    intermediate_states['activated_hidden'] = activated_hidden

    # Process outputs for each expert using W_out
    expert_outputs, output_dict_output_experts = calculate_moe_parallel_experts(activated_hidden, w['W_out'],  expert_size, num_experts)
    intermediate_states = merge_with_suffix(intermediate_states, output_dict_output_experts, 'out')

    # Scale outputs by gates
    scaled_expert_outputs = expert_outputs * batch_gates[:, None]
    intermediate_states['scaled_expert_outputs'] = scaled_expert_outputs

    # Accumulate outputs
    layer_output = np.zeros((batch_size_seq_len, input_size), dtype=scaled_expert_outputs.dtype)
    np.add.at(layer_output, batch_index, scaled_expert_outputs)  # Accumulate outputs from experts
    intermediate_states['layer_output'] = layer_output

    # Add bias
    layer_output_with_bias = layer_output + w['bias']
    intermediate_states['layer_output_with_bias'] = layer_output_with_bias

    return layer_output_with_bias, intermediate_states


def calculate_wt_moe_experts(wts, inp, w):

    relevance_input = np.zeros_like(inp)

    for i in range(wts.shape[0]):
        R = wts[i]  # shape: (tokens,)
        contribution_matrix = np.einsum('ij,j->ij', w, inp[i])  # shape: (tokens, features)
        wt_mat = np.zeros_like(contribution_matrix)

        for j in range(contribution_matrix.shape[0]):
            l1_ind1 = contribution_matrix[j]
            wt = R[j]

            # Positive and negative indices
            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0

            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1

            # Aggregate positive/negative weights
            p_agg_wt = p_sum / (p_sum + n_sum) if p_sum > 0 else 0
            n_agg_wt = n_sum / (p_sum + n_sum) if n_sum > 0 else 0

            p_sum = p_sum if p_sum != 0 else 1
            n_sum = n_sum if n_sum != 0 else 1

            # Weighted contributions
            wt_mat[j][p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_mat[j][n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        # Sum across all tokens to get the relevance row
        relevance_input[i] = wt_mat.sum(axis=0)

    return relevance_input    

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

def calculate_wt_jetmoe_feed_forward(wts, inp, w, model):
    layer_output, intermediate_states = calculate_jetmoemoe_output(w, inp, model)

    relevance_bias = wts * (w['bias'] / intermediate_states['layer_output_with_bias'])
    relevance_layer_output = wts - relevance_bias

    # 2. Relevance of `layer_output` propagated to `scaled_expert_outputs`
    gate_sums = np.zeros_like(relevance_layer_output[:, 0])  # Shape: (8,)
    np.add.at(gate_sums, intermediate_states['batch_index'], intermediate_states['batch_gates'])
    normalized_batch_gates = intermediate_states['batch_gates'] / gate_sums[intermediate_states['batch_index']]

    relevance_scaled_expert_outputs = np.zeros_like(intermediate_states['scaled_expert_outputs'])
    for i, idx in enumerate(intermediate_states['batch_index']):
        relevance_scaled_expert_outputs[i] = relevance_layer_output[idx] * normalized_batch_gates[i]

    # 3. Relevance of `scaled_expert_outputs` propagated through `batch gates`
    relevance_expert_outputs = 0.5 * relevance_scaled_expert_outputs
    relevance_batch_gates = 0.5 * relevance_scaled_expert_outputs

    # 4. Relevance calculation of Process outputs for each expert using `W_out`
    expert_size = np.round(intermediate_states['expert_size']).astype(int)
    relevance_expert_outputs_list = split_array_by_sizes(relevance_expert_outputs, expert_size)
    relevance_hidden_states_list = []
    for i in range(model.config.num_local_experts):
        relevance_hidden_states = calculate_wt_moe_experts(relevance_expert_outputs_list[i], intermediate_states['input_list_out'][i], w['W_out'][i])
        relevance_hidden_states_list.append(relevance_hidden_states)

    relevance_hidden_states = np.concatenate(relevance_hidden_states_list, axis=0)

    # 5. Relevance calculation for `chunking`
    relevance_chunked_hidden_states_0 = 0.5 * relevance_hidden_states
    relevance_chunked_hidden_states_1 = 0.5 * relevance_hidden_states

    relevance_hidden_states = np.concatenate([relevance_chunked_hidden_states_0, relevance_chunked_hidden_states_1], axis=-1)

    # 6. Relevance calculation of Process inputs for each expert using `W_in`
    relevance_hidden_states_list = split_array_by_sizes(relevance_hidden_states, expert_size)
    relevance_expert_inputs_list = []
    for i in range(model.config.num_local_experts):
        relevance_expert_inputs = calculate_wt_moe_experts(relevance_hidden_states_list[i], intermediate_states['input_list_in'][i], w['W_in'][i])
        relevance_expert_inputs_list.append(relevance_expert_inputs)

    relevance_expert_inputs = np.concatenate(relevance_expert_inputs_list, axis=0)

    # 7. Relevance of `inp` from `expert_inputs`
    relevance_inp = np.zeros(inp.shape)
    np.add.at(relevance_inp, intermediate_states['batch_index'], relevance_expert_inputs)

    # 8. Relevance of `top-k gating`
    relevance_batch_gates = relevance_batch_gates.sum(axis=-1)

    ######################  Relevance `top_k_gates`
    flattened_size = np.prod(intermediate_states['top_k_gates'].shape)
    relevance_top_k_gates = np.zeros(flattened_size)
    np.add.at(relevance_top_k_gates, intermediate_states['index_sorted_experts'], relevance_batch_gates)
    relevance_top_k_gates = relevance_top_k_gates.reshape(intermediate_states['top_k_gates'].shape)

    ######################  Relevance `router_logits`
    relevance_router_logits = np.zeros(intermediate_states['router_logits'].shape)
    np.add.at(
        relevance_router_logits,
        (np.arange(intermediate_states['top_k_indices'].shape[0])[:, None], intermediate_states['top_k_indices']),
        relevance_top_k_gates)

    ###################### Relevance `inp` from `router_logits`
    relevance_inp_from_router_logits = calculate_wt_router_logits(relevance_router_logits, inp, w['W_router'])
    relevance_inp += relevance_inp_from_router_logits

    ######## Relevance of Expert
    expert_relevance = calculate_expert_level_relevance(relevance_expert_inputs, expert_size)

    return relevance_inp, expert_relevance   

def moe_map(inp, w, model):
    moe_map_output = {}
    num_experts = model.config.num_local_experts
    input_size = model.config.hidden_size
    hidden_size = model.config.kv_channels * model.config.num_key_value_heads
    top_k = model.config.num_experts_per_tok

    # Compute gating topology
    length, emb_size = inp.shape

    index_sorted_experts, batch_index, batch_gates, expert_size, router_logits, gating_output_dict = calculate_jetmoe_topk_gating(w, inp, model)
    moe_map_output = moe_map_output | gating_output_dict
    topo_info = (index_sorted_experts, batch_index, batch_gates, expert_size)

    # Group inputs according to topology and compute query projection
    expert_inputs = inp[batch_index]
    moe_map_output['map_expert_inputs'] = expert_inputs

    expert_size = np.round(expert_size).astype(int)
    expert_outputs, output_dict_input_experts = calculate_moe_parallel_experts(expert_inputs, w['W_in'],  expert_size, num_experts)
    moe_map_output = merge_with_suffix(moe_map_output, output_dict_input_experts, 'in')
    moe_map_output = moe_map_output | output_dict_input_experts

    layer_output = np.zeros((length * top_k, hidden_size), dtype=expert_outputs.dtype)
    np.add.at(layer_output, index_sorted_experts, expert_outputs)
    moe_map_output['map_layer_output'] = layer_output

    reshaped_layer_output = layer_output.reshape(length, top_k, -1)
    moe_map_output['reshaped_map_layer_output'] = reshaped_layer_output

    return reshaped_layer_output, router_logits, topo_info, moe_map_output


def moe_reduce(inp, topo_info, w, config):
    moe_reduce_output = {}

    num_experts = config.num_local_experts
    input_size = config.hidden_size

    length, k, hidden_size = inp.shape
    layer_input = inp.reshape(-1, hidden_size)
    moe_reduce_output['reduce_layer_input'] = layer_input

    index_sorted_experts, batch_index, batch_gates, expert_size = topo_info

    expert_inputs = layer_input[index_sorted_experts]
    moe_reduce_output['reduce_expert_inputs'] = expert_inputs

    expert_size = np.round(expert_size).astype(int)
    expert_outputs, output_dict_output_experts = calculate_moe_parallel_experts(expert_inputs, w['W_out'],  expert_size, num_experts)
    moe_reduce_output = merge_with_suffix(moe_reduce_output, output_dict_output_experts, 'out')
    moe_reduce_output = moe_reduce_output | output_dict_output_experts

    scaled_expert_outputs = expert_outputs * batch_gates[:, None]
    moe_reduce_output['scaled_expert_outputs'] = scaled_expert_outputs

    layer_output = np.zeros((length , input_size), dtype=expert_outputs.dtype)
    np.add.at(layer_output, batch_index, scaled_expert_outputs)
    moe_reduce_output['reduce_layer_output'] = layer_output

    layer_output_with_bias = layer_output + w['bias']
    moe_reduce_output['reduce_layer_output_with_bias'] = layer_output_with_bias

    return layer_output_with_bias, moe_reduce_output


def calculate_moe_moa_output(inp, w, model):
    intermediate_states = {}

    q_len, _ = inp.shape
    kv_projection_size = model.config.kv_channels * model.config.num_key_value_heads

    query_states, router_logits, topo_info, moe_map_output = moe_map(inp, w, model)
    intermediate_states = intermediate_states | moe_map_output

    projected = np.dot(inp, w['W_kv'].T)
    intermediate_states['projected'] = projected

    key_states, value_states = np.split(projected, 2, axis=-1)

    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    top_k = model.config.num_experts_per_tok
    # Check if the config has 'num_key_value_heads' attribute
    if hasattr(model.config, 'num_key_value_heads'):
        num_key_value_heads = model.config.num_key_value_heads
    else:
        num_key_value_heads = model.config.num_heads
    head_dim = model.config.kv_channels  # dimension of each attention head

    query_states = query_states.reshape(q_len, num_heads, head_dim).transpose(1, 0, 2)
    key_states = key_states.reshape(q_len, num_key_value_heads, head_dim).transpose(1, 0, 2)
    value_states = value_states.reshape(q_len, num_key_value_heads, head_dim).transpose(1, 0, 2)

    key_states = np.repeat(key_states, top_k, axis=0)
    value_states = np.repeat(value_states, top_k, axis=0)

    intermediate_states['query_states'] = query_states
    intermediate_states['key_states'] = key_states
    intermediate_states['value_states'] = value_states

    QK_output = np.einsum('hqd,hkd->hqk', query_states, key_states)    # (num_heads, num_tokens, num_tokens)
    intermediate_states['QK_output'] = QK_output
    attn_weights = QK_output / np.sqrt(head_dim)

    # Apply softmax along the last dimension (softmax over key dimension)
    attn_weights = np.exp(attn_weights - np.max(attn_weights, axis=-1, keepdims=True))  # Numerically stable softmax
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    intermediate_states['attn_weights'] = attn_weights

    # Weighted sum of values (num_heads, num_tokens, head_dim)
    attn_output = np.einsum('hqk,hkl->hql', attn_weights, value_states)
    intermediate_states['attn_output'] = attn_output

    # Reshape attention output back to original shape (num_tokens, hidden_size)
    attn_output = np.einsum('hqd->qhd', attn_output)
    intermediate_states['attn_output'] = attn_output

    reshaped_attn_output = attn_output.reshape(q_len, top_k, kv_projection_size)
    intermediate_states['reshaped_attn_output'] = reshaped_attn_output

    layer_output_with_bias, moe_reduce_output = moe_reduce(reshaped_attn_output, topo_info, w, model.config)
    intermediate_states = intermediate_states | moe_reduce_output

    return intermediate_states

def calculate_wt_jetmoe_self_attention_parallel(wts, inp, w, model):
    inp = inp.detach().numpy()
    q_len, _ = inp.shape

    intermediate_states = calculate_moe_moa_output(inp, w, model)

    #########  Relevance Calculation of JetMoEMoE Output ################

    # 1. Relevance of `layer_output_with_bias`
    relevance_bias = wts * (w['bias'] / intermediate_states['reduce_layer_output_with_bias'])
    relevance_reduce_layer_output = wts - relevance_bias

    # 2. Relevance of `layer_output` propagated to `scaled_expert_outputs`
    gate_sums = np.zeros_like(relevance_reduce_layer_output[:, 0])  # Shape: (8,)
    np.add.at(gate_sums, intermediate_states['batch_index'], intermediate_states['batch_gates'])
    normalized_batch_gates = intermediate_states['batch_gates'] / gate_sums[intermediate_states['batch_index']]

    relevance_scaled_expert_outputs = np.zeros_like(intermediate_states['scaled_expert_outputs'])
    for i, idx in enumerate(intermediate_states['batch_index']):
        relevance_scaled_expert_outputs[i] = relevance_reduce_layer_output[idx] * normalized_batch_gates[i]

    # 3. Relevance of `scaled_expert_outputs` propagated through `batch gates`
    relevance_expert_outputs = 0.5 * relevance_scaled_expert_outputs
    relevance_batch_gates = 0.5 * relevance_scaled_expert_outputs

    # 4. Relevance calculation of Process outputs for each expert using `W_out`
    expert_size = np.round(intermediate_states['expert_size']).astype(int)
    relevance_expert_outputs_list = split_array_by_sizes(relevance_expert_outputs, expert_size)
    relevance_hidden_states_list = []
    for i in range(model.config.num_local_experts):
        relevance_hidden_states = calculate_wt_moe_experts(relevance_expert_outputs_list[i], intermediate_states['input_list_out'][i], w['W_out'][i])
        relevance_hidden_states_list.append(relevance_hidden_states)

    relevance_reduce_expert_inputs = np.concatenate(relevance_hidden_states_list, axis=0)

    # 5. Relevance of `layer_input` from `expert_inputs`
    relevance_reduce_layer_input = np.zeros(intermediate_states['reduce_layer_input'] .shape)
    np.add.at(relevance_reduce_layer_input, intermediate_states['index_sorted_experts'], relevance_reduce_expert_inputs)

    relevance_reshaped_attn_output = relevance_reduce_layer_input.reshape(intermediate_states['reshaped_attn_output'].shape)

    # 6. Calculate relevance of `Self-Attention`
    relevance_attn_output = relevance_reshaped_attn_output.reshape(intermediate_states['attn_output'].shape)

    stabilized_attn_output = stabilize(intermediate_states['attn_output'] * 2)
    norm_wt_mat_attn = relevance_attn_output / stabilized_attn_output
    norm_wt_mat_attn = np.einsum('qhd->hqd', norm_wt_mat_attn)
    relevance_QK = np.einsum('htd,hbd->htb', norm_wt_mat_attn, intermediate_states['value_states']) * intermediate_states['attn_weights']
    relevance_V = np.einsum('hdt,hdb->htb', intermediate_states['attn_weights'], norm_wt_mat_attn)  * intermediate_states['value_states']

    stabilized_QK_output = stabilize(intermediate_states['QK_output'] * 2)
    norm_wt_mat_QK = relevance_QK / stabilized_QK_output
    relevance_Q = np.einsum('htd,hdb->htb', norm_wt_mat_QK, intermediate_states['key_states']) * intermediate_states['query_states']
    relevance_K = np.einsum('htd,htb->hbd', intermediate_states['query_states'], norm_wt_mat_QK) * intermediate_states['key_states']

    relevance_Q = np.einsum('hqd->qhd', relevance_Q).reshape(q_len, -1)
    relevance_K = np.einsum('hqd->qhd', relevance_K).reshape(q_len, -1)
    relevance_V = np.einsum('hqd->qhd', relevance_V).reshape(q_len, -1)

    relevance_projected = relevance_K + relevance_V

    relevance_inp_from_kv = calculate_wt_moe_experts(relevance_projected, inp, w['W_kv'])  # np.dot(relevance_projected, w['W_kv'])

    # 7. Calculate relevance of `moe_map`
    relevance_map_reshaped_layer_output = relevance_Q.reshape(intermediate_states['reshaped_map_layer_output'].shape)

    ######## calculate relevance of `layer_output`
    relevance_map_layer_output = relevance_map_reshaped_layer_output.reshape(intermediate_states['map_layer_output'].shape)

    ######## calculate relevance of `expert_outputs`
    gate_sums = np.zeros_like(relevance_map_layer_output[:, 0])
    np.add.at(gate_sums, intermediate_states['index_sorted_experts'], intermediate_states['batch_gates'])
    normalized_batch_gates = intermediate_states['batch_gates'] / gate_sums[intermediate_states['index_sorted_experts']]

    relevance_expert_outputs = np.zeros_like(intermediate_states['hidden_states_in'])
    for i, idx in enumerate(intermediate_states['index_sorted_experts']):
        relevance_expert_outputs[i] = relevance_map_layer_output[idx] * normalized_batch_gates[i]

    ######## calculate relevance of Process outputs for each expert using `W_in`
    relevance_hidden_states_list = split_array_by_sizes(relevance_expert_outputs, expert_size)
    relevance_expert_inputs_list = []
    for i in range(model.config.num_local_experts):
        relevance_hidden_states = calculate_wt_moe_experts(relevance_hidden_states_list[i], intermediate_states['input_list_in'][i], w['W_in'][i])
        relevance_expert_inputs_list.append(relevance_hidden_states)

    relevance_map_expert_inputs = np.concatenate(relevance_expert_inputs_list, axis=0)

    ######## calculate relevance for `inp` from `expert_inputs`
    relevance_inp_from_expert_inputs = np.zeros(inp.shape)
    np.add.at(relevance_inp_from_expert_inputs, intermediate_states['batch_index'], relevance_map_expert_inputs)

    ######### relevance of `top-k gating`
    relevance_batch_gates = relevance_batch_gates.sum(axis=-1)

    ######################  Relevance `top_k_gates`
    flattened_size = np.prod(intermediate_states['top_k_gates'].shape)
    relevance_top_k_gates = np.zeros(flattened_size)
    np.add.at(relevance_top_k_gates, intermediate_states['index_sorted_experts'], relevance_batch_gates)
    relevance_top_k_gates = relevance_top_k_gates.reshape(intermediate_states['top_k_gates'].shape)

    ######################  Relevance `router_logits`
    relevance_router_logits = np.zeros(intermediate_states['router_logits'].shape)
    np.add.at(
        relevance_router_logits,
        (np.arange(intermediate_states['top_k_indices'].shape[0])[:, None], intermediate_states['top_k_indices']),
        relevance_top_k_gates)

    ###################### Relevance `inp` from `router_logits`
    relevance_inp_from_router_logits = calculate_wt_router_logits(relevance_router_logits, inp, w['W_router'])

    ###################### Calculate Total relevance of `inp`
    relevance_inp = relevance_inp_from_kv + relevance_inp_from_expert_inputs + relevance_inp_from_router_logits

    ######## Relevance of Expert
    expert_relevance = calculate_expert_level_relevance(relevance_map_expert_inputs, expert_size)

    return relevance_inp, expert_relevance