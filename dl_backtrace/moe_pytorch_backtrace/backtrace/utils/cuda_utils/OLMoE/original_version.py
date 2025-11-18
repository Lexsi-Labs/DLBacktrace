import torch
import torch.nn.functional as F
import numpy as np

def np_swish(x, beta=0.75):
    z = 1 / (1 + np.exp(-np.clip(beta * x, -500, 500)))
    return x * z 

def process_single_relevance_router_logits(wts, input, W_router):
    print("Running original version of process_single_relevance_router_logits")
    wt_mat_total = np.zeros(input.shape)
    
    for i in range(wts.shape[0]):
        R = wts[i]
        contribution_matrix = W_router * input[i]
        wt_mat = np.zeros(contribution_matrix.shape)
        for j in range(contribution_matrix.shape[0]):
            l1_ind1 = contribution_matrix[j]
            wt = R[j]
            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1
            t_sum = p_sum - n_sum

            if t_sum < -1:
                p_sum = 0
            if t_sum > 2:
                n_sum = 0
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
            wt_mat[j][p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_mat[j][n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0
        relevance_input = wt_mat.sum(axis=0)
        wt_mat_total += relevance_input
    
    return wt_mat_total

def process_single_relevance_gated_proj(wts, input):
    print("Running original version of process_single_relevance_gated_proj")
    wt_mat_total = np.zeros(input.shape)
    
    for i in range(wts.shape[0]):
        for j in range(wts.shape[1]):
            l1_ind1 = input
            wt = wts[i, j]
            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1
            t_sum = p_sum - n_sum

            t_act = np_swish(t_sum)
            p_act = np_swish(p_sum)
            n_act = np_swish(-1 * n_sum)

            if t_sum < -6:
                p_sum = 0
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
            wt_mat_total[p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_mat_total[n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0
    
    return wt_mat_total

def process_single_relevance_proj(wts, output):
    print("Running original version of process_single_relevance_proj")
    wt_mat_total = np.zeros(output.shape)
    
    for i in range(wts.shape[0]):
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
            wt_mat_total[p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_mat_total[n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0
    
    return wt_mat_total

def olmoe_mlp_forward(inp, w, model):
    print("Running original version of olmoe_mlp_forward")
    intermediate_outputs = {}

    _, hidden_dim = inp.shape
    top_k = model.config.num_experts_per_tok
    num_experts = model.config.num_experts

    router_logits = np.einsum('ij,jk->ik', inp, w['W_gate'].T)
    intermediate_outputs['router_logits'] = router_logits

    routing_weights = F.softmax(torch.tensor(router_logits), dim=-1)
    intermediate_outputs['softmax_routing_weights'] = routing_weights
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    intermediate_outputs['routing_weights'] = routing_weights
    intermediate_outputs['selected_experts'] = selected_experts

    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    intermediate_outputs['expert_mask'] = expert_mask

    for expert_idx in range(num_experts):
        expert_data = {} 

        idx, top_x = torch.where(expert_mask[expert_idx])
        expert_data['idx'] = idx
        expert_data['top_x'] = top_x

        current_state = inp[None, top_x].reshape(-1, hidden_dim)
        expert_data['current_state'] = current_state

        gate_proj_output = np.einsum('ij,jk->ik', current_state, w[f'{expert_idx}']['W_gate_proj'].T)
        up_proj_output = np.einsum('ij,jk->ik', current_state, w[f'{expert_idx}']['W_up_proj'].T)
        intermediate_output = np_swish(gate_proj_output) * up_proj_output
        down_proj_output = np.einsum('ij,jk->ik', intermediate_output, w[f'{expert_idx}']['W_down_proj'].T)
        current_hidden_states = down_proj_output * routing_weights[top_x, idx, None].numpy()

        expert_data['gate_proj_output'] = gate_proj_output
        expert_data['up_proj_output'] = up_proj_output
        expert_data['intermediate_output'] = intermediate_output
        expert_data['down_proj_output'] = down_proj_output
        expert_data['current_hidden_states'] = current_hidden_states

        intermediate_outputs[f'expert_{expert_idx}'] = expert_data

    return intermediate_outputs

def calculate_wt_olmoe_feed_forward_parallel(wts, inp, w, model):
    print("Running original version of calculate_wt_olmoe_feed_forward_parallel")
    num_experts = model.config.num_experts
    intermediate_outputs = olmoe_mlp_forward(inp, w, model)

    # Initialize final relevance
    final_relevance_input = np.zeros_like(inp) 

    # Initialize the relevance_expert
    relevance_expert = np.zeros((num_experts))

    # Initialize the `in_relevance`
    in_relevance = np.zeros_like(wts)

    #### Relevance calculation for each expert
    for expert_idx in range(num_experts):
        expert_data = intermediate_outputs[f'expert_{expert_idx}'] 

        _, top_x = expert_data['idx'], expert_data['top_x']
        intermediate_data = expert_data['intermediate_output']

        # If no tokens are assigned to this expert, skip processing
        if top_x.numel() == 0:
            relevance_expert[expert_idx] = 0
            continue

        in_relevance[None, top_x] = wts[None, top_x] / num_experts

        relev_half = in_relevance * 0.5

        relevance_int_output = process_single_relevance_proj(relev_half, intermediate_data)
        
        relev_proj = 0.5 * relevance_int_output

        relevance_input_gate_proj = process_single_relevance_gated_proj(relev_proj, inp)
        relevance_input_up_proj = process_single_relevance_proj(relev_proj, inp)
    
        relevance_current_state = relevance_input_gate_proj + relevance_input_up_proj

        if top_x.numel() > 0:
            final_relevance_input[top_x, :] += relevance_current_state[top_x, :]
            relevance_expert[expert_idx] = np.sum(relevance_current_state[top_x, :])

    relevance_router_logits = process_single_relevance_router_logits(relev_half, inp, w['W_gate'])

    final_relevance_input += relevance_router_logits

def process_single_relevance_QK(i, wts, QK_output):
    wt_mat_QK = np.zeros(QK_output.shape)
    for j in range(wts.shape[1]):
        l1_ind1 = QK_output
        wt = wts[i, j]

        p_ind = l1_ind1 > 0
        n_ind = l1_ind1 < 0
        p_sum = np.sum(l1_ind1[p_ind])
        n_sum = np.sum(l1_ind1[n_ind]) * -1

        t_sum = p_sum - n_sum

        # This layer has a softmax activation function
        act = {
            "name": "softmax",
            "range": {"l": -1, "u": 2},
            "type": "mono",
            "func": None,
        }

        if act["type"] == "mono":
            if act["range"]["l"] and t_sum < act["range"]["l"]:
                p_sum = 0
            if act["range"]["u"] and t_sum > act["range"]["u"]:
                n_sum = 0

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

        wt_mat_QK[p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
        wt_mat_QK[n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

    return wt_mat_QK

# Optimized parallel function
def calculate_relevance_QK_parallel(wts, QK_output):
    wt_mat_QK_total = np.zeros(QK_output.shape)

    # Parallel processing using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_relevance_QK, range(wts.shape[0]), [wts] * wts.shape[0], [QK_output] * wts.shape[0]))

    # Combine the results into the final wt_mat_QK matrix
    for result in results:
        wt_mat_QK_total += result

    return wt_mat_QK_total    

def process_single_relevance_attention_output(i, wts, proj_output):
    wt_mat_proj_output = np.zeros(proj_output.shape)
    for j in range(wts.shape[1]):
        l1_ind1 = proj_output
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

        wt_mat_proj_output[p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
        wt_mat_proj_output[n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

    return wt_mat_proj_output

# Optimized parallel function
def calculate_wt_attention_output_projection_parallel(wts, proj_output):
    wt_mat_proj_output_total = np.zeros(proj_output.shape)

    # Parallel processing using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_relevance_attention_output, range(wts.shape[0]), [wts] * wts.shape[0], [proj_output] * wts.shape[0]))

    # Combine the results into the final wt_mat_proj_output matrix
    for result in results:
        wt_mat_proj_output_total += result

    return wt_mat_proj_output_total

def calculate_wt_self_attention_parallel(wts, inp, w):
    '''
    Input:
        wts:  relevance score of the layer
        inp: input to the layer
        w: weights of the layer- ['W_q', 'W_k', 'W_v', 'W_o']

    Outputs:
        Step-1: outputs = torch.matmul(input_a, input_b)
        Step-2: outputs = F.softmax(inputs, dim=dim, dtype=dtype)
        Step-3: outputs = input_a * input_b
    '''
    # print(f"inp: {inp.shape}, wts: {wts.shape}")   # (1, 512)
    # print(f"w['W_q']: {w['W_q'].shape}, w['W_k']: {w['W_k'].shape}, w['W_v']: {w['W_v'].shape}")

    query_output = np.einsum('ij,kj->ik', inp, w['W_q'])
    key_output = np.einsum('ij,kj->ik', inp, w['W_k'])
    value_output = np.einsum('ij,kj->ik', inp, w['W_v'])
    # print(f"query_output: {query_output.shape}, key_output: {key_output.shape}, value_output: {value_output.shape}")

    # --------------- Reshape for Multi-Head Attention ----------------------
    # config = model.config
    # num_heads = config.num_attention_heads
    # hidden_size = config.hidden_size
    # num_key_value_heads = config.num_key_value_heads
    # head_dim = hidden_size // num_heads  # dimension of each attention head
    config = model.config
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    # Check if the config has 'num_key_value_heads' attribute
    if hasattr(config, 'num_key_value_heads'):
        num_key_value_heads = config.num_key_value_heads
    else:
        num_key_value_heads = config.num_heads
    head_dim = hidden_size // num_heads  # dimension of each attention head


    # query_states = query_output.view(query_output.shape[0], num_heads, head_dim).transpose(0, 1)     # (num_heads, num_tokens, head_dim)
    # key_states = key_output.view(key_output.shape[0], num_key_value_heads, head_dim).transpose(0, 1)    # (num_key_value_heads, num_tokens, head_dim)
    # value_states = value_output.view(value_output.shape[0], num_key_value_heads, head_dim).transpose(0, 1)    # (num_key_value_heads, num_tokens, head_dim)
    # print(f"query_states: {query_states.shape}, key_states: {key_states.shape}, value_states: {value_states.shape}")

    query_states = np.einsum('thd->htd', query_output.reshape(query_output.shape[0], num_heads, head_dim))  # (num_heads, num_tokens, head_dim)
    key_states = np.einsum('thd->htd', key_output.reshape(key_output.shape[0], num_key_value_heads, head_dim))  # (num_key_value_heads, num_tokens, head_dim)
    value_states = np.einsum('thd->htd', value_output.reshape(value_output.shape[0], num_key_value_heads, head_dim))  # (num_key_value_heads, num_tokens, head_dim)
    # print(f"query_states: {query_states.shape}, key_states: {key_states.shape}, value_states: {value_states.shape}")

    # calculate how many times we need to repeat the key/value heads
    n_rep = num_heads // num_key_value_heads
    key_states = np.repeat(key_states, n_rep, axis=0)
    value_states = np.repeat(value_states, n_rep, axis=0)

    # print(f"Key States Shape (after repeating): {key_states.shape}")
    # print(f"Value States Shape (after repeating): {value_states.shape}")

    QK_output = np.einsum('hqd,hkd->hqk', query_states, key_states)    # (num_heads, num_tokens, num_tokens)
    # print(f"QK_output: {QK_output.shape}")
    attn_weights = QK_output / np.sqrt(head_dim)
    # print(f"attn_weights: {attn_weights.shape}")

    # Apply softmax along the last dimension (softmax over key dimension)
    attn_weights = np.exp(attn_weights - np.max(attn_weights, axis=-1, keepdims=True))  # Numerically stable softmax
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    # Weighted sum of values (num_heads, num_tokens, head_dim)
    attn_output = np.einsum('hqk,hkl->hql', attn_weights, value_states)

    # Reshape attention output back to original shape (num_tokens, hidden_size)
    attn_output = np.einsum('hqd->qhd', attn_output)
    # print(f"attn_output: {attn_output.shape}")
    attn_output = attn_output.reshape(attn_output.shape[0], num_heads * head_dim)
    # print(f"attention_output: {attn_output.shape}")

    # Perform final linear projection (num_tokens, hidden_size)
    final_output = np.einsum('qd,dh->qh', attn_output, w['W_d'])
    # print("Final Output Shape:", final_output.shape)

    # ------------- Relevance calculation for Final Linear Projection -------------
    # wt_mat_attn_proj = calculate_wt_attention_output_projection(wts, final_output)
    wt_mat_attn_proj = calculate_wt_attention_output_projection_parallel(wts, final_output)
    # print(f"wt_mat_attn_proj: {np.sum(wt_mat_attn_proj):.2f}, shape: {wt_mat_attn_proj.shape}")

    # --------------- Relevance Calculation for Step-3 -----------------------
    relevance_V = wt_mat_attn_proj / 2
    relevance_QK = wt_mat_attn_proj / 2
    # print(f"relevance_V: {np.sum(relevance_V):.2f}, relevance_QK: {np.sum(relevance_QK):.2f}")
    # print(f"relevance_V: {relevance_V.shape}, relevance_QK: {relevance_QK.shape}")
    # relevance_V: (8, 4096), relevance_QK: (8, 4096)

    # --------------- Relevance Calculation for V --------------------------------
    wt_mat_V = calculate_wt_attention_output_projection_parallel(relevance_V, value_states)
    # print(f"wt_mat_V: {wt_mat_V.shape}, {np.sum(wt_mat_V):.2f}")

    # --------------- Transformed Relevance QK ----------------------------------
    # print(f"query_output: {query_output.shape}, key_output: {key_output.shape}")
    # query_output: (8, 4096), key_output: (8, 1024)
    # QK_output = np.einsum('ij,ik->ij', query_output, key_output)
    # QK_output = np.einsum('ij,kj->ik', query_output, key_output)
    # wt_mat_QK = calculate_relevance_QK(relevance_QK, QK_output)
    wt_mat_QK = calculate_relevance_QK_parallel(relevance_QK, QK_output)
    # print(f"wt_mat_QK: {np.sum(wt_mat_QK):.2f},  relevance_QK: {np.sum(relevance_QK):.2f}")

    # --------------- Relevance Calculation for K and Q --------------------------------
    stabilized_QK_output = stabilize(QK_output * 2)
    norm_wt_mat_QK = wt_mat_QK / stabilized_QK_output
    # print(f"wt_mat_QK: {wt_mat_QK.shape}, query_output: {query_output.shape}, key_output: {key_output.shape}, norm_wt_mat_QK: {norm_wt_mat_QK.shape}")

    # wt_mat_Q = np.einsum('ij,jk->ik', norm_wt_mat_QK, key_output) * query_output
    # wt_mat_K = np.einsum('ij,ik->kj', query_output, norm_wt_mat_QK) * key_output

    wt_mat_Q = np.einsum('htd,hdb->htb', norm_wt_mat_QK, key_states) * query_states
    wt_mat_K = np.einsum('htd,htb->hbd', query_states, norm_wt_mat_QK) * key_states
    # print(f"wt_mat_Q: {wt_mat_Q.shape}, {np.sum(wt_mat_Q):.2f}")
    # print(f"wt_mat_K: {wt_mat_K.shape}, {np.sum(wt_mat_K):.2f}")

    wt_mat = wt_mat_V + wt_mat_K + wt_mat_Q

    # Reshape wt_mat
    wt_mat = np.einsum('htd->thd', wt_mat)
    wt_mat = wt_mat.reshape(wt_mat.shape[0], wt_mat.shape[1] * wt_mat.shape[2])  # reshaped_array = array.reshape(8, 32 * 128)

    return wt_mat
    final_relevance_input = (wts / final_relevance_input) * final_relevance_input

    return final_relevance_input, relevance_expert
