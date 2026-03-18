import torch
import torch.nn.functional as F
import numpy as np
import concurrent.futures


def np_swish(x, beta=0.75):
    z = 1 / (1 + np.exp(-np.clip(beta * x, -500, 500)))
    return x * z 

def stabilize(matrix, epsilon=1e-6):
    return matrix + epsilon * np.sign(matrix)

def process_single_relevance_router_logits(wts, input, W_router):
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
    final_relevance_input = (wts / final_relevance_input) * final_relevance_input
    
    # ✅ Return relevance map and per-expert relevance
    return final_relevance_input, relevance_expert


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

def calculate_wt_self_attention_parallel(wts, inp, w, model):
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
    query_output = np.einsum('ij,kj->ik', inp, w['W_q'])
    key_output = np.einsum('ij,kj->ik', inp, w['W_k'])
    value_output = np.einsum('ij,kj->ik', inp, w['W_v'])

    # --------------- Reshape for Multi-Head Attention ----------------------
    config = model.config
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    # Check if the config has 'num_key_value_heads' attribute
    if hasattr(config, 'num_key_value_heads'):
        num_key_value_heads = config.num_key_value_heads
    else:
        num_key_value_heads = config.num_heads
    head_dim = hidden_size // num_heads  # dimension of each attention head

    query_states = np.einsum('thd->htd', query_output.reshape(query_output.shape[0], num_heads, head_dim))  # (num_heads, num_tokens, head_dim)
    key_states = np.einsum('thd->htd', key_output.reshape(key_output.shape[0], num_key_value_heads, head_dim))  # (num_key_value_heads, num_tokens, head_dim)
    value_states = np.einsum('thd->htd', value_output.reshape(value_output.shape[0], num_key_value_heads, head_dim))  # (num_key_value_heads, num_tokens, head_dim)

    # calculate how many times we need to repeat the key/value heads
    n_rep = num_heads // num_key_value_heads
    key_states = np.repeat(key_states, n_rep, axis=0)
    value_states = np.repeat(value_states, n_rep, axis=0)

    QK_output = np.einsum('hqd,hkd->hqk', query_states, key_states)    # (num_heads, num_tokens, num_tokens)
    attn_weights = QK_output / np.sqrt(head_dim)

    # Apply softmax along the last dimension (softmax over key dimension)
    attn_weights = np.exp(attn_weights - np.max(attn_weights, axis=-1, keepdims=True))  # Numerically stable softmax
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    # Weighted sum of values (num_heads, num_tokens, head_dim)
    attn_output = np.einsum('hqk,hkl->hql', attn_weights, value_states)

    # Reshape attention output back to original shape (num_tokens, hidden_size)
    attn_output = np.einsum('hqd->qhd', attn_output)
    attn_output = attn_output.reshape(attn_output.shape[0], num_heads * head_dim)

    # Perform final linear projection (num_tokens, hidden_size)
    final_output = np.einsum('qd,dh->qh', attn_output, w['W_d'])

    # ------------- Relevance calculation for Final Linear Projection -------------
    wt_mat_attn_proj = calculate_wt_attention_output_projection_parallel(wts, final_output)

    # --------------- Relevance Calculation for Step-3 -----------------------
    relevance_V = wt_mat_attn_proj / 2
    relevance_QK = wt_mat_attn_proj / 2

    # --------------- Relevance Calculation for V --------------------------------
    wt_mat_V = calculate_wt_attention_output_projection_parallel(relevance_V, value_states)

    # --------------- Transformed Relevance QK ----------------------------------
    wt_mat_QK = calculate_relevance_QK_parallel(relevance_QK, QK_output)

    # --------------- Relevance Calculation for K and Q --------------------------------
    stabilized_QK_output = stabilize(QK_output * 2)
    norm_wt_mat_QK = wt_mat_QK / stabilized_QK_output

    wt_mat_Q = np.einsum('htd,hdb->htb', norm_wt_mat_QK, key_states) * query_states
    wt_mat_K = np.einsum('htd,htb->hbd', query_states, norm_wt_mat_QK) * key_states

    wt_mat = wt_mat_V + wt_mat_K + wt_mat_Q

    # Reshape wt_mat
    wt_mat = np.einsum('htd->thd', wt_mat)
    wt_mat = wt_mat.reshape(wt_mat.shape[0], wt_mat.shape[1] * wt_mat.shape[2])

    return wt_mat