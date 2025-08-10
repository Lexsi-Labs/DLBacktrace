import torch
import torch.nn.functional as F
import numpy as np

def np_swish(x, beta=0.75):
    z = 1 / (1 + np.exp(-np.clip(beta * x, -500, 500)))
    return x * z 

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
            if t_sum > None:
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

    return final_relevance_input, relevance_expert