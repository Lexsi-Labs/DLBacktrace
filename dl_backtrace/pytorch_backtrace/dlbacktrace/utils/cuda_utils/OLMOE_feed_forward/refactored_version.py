import numpy as np
import torch
import torch.nn.functional as F

def np_swish(x, beta=0.75):
    z = 1 / (1 + np.exp(-np.clip(beta * x, -500, 500)))
    return x * z 

def process_single_relevance_router_logits(wts: np.ndarray, input: np.ndarray, W_router: np.ndarray) -> np.ndarray:
    """
    Process relevance router logits by computing weighted contributions with positive/negative aggregation.
    
    This function processes router weights and input data to compute relevance-weighted contributions,
    handling positive and negative components separately with conditional thresholding logic.
    
    Args:
        wts (np.ndarray): Router weights array of shape (n_samples, n_features)
        input (np.ndarray): Input data array of shape (n_samples, input_dim) 
        W_router (np.ndarray): Router weight matrix for computing contributions
        
    Returns:
        np.ndarray: Processed relevance logits with same shape as input[0]
        
    Notes:
        - Applies conditional thresholding based on total sum (t_sum)
        - Handles positive and negative components with separate aggregation weights
        - Replicates original division-by-zero handling by setting zero sums to 1
    """
    print("Running refactored version of process_single_relevance_router_logits")
    n_samples, n_features = wts.shape
    wt_mat_total = np.zeros(input.shape[1:])  # More efficient initialization
    
    # Process each sample
    for i in range(n_samples):
        R = wts[i]  # Shape: (n_features,)
        contribution_matrix = W_router * input[i]  # Broadcasting multiplication
        
        # Vectorized positive/negative masks for all features at once
        p_mask = contribution_matrix > 0  # Shape: (n_features, input_dim)
        n_mask = contribution_matrix < 0  # Shape: (n_features, input_dim)

        p_matrix = contribution_matrix * p_mask
        n_matrix = contribution_matrix * n_mask
        
        # Vectorized sum calculations using masks
        p_sums = np.sum(p_matrix, axis=1)  # Sum positive parts per feature
        n_sums = np.sum(n_matrix, axis=1) * -1  # Sum negative parts (make positive)
        t_sums = p_sums - n_sums  # Total sums per feature
        
        # Apply conditional thresholding logic (replicating original behavior exactly)
        p_sums = np.where(t_sums < -1, 0, p_sums)  # Set p_sum to 0 if t_sum < -1
        n_sums = np.where(t_sums > 2, 0, n_sums)   # Set n_sum to 0 if t_sum > 2
        
        # Compute aggregation weights (vectorized)
        denominators = p_sums + n_sums
        p_agg_wts = np.where(p_sums > 0, p_sums / denominators, 0)
        n_agg_wts = np.where(n_sums > 0, n_sums / denominators, 0)
        
        # Handle division by zero exactly as original (set zero sums to 1)
        p_sums_safe = np.where(p_sums == 0, 1, p_sums)
        n_sums_safe = np.where(n_sums == 0, 1, n_sums)
        
        # Vectorized weight computation for positive contributions
        # Using broadcasting: R[:, None] makes R broadcastable with masks
        p_contributions = (p_matrix) / p_sums_safe[:, None]
        p_contributions *= R[:, None] * p_agg_wts[:, None]
        
        # Vectorized weight computation for negative contributions  
        n_contributions = (n_matrix) / n_sums_safe[:, None]
        n_contributions *= R[:, None] * n_agg_wts[:, None] * -1.0
        
        # Sum contributions across all features for this sample
        relevance_input = np.sum(p_contributions + n_contributions, axis=0)
        wt_mat_total += relevance_input
    
    return wt_mat_total

def process_single_relevance_gated_proj(wts: np.ndarray, output: np.ndarray) -> np.ndarray:
    """
    Process relevance-gated projection with vectorized operations.
    
    This function applies a complex gating mechanism to weight matrices based on 
    positive and negative components of the output array. Each weight in the 
    weight matrix contributes to the final result through separate positive 
    and negative pathways with Swish activation and normalization.
    
    Args:
        wts: 2D weight matrix of shape (M, N)
        output: Input array to be processed
        
    Returns:
        Processed array of same shape as output, containing weighted contributions
        from all weight matrix elements
    """
    print("Running refactored version of process_single_relevance_gated_proj")
    # Initialize result array
    wt_mat_total = np.zeros_like(output)
    
    # Pre-compute masks and sums for positive/negative components
    # These remain constant across all weight iterations
    pos_mask = output > 0
    neg_mask = output < 0
    
    # Extract positive and negative components
    pos_values = output[pos_mask] if np.any(pos_mask) else np.array([])
    neg_values = output[neg_mask] if np.any(neg_mask) else np.array([])
    
    # Compute sums for positive and negative parts
    pos_sum = np.sum(pos_values) if len(pos_values) > 0 else 0.0
    neg_sum = np.sum(neg_values) * -1 if len(neg_values) > 0 else 0.0  # Convert to positive
    
    # Total sum calculation
    total_sum = pos_sum - neg_sum
    
    # Apply Swish activations
    total_activation = np_swish(total_sum)
    pos_activation = np_swish(pos_sum)
    neg_activation = np_swish(-1 * neg_sum)
    
    if total_sum < -6:
        pos_sum = 0
    
    if pos_sum > 0 and neg_sum > 0:
        if total_activation == pos_activation:
            neg_sum = 0
        elif total_activation == neg_activation:
            pos_sum = 0
    
    # Calculate aggregation weights
    sum_total = pos_sum + neg_sum
    pos_agg_weight = pos_sum / sum_total if pos_sum > 0 else 0.0
    neg_agg_weight = neg_sum / sum_total if neg_sum > 0 else 0.0
    
    # Set normalization denominators (avoiding division by zero)
    pos_norm_denom = pos_sum if pos_sum != 0 else 1.0
    neg_norm_denom = neg_sum if neg_sum != 0 else 1.0
    
    # Vectorized computation across all weights
    # Flatten weights for easier iteration while maintaining order
    weights_flat = wts.flatten()
    
    for weight in weights_flat:
        # Compute contributions for positive and negative parts
        if np.any(pos_mask) and pos_agg_weight != 0:
            pos_contribution = (output[pos_mask] / pos_norm_denom) * weight * pos_agg_weight
            wt_mat_total[pos_mask] += pos_contribution
            
        if np.any(neg_mask) and neg_agg_weight != 0:
            neg_contribution = (output[neg_mask] / neg_norm_denom) * weight * neg_agg_weight * -1.0
            wt_mat_total[neg_mask] += neg_contribution
    
    return wt_mat_total

def process_single_relevance_proj(wts: np.ndarray, output: np.ndarray) -> np.ndarray:
    """
    Process single relevance projection by computing weighted contributions of positive and negative values.
    
    This function applies a relevance projection algorithm that:
    1. Separates positive and negative values in the output array
    2. Computes aggregated weights based on the ratio of positive/negative sums
    3. Normalizes contributions and applies weights from the weight matrix
    
    Args:
        wts: Weight matrix of shape (M, N) containing scalar weights
        output: Input array to be processed for relevance projection
        
    Returns:
        np.ndarray: Weighted relevance projection result with same shape as output
        
    Notes:
        - Positive values are normalized by their sum and weighted by p_agg_wt
        - Negative values are normalized by their absolute sum and weighted by n_agg_wt
        - Division by zero is handled by setting denominators to 1 when sums are 0
    """
    print("Running refactored version of process_single_relevance_proj")
    # Pre-compute masks for positive and negative values (done once)
    positive_mask = output > 0
    negative_mask = output < 0
    
    # Pre-compute sums for efficiency
    positive_values = output[positive_mask]
    negative_values = output[negative_mask]
    positive_sum = np.sum(positive_values) if positive_values.size > 0 else 0.0
    negative_sum = np.sum(negative_values) * -1.0 if negative_values.size > 0 else 0.0
    
    # Compute aggregated weights (replicating original logic exactly)
    total_sum = positive_sum + negative_sum
    if positive_sum > 0:
        p_agg_wt = positive_sum / total_sum
    else:
        p_agg_wt = 0.0
        
    if negative_sum > 0:
        n_agg_wt = negative_sum / total_sum  
    else:
        n_agg_wt = 0.0
    
    # Handle division by zero exactly as original (set denominators to 1 when sums are 0)
    p_sum_for_division = 1.0 if positive_sum == 0 else positive_sum
    n_sum_for_division = 1.0 if negative_sum == 0 else negative_sum
    
    # Vectorized computation of weight contributions
    # Sum all weights to get total contribution factor
    total_weight = np.sum(wts)
    
    # Initialize result array
    wt_mat_total = np.zeros_like(output)
    
    # Vectorized assignment for positive values
    if np.any(positive_mask):
        wt_mat_total[positive_mask] = (
            (output[positive_mask] / p_sum_for_division) * total_weight * p_agg_wt
        )
    
    # Vectorized assignment for negative values (maintaining the *-1.0 factor from original)
    if np.any(negative_mask):
        wt_mat_total[negative_mask] = (
            (output[negative_mask] / n_sum_for_division) * total_weight * n_agg_wt * -1.0
        )
    
    return wt_mat_total    

def olmoe_mlp_forward(inp, w, model):
    print("Running refactored version of olmoe_mlp_forward")
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
    print("Running refactored version of calculate_wt_olmoe_feed_forward_parallel")
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