import torch
import torch.nn.functional as F
import torch._dynamo
from typing import Tuple, Dict, Any

torch._dynamo.config.suppress_errors = True

def torch_swish(x: torch.Tensor, beta: float = 0.75) -> torch.Tensor:
    """PyTorch implementation of Swish activation function."""
    z = torch.sigmoid(torch.clamp(beta * x, -500, 500))
    return x * z

def stabilize(matrix: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    abs_matrix = torch.abs(matrix)
    sign_matrix = torch.where(matrix == 0, 
                             torch.ones_like(matrix), 
                             torch.sign(matrix))
    
    return torch.where(abs_matrix < epsilon, 
                      epsilon * sign_matrix, 
                      matrix)

def process_single_relevance_router_logits(
    wts: torch.Tensor, 
    input_tensor: torch.Tensor, 
    W_router: torch.Tensor
) -> torch.Tensor:
    """
    Process relevance router logits by computing weighted contributions with positive/negative aggregation.
    
    This function processes router weights and input data to compute relevance-weighted contributions,
    handling positive and negative components separately with conditional thresholding logic.
    
    Args:
        wts: Router weights tensor of shape (n_samples, n_features)
        input_tensor: Input data tensor of shape (n_samples, input_dim) 
        W_router: Router weight matrix for computing contributions
        
    Returns:
        torch.Tensor: Processed relevance logits with same shape as input_tensor[0]
    """
    # Vectorized computation across all samples
    # Reshape for broadcasting: (n_samples, n_features, 1) * (n_samples, 1, input_dim)
    contribution_matrix = W_router.unsqueeze(0) * input_tensor.unsqueeze(1)  # (n_samples, n_features, input_dim)
    
    # Create masks for positive/negative values
    p_mask = contribution_matrix > 0
    n_mask = contribution_matrix < 0
    
    # Extract positive and negative components
    p_matrix = contribution_matrix * p_mask.float()
    n_matrix = contribution_matrix * n_mask.float()
    
    # Sum across input dimension for each sample and feature
    p_sums = torch.sum(p_matrix, dim=2)  # (n_samples, n_features)
    n_sums = torch.sum(n_matrix, dim=2) * -1  # Make positive
    t_sums = p_sums - n_sums
    
    # Apply conditional thresholding logic
    p_sums = torch.where(t_sums < -1, torch.zeros_like(p_sums), p_sums)
    n_sums = torch.where(t_sums > 2, torch.zeros_like(n_sums), n_sums)
    
    # Compute aggregation weights
    denominators = p_sums + n_sums
    p_agg_wts = torch.where(p_sums > 0, p_sums / denominators, torch.zeros_like(p_sums))
    n_agg_wts = torch.where(n_sums > 0, n_sums / denominators, torch.zeros_like(n_sums))
    
    # Handle division by zero
    p_sums_safe = torch.where(p_sums == 0, torch.ones_like(p_sums), p_sums)
    n_sums_safe = torch.where(n_sums == 0, torch.ones_like(n_sums), n_sums)
    
    total_weight = torch.sum(wts)

    # Compute contributions with broadcasting
    # Reshape for proper broadcasting: (n_samples, n_features, 1)
    p_contributions = (p_matrix / p_sums_safe.unsqueeze(2)) * (total_weight * p_agg_wts).unsqueeze(2)
    n_contributions = (n_matrix / n_sums_safe.unsqueeze(2)) * (total_weight * n_agg_wts).unsqueeze(2) * -1.0
    
    # Sum across samples and features
    relevance_input = torch.sum(p_contributions + n_contributions, dim=(0, 1))
    
    return relevance_input

def process_single_relevance_gated_proj(
    wts: torch.Tensor, 
    output: torch.Tensor
) -> torch.Tensor:
    """
    Process relevance-gated projection with vectorized operations.
    
    This function applies a complex gating mechanism to weight matrices based on 
    positive and negative components of the output tensor.
    
    Args:
        wts: 2D weight matrix of shape (M, N)
        output: Input tensor to be processed
        
    Returns:
        torch.Tensor: Processed tensor of same shape as output
    """
    # Initialize result tensor
    wt_mat_total = torch.zeros_like(output)
    
    # Pre-compute masks and components
    pos_mask = output > 0
    neg_mask = output < 0
    
    # Compute sums
    pos_sum = torch.sum(output[pos_mask]) if torch.any(pos_mask) else torch.tensor(0.0, device=output.device)
    neg_sum = torch.sum(output[neg_mask]) * -1 if torch.any(neg_mask) else torch.tensor(0.0, device=output.device)
    
    # Total sum calculation
    total_sum = pos_sum - neg_sum
    
    # Apply Swish activations
    total_activation = torch_swish(total_sum)
    pos_activation = torch_swish(pos_sum)
    neg_activation = torch_swish(-1 * neg_sum)
    
    # Conditional logic
    if total_sum < -6:
        pos_sum = torch.tensor(0.0, device=output.device)
    
    if pos_sum > 0 and neg_sum > 0:
        if torch.isclose(total_activation, pos_activation):
            neg_sum = torch.tensor(0.0, device=output.device)
        elif torch.isclose(total_activation, neg_activation):
            pos_sum = torch.tensor(0.0, device=output.device)
    
    # Calculate aggregation weights
    sum_total = pos_sum + neg_sum
    pos_agg_weight = pos_sum / sum_total if pos_sum > 0 else torch.tensor(0.0, device=output.device)
    neg_agg_weight = neg_sum / sum_total if neg_sum > 0 else torch.tensor(0.0, device=output.device)
    
    # Set normalization denominators
    pos_norm_denom = pos_sum if pos_sum != 0 else torch.tensor(1.0, device=output.device)
    neg_norm_denom = neg_sum if neg_sum != 0 else torch.tensor(1.0, device=output.device)
    
    # Vectorized computation across all weights
    total_weight = torch.sum(wts)
    
    # Apply contributions
    if torch.any(pos_mask) and pos_agg_weight != 0:
        pos_contribution = (output[pos_mask] / pos_norm_denom) * total_weight * pos_agg_weight
        wt_mat_total[pos_mask] += pos_contribution
        
    if torch.any(neg_mask) and neg_agg_weight != 0:
        neg_contribution = (output[neg_mask] / neg_norm_denom) * total_weight * neg_agg_weight * -1.0
        wt_mat_total[neg_mask] += neg_contribution
    
    return wt_mat_total

def process_single_relevance_proj(
    wts: torch.Tensor, 
    output: torch.Tensor
) -> torch.Tensor:
    """
    Process single relevance projection by computing weighted contributions of positive and negative values.
    
    Args:
        wts: Weight matrix of shape (M, N) containing scalar weights
        output: Input tensor to be processed for relevance projection
        
    Returns:
        torch.Tensor: Weighted relevance projection result with same shape as output
    """
    # Pre-compute masks for positive and negative values
    positive_mask = output > 0
    negative_mask = output < 0
    
    # Pre-compute sums for efficiency
    positive_sum = torch.sum(output[positive_mask]) if torch.any(positive_mask) else torch.tensor(0.0, device=output.device)
    negative_sum = torch.sum(output[negative_mask]) * -1.0 if torch.any(negative_mask) else torch.tensor(0.0, device=output.device)
    
    # Compute aggregated weights
    total_sum = positive_sum + negative_sum
    p_agg_wt = positive_sum / total_sum if positive_sum > 0 else torch.tensor(0.0, device=output.device)
    n_agg_wt = negative_sum / total_sum if negative_sum > 0 else torch.tensor(0.0, device=output.device)
    
    # Handle division by zero
    p_sum_for_division = torch.tensor(1.0, device=output.device) if positive_sum == 0 else positive_sum
    n_sum_for_division = torch.tensor(1.0, device=output.device) if negative_sum == 0 else negative_sum
    
    # Vectorized computation
    total_weight = torch.sum(wts)
    
    # Initialize result tensor
    wt_mat_total = torch.zeros_like(output)
    
    # Vectorized assignment for positive values
    if torch.any(positive_mask):
        wt_mat_total[positive_mask] = (
            (output[positive_mask] / p_sum_for_division) * total_weight * p_agg_wt
        )
    
    # Vectorized assignment for negative values
    if torch.any(negative_mask):
        wt_mat_total[negative_mask] = (
            (output[negative_mask] / n_sum_for_division) * total_weight * n_agg_wt * -1.0
        )
    
    return wt_mat_total

def olmoe_mlp_forward(
    inp: torch.Tensor, 
    w: Dict[str, torch.Tensor], 
    model: Any
) -> Dict[str, torch.Tensor]:
    """
    Forward pass through OLMoE MLP with expert routing.
    
    Args:
        inp: Input tensor of shape (batch_size, hidden_dim)
        w: Dictionary containing weight tensors
        model: Model configuration object
        
    Returns:
        Dict containing intermediate outputs and expert data
    """
    intermediate_outputs = {}

    _, hidden_dim = inp.shape
    top_k = model.config.num_experts_per_tok
    num_experts = model.config.num_experts

    # Router logits computation
    router_logits = torch.einsum('ij,jk->ik', inp, w['W_gate'].t())
    intermediate_outputs['router_logits'] = router_logits

    # Routing weights computation
    routing_weights = F.softmax(router_logits, dim=-1)
    intermediate_outputs['softmax_routing_weights'] = routing_weights
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    intermediate_outputs['routing_weights'] = routing_weights
    intermediate_outputs['selected_experts'] = selected_experts

    # Expert mask
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    intermediate_outputs['expert_mask'] = expert_mask

    # Process each expert
    for expert_idx in range(num_experts):
        expert_data = {} 

        idx, top_x = torch.where(expert_mask[expert_idx])
        expert_data['idx'] = idx
        expert_data['top_x'] = top_x

        if top_x.numel() == 0:
            # Handle empty expert case
            expert_data.update({
                'current_state': torch.empty(0, hidden_dim, device=inp.device, dtype=inp.dtype),
                'gate_proj_output': torch.empty(0, device=inp.device, dtype=inp.dtype),
                'up_proj_output': torch.empty(0, device=inp.device, dtype=inp.dtype),
                'intermediate_output': torch.empty(0, device=inp.device, dtype=inp.dtype),
                'down_proj_output': torch.empty(0, device=inp.device, dtype=inp.dtype),
                'current_hidden_states': torch.empty(0, device=inp.device, dtype=inp.dtype)
            })
        else:
            current_state = inp[top_x]
            expert_data['current_state'] = current_state

            # Expert computations using torch.einsum
            gate_proj_output = torch.einsum('ij,jk->ik', current_state, w[f'{expert_idx}']['W_gate_proj'].t())
            up_proj_output = torch.einsum('ij,jk->ik', current_state, w[f'{expert_idx}']['W_up_proj'].t())
            intermediate_output = torch_swish(gate_proj_output) * up_proj_output
            down_proj_output = torch.einsum('ij,jk->ik', intermediate_output, w[f'{expert_idx}']['W_down_proj'].t())
            current_hidden_states = down_proj_output * routing_weights[top_x, idx, None]

            expert_data.update({
                'gate_proj_output': gate_proj_output,
                'up_proj_output': up_proj_output,
                'intermediate_output': intermediate_output,
                'down_proj_output': down_proj_output,
                'current_hidden_states': current_hidden_states
            })

        intermediate_outputs[f'expert_{expert_idx}'] = expert_data

    return intermediate_outputs

@torch.compile
def calculate_wt_olmoe_feed_forward_parallel(
    wts: torch.Tensor, 
    inp: torch.Tensor, 
    w: Dict[str, torch.Tensor], 
    model: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate weighted OLMoE feed-forward pass with relevance propagation in parallel.
    
    This function performs relevance analysis across multiple experts in parallel,
    computing weighted contributions for each expert and aggregating the results.
    
    Args:
        wts: Weight tensor for relevance calculation
        inp: Input tensor to the MLP layer
        w: Dictionary containing all weight matrices for experts and router
        model: Model configuration object containing expert parameters
        
    Returns:
        Tuple containing:
            - final_relevance_input: Final relevance tensor with same shape as inp
            - relevance_expert: Per-expert relevance scores of shape (num_experts,)
    """

    device = torch.device('cuda')

    # Handle the conversion more carefully
    w_torch = {}
    for k, v in w.items():
        if isinstance(v, dict):
            # If it's a nested dictionary, convert each sub-tensor
            w_torch[k] = {sub_k: torch.tensor(sub_v, dtype=torch.float32, device=device) 
                         for sub_k, sub_v in v.items()}
        else:
            # If it's an array/tensor, convert directly
            w_torch[k] = torch.tensor(v, dtype=torch.float32, device=device)
    
    num_experts = model.config.num_experts
    intermediate_outputs = olmoe_mlp_forward(inp, w_torch, model)

    # Initialize tensors with proper device and dtype
    final_relevance_input = torch.zeros_like(inp)
    relevance_expert = torch.zeros(num_experts, dtype=inp.dtype, device=device)
    in_relevance = torch.zeros_like(wts)

    # Process each expert
    for expert_idx in range(num_experts):
        expert_data = intermediate_outputs[f'expert_{expert_idx}'] 
        top_x = expert_data['top_x']
        intermediate_data = expert_data['intermediate_output']

        # Skip if no tokens assigned to this expert
        if top_x.numel() == 0:
            continue

        # Update in_relevance for assigned tokens
        in_relevance[top_x] = wts[top_x] / num_experts
        relev_half = in_relevance * 0.5

        # Process relevance through the network
        relevance_int_output = process_single_relevance_proj(relev_half, intermediate_data)
        relev_proj = 0.5 * relevance_int_output

        # Compute input relevances
        relevance_input_gate_proj = process_single_relevance_gated_proj(relev_proj, inp)
        relevance_input_up_proj = process_single_relevance_proj(relev_proj, inp)
        
        relevance_current_state = relevance_input_gate_proj + relevance_input_up_proj

        # Update final relevance and expert scores
        if top_x.numel() > 0:
            final_relevance_input[top_x] += relevance_current_state[top_x]
            relevance_expert[expert_idx] = torch.sum(relevance_current_state[top_x])

    # Process router logits relevance
    relevance_router_logits = process_single_relevance_router_logits(
        in_relevance * 0.5, inp, w_torch['W_gate']
    )

    final_relevance_input += relevance_router_logits

    # Final normalization (preserving original logic)
    final_relevance_input = (wts / final_relevance_input) * final_relevance_input

    return final_relevance_input.cpu().numpy(), relevance_expert.cpu().numpy()

def calculate_relevance_QK(wts: torch.Tensor, QK_output: torch.Tensor) -> torch.Tensor:
    
    # Softmax activation properties (constant across all iterations)
    act_range_lower = -1
    act_range_upper = 2
    
    # Create positive and negative masks (boolean tensors for efficiency)
    p_mask = QK_output > 0
    n_mask = QK_output < 0
    
    # Compute positive and negative values using masking
    p_values = torch.where(p_mask, QK_output, torch.tensor(0.0, device=QK_output.device, dtype=QK_output.dtype))
    n_values = torch.where(n_mask, QK_output, torch.tensor(0.0, device=QK_output.device, dtype=QK_output.dtype))
    
    # Scalar sums
    p_sum = torch.sum(p_values)
    n_sum = -torch.sum(n_values)
    t_sum = p_sum - n_sum
    
    # Apply monotonic activation range constraints
    if t_sum < act_range_lower:
        p_sum = torch.tensor(0.0, device=p_sum.device, dtype=p_sum.dtype)
    if t_sum > act_range_upper:
        n_sum = torch.tensor(0.0, device=n_sum.device, dtype=n_sum.dtype)
    
    # Calculate aggregate weights
    p_agg_wt = p_sum / (p_sum + n_sum) if p_sum > 0 else torch.tensor(0.0, device=p_sum.device, dtype=p_sum.dtype)
    n_agg_wt = n_sum / (p_sum + n_sum) if n_sum > 0 else torch.tensor(0.0, device=n_sum.device, dtype=n_sum.dtype)
    
    # Safe denominators for normalization
    p_sum_safe = p_sum if p_sum != 0 else torch.tensor(1.0, device=p_sum.device, dtype=p_sum.dtype)
    n_sum_safe = n_sum if n_sum != 0 else torch.tensor(1.0, device=n_sum.device, dtype=n_sum.dtype)
    
    # Precompute normalized contributions
    p_normalized = torch.where(p_mask, QK_output / p_sum_safe, torch.tensor(0.0, device=QK_output.device, dtype=QK_output.dtype))
    n_normalized = torch.where(n_mask, QK_output / n_sum_safe, torch.tensor(0.0, device=QK_output.device, dtype=QK_output.dtype))
    
    # KEY OPTIMIZATION: Vectorize the nested loops
    # The original nested loop computes:
    # sum over i,j of: (p_normalized * wts[i,j] * p_agg_wt + n_normalized * wts[i,j] * n_agg_wt * -1)
    # This simplifies to: (p_normalized * p_agg_wt - n_normalized * n_agg_wt) * sum(wts)
    total_weight = torch.sum(wts)
    
    wt_mat_QK_total = (p_normalized * p_agg_wt * total_weight - 
                       n_normalized * n_agg_wt * total_weight)
    
    return wt_mat_QK_total

def calculate_wt_attention_output_projection(
    wts: torch.Tensor, 
    proj_output: torch.Tensor
) -> torch.Tensor:

    # Expand dimensions for broadcasting: wts[i,j] -> wts[i,j,1,1,...]
    # This allows wts to broadcast against proj_output for all operations
    wts_expanded = wts.view(wts.shape[0], wts.shape[1], *([1] * proj_output.ndim))
    
    # Create boolean masks for positive and negative values
    # Shape: same as proj_output
    p_mask = proj_output > 0
    n_mask = proj_output < 0
    
    # Get positive and negative components
    # Use torch.where to avoid in-place operations that could break autograd
    p_values = torch.where(p_mask, proj_output, torch.zeros_like(proj_output))
    n_values = torch.where(n_mask, proj_output, torch.zeros_like(proj_output))
    
    # Calculate sums: shape (1, 1) for broadcasting
    p_sum = p_values.sum()
    n_sum = n_values.sum().abs()  # Take absolute value of negative sum
    
    # Calculate total sum for normalization
    total_sum = p_sum + n_sum
    
    # Calculate aggregation weights with safe division
    # If total_sum is 0, both weights should be 0
    p_agg_wt = torch.where(
        p_sum > 0,
        p_sum / total_sum,
        torch.zeros_like(p_sum)
    )
    n_agg_wt = torch.where(
        n_sum > 0,
        n_sum / total_sum,
        torch.zeros_like(n_sum)
    )
    
    # Avoid division by zero for normalization denominators
    # If sum is 0, set to 1 (the values will be 0 anyway, so division result is 0)
    p_sum_safe = torch.where(p_sum > 0, p_sum, torch.ones_like(p_sum))
    n_sum_safe = torch.where(n_sum > 0, n_sum, torch.ones_like(n_sum))
    
    # Normalize positive and negative components
    p_normalized = p_values / p_sum_safe
    n_normalized = n_values / n_sum_safe
    
    # Apply weights and aggregation factors
    # Shape: (m, n, *proj_output.shape)
    p_weighted = p_normalized.unsqueeze(0).unsqueeze(0) * wts_expanded * p_agg_wt
    n_weighted = n_normalized.unsqueeze(0).unsqueeze(0) * wts_expanded * n_agg_wt * -1.0
    
    # Sum over the first two dimensions (corresponding to wts dimensions)
    # This replaces the nested loop and accumulation
    result = (p_weighted + n_weighted).sum(dim=(0, 1))
    
    return result

@torch.compile
def calculate_wt_self_attention_parallel(
    wts: torch.Tensor,
    inp: torch.Tensor,
    w: Dict[str, torch.Tensor],
    model: Any
) -> torch.Tensor:

    device = inp.device
    
    # Compute Q, K, V projections using einsum
    query_output = torch.einsum('ij,kj->ik', inp, w['W_q'])
    key_output = torch.einsum('ij,kj->ik', inp, w['W_k'])
    value_output = torch.einsum('ij,kj->ik', inp, w['W_v'])
    
    # Get configuration parameters
    config = model.config
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    
    # Handle grouped query attention
    if hasattr(config, 'num_key_value_heads'):
        num_key_value_heads = config.num_key_value_heads
    else:
        num_key_value_heads = config.num_heads
    
    head_dim = hidden_size // num_heads
    
    # Reshape to multi-head format and transpose
    # (num_tokens, num_heads, head_dim) -> (num_heads, num_tokens, head_dim)
    query_states = query_output.reshape(query_output.shape[0], num_heads, head_dim).permute(1, 0, 2)
    key_states = key_output.reshape(key_output.shape[0], num_key_value_heads, head_dim).permute(1, 0, 2)
    value_states = value_output.reshape(value_output.shape[0], num_key_value_heads, head_dim).permute(1, 0, 2)
    
    # Repeat key/value heads for grouped query attention
    n_rep = num_heads // num_key_value_heads
    key_states = key_states.repeat_interleave(n_rep, dim=0)
    value_states = value_states.repeat_interleave(n_rep, dim=0)
    
    # Compute attention scores: (num_heads, num_tokens, num_tokens)
    QK_output = torch.einsum('hqd,hkd->hqk', query_states, key_states)
    attn_weights = QK_output / torch.sqrt(torch.tensor(head_dim, dtype=QK_output.dtype, device=device))
    
    # Apply softmax with numerical stability
    attn_weights = attn_weights - torch.max(attn_weights, dim=-1, keepdim=True).values
    attn_weights = torch.exp(attn_weights)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    
    # Weighted sum of values: (num_heads, num_tokens, head_dim)
    attn_output = torch.einsum('hqk,hkl->hql', attn_weights, value_states)
    
    # Reshape attention output back to original shape: (num_tokens, hidden_size)
    attn_output = attn_output.permute(1, 0, 2)  # (num_tokens, num_heads, head_dim)
    attn_output = attn_output.reshape(attn_output.shape[0], num_heads * head_dim)
    
    # Perform final linear projection: (num_tokens, hidden_size)
    final_output = torch.einsum('qd,dh->qh', attn_output, w['W_d'])
    
    # ============= Relevance Calculation =============
    
    # Step 1: Relevance for final linear projection
    # wts shape: (num_heads, num_tokens)
    # final_output shape: (num_tokens, hidden_size)
    wt_mat_attn_proj = calculate_wt_attention_output_projection(wts, final_output)
    
    # Step 2: Split relevance for V and QK paths
    # wt_mat_attn_proj shape: (num_tokens, hidden_size)
    relevance_V = wt_mat_attn_proj / 2
    relevance_QK = wt_mat_attn_proj / 2
    
    # Reshape to multi-head format for relevance propagation
    # (num_tokens, hidden_size) -> (num_tokens, num_heads, head_dim) -> (num_heads, num_tokens, head_dim)
    relevance_V_reshaped = relevance_V.reshape(relevance_V.shape[0], num_heads, head_dim).permute(1, 0, 2)
    relevance_QK_reshaped = relevance_QK.reshape(relevance_QK.shape[0], num_heads, head_dim).permute(1, 0, 2)
    
    # Step 3: Relevance calculation for V
    # We need to create dummy wts for the helper function - using attention weights shape
    dummy_wts = torch.ones_like(attn_weights[:, :, 0])  # (num_heads, num_tokens)
    wt_mat_V = relevance_V_reshaped  # Direct propagation through value path
    
    # Step 4: Relevance calculation for QK
    # Use relevance_QK to compute weighted QK matrix
    wt_mat_QK = calculate_relevance_QK(dummy_wts, QK_output)
    
    # Scale by actual relevance
    wt_mat_QK = wt_mat_QK * relevance_QK_reshaped.sum() / wt_mat_QK.sum() if wt_mat_QK.sum() != 0 else wt_mat_QK
    
    # Step 5: Relevance calculation for K and Q
    stabilized_QK_output = stabilize(QK_output * 2)
    norm_wt_mat_QK = wt_mat_QK / stabilized_QK_output
    
    wt_mat_Q = torch.einsum('htd,hdb->htb', norm_wt_mat_QK, key_states) * query_states
    wt_mat_K = torch.einsum('htd,htb->hbd', query_states, norm_wt_mat_QK) * key_states
    
    # Combine all relevance contributions
    wt_mat = wt_mat_V + wt_mat_K + wt_mat_Q
    
    # Reshape back to (num_tokens, hidden_size)
    wt_mat = wt_mat.permute(1, 0, 2)  # (num_tokens, num_heads, head_dim)
    wt_mat = wt_mat.reshape(wt_mat.shape[0], wt_mat.shape[1] * wt_mat.shape[2])
    
    return wt_mat
