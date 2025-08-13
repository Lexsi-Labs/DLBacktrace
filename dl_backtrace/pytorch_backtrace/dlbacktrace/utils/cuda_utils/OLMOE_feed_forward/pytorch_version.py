import torch
import torch.nn.functional as F
import torch._dynamo
from typing import Tuple, Dict, Any

torch._dynamo.config.suppress_errors = True

def torch_swish(x: torch.Tensor, beta: float = 0.75) -> torch.Tensor:
    """PyTorch implementation of Swish activation function."""
    z = torch.sigmoid(torch.clamp(beta * x, -500, 500))
    return x * z

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wts_torch = torch.tensor(wts, dtype=torch.float32, device=device)
    inp_torch = torch.tensor(inp, dtype=torch.float32, device=device)

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
    intermediate_outputs = olmoe_mlp_forward(inp_torch, w_torch, model)

    # Initialize tensors with proper device and dtype
    final_relevance_input = torch.zeros_like(inp_torch)
    relevance_expert = torch.zeros(num_experts, dtype=inp_torch.dtype, device=device)
    in_relevance = torch.zeros_like(wts_torch)

    # Process each expert
    for expert_idx in range(num_experts):
        expert_data = intermediate_outputs[f'expert_{expert_idx}'] 
        top_x = expert_data['top_x']
        intermediate_data = expert_data['intermediate_output']

        # Skip if no tokens assigned to this expert
        if top_x.numel() == 0:
            continue

        # Update in_relevance for assigned tokens
        in_relevance[top_x] = wts_torch[top_x] / num_experts
        relev_half = in_relevance * 0.5

        # Process relevance through the network
        relevance_int_output = process_single_relevance_proj(relev_half, intermediate_data)
        relev_proj = 0.5 * relevance_int_output

        # Compute input relevances
        relevance_input_gate_proj = process_single_relevance_gated_proj(relev_proj, inp_torch)
        relevance_input_up_proj = process_single_relevance_proj(relev_proj, inp_torch)
        
        relevance_current_state = relevance_input_gate_proj + relevance_input_up_proj

        # Update final relevance and expert scores
        if top_x.numel() > 0:
            final_relevance_input[top_x] += relevance_current_state[top_x]
            relevance_expert[expert_idx] = torch.sum(relevance_current_state[top_x])

    # Process router logits relevance
    relevance_router_logits = process_single_relevance_router_logits(
        in_relevance * 0.5, inp_torch, w_torch['W_gate']
    )

    final_relevance_input += relevance_router_logits

    # Final normalization (preserving original logic)
    final_relevance_input = (wts_torch / final_relevance_input) * final_relevance_input

    return final_relevance_input.cpu().numpy(), relevance_expert.cpu().numpy()