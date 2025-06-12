from typing import Callable, Union, Optional, Dict, Any
import torch

# For type hinting the activation function callable
ActivationCallable = Callable[[torch.Tensor], torch.Tensor]

def calculate_wt_fc(
    row_specific_weights: torch.Tensor,
    input_activations: torch.Tensor,
    weights_matrix: torch.Tensor,
    bias_vector: torch.Tensor,
    activation_params: Dict[str, Any]
) -> torch.Tensor:
    """
    Calculates feature contributions in PyTorch based on weights, inputs, and activation rules.

    This function is a PyTorch refactoring of a NumPy-based calculation,
    optimized for performance and potential autograd compatibility. It aims for strict
    logical and numerical equivalence to the original NumPy version.

    Args:
        row_specific_weights (torch.Tensor): 1D Tensor of shape (D_in,).
            Scalar weight associated with each row of the (conceptual) scaled_weights_matrix.
        input_activations (torch.Tensor): 1D Tensor of shape (D_in,).
            Input feature values.
        weights_matrix (torch.Tensor): A 2D Tensor of shape (D_in, D_out).
        bias_vector (torch.Tensor): A 1D Tensor of shape (D_in,).
        activation_params (Dict[str, Any]): Dictionary configuring activation logic.
            Expected keys:
            - "type": str, either "mono" or "non_mono".
            - "range": Dict[str, float], with optional keys "l" and "u" for lower/upper thresholds.
                       The values of "l" and "u" are checked for truthiness (non-zero).
            - "func": ActivationCallable (only for "non_mono"), a JIT-compatible function
                      (e.g., torch.relu, torch.sigmoid, or a @torch.jit.script function)
                      to apply for activation. It should operate element-wise on Tensors.

    Returns:
        torch.Tensor: A 1D Tensor of shape (D_out,) representing the summed contributions.
    """

    dtype = weights_matrix.dtype
    device = weights_matrix.device

    # Scalar constants typed to match dtype and device
    _zero = torch.tensor(0.0, dtype=dtype, device=device)
    _one = torch.tensor(1.0, dtype=dtype, device=device)
    _minus_one = torch.tensor(-1.0, dtype=dtype, device=device)

    # Calculate scaled_weights_matrix: W[r,c] * inp[r]
    scaled_weights_matrix = weights_matrix * input_activations[None, :]

    # Positive and negative part identification and sums
    # is_positive_part/is_negative_part are boolean masks of shape (D_in, D_out)
    is_positive_part = scaled_weights_matrix > _zero
    is_negative_part = scaled_weights_matrix < _zero

    # Summing parts: scaled_weights_matrix * boolean_mask zeros out irrelevant parts.
    # Sum over axis 1 (D_out dimension) to get per-row (D_in dimension) sums.
    p_sum_vec = torch.sum(scaled_weights_matrix * is_positive_part, dim=1) # Shape (D_in,)
    n_sum_vec = torch.sum(scaled_weights_matrix * is_negative_part, dim=1) * _minus_one # Sum of abs values; Shape (D_in,)

    # Bias handling (vectorized)
    pbias_vec = torch.maximum(_zero, bias_vector) # Shape (D_in,)
    nbias_vec = torch.maximum(_zero, -bias_vector) # Shape (D_in,)
    
    # Total sum used for activation range checks
    t_sum_vec = p_sum_vec + pbias_vec - (n_sum_vec + nbias_vec) # Shape (D_in,)

    # --- Activation Logic ---
    # Copies of p_sum_vec and n_sum_vec for act["func"] arguments if "non_mono",
    # using values *before* range modifications, as per original logic.
    p_sum_for_act_func = p_sum_vec.clone()
    n_sum_for_act_func = n_sum_vec.clone()
    
    activation_range = activation_params["range"]

    # Lower bound check
    # Original checks truthiness of the threshold value itself (e.g. 0 or 0.0 is falsy).
    if activation_range["l"]:
        lower_threshold = torch.tensor(activation_range["l"], dtype=dtype, device=device)
        condition_lower_bound = t_sum_vec < lower_threshold
        # p_sum_vec is updated here
        p_sum_vec = torch.where(condition_lower_bound, _zero, p_sum_vec)

    # Upper bound check
    if activation_range["u"]:
        upper_threshold = torch.tensor(activation_range["u"], dtype=dtype, device=device)
        condition_upper_bound = t_sum_vec > upper_threshold
        # n_sum_vec is updated here
        n_sum_vec = torch.where(condition_upper_bound, _zero, n_sum_vec)

    if activation_params["type"] == "non_mono":
        # This cast is necessary for JIT if "func" is stored as Any.
        # Assumes activation_params["func"] is a JIT-compatible callable.
        activation_function : ActivationCallable = activation_params["func"]

        t_act_vec = activation_function(t_sum_vec)
        p_act_vec = activation_function(p_sum_for_act_func + pbias_vec)
        n_act_vec = activation_function(_minus_one * (n_sum_for_act_func + nbias_vec))
        
        # Conditions for zeroing out, using p_sum_vec/n_sum_vec *after* range modifications.
        cond_both_sums_positive = (p_sum_vec > _zero) & (n_sum_vec > _zero)
        cond_t_act_equals_p_act = (t_act_vec == p_act_vec)
        # For "elif" logic: ensure t_act != p_act before checking t_act == n_act
        cond_t_act_equals_n_act_for_elif = ~cond_t_act_equals_p_act & (t_act_vec == n_act_vec)

        # if t_act == p_act: n_sum = 0 (where both sums were positive)
        n_sum_vec = torch.where(cond_both_sums_positive & cond_t_act_equals_p_act, _zero, n_sum_vec)
        
        # elif t_act == n_act: p_sum = 0 (where both sums were positive)
        p_sum_vec = torch.where(cond_both_sums_positive & cond_t_act_equals_n_act_for_elif, _zero, p_sum_vec)

    # --- Calculate Aggregate Weights (p_agg_wt, n_agg_wt) ---
    # These use p_sum_vec/n_sum_vec *after* all activation modifications.
    
    condition_p_sum_gt_zero = p_sum_vec > _zero
    condition_n_sum_gt_zero = n_sum_vec > _zero
    
    den1_common = p_sum_vec + n_sum_vec + pbias_vec + nbias_vec

    # Positive aggregate weight calculation
    ratio1_p = (p_sum_vec + pbias_vec) / den1_common # Can be inf/nan if den1_common is 0

    ratio2_p_denominator = p_sum_vec + pbias_vec
    safe_ratio2_p_denominator = torch.where(ratio2_p_denominator == _zero, _one, ratio2_p_denominator)
    ratio2_p = p_sum_vec / safe_ratio2_p_denominator
    
    p_agg_wt_val_candidate = ratio1_p * ratio2_p
    p_agg_wt_vec = torch.where(condition_p_sum_gt_zero, p_agg_wt_val_candidate, _zero)

    # Negative aggregate weight calculation
    ratio1_n = (n_sum_vec + nbias_vec) / den1_common # Can be inf/nan

    ratio2_n_denominator = n_sum_vec + nbias_vec
    safe_ratio2_n_denominator = torch.where(ratio2_n_denominator == _zero, _one, ratio2_n_denominator)
    ratio2_n = n_sum_vec / safe_ratio2_n_denominator

    n_agg_wt_val_candidate = ratio1_n * ratio2_n
    n_agg_wt_vec = torch.where(condition_n_sum_gt_zero, n_agg_wt_val_candidate, _zero)

    p_sum_div_vec = torch.where(p_sum_vec == _zero, _one, p_sum_vec)
    n_sum_div_vec = torch.where(n_sum_vec == _zero, _one, n_sum_vec)
    
    output_contributions = torch.zeros_like(scaled_weights_matrix, dtype=dtype, device=device)

    term_p_norm = scaled_weights_matrix / p_sum_div_vec[:, None]
    update_p_values = term_p_norm * row_specific_weights[:, None] * p_agg_wt_vec[:, None]
    output_contributions = torch.where(is_positive_part, update_p_values, output_contributions)

    term_n_norm = scaled_weights_matrix / n_sum_div_vec[:, None]
    update_n_values = term_n_norm * row_specific_weights[:, None] * n_agg_wt_vec[:, None] * _minus_one
    output_contributions = torch.where(is_negative_part, update_n_values, output_contributions)
    
    final_output_weights = torch.sum(output_contributions, dim=0)

    return final_output_weights

compiled_calculate_wt_fc = torch.compile(calculate_wt_fc)
