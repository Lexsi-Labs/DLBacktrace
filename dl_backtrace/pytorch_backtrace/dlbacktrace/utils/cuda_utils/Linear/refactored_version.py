import numpy as np
from typing import Dict, Callable, Any, Union

def calculate_wt_fc(
    row_specific_weights: np.ndarray,
    input_activations: np.ndarray,
    weights_matrix: Union[np.ndarray, Any],
    bias_vector: Union[np.ndarray, Any],
    activation_params: Dict[str, Any]
) -> np.ndarray:
    """
    Calculates feature contributions based on weights, inputs, and activation rules.

    This function is a refactored and vectorized version of an original loop-based
    calculation. It aims for improved performance and clarity while maintaining
    strict logical and numerical equivalence to the original.

    Args:
        row_specific_weights (np.ndarray): 1D NumPy array of shape (D_in,).
            Scalar weight associated with each row of the (conceptual) scaled_weights_matrix.
            Corresponds to `wts` in the original.
        input_activations (np.ndarray): 1D NumPy array of shape (D_in,).
            Input feature values. Corresponds to `inp` in the original.
        weights_matrix (Tensorable): A 2D array-like (NumPy array or object with
            .numpy() method) of shape (D_in, D_out). Corresponds to `w` in the original.
        bias_vector (Tensorable): A 1D array-like (NumPy array or object with
            .numpy() method) of shape (D_in,). Corresponds to `b` in the original.
        activation_params (Dict[str, Any]): Dictionary configuring activation logic.
            Expected keys:
            - "type": str, either "mono" or "non_mono".
            - "range": Dict, with optional keys "l" and "u" for lower/upper thresholds.
                       The values of "l" and "u" are themselves checked for truthiness
                       before being used as thresholds, replicating original behavior.
            - "func": Callable (only for "non_mono"), a function to apply for activation.
                      This function should be compatible with NumPy array inputs (ufunc-like).
            Corresponds to `act` in the original.

    Returns:
        np.ndarray: A 1D NumPy array of shape (D_out,) representing the summed
                    contributions. Corresponds to `wt_mat.sum(axis=0)` in the original.

    """
    # Convert inputs to NumPy arrays if they have a .numpy() method
    # Use np.asarray to avoid copying if already a NumPy array.
    w_arr = weights_matrix.numpy() if hasattr(weights_matrix, 'numpy') else np.asarray(weights_matrix)
    b_arr = bias_vector.numpy() if hasattr(bias_vector, 'numpy') else np.asarray(bias_vector)
    
    row_specific_weights_arr = np.asarray(row_specific_weights)
    input_activations_arr = np.asarray(input_activations)

    # Determine the primary float data type for calculations.
    # If w_arr is not float, default to float32. This ensures float arithmetic.
    # This `dtype` will be used for creating new arrays and typed scalar constants.
    calc_dtype = w_arr.dtype
    if not np.issubdtype(calc_dtype, np.floating):
        calc_dtype = np.float32
    
    # Scalar constants typed to match calc_dtype, preventing unintended type promotions.
    _zero = np.array(0.0, dtype=calc_dtype)
    _one = np.array(1.0, dtype=calc_dtype)
    _minus_one = np.array(-1.0, dtype=calc_dtype)

    # Calculate scaled_weights_matrix: w_arr * input_activations_arr[:, np.newaxis]
    # Original: mul_mat = np.einsum("ij,i->ij", w_arr.T, input_activations_arr).T
    # This is equivalent to W[r,c] * inp[r] for W (D_in, D_out) and inp (D_in,)
    scaled_weights_matrix = w_arr * input_activations_arr[np.newaxis, :] # Shape (D_out, D_in)

    # Positive and negative part identification and sums
    # is_positive_part/is_negative_part are boolean masks of shape (D_out, D_in)
    is_positive_part = scaled_weights_matrix > _zero
    is_negative_part = scaled_weights_matrix < _zero

    # Summing parts: scaled_weights_matrix * boolean_mask zeros out irrelevant parts.
    p_sum_vec = np.sum(scaled_weights_matrix * is_positive_part, axis=1) # Shape (D_in,)
    n_sum_vec = np.sum(scaled_weights_matrix * is_negative_part, axis=1) * _minus_one # Sum of abs values; Shape (D_in,)

    # Bias handling (vectorized)
    pbias_vec = np.maximum(_zero, b_arr) # Shape (D_in,)
    nbias_vec = np.maximum(_zero, -b_arr) # Shape (D_in,)
    
    # Total sum used for activation range checks
    t_sum_vec = p_sum_vec + pbias_vec - (n_sum_vec + nbias_vec) # Shape (D_in,)

    # --- Activation Logic ---
    # Copies of p_sum_vec and n_sum_vec for act["func"] arguments if "non_mono",
    # using values *before* range modifications, as per original logic.
    p_sum_for_act_func = p_sum_vec.copy()
    n_sum_for_act_func = n_sum_vec.copy()
    
    # Apply range-based modifications. Original uses `if act["range"]["l"]:`
    # which checks truthiness of the threshold value itself (e.g. 0 is falsy).
    # Direct dictionary access (e.g. `activation_params["range"]["l"]`) is used
    # to match original's potential KeyError if keys are missing.
    act_range = activation_params["range"] # Cache for minor optimization

    # Lower bound check
    if "l" in act_range and act_range["l"]: # Checks key existence and truthiness of value
        lower_threshold = act_range["l"]
        condition_lower_bound = t_sum_vec < lower_threshold
        p_sum_vec = np.where(condition_lower_bound, _zero, p_sum_vec)

    # Upper bound check
    if "u" in act_range and act_range["u"]: # Checks key existence and truthiness of value
        upper_threshold = act_range["u"]
        condition_upper_bound = t_sum_vec > upper_threshold
        n_sum_vec = np.where(condition_upper_bound, _zero, n_sum_vec)

    if activation_params["type"] == "non_mono":
        act_func = activation_params["func"] # Expected to be a ufunc or vectorized callable

        # Calculate activations. For p_act and n_act, use sums *before* range modifications.
        t_act_vec = act_func(t_sum_vec) 
        p_act_vec = act_func(p_sum_for_act_func + pbias_vec)
        n_act_vec = act_func(_minus_one * (n_sum_for_act_func + nbias_vec))
        
        # Conditions for zeroing out, using p_sum_vec/n_sum_vec *after* range modifications.
        cond_both_sums_positive = (p_sum_vec > _zero) & (n_sum_vec > _zero)
        cond_t_act_equals_p_act = (t_act_vec == p_act_vec)
        # For "elif" logic: ensure t_act != p_act before checking t_act == n_act
        cond_t_act_equals_n_act_for_elif = ~cond_t_act_equals_p_act & (t_act_vec == n_act_vec)

        # if t_act == p_act: n_sum = 0
        n_sum_vec = np.where(cond_both_sums_positive & cond_t_act_equals_p_act, _zero, n_sum_vec)
        
        # elif t_act == n_act: p_sum = 0
        p_sum_vec = np.where(cond_both_sums_positive & cond_t_act_equals_n_act_for_elif, _zero, p_sum_vec)

    # --- Calculate Aggregate Weights (p_agg_wt, n_agg_wt) ---
    # These use p_sum_vec/n_sum_vec *after* all activation modifications.
    # Original behavior for division by zero (potential inf/nan) is preserved by np.errstate.
    
    p_agg_wt_vec = np.zeros_like(p_sum_vec, dtype=calc_dtype) # Initialize with zeros of correct dtype
    n_agg_wt_vec = np.zeros_like(n_sum_vec, dtype=calc_dtype) # Initialize with zeros of correct dtype

    condition_p_sum_gt_zero = p_sum_vec > _zero
    condition_n_sum_gt_zero = n_sum_vec > _zero
    
    # Common denominator for ratio1: (p_sum + n_sum + pbias + nbias)
    # This can be zero, leading to inf/nan by standard float arithmetic.
    den1_common = p_sum_vec + n_sum_vec + pbias_vec + nbias_vec

    with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings, not errors; results are inf/nan
        # Positive aggregate weight calculation
        # Denominator for ratio2_p: (p_sum + pbias). If p_sum > 0 and pbias >=0, this is > 0.
        # So ratio2_p is finite where p_sum > 0.
        # ratio1_p can be inf/nan. Product will propagate inf/nan.
        ratio1_p = (p_sum_vec + pbias_vec) / den1_common
        ratio2_p = p_sum_vec / (p_sum_vec + pbias_vec) # Safe from 0/0 if p_sum_vec > 0
        p_agg_wt_val = ratio1_p * ratio2_p
        p_agg_wt_vec = np.where(condition_p_sum_gt_zero, p_agg_wt_val, _zero)

        # Negative aggregate weight calculation (similar logic)
        ratio1_n = (n_sum_vec + nbias_vec) / den1_common
        ratio2_n = n_sum_vec / (n_sum_vec + nbias_vec) # Safe from 0/0 if n_sum_vec > 0
        n_agg_wt_val = ratio1_n * ratio2_n
        n_agg_wt_vec = np.where(condition_n_sum_gt_zero, n_agg_wt_val, _zero)

    # --- Denominators for final normalization step ---
    # Original: if p_sum == 0, p_sum_div = 1.0. Same for n_sum.
    # This uses p_sum_vec/n_sum_vec *after* activation modifications.
    p_sum_div_vec = p_sum_vec.copy()
    p_sum_div_vec[p_sum_vec == _zero] = _one # Item assignment casts _one to array's dtype

    n_sum_div_vec = n_sum_vec.copy()
    n_sum_div_vec[n_sum_vec == _zero] = _one
    
    # --- Calculate final weighted contributions ---
    output_contributions = np.zeros_like(scaled_weights_matrix, dtype=calc_dtype)

    # Positive contributions
    # (scaled_val / p_sum_div) * row_weight * p_agg_factor
    # Division by p_sum_div_vec is safe as it's been set to 1.0 where it was 0.
    term_p_norm = scaled_weights_matrix / p_sum_div_vec[:, np.newaxis]
    update_p_values = term_p_norm * row_specific_weights_arr[:, np.newaxis] * p_agg_wt_vec[:, np.newaxis]
    # Only apply update where scaled_weights_matrix was originally positive
    output_contributions = np.where(is_positive_part, update_p_values, output_contributions)

    # Negative contributions
    # (scaled_val / n_sum_div) * row_weight * n_agg_factor * -1.0
    term_n_norm = scaled_weights_matrix / n_sum_div_vec[:, np.newaxis]
    update_n_values = term_n_norm * row_specific_weights_arr[:, np.newaxis] * n_agg_wt_vec[:, np.newaxis] * _minus_one
    # Only apply update where scaled_weights_matrix was originally negative
    output_contributions = np.where(is_negative_part, update_n_values, output_contributions)
    
    # Sum contributions along axis 0 (rows) to get final result per output feature
    final_output_weights = np.sum(output_contributions, axis=0) # Shape (D_out,)

    return final_output_weights
