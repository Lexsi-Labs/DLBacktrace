import numpy as np

def calculate_wt_fc(relevance_y, input_array, w, b, act):
    """
    DL Backtrace-style relevance tracing for fully connected (linear) layers.

    Args:
        relevance_y: relevance at the output (same shape as linear output)
        input_array: input to the linear layer (can be any shape: [B, D], [B, T, D], etc.)
        w: weight matrix of the linear layer (shape: [out_dim, in_dim])
        b: bias vector (shape: [out_dim]) or None
        act: dict containing activation info with keys: "type", "range", "func"

    Returns:
        relevance_x: relevance at the input, same shape as input_array
    """

    # Flatten input except for last dim
    original_shape = input_array.shape
    batch_dims = original_shape[:-1]
    feature_dim = original_shape[-1]

    input_flat = input_array.reshape(-1, feature_dim)
    relevance_flat = relevance_y.reshape(-1, relevance_y.shape[-1])

    relevance_x_flat = []

    for i in range(input_flat.shape[0]):
        inp = input_flat[i]            # shape: (input_dim,)
        wts = relevance_flat[i]        # shape: (output_dim,)

        # Contribution matrix: (output_dim, input_dim)
        mul_mat = np.einsum("ij,i->ij", w.T, inp).T
        wt_mat = np.zeros_like(mul_mat)

        for j in range(mul_mat.shape[0]):  # over output neurons
            contribs = mul_mat[j]          # shape: (input_dim,)
            wt_ind = wt_mat[j]
            wt = wts[j]

            # Positive and negative contributions
            p_ind = contribs > 0
            n_ind = contribs < 0
            p_sum = contribs[p_ind].sum()
            n_sum = -contribs[n_ind].sum()

            # Handle bias
            if b is not None:
                bias_val = b[j]
            else:
                bias_val = 0.0

            pbias = max(bias_val, 0)
            nbias = -min(bias_val, 0)

            t_sum = p_sum + pbias - n_sum - nbias

            # DL Backtrace activation-aware handling
            if act["type"] == "mono":
                if act["range"]["l"] is not None and t_sum < act["range"]["l"]:
                    p_sum = 0
                if act["range"]["u"] is not None and t_sum > act["range"]["u"]:
                    n_sum = 0

            elif act["type"] == "non_mono":
                t_act = act["func"](t_sum)
                p_act = act["func"](p_sum + pbias)
                n_act = act["func"](-1 * (n_sum + nbias))

                if act["range"]["l"] is not None and t_sum < act["range"]["l"]:
                    p_sum = 0
                if act["range"]["u"] is not None and t_sum > act["range"]["u"]:
                    n_sum = 0

                if p_sum > 0 and n_sum > 0:
                    if t_act == p_act:
                        n_sum = 0
                    elif t_act == n_act:
                        p_sum = 0

            # Avoid divide by zero
            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1

            # Aggregated weights
            p_agg_wt = ((p_sum + pbias) / (p_sum + n_sum + pbias + nbias)) * (p_sum / (p_sum + pbias)) if p_sum > 0 else 0
            n_agg_wt = ((n_sum + nbias) / (p_sum + n_sum + pbias + nbias)) * (n_sum / (n_sum + nbias)) if n_sum > 0 else 0

            # Redistribute relevance
            wt_ind[p_ind] = (contribs[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind[n_ind] = (contribs[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        relevance_vec = wt_mat.sum(axis=0)  # shape: (input_dim,)
        relevance_x_flat.append(relevance_vec)

    relevance_x_flat = np.array(relevance_x_flat)
    relevance_x = relevance_x_flat.reshape(*batch_dims, feature_dim)
    return relevance_x
