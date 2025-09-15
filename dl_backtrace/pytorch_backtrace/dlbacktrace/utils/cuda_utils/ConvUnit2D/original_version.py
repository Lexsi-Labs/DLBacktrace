import numpy as np

def calculate_wt_conv_unit(patch, wts, w, b, act):
    k = w
    bias = b
    if bias is not None:
        b_ind = bias > 0
        bias_pos = bias * b_ind
        b_ind = bias < 0
        bias_neg = bias * b_ind * -1.0  

        conv_out = np.einsum("ijkl,ijk->ijkl", k, patch)
        p_ind = conv_out > 0
        p_ind = conv_out * p_ind
        p_sum = np.einsum("ijkl->l", p_ind)
        n_ind = conv_out < 0
        n_ind = conv_out * n_ind
        n_sum = np.einsum("ijkl->l", n_ind) * -1.0
        t_sum = p_sum + n_sum

        wt_mat = np.zeros_like(k)
        p_saturate = p_sum > 0
        n_saturate = n_sum > 0
        
        if act["type"]=='mono':
            if act["range"]["l"]:
                temp_ind = t_sum > act["range"]["l"]
                p_saturate = temp_ind
            if act["range"]["u"]:
                temp_ind = t_sum < act["range"]["u"]
                n_saturate = temp_ind
        elif act["type"]=='non_mono':
            t_act = act["func"](t_sum)
            p_act = act["func"](p_sum + bias_pos)
            n_act = act["func"](-1*(n_sum + bias_neg))
            if act["range"]["l"]:
                temp_ind = t_sum > act["range"]["l"]
                p_saturate = p_saturate*temp_ind
            if act["range"]["u"]:
                temp_ind = t_sum < act["range"]["u"]
                n_saturate = n_saturate*temp_ind
            temp_ind = np.abs(t_act - p_act)>1e-5
            n_saturate = n_saturate*temp_ind
            temp_ind = np.abs(t_act - n_act)>1e-5
            p_saturate = p_saturate*temp_ind

        denom = p_sum + n_sum + bias_pos + bias_neg
        denom = np.where(denom == 0, 1e-12, denom)

        p_agg_wt = (1.0 / denom) * wts * p_saturate
        n_agg_wt = (1.0 / denom) * wts * n_saturate

    else:
        conv_out = np.einsum("ijkl,ijk->ijkl", k, patch)
        p_ind = conv_out > 0
        p_ind = conv_out * p_ind
        p_sum = np.einsum("ijkl->l", p_ind)
        n_ind = conv_out < 0
        n_ind = conv_out * n_ind
        n_sum = np.einsum("ijkl->l", n_ind) * -1.0
        t_sum = p_sum + n_sum

        wt_mat = np.zeros_like(k)
        p_saturate = p_sum > 0
        n_saturate = n_sum > 0

        if act["type"] == 'mono':
            if act["range"]["l"]:
                temp_ind = t_sum > act["range"]["l"]
                p_saturate = temp_ind
            if act["range"]["u"]:
                temp_ind = t_sum < act["range"]["u"]
                n_saturate = temp_ind
        elif act["type"]=='non_mono':
            # For bias-less case, use dummy bias_pos/neg as zeros
            bias_pos = np.zeros_like(p_sum)
            bias_neg = np.zeros_like(n_sum)

            t_act = act["func"](t_sum)
            p_act = act["func"](p_sum + bias_pos)
            n_act = act["func"](-1 * (n_sum + bias_neg))
            if act["range"]["l"]:
                temp_ind = t_sum > act["range"]["l"]
                p_saturate = p_saturate * temp_ind
            if act["range"]["u"]:
                temp_ind = t_sum < act["range"]["u"]
                n_saturate = n_saturate * temp_ind
            temp_ind = np.abs(t_act - p_act) > 1e-5
            n_saturate = n_saturate * temp_ind
            temp_ind = np.abs(t_act - n_act) > 1e-5
            p_saturate = p_saturate * temp_ind

        denom = p_sum + n_sum
        denom = np.where(denom == 0, 1e-12, denom)

        p_agg_wt = (1.0 / denom) * wts * p_saturate
        n_agg_wt = (1.0 / denom) * wts * n_saturate

    wt_mat = wt_mat + (p_ind * p_agg_wt)
    wt_mat = wt_mat + (n_ind * n_agg_wt * -1.0)
    wt_mat = np.sum(wt_mat,axis=-1)
    return wt_mat
