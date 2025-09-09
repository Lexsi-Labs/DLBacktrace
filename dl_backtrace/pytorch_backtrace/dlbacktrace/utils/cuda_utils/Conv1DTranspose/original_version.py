import numpy as np

def calculate_output_padding_conv1d_transpose(input_shape, kernel_size, padding, strides,dilation):
    if padding == 'valid':
        out_shape = [(input_shape[0] - 1) * strides + kernel_size[0]]
        paddings = [[0, 0], [0, 0]]
    elif padding == 0:
        out_shape = [(input_shape[0] - 1) * strides + kernel_size[0]]
        paddings = [[0, 0], [0, 0]]
    elif isinstance(padding, int):
        out_shape = [input_shape[0] * strides]
        pad_v = (dilation * (kernel_size[0] - 1)) - padding
        out_shape = [input_shape[0] * strides + pad_v]
        paddings = [[pad_v, pad_v], 
                    [0, 0]]
    else:  # 'same' padding
        out_shape = [input_shape[0] * strides]
        pad_h = max(0, (input_shape[0] - 1) * strides + kernel_size[0] - out_shape[0])
        paddings = [[pad_h // 2, pad_h // 2], 
                    [0, 0]]
    
    return out_shape, paddings

def calculate_wt_conv1d_transpose_unit(patch, wts, w, b, act):
    if patch.ndim == 1:
        patch = patch.reshape(1, -1)
    elif patch.ndim != 2:
        raise ValueError(f"Unexpected patch shape: {patch.shape}")
    
    k = w.permute(0, 2, 1).numpy()
    bias = b.numpy()
    b_ind = bias > 0
    bias_pos = bias * b_ind
    b_ind = bias < 0
    bias_neg = bias * b_ind * -1.0  
    conv_out = np.einsum('ijk,mj->ijk', k, patch)
    p_ind = conv_out > 0
    p_ind = conv_out * p_ind
    n_ind = conv_out < 0
    n_ind = conv_out * n_ind
    
    p_sum = np.einsum("ijl->l", p_ind)
    n_sum = np.einsum("ijl->l", n_ind) * -1.0
    t_sum = p_sum + n_sum
    
    wt_mat = np.zeros_like(k)
    p_saturate = p_sum > 0
    n_saturate = n_sum > 0
    
    if act["type"] == 'mono':
        if act["range"]["l"]:
            p_saturate = t_sum > act["range"]["l"]
        if act["range"]["u"]:
            n_saturate = t_sum < act["range"]["u"]
    elif act["type"] == 'non_mono':
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
    
    p_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * p_saturate
    n_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * n_saturate
    wt_mat = wt_mat + (p_ind * p_agg_wt)
    wt_mat = wt_mat + (n_ind * n_agg_wt * -1.0)
    wt_mat = np.sum(wt_mat, axis=-1)
    return wt_mat

def calculate_wt_conv1d_transpose(wts, inp, w, b, padding, strides, dilation, act):
    wts = wts.T
    inp = inp.T
    w = w.T
    out_shape, paddings = calculate_output_padding_conv1d_transpose(inp.shape, w.shape, padding, strides, dilation)
    out_ds = np.zeros(out_shape + [w.shape[2]])

    for ind in range(inp.shape[0]):
        out_ind = ind * strides
        tmp_patch = inp[ind, :]
        updates = calculate_wt_conv1d_transpose_unit(tmp_patch, wts[ind, :], w, b, act)
        end_ind = min(out_ind + w.shape[0], out_shape[0])
        valid_updates = updates[:end_ind - out_ind, :]
        out_ds[out_ind:end_ind, :] += valid_updates
    
    if padding == 'same':
        adjusted_out_ds = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            start_i = max(0, i * strides)
            end_i = min(out_ds.shape[0], (i + 1) * strides)
            relevant_area = out_ds[start_i:end_i, :]
            adjusted_out_ds[i, :] = np.sum(relevant_area, axis=0)
        out_ds = adjusted_out_ds
    elif padding == 0:
        adjusted_out_ds = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            start_i = max(0, i * strides)
            end_i = min(out_ds.shape[0], (i + 1) * strides)
            relevant_area = out_ds[start_i:end_i, :]
            adjusted_out_ds[i, :] = np.sum(relevant_area, axis=0)
        out_ds = adjusted_out_ds
    else:
        out_ds = out_ds[paddings[0][0]:(paddings[0][0] + inp.shape[0]), :]
    return out_ds