import numpy as np

def calculate_output_padding_conv2d_transpose(input_shape, kernel_size, padding, strides):
    if padding == 'valid':
        out_shape = [(input_shape[0] - 1) * strides[0] + kernel_size[0],
                     (input_shape[1] - 1) * strides[1] + kernel_size[1]]
        paddings = [[0, 0], [0, 0], [0, 0]]
    elif padding == (0,0):
        out_shape = [(input_shape[0] - 1) * strides[0] + kernel_size[0],
                     (input_shape[1] - 1) * strides[1] + kernel_size[1]]
        paddings = [[0, 0], [0, 0], [0, 0]]
    elif isinstance(padding, tuple) and padding != (None, None):
        out_shape = [input_shape[0] * strides[0], input_shape[1] * strides[1]]
        pad_h = padding[0]
        pad_v = padding[1]
        paddings = [[pad_h, pad_h], [pad_v, pad_v], [0, 0]]
    else:  # 'same' padding
        out_shape = [input_shape[0] * strides[0], input_shape[1] * strides[1]]
        pad_h = max(0, (input_shape[0] - 1) * strides[0] + kernel_size[0] - out_shape[0])
        pad_v = max(0, (input_shape[1] - 1) * strides[1] + kernel_size[1] - out_shape[1])
        paddings = [[pad_h // 2, pad_h - pad_h // 2], 
                    [pad_v // 2, pad_v - pad_v // 2], 
                    [0, 0]]
    
    return out_shape, paddings

def calculate_wt_conv2d_transpose_unit(patch, wts, w, b, act):
    if patch.ndim == 1:
        patch = patch.reshape(1, 1, -1)
    elif patch.ndim == 2:
        patch = patch.reshape(1, *patch.shape)
    elif patch.ndim != 3:
        raise ValueError(f"Unexpected patch shape: {patch.shape}")

    k = w.permute(0, 1, 3, 2).numpy()
    bias = b.numpy()
    b_ind = bias > 0
    bias_pos = bias * b_ind
    b_ind = bias < 0
    bias_neg = bias * b_ind * -1.0  
    
    conv_out = np.einsum('ijkl,mnk->ijkl', k, patch)    
    p_ind = conv_out > 0
    p_ind = conv_out * p_ind
    n_ind = conv_out < 0
    n_ind = conv_out * n_ind
    
    p_sum = np.einsum("ijkl->l", p_ind)
    n_sum = np.einsum("ijkl->l", n_ind) * -1.0
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

def calculate_wt_conv2d_transpose(wts, inp, w, b, padding, strides, act):
    wts = wts.T
    inp = inp.T
    w = w.T
    out_shape, paddings = calculate_output_padding_conv2d_transpose(inp.shape, w.shape, padding, strides)
    out_ds = np.zeros(out_shape + [w.shape[3]])
    
    for ind1 in range(inp.shape[0]):
        for ind2 in range(inp.shape[1]):
            out_ind1 = ind1 * strides[0]
            out_ind2 = ind2 * strides[1]
            tmp_patch = inp[ind1, ind2, :]
            updates = calculate_wt_conv2d_transpose_unit(tmp_patch, wts[ind1, ind2, :], w, b, act)
            end_ind1 = min(out_ind1 + w.shape[0], out_shape[0])
            end_ind2 = min(out_ind2 + w.shape[1], out_shape[1])
            valid_updates = updates[:end_ind1 - out_ind1, :end_ind2 - out_ind2, :]
            out_ds[out_ind1:end_ind1, out_ind2:end_ind2, :] += valid_updates
    
    if padding == 'same':
        adjusted_out_ds = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                start_i = max(0, i * strides[0])
                start_j = max(0, j * strides[1])
                end_i = min(out_ds.shape[0], (i+1) * strides[0])
                end_j = min(out_ds.shape[1], (j+1) * strides[1])
                relevant_area = out_ds[start_i:end_i, start_j:end_j, :]
                adjusted_out_ds[i, j, :] = np.sum(relevant_area, axis=(0, 1))
        out_ds = adjusted_out_ds
    elif isinstance(padding, tuple) and padding != (None, None):
        adjusted_out_ds = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                start_i = max(0, i * strides[0])
                start_j = max(0, j * strides[1])
                end_i = min(out_ds.shape[0], (i+1) * strides[0])
                end_j = min(out_ds.shape[1], (j+1) * strides[1])
                relevant_area = out_ds[start_i:end_i, start_j:end_j, :]
                adjusted_out_ds[i, j, :] = np.sum(relevant_area, axis=(0, 1))
        out_ds = adjusted_out_ds
    else:
        out_ds = out_ds[paddings[0][0]:(paddings[0][0] + inp.shape[0]),
                        paddings[1][0]:(paddings[1][0] + inp.shape[1]), :]
    
    return out_ds