import numpy as np

def calculate_padding_1d(kernel_size, inp, padding, strides, const_val=0.0):
    if padding == 'valid':
        return inp, [[0, 0],[0,0]]
    elif padding == 0:
        return inp,  [[0, 0],[0,0]]
    elif isinstance(padding, int):
        inp_pad = np.pad(inp, ((padding, padding), (0,0)), 'constant', constant_values=const_val)
        return inp_pad, [[padding, padding],[0,0]]
    else:
        remainder = inp.shape[0] % strides
        if remainder == 0:
            pad_total = max(0, kernel_size - strides)
        else:
            pad_total = max(0, kernel_size - remainder)
        
        pad_left = int(np.floor(pad_total / 2.0))
        pad_right = int(np.ceil(pad_total / 2.0))
        
        inp_pad = np.pad(inp, ((pad_left, pad_right),(0,0)), 'constant', constant_values=const_val)
        return inp_pad, [[pad_left, pad_right],[0,0]]

def calculate_wt_avg_unit_1d(patch, wts):
    p_ind = patch > 0
    p_ind = patch * p_ind
    p_sum = np.sum(p_ind, axis=0)
    n_ind = patch < 0
    n_ind = patch * n_ind
    n_sum = np.sum(n_ind, axis=0) * -1.0
    t_sum = p_sum + n_sum
    wt_mat = np.zeros_like(patch)
    p_saturate = p_sum > 0
    n_saturate = n_sum > 0
    t_sum[t_sum == 0] = 1.0
    p_agg_wt = (1.0 / t_sum) * wts * p_saturate
    n_agg_wt = (1.0 / t_sum) * wts * n_saturate
    wt_mat = wt_mat + (p_ind * p_agg_wt)
    wt_mat = wt_mat + (n_ind * n_agg_wt * -1.0)
    return wt_mat

def calculate_wt_avgpool_1d(wts, inp, pool_size, padding, stride):
    wts = wts.T
    inp = inp.T
    stride=stride
    pool_size=pool_size
    input_padded, paddings = calculate_padding_1d(pool_size, inp, padding[0], stride[0], 0)
    out_ds = np.zeros_like(input_padded)
    for ind in range(wts.shape[0]):
        indexes = np.arange(ind * stride[0], ind * stride[0] + pool_size[0])
        tmp_patch = input_padded[indexes]
        updates = calculate_wt_avg_unit_1d(tmp_patch, wts[ind, :])
        out_ds[indexes] += updates
    out_ds = out_ds[paddings[0][0]:(paddings[0][0] + inp.shape[0])]
    return out_ds