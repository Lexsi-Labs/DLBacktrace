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

def calculate_wt_max_unit_1d(patch, wts):
    pmax = np.max(patch, axis=0)
    indexes = (patch - pmax) == 0
    indexes = indexes.astype(np.float32)
    indexes_norm = 1.0 / np.sum(indexes, axis=0)
    indexes = np.einsum("ij,j->ij", indexes, indexes_norm)
    out = np.einsum("ij,j->ij", indexes, wts)
    return out

def calculate_wt_maxpool_1d(wts, inp, pool_size, padding, stride):
    inp = inp.T
    wts = wts.T
    input_padded, paddings = calculate_padding_1d(pool_size, inp, padding, stride, -np.inf)
    out_ds = np.zeros_like(input_padded)
    stride=stride
    pool_size=pool_size
    for ind in range(wts.shape[0]):
        indexes = np.arange(ind * stride, ind * stride + pool_size)
        tmp_patch = input_padded[indexes]
        updates = calculate_wt_max_unit_1d(tmp_patch, wts[ind, :])
        out_ds[indexes] += updates
    out_ds = out_ds[paddings[0][0]:(paddings[0][0] + inp.shape[0])]
    return out_ds