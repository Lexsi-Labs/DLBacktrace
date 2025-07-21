import numpy as np
from ..WtMaxunit2D.original_version import calculate_wt_max_unit
from ..Padding.original import calculate_padding

def calculate_wt_maxpool(wts, inp, pool_size, padding, strides):
    wts=wts.T
    inp=inp.T
    if isinstance(strides, int):
        strides = (strides,strides)
    if isinstance(padding, int):
        padding = (padding,padding)
    input_padded, paddings = calculate_padding(pool_size, inp, padding, strides, -np.inf)
    out_ds = np.zeros_like(input_padded)
    for ind1 in range(wts.shape[0]):
        for ind2 in range(wts.shape[1]):
            indexes = [np.arange(ind1*strides[0], ind1*(strides[0])+pool_size[0]),
                       np.arange(ind2*strides[1], ind2*(strides[1])+pool_size[1])]
            tmp_patch = input_padded[np.ix_(indexes[0],indexes[1])]
            updates = calculate_wt_max_unit(tmp_patch, wts[ind1,ind2,:], pool_size)
            out_ds[np.ix_(indexes[0],indexes[1])]+=updates
    out_ds = out_ds[paddings[0][0]:(paddings[0][0]+inp.shape[0]),
                    paddings[1][0]:(paddings[1][0]+inp.shape[1]),:]
    return out_ds
