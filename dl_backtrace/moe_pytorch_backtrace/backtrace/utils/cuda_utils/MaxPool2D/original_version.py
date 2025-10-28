import numpy as np
from ..WtMaxunit2D.original_version import calculate_wt_max_unit
from ..Padding.original import calculate_padding

def calculate_wt_maxpool(relevance_y, input_array, pool_size, pad, stride):
    bs,_,_,_ = input_array.shape
    relevance_x =[]
    for i in range(bs):
        wts = relevance_y[i]
        inp = input_array[i]
        inp = inp.T
        wts = wts.T
        if isinstance(stride,tuple):
            strides = stride
            padding = pad
        else:
            strides = (stride,stride)
            padding = (pad,pad)
        input_padded, paddings = calculate_padding(pool_size, inp, padding, strides)
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
        relevance_x.append(out_ds.T)
    return np.array(relevance_x)
