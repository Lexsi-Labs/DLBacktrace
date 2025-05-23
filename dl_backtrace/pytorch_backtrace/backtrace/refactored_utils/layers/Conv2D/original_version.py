import numpy as np  
from ..ConvUnit2D.original_version import calculate_wt_conv_unit
from ..Padding.original import calculate_padding

def calculate_wt_conv(wts, inp, w, b, padding, strides, act):
    wts = wts.T
    inp = inp.T
    w = w.T
    input_padded, paddings = calculate_padding(w.shape, inp, padding, strides)
    out_ds = np.zeros_like(input_padded)
    for ind1 in range(wts.shape[0]):
        for ind2 in range(wts.shape[1]):
            indexes = [np.arange(ind1*strides[0], ind1*(strides[0])+w.shape[0]),
                       np.arange(ind2*strides[1], ind2*(strides[1])+w.shape[1])]
            # Take slice
            tmp_patch = input_padded[np.ix_(indexes[0],indexes[1])]
            updates = calculate_wt_conv_unit(tmp_patch, wts[ind1,ind2,:], w, b, act)
            # Build tensor with "filtered" gradient
            out_ds[np.ix_(indexes[0],indexes[1])]+=updates
    out_ds = out_ds[paddings[0][0]:(paddings[0][0]+inp.shape[0]),
                    paddings[1][0]:(paddings[1][0]+inp.shape[1]),:]
    return out_ds
