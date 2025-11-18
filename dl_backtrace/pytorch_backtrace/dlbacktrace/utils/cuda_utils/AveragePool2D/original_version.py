import numpy as np

def calculate_padding(kernel_size, inp, padding, strides, const_val=0.0):
    if padding=='valid':
        return (inp, [[0,0],[0,0],[0,0]])
    elif padding == 'same':
        h = inp.shape[0]%strides[0]
        if h==0:
            pad_h = np.max([0,kernel_size[0]-strides[0]]) 
        else:
            pad_h = np.max([0,kernel_size[0]-h])

        v = inp.shape[1]%strides[1]
        if v==0:
            pad_v = np.max([0,kernel_size[1]-strides[1]]) 
        else:
            pad_v = np.max([0,kernel_size[1]-v]) 

        paddings = [np.floor([pad_h/2.0,(pad_h+1)/2.0]).astype("int32"),
                    np.floor([pad_v/2.0,(pad_v+1)/2.0]).astype("int32"),
                    np.zeros((2)).astype("int32")]
        inp_pad = np.pad(inp, paddings, 'constant', constant_values=const_val)
        return (inp_pad,paddings)
    else:
        if isinstance(padding, tuple) and padding != (None, None):
            pad_h = padding[0]
            pad_v = padding[1]
            paddings = [np.floor([pad_h,pad_h]).astype("int32"),
                    np.floor([pad_v,pad_v]).astype("int32"),
                    np.zeros((2)).astype("int32")]
            inp_pad = np.pad(inp, paddings, 'constant', constant_values=const_val)
            return (inp_pad,paddings)
        else:
            return (inp, [[0,0],[0,0],[0,0]])

def calculate_wt_avg_unit(patch, wts):
    p_ind = patch>0
    p_ind = patch*p_ind
    p_sum = np.einsum("ijk->k",p_ind)
    n_ind = patch<0
    n_ind = patch*n_ind
    n_sum = np.einsum("ijk->k",n_ind)*-1.0
    t_sum = p_sum+n_sum
    wt_mat = np.zeros_like(patch)
    p_saturate = p_sum>0
    n_saturate = n_sum>0
    t_sum[t_sum==0] = 1.0
    p_agg_wt = (1.0/(t_sum))*wts*p_saturate
    n_agg_wt = (1.0/(t_sum))*wts*n_saturate
    wt_mat = wt_mat+(p_ind*p_agg_wt)
    wt_mat = wt_mat+(n_ind*n_agg_wt*-1.0)
    return wt_mat

def calculate_wt_avgpool(relevance_y, input_array, pool_size, pad, stride):
    bs,_,_,_ = input_array.shape
    relevance_x =[]
    for i in range(bs):
        wts = relevance_y[i]
        inp = input_array[i]
        inp = inp.T
        wts = wts.T
        strides = (stride,stride)
        padding = (pad,pad)
        input_padded, paddings = calculate_padding(pool_size, inp, padding, strides, -np.inf)
        out_ds = np.zeros_like(input_padded)
        for ind1 in range(wts.shape[0]):
            for ind2 in range(wts.shape[1]):
                indexes = [np.arange(ind1*strides[0], ind1*(strides[0])+pool_size[0]),
                        np.arange(ind2*strides[1], ind2*(strides[1])+pool_size[1])]
                # Take slice
                tmp_patch = input_padded[np.ix_(indexes[0],indexes[1])]
                updates = calculate_wt_avg_unit(tmp_patch, wts[ind1,ind2,:])
                # Build tensor with "filtered" gradient
                out_ds[np.ix_(indexes[0],indexes[1])]+=updates
        out_ds = out_ds[paddings[0][0]:(paddings[0][0]+inp.shape[0]),
                        paddings[1][0]:(paddings[1][0]+inp.shape[1]),:]
        relevance_x.append(out_ds.T)
    return np.array(relevance_x)