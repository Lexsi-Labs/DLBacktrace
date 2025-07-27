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

def calculate_wt_conv(relevance_y, input_array, w, b, padding, strides, act):
    bs,_,_,_ = input_array.shape #bs, c, h , w
    w = w.T
    relevance_x = []
    for i in range(bs):
        wts = relevance_y[i] #bs, c, h, w
        inp = input_array[i]
        wts = wts.T
        inp = inp.T
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
        relevance_x.append(out_ds.T)
    return np.array(relevance_x)
