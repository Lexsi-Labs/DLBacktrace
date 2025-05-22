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
