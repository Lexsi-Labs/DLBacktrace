import numpy as np

def calculate_wt_conv_unit_1d_v2(patch, wts, w, b, act):
    """
    Compute relevance for a single patch of the input tensor.

    Parameters:
        patch (ndarray): Patch of input corresponding to the receptive field of the kernel.
        wts (ndarray): Relevance values from the next layer for this patch.
        w (ndarray): Weights of the convolutional kernel.
        b (ndarray): Bias values for the convolution.
        act (dict): Activation function details. Should contain:
            - "type": Type of activation ('mono' or 'non_mono').
            - "range": Range dictionary with "l" (lower bound) and "u" (upper bound).
            - "func": Function to apply for activation.

    Returns:
        wt_mat (ndarray): Weighted relevance matrix for the patch.
    """
    kernel = w
    bias = b
    wt_mat = np.zeros_like(kernel)
    # Compute convolution output
    conv_out = np.einsum("ijk,ij->ijk", kernel, patch)
    # Separate positive and negative contributions
    p_ind = conv_out > 0
    p_ind = conv_out * p_ind
    p_sum = np.einsum("ijk->k",p_ind)
    n_ind = conv_out < 0
    n_ind = conv_out * n_ind
    n_sum = np.einsum("ijk->k",n_ind) * -1.0
    t_sum = p_sum + n_sum
    # Handle positive and negative bias
    bias_pos = bias * (bias > 0)
    bias_neg = bias * (bias < 0) * -1.0
    # Activation handling (saturate weights if specified)
    p_saturate = p_sum > 0
    n_saturate = n_sum > 0
    if act["type"] == 'mono':
        if act["range"]["l"]:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = temp_ind
        if act["range"]["u"]:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = temp_ind
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

    # Aggregate weights
    p_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * p_saturate
    n_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * n_saturate

    wt_mat = wt_mat + (p_ind * p_agg_wt)
    wt_mat = wt_mat + (n_ind * n_agg_wt * -1.0)
    wt_mat = np.sum(wt_mat, axis=-1)
    return wt_mat

def calculate_padding_1d_v2(kernel_size, input_length, padding, strides, dilation=1, const_val=0.0):
    """
    Calculate and apply padding to match TensorFlow Keras behavior for 'same', 'valid', and custom padding.
    
    Parameters:
        kernel_size (int): Size of the convolutional kernel.
        input_length (int): Length of the input along the spatial dimension.
        padding (str/int/tuple): Padding type. Can be:
            - 'valid': No padding.
            - 'same': Pads to maintain output length equal to input length (stride=1).
            - int: Symmetric padding on both sides.
            - tuple/list: Explicit padding [left, right].
        strides (int): Stride size of the convolution.
        dilation (int): Dilation rate for the kernel.
        const_val (float): Value used for padding. Defaults to 0.0.
    
    Returns:
        padded_length (int): Length of the input after padding.
        paddings (list): Padding applied on left and right sides.
    """
    effective_kernel_size = (kernel_size - 1) * dilation + 1  # Effective size considering dilation

    if padding == 'valid':
        return input_length, [0, 0]
    elif padding == 'same':
        # Total padding required to keep output size same as input
        pad_total = max(0, (input_length - 1) * strides + effective_kernel_size - input_length)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
    elif isinstance(padding, int):
        pad_left = padding
        pad_right = padding
    elif isinstance(padding, (list, tuple)) and len(padding) == 2:
        pad_left, pad_right = padding
    else:
        raise ValueError("Invalid padding. Use 'valid', 'same', an integer, or a tuple/list of two integers.")

    padded_length = input_length + pad_left + pad_right
    return padded_length, [pad_left, pad_right]

def calculate_wt_conv_1d(wts, inp, w, b, padding, stride, dilation, groups, act):
    """
    Perform relevance propagation for a 1D convolution layer with support for groups and dilation.

    Parameters:
        wts (ndarray): Relevance values from the next layer (shape: [output_length, output_channels]).
        inp (ndarray): Input tensor for the current layer (shape: [input_length, input_channels]).
        w (ndarray): Weights of the convolutional kernel (shape: [kernel_size, input_channels/groups, output_channels/groups]).
        b (ndarray): Bias values for the convolution (shape: [output_channels]).
        padding (str/int/tuple): Padding mode. Supports 'same', 'valid', integer, or tuple of (left, right).
        stride (int): Stride of the convolution.
        dilation (int): Dilation rate for the kernel.
        groups (int): Number of groups for grouped convolution.
        act (dict): Activation function details.

    Returns:
        out_ds (ndarray): Propagated relevance for the input tensor.
    """
    wts = wts.T
    inp = inp.T
    w = w.T    
    kernel_size = w.shape[0]
    input_length = inp.shape[0]

    # Compute and apply padding
    padded_length, paddings = calculate_padding_1d_v2(kernel_size, input_length, padding, stride, dilation)
    inp_padded = np.pad(inp, ((paddings[0], paddings[1]), (0, 0)), 'constant', constant_values=0)
    # Initialize output relevance map
    out_ds = np.zeros_like(inp_padded)

    # Handle grouped convolution
    input_channels_per_group = inp.shape[1] // groups
    output_channels_per_group = wts.shape[1] // groups

    for g in range(groups):
        input_start = g * input_channels_per_group
        input_end = (g + 1) * input_channels_per_group
        output_start = g * output_channels_per_group
        output_end = (g + 1) * output_channels_per_group

        for ind in range(wts.shape[0]):
            start_idx = ind * stride
            tmp_patch = inp_padded[start_idx:start_idx + kernel_size * dilation:dilation, input_start:input_end]
            updates = calculate_wt_conv_unit_1d_v2(tmp_patch, wts[ind, output_start:output_end], w[:, :, output_start:output_end], b[output_start:output_end], act)
            out_ds[start_idx:start_idx + kernel_size * dilation:dilation, input_start:input_end] += updates

    # Remove padding
    out_ds = out_ds[paddings[0]:(paddings[0] + input_length), :]
    return out_ds