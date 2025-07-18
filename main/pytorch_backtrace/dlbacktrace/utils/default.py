import gc
import torch
import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Tuple, List

from .cuda_utils.Linear.cuda_version.wt_fc_ops import calculate_wt_fc_interface as calculate_wt_fc_kernel_cuda

# Toggle debug prints
DEBUG = True
def log(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)
        
def calculate_wt_mul(
    R: np.ndarray, 
    X: np.ndarray, 
    Y: np.ndarray, 
    epsilon: float = 1e-12, 
    clip_negative: bool = True, 
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stable and safe distribution of relevance for elementwise multiplication.
    
    This function distributes relevance scores equally between two input arrays
    that participate in elementwise multiplication, effectively splitting the
    relevance 50-50 between X and Y inputs.
    
    Parameters
    ----------
    R : np.ndarray
        Relevance from the output. Will be converted to float32.
    X : np.ndarray
        First input to the multiplication. Will be converted to float32.
    Y : np.ndarray
        Second input to the multiplication. Will be converted to float32.
    epsilon : float, optional
        Small constant to avoid divide-by-zero (default: 1e-12).
        Note: Currently unused in this implementation.
    clip_negative : bool, optional
        If True, clips negative relevance to zero (default: True).
        Note: Currently unused in this implementation.
    normalize : bool, optional
        If True, ensures Rx + Ry ≈ sum(R) (default: True).
        Note: Currently unused in this implementation.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        R_x : Relevance assigned to X (R / 2)
        R_y : Relevance assigned to Y (R / 2)
    
    Notes
    -----
    The current implementation performs a simple 50-50 split of relevance,
    ignoring the epsilon, clip_negative, and normalize parameters.
    """
    # Convert inputs to float32 arrays 
    R = np.asarray(R, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    
    # Vectorized 50-50 relevance split using broadcasting
    relevance_half = R * 0.5
    
    return relevance_half, relevance_half

def np_swish(x, beta=0.75):
    z = 1 / (1 + np.exp(-(beta * x)))
    return x * z

def np_wave(x, alpha=1.0):
    return (alpha * x * np.exp(1.0)) / (np.exp(-x) + np.exp(x))

def np_pulse(x, alpha=1.0):
    return alpha * (1 - np.tanh(x) * np.tanh(x))

def np_absolute(x, alpha=1.0):
    return alpha * x * np.tanh(x)

def np_hard_sigmoid(x):
    return np.clip(0.2 * x + 0.5, 0, 1)

def np_sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z

def np_tanh(x):
    z = np.tanh(x)
    return z.astype(np.float32)

def calculate_start_wt(arg, scaler=1,*args, **kwargs):
    task = kwargs.get('task', None)  # Access 'task' from kwargs, default to None if not provided
    
    if arg.ndim == 1:
        arg = arg.reshape(1,-1)

    elif arg.ndim == 2:
        if task == "binary-classification":
            x = np.argmax(arg, axis=1)  # Get the index of the max value for each row
            m = np.max(arg, axis=1)  # Get the max value for each row
            y = np.zeros_like(arg)  # Initialize output array with zeros

            if scaler:
                y[np.arange(arg.shape[0]), x] = scaler  # Set the max index to scaler
            else:
                y[np.arange(arg.shape[0]), x] = m  # Set the max index to max value

        elif task == "generation":
            x = np.argmax(arg, axis=1) 
            y = np.zeros_like(arg)
            value = 1 / arg.shape[0]

            y[np.arange(arg.shape[0]), x] = value

    elif arg.ndim == 3:
        # Shape: (batch, sequence, features)
        if task == "binary-classification":
            x = np.argmax(arg, axis=2)  # max along features
            m = np.max(arg, axis=2)
            y = np.zeros_like(arg)

            batch_size, seq_len, _ = arg.shape
            for i in range(batch_size):
                for j in range(seq_len):
                    if scaler:
                        y[i, j, x[i, j]] = scaler
                    else:
                        y[i, j, x[i, j]] = m[i, j]
            log(y.shape,"y")

        elif task == "generation":
            # code here
            x = np.argmax(arg, axis=2)
            y = np.zeros_like(arg)
            value = 1 / arg.shape[1]

            batch_size, seq_len, _ = arg.shape
            for i in range(batch_size):
                for j in range(seq_len):
                    y[i, j, x[i, j]] = value 

    else:
        print(arg.shape)


    return y


def calculate_wt_add(wts, inp=None):
    wts_shape = wts.shape
    batch_size = wts_shape[0]
    batch_expanded_wts = []

    # Flatten weights
    for i in range(batch_size):
        wts_matrix = wts[i]
        expanded_wts_matrix = wts_matrix.reshape(-1)
        batch_expanded_wts.append(expanded_wts_matrix)

    batch_expanded_wts = np.array(batch_expanded_wts)

    batch_wt_mat = []
    batch_inp_list = []

    original_input_shapes = [x.shape for x in inp]

    # Compute broadcast shape
    target_shape = np.broadcast_shapes(*[x.shape for x in inp])
    inp_bcast = [np.broadcast_to(x, target_shape) for x in inp]

    for i, x in enumerate(inp_bcast):
        wt_mat = []
        inp_list = []
        for j in range(x.shape[0]):
            expanded_input = x[j].reshape(-1)
            inp_list.append(expanded_input)
            wt_mat.append(np.zeros_like(expanded_input))
        batch_inp_list.append(inp_list)
        batch_wt_mat.append(wt_mat)

    batch_inp_list = list(map(list, zip(*batch_inp_list)))
    batch_wt_mat = list(map(list, zip(*batch_wt_mat)))

    for i in range(batch_size):
        inp_list = [np.array(x) for x in batch_inp_list[i]]
        expanded_wts = batch_expanded_wts[i]

        for j in range(len(expanded_wts)):
            wt_ind1 = np.array(batch_wt_mat[i])[:, j]
            wt = expanded_wts[j]
            l1_ind1 = np.array(inp_list)[:, j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0

            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1

            p_agg_wt = n_agg_wt = 0
            total = p_sum + n_sum
            if total > 0:
                p_agg_wt = p_sum / total
                n_agg_wt = n_sum / total

            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1

            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

            for k in range(len(batch_wt_mat[i])):
                batch_wt_mat[i][k][j] = wt_ind1[k]

    # Reshape and reduce to original input shapes
    result = []
    for input_idx, orig_shape in enumerate(original_input_shapes):
        relevance_per_batch = []
        for batch_idx in range(batch_size):
            rel_flat = np.array(batch_wt_mat[batch_idx][input_idx])
            input_shape = inp_bcast[input_idx][batch_idx].shape

            if rel_flat.size != np.prod(input_shape):
                raise ValueError(f"[calculate_wt_add] ❌ Mismatch: trying to reshape {rel_flat.size} elements into {input_shape}")

            rel = rel_flat.reshape(input_shape)

            # Reduce back to original input shape
            reduced_rel = rel
            for axis in reversed(range(len(orig_shape))):
                if orig_shape[axis] == 1 and rel.shape[axis] > 1:
                    reduced_rel = np.sum(reduced_rel, axis=axis, keepdims=True)

            relevance_per_batch.append(reduced_rel)

        result.append(np.stack(relevance_per_batch, axis=0))

    return result


def calculate_wt_add_equal(R, inp):
    num_inputs = len(inp)
    input_shapes = [x.shape for x in inp]

    # Split relevance equally
    equal_relevance = [R / num_inputs for _ in range(num_inputs)]

    result = []
    for idx, orig_shape in enumerate(input_shapes):
        per_input_rel = []

        # For each batch entry
        for batch_idx in range(R.shape[0]):
            # Slice out the batch dimension
            rel_slice = equal_relevance[idx][batch_idx]  # shape == orig_shape[1:]
            reduced_rel = rel_slice

            # Only iterate over non-batch dims
            for axis, orig_dim in enumerate(orig_shape[1:]):
                # if this original dim was 1 but got broadcast, sum it back
                if orig_dim == 1 and reduced_rel.shape[axis] > 1:
                    reduced_rel = reduced_rel.sum(axis=axis, keepdims=True)

            per_input_rel.append(reduced_rel)

        # Reassemble batch dimension
        result.append(np.stack(per_input_rel, axis=0))

    return result 

@torch.compile
def calculate_wt_add_equal_vectorized(
    R: torch.Tensor,
    inp: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Highly optimized vectorized version of calculate_wt_add_equal.
    
    This version pre-computes reduction operations and uses more efficient
    tensor operations to minimize memory allocations and improve performance.
    
    Args:
        R: Relevance tensor of shape ``(batch, d1, d2, …)``.
        inp: List of input tensors, each with shape ``(batch, …)``. The first
            dimension of every tensor must equal ``R.shape[0]``.

    Returns:
        List of relevance tensors, one per entry in ``inp``, each having the
        same shape as the corresponding entry in ``inp``.
    """
    num_inputs = len(inp)
    device = R.device
    dtype = R.dtype
    
    # Pre-compute equal relevance once
    equal_relevance = R.div_(num_inputs) if R.requires_grad else R / num_inputs
    
    result: List[torch.Tensor] = []
    
    for tensor in inp:
        # Start with broadcasted relevance
        reduced_rel = equal_relevance
        
        # Collect all dimensions that need reduction
        dims_to_reduce = []
        for axis in range(1, len(tensor.shape)):
            if tensor.shape[axis] == 1 and reduced_rel.shape[axis] > 1:
                dims_to_reduce.append(axis)
        
        # Perform all reductions at once if possible
        if dims_to_reduce:
            # Sort dimensions in descending order to avoid index shifting
            dims_to_reduce.sort(reverse=True)
            
            for dim in dims_to_reduce:
                reduced_rel = torch.sum(reduced_rel, dim=dim, keepdim=True)
        
        result.append(reduced_rel)
    
    return result

def calculate_wt_fc_cuda(relevance_y, input_array, w, b, act):
    """
    CUDA-accelerated version that maintains the original algorithm structure.
    Handles batch processing and multi-dimensional inputs in Python,
    delegates core computation to CUDA kernel.
    
    Args:
        relevance_y: relevance at the output (same shape as linear output)
        input_array: input to the linear layer (can be any shape: [B, D], [B, T, D], etc.)
        w: weight matrix of the linear layer (shape: [out_dim, in_dim])
        b: bias vector (shape: [out_dim]) or None
        act: dict containing activation info with keys: "type", "range", "func"

    Returns:
        relevance_x: relevance at the input, same shape as input_array
    """
    # Flatten input except for last dim (same as original)
    original_shape = input_array.shape
    batch_dims = original_shape[:-1]
    feature_dim = original_shape[-1]

    input_flat = input_array.reshape(-1, feature_dim)
    relevance_flat = relevance_y.reshape(-1, relevance_y.shape[-1])

    # Process each batch element individually (maintains original logic)
    relevance_x_flat = []
    cuda_device = torch.device("cuda")
    
    # Convert weights to CUDA once (they're the same for all batch elements)
    w_torch = torch.tensor(w, dtype=torch.float32, device=cuda_device)
    b_torch = torch.tensor(b, dtype=torch.float32, device=cuda_device) if b is not None else torch.empty(0, device=cuda_device)
    
    for i in range(input_flat.shape[0]):
        inp = input_flat[i]            # shape: (input_dim,)
        wts = relevance_flat[i]        # shape: (output_dim,)
        
        # Convert to CUDA tensors for single batch element
        inp_torch = torch.tensor(inp, dtype=torch.float32, device=cuda_device)
        wts_torch = torch.tensor(wts, dtype=torch.float32, device=cuda_device)
        
        # Call CUDA kernel for single batch element
        result = calculate_wt_fc_kernel_cuda(
            wts_torch, inp_torch, w_torch, b_torch, act
        )
        
        relevance_x_flat.append(result.cpu().numpy())
    
    # Reshape back to original dimensions
    relevance_x_flat = np.array(relevance_x_flat)
    relevance_x = relevance_x_flat.reshape(*batch_dims, feature_dim)
    return relevance_x

def calculate_wt_fc(relevance_y, input_array, w, b, act):
    """
    DL Backtrace-style relevance tracing for fully connected (linear) layers.

    Args:
        relevance_y: relevance at the output (same shape as linear output)
        input_array: input to the linear layer (can be any shape: [B, D], [B, T, D], etc.)
        w: weight matrix of the linear layer (shape: [out_dim, in_dim])
        b: bias vector (shape: [out_dim]) or None
        act: dict containing activation info with keys: "type", "range", "func"

    Returns:
        relevance_x: relevance at the input, same shape as input_array
    """

    # Flatten input except for last dim
    original_shape = input_array.shape
    batch_dims = original_shape[:-1]
    feature_dim = original_shape[-1]

    input_flat = input_array.reshape(-1, feature_dim)
    relevance_flat = relevance_y.reshape(-1, relevance_y.shape[-1])

    relevance_x_flat = []

    for i in range(input_flat.shape[0]):
        inp = input_flat[i]            # shape: (input_dim,)
        wts = relevance_flat[i]        # shape: (output_dim,)

        # Contribution matrix: (output_dim, input_dim)
        mul_mat = np.einsum("ij,i->ij", w.T, inp).T
        wt_mat = np.zeros_like(mul_mat)

        for j in range(mul_mat.shape[0]):  # over output neurons
            contribs = mul_mat[j]          # shape: (input_dim,)
            wt_ind = wt_mat[j]
            wt = wts[j]

            # Positive and negative contributions
            p_ind = contribs > 0
            n_ind = contribs < 0
            p_sum = contribs[p_ind].sum()
            n_sum = -contribs[n_ind].sum()

            # Handle bias
            if b is not None:
                bias_val = b[j]
            else:
                bias_val = 0.0

            pbias = max(bias_val, 0)
            nbias = -min(bias_val, 0)

            t_sum = p_sum + pbias - n_sum - nbias

            # DL Backtrace activation-aware handling
            if act["type"] == "mono":
                if act["range"]["l"] is not None and t_sum < act["range"]["l"]:
                    p_sum = 0
                if act["range"]["u"] is not None and t_sum > act["range"]["u"]:
                    n_sum = 0

            elif act["type"] == "non_mono":
                t_act = act["func"](t_sum)
                p_act = act["func"](p_sum + pbias)
                n_act = act["func"](-1 * (n_sum + nbias))

                if act["range"]["l"] is not None and t_sum < act["range"]["l"]:
                    p_sum = 0
                if act["range"]["u"] is not None and t_sum > act["range"]["u"]:
                    n_sum = 0

                if p_sum > 0 and n_sum > 0:
                    if t_act == p_act:
                        n_sum = 0
                    elif t_act == n_act:
                        p_sum = 0

            # Avoid divide by zero
            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1

            # Aggregated weights
            p_agg_wt = ((p_sum + pbias) / (p_sum + n_sum + pbias + nbias)) * (p_sum / (p_sum + pbias)) if p_sum > 0 else 0
            n_agg_wt = ((n_sum + nbias) / (p_sum + n_sum + pbias + nbias)) * (n_sum / (n_sum + nbias)) if n_sum > 0 else 0

            # Redistribute relevance
            wt_ind[p_ind] = (contribs[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind[n_ind] = (contribs[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        relevance_vec = wt_mat.sum(axis=0)  # shape: (input_dim,)
        relevance_x_flat.append(relevance_vec)

    relevance_x_flat = np.array(relevance_x_flat)
    relevance_x = relevance_x_flat.reshape(*batch_dims, feature_dim)
    return relevance_x

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
    bs,_,_,_ = input_array.shape
    w = w.T
    relevance_x = []
    for i in range(bs):
        wts = relevance_y[i]
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

def calculate_wt_gavgpool(relevance_y, input_array):
    bs,_,_,_ = input_array.shape
    relevance_x =[]
    for i in range(bs):
        wts = relevance_y[i]
        inp = input_array[i]
        channels = wts.shape[0]
        inp = inp.T
        wts = wts.T
        wt_mat = np.zeros_like(inp)
        for c in range(channels):
            wt = wts[..., c]
            temp_wt = wt_mat[..., c]
            x = inp[..., c]
            p_mat = np.copy(x)
            n_mat = np.copy(x)
            p_mat[x < 0] = 0
            n_mat[x > 0] = 0
            p_sum = np.sum(p_mat)
            n_sum = np.sum(n_mat) * -1
            p_agg_wt = 0.0
            n_agg_wt = 0.0
            if p_sum + n_sum > 0.0:
                p_agg_wt = p_sum / (p_sum + n_sum)
                n_agg_wt = n_sum / (p_sum + n_sum)
            if p_sum == 0.0:
                p_sum = 1.0
            if n_sum == 0.0:
                n_sum = 1.0
            temp_wt = temp_wt + ((p_mat / p_sum) * wt * p_agg_wt)
            temp_wt = temp_wt + ((n_mat / n_sum) * wt * n_agg_wt * -1.0)
            wt_mat[..., c] = temp_wt
        relevance_x.append(wt_mat.T)
    return np.array(relevance_x)

def calculate_wt_max_unit(patch, wts, pool_size):
    pmax = np.einsum("ijk,k->ijk",np.ones_like(patch),np.max(np.max(patch,axis=0),axis=0))
    indexes = (patch-pmax)==0
    indexes = indexes.astype(np.float32)
    indexes_norm = 1.0/np.einsum("mnc->c",indexes)
    indexes = np.einsum("ijk,k->ijk",indexes,indexes_norm)
    out = np.einsum("ijk,k->ijk",indexes,wts)
    return out

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


def calculate_wt_avg_unit(patch, wts, pool_size):
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
        pad1 = pool_size[0]
        pad2 = pool_size[1]
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
                updates = calculate_wt_avg_unit(tmp_patch, wts[ind1,ind2,:], pool_size)
                # Build tensor with "filtered" gradient
                out_ds[np.ix_(indexes[0],indexes[1])]+=updates
        out_ds = out_ds[paddings[0][0]:(paddings[0][0]+inp.shape[0]),
                        paddings[1][0]:(paddings[1][0]+inp.shape[1]),:]
        relevance_x.append(out_ds.T)
    return np.array(relevance_x)

def stabilize(matrix, epsilon=1e-6):
    return matrix + epsilon * np.sign(matrix) 

def calculate_wt_self_attention(R_out, Q, K, V, masked_fill=None, scale=None, epsilon=1e-9):
    """
    Handles multi-head attention relevance backtrace.
    Args:
        R_out: [B, H, T_q, D]
        Q, K, V: [B, H, T_q, D] or [B, H, T_k, D]
        masked_fill: Optional additive mask of shape [B, 1, T_q, T_k] or [B, H, T_q, T_k]
        scale: Optional scaling factor (default: sqrt(D))
        epsilon: Small constant for numerical stability
    Returns:
        R_Q, R_K, R_V, R_masked_fill: same shape as Q, K, V, masked_fill
    """
    B, H, T_q, D = Q.shape
    T_k = K.shape[2]  # number of key tokens
    scale = scale or np.sqrt(D)

    # Step 1: Raw attention logits
    QK_output = np.matmul(Q, K.transpose(0, 1, 3, 2))  # [B, H, T_q, T_k]
    logits_unmasked = QK_output / scale 

    # Step 2: Softmax over unmasked logits (for debugging or interpretability)
    A = np.exp(logits_unmasked - np.max(logits_unmasked, axis=-1, keepdims=True))
    A = A / (np.sum(A, axis=-1, keepdims=True) + epsilon) 
    log(f"A (unmasked attention weights): {A.shape}")

    # Step 3: Apply additive attention mask (optional)
    masked_fill = None 
    if masked_fill is not None:
        logits_masked = logits_unmasked + masked_fill  # [B, H, T, T] + [B, 1, T, T]
        log(f"masked_fill--- minimum: {np.min(masked_fill)}, maximum: {np.max(masked_fill)}") 
    else:
        logits_masked = logits_unmasked.copy()

    # Step 4: Softmax over masked logits
    A_masked = np.exp(logits_masked - np.max(logits_masked, axis=-1, keepdims=True))
    A_masked = A_masked / (np.sum(A_masked, axis=-1, keepdims=True) + epsilon) 
    log(f"A_masked (masked attention weights): {A_masked.shape}")

    # Step 5: Compute attention output using masked weights
    attention_output = np.matmul(A_masked, V)  # [B, H, T_q, D] 
    log(f"attention_output (using A_masked): {attention_output.shape}")

    # Step 6: Relevance propagation to V and attention weights
    relevance_norm_attn_out = R_out / stabilize(attention_output * 2, epsilon)
    log(f"relevance_norm_attn_out---  value: {np.sum(relevance_norm_attn_out):.2f},  shape: {relevance_norm_attn_out.shape}")
    log(f"V: {V.shape}, A_masked: {A_masked.shape}")
    log(f"type(A_masked): {type(A_masked)}, type(V): {type(V)}, type(relevance_norm_attn_out): {type(relevance_norm_attn_out)}")

    R_QK = np.matmul(relevance_norm_attn_out, np.transpose(V, (0, 1, 3, 2))) * A
    R_V = np.matmul(np.transpose(A, (0, 1, 3, 2)), relevance_norm_attn_out) * V
    log(f"updated R_QK--- rel: {np.sum(R_QK):.2f}, shape: {R_QK.shape}")
    log(f"updated R_V--- rel: {np.sum(R_V):.2f}, shape: {R_V.shape}") 

    # Relevance Calculation for K and Q
    relevance_norm_QK_out = R_QK / stabilize(QK_output *2, epsilon)
    log(f"relevance_norm_QK_out---  value: {np.sum(relevance_norm_QK_out):.2f},  shape: {relevance_norm_QK_out.shape}")
    log(f"Q: {Q.shape},  K: {K.shape}")

    R_Q = np.matmul(relevance_norm_QK_out, K) * Q
    R_K = np.transpose(np.matmul(np.transpose(Q, (0, 1, 3, 2)), relevance_norm_QK_out), (0, 1, 3, 2)) * K

    log(f"updated R_Q--- rel: {np.sum(R_Q):.2f}, shape: {R_Q.shape}")
    log(f"updated R_K--- rel: {np.sum(R_K):.2f}, shape: {R_K.shape}")

    # Relevance `masked_fill`
    delta_A = A - A_masked
    R_blocked_per_qk = np.einsum('bhqk,bhkd->bhqk', delta_A, V)
    R_masked_fill = R_blocked_per_qk.sum(axis=1, keepdims=True)
    log(f"R_masked_fill: {R_masked_fill.shape}")

    total_out = np.sum(R_out)
    total_prop = np.sum(R_Q) + np.sum(R_K) + np.sum(R_V) + np.sum(R_masked_fill)

    log(f"R_out sum:         {total_out:.4f}")
    log(f"Propagated sum:    {total_prop:.4f}")
    log(f"Difference (leak): {total_out - total_prop:.4f}")

    return R_Q, R_K, R_V, R_masked_fill


def calculate_wt_embedding(R_out, input_ids, vocab_size=None, aggregate="sum"):
    """
    Backtrace relevance for embedding layers.
    
    Args:
        R_out: relevance at output of embedding [B, T, D]
        input_ids: input token ids [B, T] (int)
        vocab_size: optional; default inferred from max token ID
        aggregate: 'sum' or 'mean' for token-wise relevance
    
    Returns:
        relevance_matrix: [vocab_size, D] or [B, T] if reduced
    """
    B, T, D = R_out.shape
    if vocab_size is None:
        vocab_size = np.max(input_ids) + 1

    relevance_matrix = np.zeros((vocab_size, D), dtype=np.float32)

    for b in range(B):
        for t in range(T):
            token_id = input_ids[b, t]
            relevance_matrix[token_id] += R_out[b, t]  # accumulate vector relevance

    if aggregate == "sum":
        return relevance_matrix  # [V, D]
    elif aggregate == "mean":
        return np.mean(relevance_matrix, axis=-1)  # [V]
    else:
        raise ValueError(f"Unsupported aggregation: {aggregate}")
