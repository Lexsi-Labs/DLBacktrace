import torch
import numpy as np

# Linear Layer
from .cuda_utils.Linear_v2.original_version import calculate_wt_fc as calculate_wt_fc_original_linear
from .cuda_utils.Linear_v2.pytorch_version import calculate_wt_fc as calculate_wt_fc_pytorch_linear
from .cuda_utils.Linear_v3.cuda_v3 import calculate_wt_fc_cuda as calculate_wt_fc_cuda_linear

# Conv2D Layer
from .cuda_utils.Conv2D.original_version import calculate_wt_conv as calculate_wt_conv_original
from .cuda_utils.Conv2D.refactored_version import calculate_wt_conv as calculate_wt_conv_refactored
from .cuda_utils.Conv2D.pytorch_version import calculate_wt_conv as calculate_wt_conv_parallel

# MaxPool2D Layer
from .cuda_utils.MaxPool2D.original_version import calculate_wt_maxpool as calculate_wt_maxpool_original
from .cuda_utils.MaxPool2D.refactored_version import calculate_wt_maxpool as calculate_wt_maxpool_refactored
from .cuda_utils.MaxPool2D.pytorch_version import calculate_wt_maxpool as calculate_wt_maxpool_pytorch

# AdaptiveAvgPool2D Layer
from .cuda_utils.AdaptiveAvgPool2D.original_version import calculate_wt_gavgpool as calculate_wt_gavgpool_original
from .cuda_utils.AdaptiveAvgPool2D.refactored_version import calculate_wt_gavgpool as calculate_wt_gavgpool_refactored
from .cuda_utils.AdaptiveAvgPool2D.pytorch_version import calculate_wt_gavgpool as calculate_wt_gavgpool_pytorch

# Embedded Layer
from .cuda_utils.Embedded.original_version import calculate_wt_embedding as calculate_wt_embedding_original
from .cuda_utils.Embedded.refactored_version import calculate_wt_embedding as calculate_wt_embedding_refactored
from .cuda_utils.Embedded.pytorch_version import calculate_wt_embedding as calculate_wt_embedding_pytorch
from .cuda_utils.Embedded.cuda_v2 import calculate_wt_embedding_cuda as calculate_wt_embedding_cuda

# SelfAttention Layer
from .cuda_utils.SelfAttention.original_version import calculate_wt_self_attention as calculate_wt_self_attention_original
from .cuda_utils.SelfAttention.pytorch_v2 import calculate_wt_self_attention as calculate_wt_self_attention_pytorch
from .cuda_utils.SelfAttention.cuda_v3 import calculate_wt_self_attention_cuda as calculate_wt_self_attention_cuda

# Wt_add_equal Layer
from .cuda_utils.Wt_add_equal.original_version import calculate_wt_add_equal as calculate_wt_add_original
from .cuda_utils.Wt_add_equal.refactored_version import calculate_wt_add_equal as calculate_wt_add_refactored

# Wt_mul Layer
from .cuda_utils.Wt_mul.original_version import calculate_wt_mul as calculate_wt_mul_original
from .cuda_utils.Wt_mul.refactored_version import calculate_wt_mul as calculate_wt_mul_refactored
from .cuda_utils.Wt_mul.pytorch_version import calculate_wt_mul_gpu as calculate_wt_mul_pytorch

def _prepare_tensors(device, *arrays):
    return [torch.tensor(arr, dtype=torch.float32, device=device) for arr in arrays]

def launch_linear(version, wts, inp, w, b, act):
    if version == 'original':
        func = calculate_wt_fc_original_linear
        return func(wts, inp, w, b, act)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if version == 'pytorch':
        wts_t, inp_t, w_t, b_t = _prepare_tensors(device, wts, inp, w, b)
        return calculate_wt_fc_pytorch_linear(wts_t, inp_t, w_t, b_t, act)
    elif version == 'cuda':
        try:
            # Try CUDA implementation with error handling
            result = calculate_wt_fc_cuda_linear(wts, inp, w, b, act)
            if result is None:
                print(f"⚠️  CUDA linear implementation returned None, falling back to original")
                return calculate_wt_fc_original_linear(wts, inp, w, b, act)
            return result
        except Exception as e:
            print(f"⚠️  CUDA linear implementation failed: {e}")
            print(f"   Falling back to original implementation")
            return calculate_wt_fc_original_linear(wts, inp, w, b, act)
    else:
        raise ValueError(f"Unknown version for Linear layer: {version}")

def launch_conv2d(version, wts, inp, w, b, padding, strides, act):
    if version in ['original', 'refactored']:
        func = calculate_wt_conv_original if version == 'original' else calculate_wt_conv_refactored
        return func(wts, inp, w, b, padding, strides, act)
    
    elif version == 'cuda':
        return calculate_wt_conv_parallel(wts, inp, w, b, padding, strides, act)

    else:
        raise ValueError(f"Unknown version for Conv2D layer: {version}")

def launch_embedding(version, R_out, inp, vocab_size, aggregate):
    if version in ['original', 'refactored']:
        func = calculate_wt_embedding_original if version == 'original' else calculate_wt_embedding_refactored
        return func(R_out, inp, vocab_size, aggregate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R_out_t = torch.tensor(R_out, dtype=torch.float32, device=device)
    inp_t = torch.tensor(inp, dtype=torch.long, device=device)

    if version == 'pytorch':
        return calculate_wt_embedding_pytorch(R_out_t, inp_t, vocab_size, aggregate)[0].cpu().numpy()
    elif version == 'cuda':
        return calculate_wt_embedding_cuda(R_out_t, inp_t, vocab_size, aggregate)[0].cpu().numpy()
    else:
        raise ValueError(f"Unknown version for Embedding layer: {version}")

def launch_self_attention(version, R_out, Q, K, V, masked_fill, scale = None, epsilon=1e-9):
    if version == 'original':
        return calculate_wt_self_attention_original(R_out, Q, K, V, masked_fill, scale, epsilon)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R_out_t, Q_t, K_t, V_t = _prepare_tensors(device, R_out, Q, K, V)
    masked_fill_t = torch.tensor(masked_fill, dtype=torch.float32, device=device) if masked_fill is not None else None
    scale_t = torch.tensor(scale, dtype=torch.float32, device=device) if scale is not None else None
    
    if version == 'pytorch':
        result_torch = calculate_wt_self_attention_pytorch(R_out_t, Q_t, K_t, V_t, masked_fill_t, scale_t, epsilon)
        return [arr.cpu().numpy() for arr in result_torch]
    elif version == 'cuda':
        result_cuda = calculate_wt_self_attention_cuda(R_out_t, Q_t, K_t, V_t, masked_fill_t, scale_t)
        return [arr.cpu().numpy() for arr in result_cuda]
    else:
        raise ValueError(f"Unknown version for SelfAttention layer: {version}")

def launch_wt_add_equal(version, R_out, inp):
    if version in ['original', 'refactored']:
        func = calculate_wt_add_original if version == 'original' else calculate_wt_add_refactored
        return func(R_out, inp)

def launch_wt_mul(version, R_out):
    if version in ['original', 'refactored']:
        func = calculate_wt_mul_original if version == 'original' else calculate_wt_mul_refactored
        return func(R_out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R_out_t = torch.tensor(R_out, dtype=torch.float32, device=device)

    if version == 'pytorch':
        result = calculate_wt_mul_pytorch(R_out_t)
        return tuple(tensor.cpu().numpy() for tensor in result)

    elif version == 'cuda':
        #return calculate_wt_mul_cuda(R_out)
        return None
    else:
        # Fallback to original for unsupported implementations
        print(f"⚠️  {version} implementation not available for Wt_mul, using original")
        return calculate_wt_mul_original(R_out)

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

def calculate_start_wt(arg, scaler=1, *args, **kwargs):
    """
    Initialize relevance for backprop.

    Expected shapes in current code:
      - task == "binary-classification":
            arg: [B, C]
            we create target_relevance: [B, C]
      - task == "generation":
            arg: [B, T, V]   (logits over vocab for each time step)
            we only care about the *last* position T-1
            we create target_relevance: [B, T, V] with 1.0 at (b, T-1, chosen_token)

    NEW:
      kwargs may include `target_indices` (or `target_token_ids`) for generation.
      If provided, we seed relevance at those indices instead of greedy argmax.
      This is ONLY applied for task == "generation".
      Classification path is unchanged.
    """

    task = kwargs.get("task", None)
    # Allow either name for clarity
    target_indices = kwargs.get("target_indices", None)
    if target_indices is None:
        target_indices = kwargs.get("target_token_ids", None)

    if task == "binary-classification":
        # --- ORIGINAL BEHAVIOR (unchanged) ---
        predicted_class = np.argmax(arg, axis=-1, keepdims=True)  # [B, 1]
        print(f"predicted_class: {predicted_class}")

        target_relevance = np.zeros_like(arg, dtype=np.float32)

        for b, t in enumerate(predicted_class):
            print(f"batch: {b}, predicted_class: {t.item()}")
            target_relevance[b, t] = 1.0

        print(
            f"target_relevance --- original array: {target_relevance}, "
            f"value: {np.sum(target_relevance):.4f}, shape: {target_relevance.shape}"
        )

    elif task == "generation":
        # --- MOSTLY ORIGINAL BEHAVIOR ---
        # arg is [B, T, V]
        print("======arg.shape=====", arg.shape)

        # Focus only on logits for the last generated position
        next_token_logit = arg[:, -1, :]           # [B, V]
        print(f"next_token_logit: {next_token_logit.shape}")

        if target_indices is not None:
            # We were told exactly which token(s) got chosen by sampling / beam.
            # Normalize to numpy array of shape [B]
            ti = np.array(target_indices).reshape(-1)
            if ti.shape[0] == 1 and next_token_logit.shape[0] > 1:
                # broadcast single id across batch if needed
                ti = np.repeat(ti, next_token_logit.shape[0])

            if ti.shape[0] != next_token_logit.shape[0]:
                raise ValueError(
                    f"target_indices batch mismatch: got {ti.shape[0]} "
                    f"for batch {next_token_logit.shape[0]}"
                )

            predicted_token = ti[:, None]  # [B, 1]
            print(f"predicted_token (from target_indices): {predicted_token}")

        else:
            # Greedy fallback = original behavior
            predicted_token = np.argmax(next_token_logit, axis=-1, keepdims=True)  # [B, 1]
            print(f"predicted_token (greedy argmax): {predicted_token}")

        # Create relevance tensor same shape as arg
        target_relevance = np.zeros_like(arg, dtype=np.float32)

        # Set 1.0 at the chosen token index for the LAST position only
        for b, t in enumerate(predicted_token):
            print(f"batch: {b}, token: {t}")
            target_relevance[b, -1, t] = 1.0

        print(
            f"target_relevance --- value: {np.sum(target_relevance):.4f}, "
            f"shape: {target_relevance.shape}"
        )

    else:
        # Fallback: if task wasn't passed or is unknown,
        # just mimic the binary-classification logic (argmax on last dim).
        predicted_class = np.argmax(arg, axis=-1, keepdims=True)
        target_relevance = np.zeros_like(arg, dtype=np.float32)
        for b, t in enumerate(predicted_class):
            target_relevance[b, t] = 1.0

    return target_relevance


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