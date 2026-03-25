import torch
import numpy as np
from numpy.lib.stride_tricks import as_strided

# ------------- Add `import` and `launching function` for all 4 MoEs ----------------
# JetMoE Layer
from .cuda_utils.JetMoE.original_version import calculate_wt_jetmoe_feed_forward as calculate_wt_jetmoe_feed_forward_original, calculate_wt_jetmoe_self_attention_parallel as calculate_wt_jetmoe_self_attention_parallel_original
from .cuda_utils.JetMoE.refactored_version import calculate_wt_jetmoe_feed_forward as calculate_wt_jetmoe_feed_forward_refactored, calculate_wt_jetmoe_self_attention_parallel as calculate_wt_jetmoe_self_attention_parallel_refactored
from .cuda_utils.JetMoE.pytorch_version import calculate_wt_jetmoe_feed_forward as calculate_wt_jetmoe_feed_forward_pytorch, calculate_wt_jetmoe_self_attention_parallel as calculate_wt_jetmoe_self_attention_parallel_pytorch

# OLMoE Layer
from .cuda_utils.OLMoE.original_version import calculate_wt_olmoe_feed_forward_parallel as calculate_wt_olmoe_feed_forward_original, calculate_wt_self_attention_parallel as calculate_wt_olmoe_self_attention_parallel_original
from .cuda_utils.OLMoE.refactored_version import calculate_wt_olmoe_feed_forward_parallel as calculate_wt_olmoe_feed_forward_refactored, calculate_wt_self_attention_parallel as calculate_wt_olmoe_self_attention_parallel_refactored
from .cuda_utils.OLMoE.pytorch_version import calculate_wt_olmoe_feed_forward_parallel as calculate_wt_olmoe_feed_forward_pytorch, calculate_wt_self_attention_parallel as calculate_wt_olmoe_self_attention_parallel_pytorch

# Qwen3-MoE Layer
from .cuda_utils.Qwen_MoE.original_version import calculate_wt_feed_forward as calculate_wt_qwen3_moe_feed_forward_original, calculate_wt_self_attention_parallel as calculate_wt_qwen3_moe_self_attention_parallel_original
from .cuda_utils.Qwen_MoE.refactored_version import calculate_wt_feed_forward as calculate_wt_qwen3_moe_feed_forward_refactored, calculate_wt_self_attention as calculate_wt_qwen3_moe_self_attention_refactored
from .cuda_utils.Qwen_MoE.pytorch_version import calculate_wt_feed_forward as calculate_wt_qwen3_moe_feed_forward_pytorch, calculate_wt_self_attention as calculate_wt_qwen3_moe_self_attention_pytorch 

# GPT-OSS Layer
from .cuda_utils.GPT_oss.original_version import calculate_wt_lm_head as calculate_wt_lm_head_original, calculate_wt_gpt_oss_feed_forward_parallel as calculate_wt_gpt_oss_feed_forward_parallel_original, calculate_wt_self_attention_parallel as calculate_wt_gpt_oss_self_attention_parallel_original 
from .cuda_utils.GPT_oss.refactored_version import calculate_wt_lm_head as calculate_wt_lm_head_refactored, calculate_wt_gpt_oss_feed_forward_parallel as calculate_wt_gpt_oss_feed_forward_parallel_refactored, calculate_wt_self_attention_parallel as calculate_wt_gpt_oss_self_attention_parallel_refactored 
from .cuda_utils.GPT_oss.pytorch_version import calculate_wt_lm_head as calculate_wt_lm_head_pytorch, calculate_wt_gpt_oss_feed_forward_parallel as calculate_wt_gpt_oss_feed_forward_parallel_pytorch, calculate_wt_self_attention_parallel_torch as calculate_wt_gpt_oss_self_attention_parallel_pytorch


# ═══════════════════════════════════════════════════════════════════════
#  Shared conversion helpers (used by launch functions)
# ═══════════════════════════════════════════════════════════════════════

def _to_np(x):
    """Convert anything to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x)


def _to_np_f32(x):
    """Convert to numpy float32."""
    a = _to_np(x)
    return a.astype(np.float32, copy=False)


def _to_tensor(x, device, dtype=torch.float32):
    """Convert anything to a torch tensor on device."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(_to_np(x), device=device, dtype=dtype)


def _weights_to_tensors(w, device):
    """Convert a weight dict (possibly nested) to tensors on device."""
    w_t = {}
    for k, v in w.items():
        if isinstance(v, dict):
            w_t[k] = {sk: _to_tensor(sv, device) for sk, sv in v.items()}
        elif isinstance(v, np.ndarray):
            w_t[k] = torch.tensor(v, dtype=torch.float32, device=device)
        elif torch.is_tensor(v):
            w_t[k] = v.to(device=device, dtype=torch.float32)
        else:
            w_t[k] = v
    return w_t


def _squeeze_pair(wts, inp):
    """Squeeze leading batch dim=1 from inp (and possibly wts).
    
    Works for both numpy and torch. Returns (need_unsqueeze, wts_squeezed, inp_squeezed).
    """
    is_tensor = torch.is_tensor(inp)
    
    if is_tensor:
        ndim = inp.dim()
        need_unsq = (ndim >= 1 and inp.size(0) == 1)
        inp_call = inp[0] if need_unsq else inp
        if wts.dim() == inp_call.dim() + 1 and wts.size(0) == 1:
            wts_call = wts[0]
        else:
            wts_call = wts
    else:
        ndim = inp.ndim
        need_unsq = (ndim >= 1 and inp.shape[0] == 1)
        inp_call = inp[0] if need_unsq else inp
        if wts.ndim == inp_call.ndim + 1 and wts.shape[0] == 1:
            wts_call = wts[0]
        else:
            wts_call = wts
    
    return need_unsq, wts_call, inp_call


def _unsqueeze_result(result, need_unsq):
    """Re-add leading batch dim to result (single value or tuple)."""
    if not need_unsq:
        return result
    if isinstance(result, tuple):
        main, *rest = result
        if torch.is_tensor(main):
            main = main.unsqueeze(0)
        elif isinstance(main, np.ndarray):
            main = np.expand_dims(main, 0)
        return (main, *rest)
    else:
        if torch.is_tensor(result):
            return result.unsqueeze(0)
        elif isinstance(result, np.ndarray):
            return np.expand_dims(result, 0)
        return result


def _result_to_np(out):
    """Convert launch result (single or tuple) to numpy."""
    if isinstance(out, tuple):
        main, *rest = out
        main_np = _to_np(main)
        rest_out = tuple(_to_np(x) if torch.is_tensor(x) else x for x in rest)
        return (main_np, *rest_out)
    return _to_np(out)


# ═══════════════════════════════════════════════════════════════════════
#  Launch functions
#  - 'original' path: numpy in, numpy out (unchanged)
#  - 'cuda' path: tensors in, tensors out (no back-conversion)
# ═══════════════════════════════════════════════════════════════════════

def launch_lm_head(version, wts, inp, w):
    if version == 'original':
        return calculate_wt_lm_head_original(
            _to_np(wts), _to_np(inp), w
        )
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            wts_t = _to_tensor(wts, device)
            inp_t = _to_tensor(inp, device)
            w_t = _weights_to_tensors(w, device)
            result = calculate_wt_lm_head_pytorch(wts_t, inp_t, w_t)
            if result is None:
                return calculate_wt_lm_head_original(
                    _to_np(wts), _to_np(inp), w
                )
            return result  # tensor on GPU
        except Exception as e:
            print(f"⚠️  CUDA LM head failed: {e}, falling back to original")
            return calculate_wt_lm_head_original(
                _to_np(wts), _to_np(inp), w
            )
    else:
        raise ValueError(f"Unknown version for LM head: {version}")


def launch_gpt_oss_self_attention(version, wts, inp, w, config, attn_type="full", sliding_window=None):
    if version == 'original':
        return calculate_wt_gpt_oss_self_attention_parallel_original(
            _to_np(wts), _to_np(inp), w, config,
            attn_type=attn_type, sliding_window=sliding_window
        )
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            wts_t = _to_tensor(wts, device)
            inp_t = _to_tensor(inp, device)
            w_t = _weights_to_tensors(w, device)
            result = calculate_wt_gpt_oss_self_attention_parallel_pytorch(
                wts_t, inp_t, w_t, config, attn_type, sliding_window
            )
            if result is None:
                return calculate_wt_gpt_oss_self_attention_parallel_original(
                    _to_np(wts), _to_np(inp), w, config,
                    attn_type=attn_type, sliding_window=sliding_window
                )
            return result  # tensor on GPU
        except Exception as e:
            print(f"⚠️  CUDA GPT-OSS attn failed: {e}, falling back to original")
            return calculate_wt_gpt_oss_self_attention_parallel_original(
                _to_np(wts), _to_np(inp), w, config,
                attn_type=attn_type, sliding_window=sliding_window
            )
    else:
        raise ValueError(f"Unknown version for GPT-OSS self attention: {version}")


def launch_gpt_oss_feed_forward(version, wts, inp, w, config):
    if version == 'original':
        return calculate_wt_gpt_oss_feed_forward_parallel_original(
            _to_np(wts), _to_np(inp), w, config
        )
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            wts_t = _to_tensor(wts, device)
            inp_t = _to_tensor(inp, device)
            w_t = _weights_to_tensors(w, device)
            result = calculate_wt_gpt_oss_feed_forward_parallel_pytorch(
                wts_t, inp_t, w_t, config
            )
            if result is None:
                return calculate_wt_gpt_oss_feed_forward_parallel_original(
                    _to_np(wts), _to_np(inp), w, config
                )
            return result  # tensor (or tuple of tensor + meta)
        except Exception as e:
            print(f"⚠️  CUDA GPT-OSS FF failed: {e}, falling back to original")
            return calculate_wt_gpt_oss_feed_forward_parallel_original(
                _to_np(wts), _to_np(inp), w, config
            )
    else:
        raise ValueError(f"Unknown version for GPT-OSS feed forward: {version}")


def launch_qwen3_moe_self_attention(version, wts, inp, w, config):
    if version == 'original':
        return calculate_wt_qwen3_moe_self_attention_parallel_original(
            _to_np(wts), _to_np(inp), w, config
        )
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            wts_t = _to_tensor(wts, device)
            inp_t = _to_tensor(inp, device)
            w_t = _weights_to_tensors(w, device)
            result = calculate_wt_qwen3_moe_self_attention_pytorch(
                wts_t, inp_t, w_t, config
            )
            if result is None:
                return calculate_wt_qwen3_moe_self_attention_parallel_original(
                    _to_np(wts), _to_np(inp), w, config
                )
            return result
        except Exception as e:
            print(f"⚠️  CUDA Qwen3 attn failed: {e}, falling back to original")
            return calculate_wt_qwen3_moe_self_attention_parallel_original(
                _to_np(wts), _to_np(inp), w, config
            )
    else:
        raise ValueError(f"Unknown version for Qwen3-MoE self attention: {version}")


def launch_qwen3_moe_feed_forward(version, wts, inp, w, config):
    if version == 'original':
        return calculate_wt_qwen3_moe_feed_forward_original(
            _to_np(wts), _to_np(inp), w, config
        )
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            wts_t = _to_tensor(wts, device)
            inp_t = _to_tensor(inp, device)
            w_t = _weights_to_tensors(w, device)
            result = calculate_wt_qwen3_moe_feed_forward_pytorch(
                wts_t, inp_t, w_t, config
            )
            if result is None:
                return calculate_wt_qwen3_moe_feed_forward_original(
                    _to_np(wts), _to_np(inp), w, config
                )
            return result
        except Exception as e:
            print(f"⚠️  CUDA Qwen3 FF failed: {e}, falling back to original")
            return calculate_wt_qwen3_moe_feed_forward_original(
                _to_np(wts), _to_np(inp), w, config
            )
    else:
        raise ValueError(f"Unknown version for Qwen3-MoE feed forward: {version}")


def _launch_squeeze_pattern(version, wts, inp, w, extra_arg, 
                             original_fn, pytorch_fn, name):
    """Shared pattern for OLMoE/JetMoE launch functions that need squeeze/unsqueeze."""
    if version == 'original':
        wts_np = _to_np(wts)
        inp_np = _to_np_f32(inp)
        need_unsq, wts_call, inp_call = _squeeze_pair(wts_np, inp_np)
        out = original_fn(wts_call, inp_call, w, extra_arg)
        out = _unsqueeze_result(out, need_unsq)
        return _result_to_np(out)

    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            wts_t = _to_tensor(wts, device)
            inp_t = _to_tensor(inp, device)
            w_t = _weights_to_tensors(w, device)

            need_unsq, wts_call, inp_call = _squeeze_pair(wts_t, inp_t)
            out = pytorch_fn(wts_call, inp_call, w_t, extra_arg)

            if out is None:
                # Fallback to original
                wts_np = _to_np(wts)
                inp_np = _to_np_f32(inp)
                need_unsq_np, wts_call_np, inp_call_np = _squeeze_pair(wts_np, inp_np)
                out = original_fn(wts_call_np, inp_call_np, w, extra_arg)
                out = _unsqueeze_result(out, need_unsq_np)
                return _result_to_np(out)

            out = _unsqueeze_result(out, need_unsq)
            return out  # tensor(s) on GPU

        except Exception as e:
            import traceback
            print(f"⚠️  CUDA {name} failed: {e}, falling back to original")
            print(f"   {traceback.format_exc()}")
            wts_np = _to_np(wts)
            inp_np = _to_np_f32(inp)
            need_unsq, wts_call, inp_call = _squeeze_pair(wts_np, inp_np)
            out = original_fn(wts_call, inp_call, w, extra_arg)
            out = _unsqueeze_result(out, need_unsq)
            return _result_to_np(out)
    else:
        raise ValueError(f"Unknown version for {name}: {version}")


def launch_olmoe_feed_forward(version, wts, inp, w, model):
    return _launch_squeeze_pattern(
        version, wts, inp, w, model,
        calculate_wt_olmoe_feed_forward_original,
        calculate_wt_olmoe_feed_forward_pytorch,
        "OLMoE FF"
    )


def launch_olmoe_self_attention(version, wts, inp, w, model):
    return _launch_squeeze_pattern(
        version, wts, inp, w, model,
        calculate_wt_olmoe_self_attention_parallel_original,
        calculate_wt_olmoe_self_attention_parallel_pytorch,
        "OLMoE SA"
    )


def launch_jetmoe_self_attention(version, wts, inp, w, model):
    return _launch_squeeze_pattern(
        version, wts, inp, w, model,
        calculate_wt_jetmoe_self_attention_parallel_original,
        calculate_wt_jetmoe_self_attention_parallel_pytorch,
        "JetMoE SA"
    )


def launch_jetmoe_feed_forward(version, wts, inp, w, model):
    return _launch_squeeze_pattern(
        version, wts, inp, w, model,
        calculate_wt_jetmoe_feed_forward_original,
        calculate_wt_jetmoe_feed_forward_pytorch,
        "JetMoE FF"
    )


# ═══════════════════════════════════════════════════════════════════════
#  Activation functions (numpy-based, used by original path)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
#  Start weight + Residual (torch-aware)
# ═══════════════════════════════════════════════════════════════════════

def calculate_start_wt(arg, scaler=1, *args, **kwargs):
    """Calculate initial relevance. Accepts both numpy arrays and torch tensors."""
    task = kwargs.get('task', None)

    # Convert to numpy for computation (this happens once, not per-layer)
    if torch.is_tensor(arg):
        arg_np = arg.detach().to(torch.float32).cpu().numpy()
    else:
        arg_np = arg

    if task == "binary-classification":
        predicted_class = np.argmax(arg_np, axis=-1, keepdims=True)
        print(f"predicted_class: {predicted_class}")
        target_relevance = np.zeros_like(arg_np, dtype=np.float32)
        for b, t in enumerate(predicted_class):
            print(f"batch: {b}, predicted_class: {t.item()}")
            target_relevance[b, t] = 1.0
        print(f"target_relevance --- original array: {target_relevance}, value: {np.sum(target_relevance):.4f}, shape: {target_relevance.shape}")

    elif task == "generation":
        print("======arg.shape=====", arg_np.shape)
        next_token_logit = arg_np[:, -1, :]
        print(f"next_token_logit: {next_token_logit.shape}")
        predicted_token = np.argmax(next_token_logit, axis=-1, keepdims=True)
        print(f"predicted_token: {predicted_token}")
        target_relevance = np.zeros_like(arg_np, dtype=np.float32)
        for b, t in enumerate(predicted_token):
            print(f"batch: {b}, token: {t}")
            target_relevance[b, -1, t] = 1.0
        print(f"target_relevance --- value: {np.sum(target_relevance):.4f}, shape: {target_relevance.shape}")

    return target_relevance


def calculate_wt_residual(wts, inp=None):
    """Compute residual relevance distribution. Accepts numpy or torch tensors.
    
    On CUDA: works with torch tensors, returns torch tensors.
    On CPU: works with numpy, returns numpy (original behavior).
    """
    # Check if inputs are tensors
    if torch.is_tensor(wts):
        return _calculate_wt_residual_torch(wts, inp)
    
    # Original numpy path
    wt_mat = []
    inp_list = []
    expanded_wts = as_strided(
        wts,
        shape=(np.prod(wts.shape),),
        strides=(wts.strides[-1],),
        writeable=False,
    )

    for x in inp:
        if torch.is_tensor(x):
            x = x.detach().to(torch.float32).cpu().numpy()
        expanded_input = as_strided(
            x,
            shape=(np.prod(x.shape),),
            strides=(x.strides[-1],),
            writeable=False,
        )
        inp_list.append(expanded_input)
        wt_mat.append(np.zeros_like(expanded_input))
    wt_mat = np.array(wt_mat)
    inp_list = np.array(inp_list)
    for i in range(wt_mat.shape[1]):
        wt_ind1 = wt_mat[:, i]
        wt = expanded_wts[i]
        l1_ind1 = inp_list[:, i]
        p_ind = l1_ind1 > 0
        n_ind = l1_ind1 < 0
        p_sum = np.sum(l1_ind1[p_ind])
        n_sum = np.sum(l1_ind1[n_ind]) * -1
        p_agg_wt = 0
        n_agg_wt = 0
        if p_sum + n_sum > 0:
            p_agg_wt = p_sum / (p_sum + n_sum)
            n_agg_wt = n_sum / (p_sum + n_sum)
        if p_sum == 0:
            p_sum = 1
        if n_sum == 0:
            n_sum = 1
        wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
        wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0
        wt_mat[:, i] = wt_ind1
    wt_mat = [i.reshape(wts.shape) for i in list(wt_mat)]
    return wt_mat


def _calculate_wt_residual_torch(wts, inp):
    """Torch-native residual relevance distribution (stays on GPU)."""
    device = wts.device
    flat_wts = wts.reshape(-1)
    num_elements = flat_wts.numel()
    
    # Flatten all inputs
    flat_inputs = []
    for x in inp:
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device=device, dtype=torch.float32)
        flat_inputs.append(x.reshape(-1))
    
    # Stack: [num_inputs, num_elements]
    inp_stack = torch.stack(flat_inputs, dim=0)
    wt_mat = torch.zeros_like(inp_stack)
    
    for i in range(num_elements):
        wt = flat_wts[i]
        col = inp_stack[:, i]
        
        p_mask = col > 0
        n_mask = col < 0
        
        p_sum = col[p_mask].sum()
        n_sum = col[n_mask].sum().abs()
        
        total = p_sum + n_sum
        p_agg = p_sum / total if total > 0 else 0.0
        n_agg = n_sum / total if total > 0 else 0.0
        
        p_denom = p_sum if p_sum > 0 else 1.0
        n_denom = n_sum if n_sum > 0 else 1.0
        
        result_col = torch.zeros_like(col)
        if p_mask.any():
            result_col[p_mask] = (col[p_mask] / p_denom) * wt * p_agg
        if n_mask.any():
            result_col[n_mask] = (col[n_mask] / n_denom) * wt * n_agg * -1.0
        
        wt_mat[:, i] = result_col
    
    return [row.reshape(wts.shape) for row in wt_mat]


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

        for batch_idx in range(R.shape[0]):
            rel_slice = equal_relevance[idx][batch_idx]
            reduced_rel = rel_slice

            for axis, orig_dim in enumerate(orig_shape[1:]):
                if orig_dim == 1 and reduced_rel.shape[axis] > 1:
                    reduced_rel = reduced_rel.sum(axis=axis, keepdims=True)

            per_input_rel.append(reduced_rel)

        result.append(np.stack(per_input_rel, axis=0))

    return result        


def weight_scaler(arg, scaler=100.0):
    s1 = np.sum(arg)
    scale_factor = s1 / scaler
    return arg / scale_factor

def weight_normalize(arg, max_val=1.0):
    arg_max = np.max(arg)
    arg_min = np.abs(np.min(arg))
    if arg_max > arg_min:
        return (arg / arg_max) * max_val
    elif arg_min > 0:
        return (arg / arg_min) * max_val
    else:
        return arg    