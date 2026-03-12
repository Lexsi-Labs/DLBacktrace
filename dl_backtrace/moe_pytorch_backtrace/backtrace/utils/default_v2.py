import torch
import numpy as np
from numpy.lib.stride_tricks import as_strided

# ------------- Add `import` and `launching function` for all 4 MoEs ----------------
# MoE Utils Layers
from .cuda_utils.MoE_utils.cuda_version.nonneg_conserve import nonneg_conserve_ops as dlb_style_signed_conserve_ops
from .cuda_utils.MoE_utils.cuda_version.relevance_gated_proj import relevance_gated_proj_ops as calculate_relevance_gated_proj_ops
from .cuda_utils.MoE_utils.cuda_version.relevance_proj import relevance_proj_ops as calculate_relevance_proj_ops
from .cuda_utils.MoE_utils.cuda_version.relevance_single import relevance_single_ops as calculate_relevance_ops
from .cuda_utils.MoE_utils.cuda_version.wt_router_logits import wt_router_logits_ops as calculate_wt_router_logits_ops

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

def _prepare_tensors(device, *arrays):
    return [torch.tensor(arr, dtype=torch.float32, device=device) for arr in arrays]

# ---------------------------------------------------------------------------
# Shared conversion helpers — inputs are now torch tensors from the pipeline
# ---------------------------------------------------------------------------

def _val_to_np(x):
    """Convert any value to a float32 numpy array (for 'original' CPU backends)."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if torch.is_tensor(x):
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _val_to_tensor(x, device):
    """Convert any value to a float32 torch.Tensor on `device` (for 'cuda' backends)."""
    if torch.is_tensor(x):
        return x.to(dtype=torch.float32, device=device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32, copy=False)).to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def _dict_to_np(w):
    """Recursively convert weight dict values to numpy for original backends."""
    out = {}
    for k, v in w.items():
        if isinstance(v, dict):
            out[k] = _dict_to_np(v)
        elif torch.is_tensor(v):
            out[k] = v.detach().to(torch.float32).cpu().numpy()
        elif isinstance(v, np.ndarray):
            out[k] = v.astype(np.float32, copy=False)
        else:
            out[k] = v
    return out


def _dict_to_tensor(w, device):
    """Recursively convert weight dict values to CUDA tensors — skip if already on device."""
    out = {}
    for k, v in w.items():
        if isinstance(v, dict):
            out[k] = _dict_to_tensor(v, device)
        elif torch.is_tensor(v):
            out[k] = v.to(dtype=torch.float32, device=device)
        elif isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v.astype(np.float32, copy=False)).to(device)
        else:
            out[k] = v
    return out


def launch_lm_head(version, wts, inp, w):
    if version == 'original':
        return calculate_wt_lm_head_original(
            _val_to_np(wts), _val_to_np(inp), _dict_to_np(w)
        )
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            result = calculate_wt_lm_head_pytorch(
                _val_to_tensor(wts, device),
                _val_to_tensor(inp, device),
                _dict_to_tensor(w, device),
            )
            if result is None:
                print("⚠️  CUDA LM head returned None, falling back to original")
                return calculate_wt_lm_head_original(
                    _val_to_np(wts), _val_to_np(inp), _dict_to_np(w)
                )
            return result  # keep as tensor — caller handles type
        except Exception as e:
            print(f"⚠️  CUDA LM head failed: {e}, falling back to original")
            return calculate_wt_lm_head_original(
                _val_to_np(wts), _val_to_np(inp), _dict_to_np(w)
            )
    else:
        raise ValueError(f"Unknown version for LM head: {version}")

def launch_gpt_oss_self_attention(version, wts, inp, w, config, attn_type="full", sliding_window=None):
    if version == 'original':
        return calculate_wt_gpt_oss_self_attention_parallel_original(
            _val_to_np(wts), _val_to_np(inp), _dict_to_np(w), config,
            attn_type=attn_type, sliding_window=sliding_window,
        )
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            result = calculate_wt_gpt_oss_self_attention_parallel_pytorch(
                _val_to_tensor(wts, device), _val_to_tensor(inp, device),
                _dict_to_tensor(w, device), config, attn_type, sliding_window,
            )
            if result is None:
                print("⚠️  CUDA GPT-OSS self attention returned None, falling back to original")
                return calculate_wt_gpt_oss_self_attention_parallel_original(
                    _val_to_np(wts), _val_to_np(inp), _dict_to_np(w), config,
                    attn_type=attn_type, sliding_window=sliding_window,
                )
            return result
        except Exception as e:
            print(f"⚠️  CUDA GPT-OSS self attention failed: {e}, falling back to original")
            return calculate_wt_gpt_oss_self_attention_parallel_original(
                _val_to_np(wts), _val_to_np(inp), _dict_to_np(w), config,
                attn_type=attn_type, sliding_window=sliding_window,
            )
    else:
        raise ValueError(f"Unknown version for GPT-OSS self attention: {version}")

def launch_gpt_oss_feed_forward(version, wts, inp, w, config):
    if version == 'original':
        return calculate_wt_gpt_oss_feed_forward_parallel_original(
            _val_to_np(wts), _val_to_np(inp), _dict_to_np(w), config,
        )
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            result = calculate_wt_gpt_oss_feed_forward_parallel_pytorch(
                _val_to_tensor(wts, device), _val_to_tensor(inp, device),
                _dict_to_tensor(w, device), config,
            )
            if result is None:
                print("⚠️  CUDA GPT-OSS feed forward returned None, falling back to original")
                return calculate_wt_gpt_oss_feed_forward_parallel_original(
                    _val_to_np(wts), _val_to_np(inp), _dict_to_np(w), config,
                )
            return result
        except Exception as e:
            print(f"⚠️  CUDA GPT-OSS feed forward failed: {e}, falling back to original")
            return calculate_wt_gpt_oss_feed_forward_parallel_original(
                _val_to_np(wts), _val_to_np(inp), _dict_to_np(w), config,
            )
    else:
        raise ValueError(f"Unknown version for GPT-OSS feed forward: {version}")

def launch_qwen3_moe_self_attention(version, wts, inp, w, config):
    if version == 'original':
        return calculate_wt_qwen3_moe_self_attention_parallel_original(
            _val_to_np(wts), _val_to_np(inp), _dict_to_np(w), config,
        )
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            result = calculate_wt_qwen3_moe_self_attention_pytorch(
                _val_to_tensor(wts, device), _val_to_tensor(inp, device),
                _dict_to_tensor(w, device), config,
            )
            if result is None:
                print("⚠️  CUDA Qwen3-MoE self attention returned None, falling back to original")
                return calculate_wt_qwen3_moe_self_attention_parallel_original(
                    _val_to_np(wts), _val_to_np(inp), _dict_to_np(w), config,
                )
            return result
        except Exception as e:
            print(f"⚠️  CUDA Qwen3-MoE self attention failed: {e}, falling back to original")
            return calculate_wt_qwen3_moe_self_attention_parallel_original(
                _val_to_np(wts), _val_to_np(inp), _dict_to_np(w), config,
            )
    else:
        raise ValueError(f"Unknown version for Qwen3-MoE self attention: {version}")

def launch_qwen3_moe_feed_forward(version, wts, inp, w, config):
    if version == 'original':
        return calculate_wt_qwen3_moe_feed_forward_original(
            _val_to_np(wts), _val_to_np(inp), _dict_to_np(w), config,
        )
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            result = calculate_wt_qwen3_moe_feed_forward_pytorch(
                _val_to_tensor(wts, device), _val_to_tensor(inp, device),
                _dict_to_tensor(w, device), config,
            )
            if result is None:
                print("⚠️  CUDA Qwen3-MoE feed forward returned None, falling back to original")
                return calculate_wt_qwen3_moe_feed_forward_original(
                    _val_to_np(wts), _val_to_np(inp), _dict_to_np(w), config,
                )
            return result
        except Exception as e:
            print(f"⚠️  CUDA Qwen3-MoE feed forward failed: {e}, falling back to original")
            return calculate_wt_qwen3_moe_feed_forward_original(
                _val_to_np(wts), _val_to_np(inp), _dict_to_np(w), config,
            )
    else:
        raise ValueError(f"Unknown version for Qwen3-MoE feed forward: {version}")

def launch_olmoe_feed_forward(version, wts, inp, w, model):
    """
    Squeeze leading batch dim (=1) for BOTH `inp` and `wts` before calling impl,
    then unsqueeze the main result back to (1, ...). Works for original/cuda/fallback.
    Also supports backends that return either `arr` or `(arr, meta)`.
    """

    def _maybe_squeeze_pair(wts_v, inp_v):
        is_tensor = torch.is_tensor(inp_v)
        ndim = inp_v.dim() if is_tensor else inp_v.ndim
        s0 = inp_v.size(0) if is_tensor else inp_v.shape[0]
        need_unsq = (ndim >= 1 and s0 == 1)
        inp_call = inp_v[0] if need_unsq else inp_v
        ic_ndim = inp_call.dim() if torch.is_tensor(inp_call) else inp_call.ndim
        wt_ndim = wts_v.dim() if torch.is_tensor(wts_v) else wts_v.ndim
        wt_s0 = wts_v.size(0) if torch.is_tensor(wts_v) else wts_v.shape[0]
        if wt_ndim == ic_ndim + 1 and wt_s0 == 1:
            wts_call = wts_v[0]
        else:
            wts_call = wts_v
        return need_unsq, wts_call, inp_call

    def _unsqueeze_first(result, need_unsq):
        if not need_unsq:
            return result
        if isinstance(result, tuple):
            main, *rest = result
            main = main.unsqueeze(0) if torch.is_tensor(main) else np.expand_dims(main, 0)
            return (main, *rest)
        return result.unsqueeze(0) if torch.is_tensor(result) else np.expand_dims(result, 0)

    if version == 'original':
        wts_np, inp_np = _val_to_np(wts), _val_to_np(inp)
        w_np = _dict_to_np(w)
        need_unsq, wts_call, inp_call = _maybe_squeeze_pair(wts_np, inp_np)
        out = calculate_wt_olmoe_feed_forward_original(wts_call, inp_call, w_np, model)
        return _unsqueeze_first(out, need_unsq)

    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            wts_t = _val_to_tensor(wts, device)
            inp_t = _val_to_tensor(inp, device)
            w_torch = _dict_to_tensor(w, device)
            need_unsq, wts_call, inp_call = _maybe_squeeze_pair(wts_t, inp_t)
            out = calculate_wt_olmoe_feed_forward_pytorch(wts_call, inp_call, w_torch, model)
            if out is None:
                print("[DEBUG olmoe][cuda] impl returned None -> fallback to original")
                wts_np, inp_np = _val_to_np(wts_call), _val_to_np(inp_call)
                out = calculate_wt_olmoe_feed_forward_original(wts_np, inp_np, _dict_to_np(w), model)
            return _unsqueeze_first(out, need_unsq)
        except Exception as e:
            print(f"[DEBUG olmoe][cuda] exception: {e} -> fallback to original")
            wts_np, inp_np = _val_to_np(wts), _val_to_np(inp)
            w_np = _dict_to_np(w)
            need_unsq, wts_call, inp_call = _maybe_squeeze_pair(wts_np, inp_np)
            out = calculate_wt_olmoe_feed_forward_original(wts_call, inp_call, w_np, model)
            return _unsqueeze_first(out, need_unsq)

    else:
        raise ValueError(f"Unknown version for OLMoE feed forward: {version}")





def launch_olmoe_self_attention(version, wts, inp, w, model):
    """Squeeze leading batch dim (=1), dispatch, unsqueeze back."""

    def _maybe_squeeze_pair(wts_v, inp_v):
        is_tensor = torch.is_tensor(inp_v)
        ndim = inp_v.dim() if is_tensor else inp_v.ndim
        s0 = inp_v.size(0) if is_tensor else inp_v.shape[0]
        need_unsq = (ndim >= 1 and s0 == 1)
        inp_call = inp_v[0] if need_unsq else inp_v
        ic_ndim = inp_call.dim() if torch.is_tensor(inp_call) else inp_call.ndim
        wt_ndim = wts_v.dim() if torch.is_tensor(wts_v) else wts_v.ndim
        wt_s0 = wts_v.size(0) if torch.is_tensor(wts_v) else wts_v.shape[0]
        if wt_ndim == ic_ndim + 1 and wt_s0 == 1:
            wts_call = wts_v[0]
        else:
            wts_call = wts_v
        return need_unsq, wts_call, inp_call

    def _unsqueeze_first(result, need_unsq):
        if not need_unsq:
            return result
        if isinstance(result, tuple):
            main, *rest = result
            main = main.unsqueeze(0) if torch.is_tensor(main) else np.expand_dims(main, 0)
            return (main, *rest)
        return result.unsqueeze(0) if torch.is_tensor(result) else np.expand_dims(result, 0)

    if version == 'original':
        wts_np, inp_np = _val_to_np(wts), _val_to_np(inp)
        w_np = _dict_to_np(w)
        need_unsq, wts_call, inp_call = _maybe_squeeze_pair(wts_np, inp_np)
        out = calculate_wt_olmoe_self_attention_parallel_original(wts_call, inp_call, w_np, model)
        return _unsqueeze_first(out, need_unsq)

    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            wts_t = _val_to_tensor(wts, device)
            inp_t = _val_to_tensor(inp, device)
            w_torch = _dict_to_tensor(w, device)
            need_unsq, wts_call, inp_call = _maybe_squeeze_pair(wts_t, inp_t)
            out = calculate_wt_olmoe_self_attention_parallel_pytorch(wts_call, inp_call, w_torch, model)
            if out is None:
                print("[DEBUG olmoe_sa][cuda] impl returned None -> fallback to original")
                out = calculate_wt_olmoe_self_attention_parallel_original(
                    _val_to_np(wts_call), _val_to_np(inp_call), _dict_to_np(w), model)
            return _unsqueeze_first(out, need_unsq)
        except Exception as e:
            print(f"[DEBUG olmoe_sa][cuda] exception: {e} -> fallback to original")
            wts_np, inp_np = _val_to_np(wts), _val_to_np(inp)
            w_np = _dict_to_np(w)
            need_unsq, wts_call, inp_call = _maybe_squeeze_pair(wts_np, inp_np)
            out = calculate_wt_olmoe_self_attention_parallel_original(wts_call, inp_call, w_np, model)
            return _unsqueeze_first(out, need_unsq)

    else:
        raise ValueError(f"Unknown version for OLMoE self attention: {version}") 


def launch_jetmoe_self_attention(version, wts, inp, w, model):
    """Squeeze leading batch dim (=1), dispatch, unsqueeze back."""

    def _maybe_squeeze_pair(wts_v, inp_v):
        is_tensor = torch.is_tensor(inp_v)
        ndim = inp_v.dim() if is_tensor else inp_v.ndim
        s0 = inp_v.size(0) if is_tensor else inp_v.shape[0]
        need_unsq = (ndim >= 1 and s0 == 1)
        inp_call = inp_v[0] if need_unsq else inp_v
        ic_ndim = inp_call.dim() if torch.is_tensor(inp_call) else inp_call.ndim
        wt_ndim = wts_v.dim() if torch.is_tensor(wts_v) else wts_v.ndim
        wt_s0 = wts_v.size(0) if torch.is_tensor(wts_v) else wts_v.shape[0]
        if wt_ndim == ic_ndim + 1 and wt_s0 == 1:
            wts_call = wts_v[0]
        else:
            wts_call = wts_v
        return need_unsq, wts_call, inp_call

    def _unsqueeze_first(result, need_unsq):
        if not need_unsq:
            return result
        if isinstance(result, tuple):
            main, *rest = result
            main = main.unsqueeze(0) if torch.is_tensor(main) else np.expand_dims(main, 0)
            return (main, *rest)
        return result.unsqueeze(0) if torch.is_tensor(result) else np.expand_dims(result, 0)

    if version == 'original':
        wts_np, inp_np = _val_to_np(wts), _val_to_np(inp)
        w_np = _dict_to_np(w)
        need_unsq, wts_call, inp_call = _maybe_squeeze_pair(wts_np, inp_np)
        out = calculate_wt_jetmoe_self_attention_parallel_original(wts_call, inp_call, w_np, model)
        return _unsqueeze_first(out, need_unsq)

    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            wts_t = _val_to_tensor(wts, device)
            inp_t = _val_to_tensor(inp, device)
            w_torch = _dict_to_tensor(w, device)
            need_unsq, wts_call, inp_call = _maybe_squeeze_pair(wts_t, inp_t)
            out = calculate_wt_jetmoe_self_attention_parallel_pytorch(wts_call, inp_call, w_torch, model)
            if out is None:
                print("[DEBUG jetmoe_sa][cuda] impl returned None -> fallback to original")
                out = calculate_wt_jetmoe_self_attention_parallel_original(
                    _val_to_np(wts_call), _val_to_np(inp_call), _dict_to_np(w), model)
            return _unsqueeze_first(out, need_unsq)
        except Exception as e:
            print(f"[DEBUG jetmoe_sa][cuda] exception: {e} -> fallback to original")
            wts_np, inp_np = _val_to_np(wts), _val_to_np(inp)
            w_np = _dict_to_np(w)
            need_unsq, wts_call, inp_call = _maybe_squeeze_pair(wts_np, inp_np)
            out = calculate_wt_jetmoe_self_attention_parallel_original(wts_call, inp_call, w_np, model)
            return _unsqueeze_first(out, need_unsq)

    else:
        raise ValueError(f"Unknown version for JetMoE self attention: {version}")


def launch_jetmoe_feed_forward(version, wts, inp, w, model):
    """Squeeze leading batch dim (=1), dispatch, unsqueeze back."""

    def _maybe_squeeze_pair(wts_v, inp_v):
        is_tensor = torch.is_tensor(inp_v)
        ndim = inp_v.dim() if is_tensor else inp_v.ndim
        s0 = inp_v.size(0) if is_tensor else inp_v.shape[0]
        need_unsq = (ndim >= 1 and s0 == 1)
        inp_call = inp_v[0] if need_unsq else inp_v
        ic_ndim = inp_call.dim() if torch.is_tensor(inp_call) else inp_call.ndim
        wt_ndim = wts_v.dim() if torch.is_tensor(wts_v) else wts_v.ndim
        wt_s0 = wts_v.size(0) if torch.is_tensor(wts_v) else wts_v.shape[0]
        if wt_ndim == ic_ndim + 1 and wt_s0 == 1:
            wts_call = wts_v[0]
        else:
            wts_call = wts_v
        return need_unsq, wts_call, inp_call

    def _unsqueeze_first(result, need_unsq):
        if not need_unsq:
            return result
        if isinstance(result, tuple):
            main, *rest = result
            main = main.unsqueeze(0) if torch.is_tensor(main) else np.expand_dims(main, 0)
            return (main, *rest)
        return result.unsqueeze(0) if torch.is_tensor(result) else np.expand_dims(result, 0)

    if version == 'original':
        wts_np, inp_np = _val_to_np(wts), _val_to_np(inp)
        w_np = _dict_to_np(w)
        need_unsq, wts_call, inp_call = _maybe_squeeze_pair(wts_np, inp_np)
        out = calculate_wt_jetmoe_feed_forward_original(wts_call, inp_call, w_np, model)
        return _unsqueeze_first(out, need_unsq)

    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            wts_t = _val_to_tensor(wts, device)
            inp_t = _val_to_tensor(inp, device)
            w_torch = _dict_to_tensor(w, device)
            need_unsq, wts_call, inp_call = _maybe_squeeze_pair(wts_t, inp_t)
            out = calculate_wt_jetmoe_feed_forward_pytorch(wts_call, inp_call, w_torch, model)
            if out is None:
                print("[DEBUG jetmoe][cuda] impl returned None -> fallback to original")
                out = calculate_wt_jetmoe_feed_forward_original(
                    _val_to_np(wts_call), _val_to_np(inp_call), _dict_to_np(w), model)
            return _unsqueeze_first(out, need_unsq)
        except Exception as e:
            print(f"[DEBUG jetmoe][cuda] exception: {e} -> fallback to original")
            wts_np, inp_np = _val_to_np(wts), _val_to_np(inp)
            w_np = _dict_to_np(w)
            need_unsq, wts_call, inp_call = _maybe_squeeze_pair(wts_np, inp_np)
            out = calculate_wt_jetmoe_feed_forward_original(wts_call, inp_call, w_np, model)
            return _unsqueeze_first(out, need_unsq)

    else:
        raise ValueError(f"Unknown version for JetMoE feed forward: {version}")

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
    task = kwargs.get('task', None)
    device = kwargs.get('device', 'cpu')

    # Convert input to float32 torch tensor
    if torch.is_tensor(arg):
        arg_t = arg.to(dtype=torch.float32)
        device = arg_t.device
    elif isinstance(arg, np.ndarray):
        arg_t = torch.from_numpy(arg.astype(np.float32, copy=False)).to(device)
    else:
        arg_t = torch.tensor(arg, dtype=torch.float32, device=device)

    if task == "binary-classification":
        predicted_class = torch.argmax(arg_t, dim=-1, keepdim=True)  # [B, 1]
        print(f"predicted_class: {predicted_class}")

        target_relevance = torch.zeros_like(arg_t)
        for b in range(predicted_class.shape[0]):
            t = predicted_class[b].item()
            print(f"batch: {b}, predicted_class: {t}")
            target_relevance[b, t] = 1.0

        print(f"target_relevance --- value: {target_relevance.sum().item():.4f}, shape: {tuple(target_relevance.shape)}")

    elif task == "generation":
        print("======arg.shape=====", tuple(arg_t.shape))
        next_token_logit = arg_t[:, -1, :]
        print(f"next_token_logit: {tuple(next_token_logit.shape)}")
        predicted_token = torch.argmax(next_token_logit, dim=-1, keepdim=True)  # [B, 1]
        print(f"predicted_token: {predicted_token}")

        target_relevance = torch.zeros_like(arg_t)
        for b in range(predicted_token.shape[0]):
            t = predicted_token[b].item()
            print(f"batch: {b}, token: {t}")
            target_relevance[b, -1, t] = 1.0

        print(f"target_relevance --- value: {target_relevance.sum().item():.4f}, shape: {tuple(target_relevance.shape)}")

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

def calculate_wt_residual(wts, inp=None):
    """Proportional relevance redistribution for residual connections.

    Accepts both torch tensors and numpy arrays. Returns a list of
    tensors/arrays matching the type of `wts`.
    """
    use_torch = torch.is_tensor(wts)
    if use_torch:
        device = wts.device
        expanded_wts = wts.reshape(-1)
        n_inp = len(inp)
        n_elem = expanded_wts.numel()
        # Stack flattened inputs → [n_inp, n_elem]
        inp_flat = torch.stack([x.reshape(-1) for x in inp], dim=0)
        wt_mat = torch.zeros_like(inp_flat)
        for i in range(n_elem):
            wt = expanded_wts[i]
            col = inp_flat[:, i]
            p_mask = col > 0
            n_mask = col < 0
            p_sum = col[p_mask].sum()
            n_sum = col[n_mask].sum() * -1
            total = p_sum + n_sum
            p_agg = p_sum / total if total > 0 else 0.0
            n_agg = n_sum / total if total > 0 else 0.0
            if p_sum == 0:
                p_sum = 1.0
            if n_sum == 0:
                n_sum = 1.0
            out_col = torch.zeros_like(col)
            out_col[p_mask] = (col[p_mask] / p_sum) * wt * p_agg
            out_col[n_mask] = (col[n_mask] / n_sum) * wt * n_agg * -1.0
            wt_mat[:, i] = out_col
        return [wt_mat[j].reshape(wts.shape) for j in range(n_inp)]
    else:
        # numpy fallback (unchanged logic, uses as_strided)
        wt_mat = []
        inp_list = []
        expanded_wts = as_strided(
            wts,
            shape=(np.prod(wts.shape),),
            strides=(wts.strides[-1],),
            writeable=False,
        )
        for x in inp:
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



def weight_normalize(arg, max_val=1.0):
    arg_max = np.max(arg)
    arg_min = np.abs(np.min(arg))
    if arg_max > arg_min:
        return (arg / arg_max) * max_val
    elif arg_min > 0:
        return (arg / arg_min) * max_val
    else:
        return arg    