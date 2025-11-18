import torch
import numpy as np


# ------------- Add `import` and `launching function` for all 4 MoEs ----------------
# MoE Utils Layers
from .cuda_utils.MoE_utils.nonneg_conserve import dlb_style_signed_conserve_cuda as dlb_style_signed_conserve_cuda
from .cuda_utils.MoE_utils.relevance_gated_proj import calculate_relevance_gated_proj_cuda as calculate_relevance_gated_proj_cuda
from .cuda_utils.MoE_utils.relevance_proj import calculate_relevance_proj_cuda as calculate_relevance_proj_cuda
from .cuda_utils.MoE_utils.relevance_single import calculate_relevance_cuda as calculate_relevance_cuda
from .cuda_utils.MoE_utils.wt_router_logits import calculate_wt_router_logits_cuda as calculate_wt_router_logits_cuda

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

def launch_lm_head(version, wts, inp, w):
    if version == 'original':
        # CPU mode: use original implementation
        print("[DEBUG default_v2] launch_lm_head -> calling original with:")
        return calculate_wt_lm_head_original(wts, inp, w)
    elif version == 'cuda':
        # CUDA mode: use PyTorch implementation 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Convert numpy arrays to tensors
            wts_t = torch.tensor(wts, dtype=torch.float32, device=device) if isinstance(wts, np.ndarray) else wts
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device) if isinstance(inp, np.ndarray) else inp
            
            # Convert weight dictionary to tensors on correct device
            w_torch = {}
            for k, v in w.items():
                if isinstance(v, dict):
                    w_torch[k] = {sub_k: torch.tensor(sub_v, dtype=torch.float32, device=device) if isinstance(sub_v, np.ndarray) else sub_v.to(device) if torch.is_tensor(sub_v) else sub_v
                                 for sub_k, sub_v in v.items()}
                elif isinstance(v, np.ndarray):
                    w_torch[k] = torch.tensor(v, dtype=torch.float32, device=device)
                elif torch.is_tensor(v):
                    w_torch[k] = v.to(device)
                else:
                    w_torch[k] = v
            
            result = calculate_wt_lm_head_pytorch(wts_t, inp_t, w_torch)
            if result is None:
                print(f"⚠️  CUDA LM head implementation returned None, falling back to original")
                # Convert back to numpy for original implementation
                wts_np = wts if isinstance(wts, np.ndarray) else wts.cpu().numpy()
                inp_np = inp if isinstance(inp, np.ndarray) else inp.cpu().numpy()
                return calculate_wt_lm_head_original(wts_np, inp_np, w)
            # Convert result back to numpy if it's a tensor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            print(f"⚠️  CUDA LM head implementation failed: {e}")
            print(f"   Falling back to original implementation")
            # Ensure inputs are numpy arrays for fallback
            wts_np = wts if isinstance(wts, np.ndarray) else wts.cpu().numpy()
            inp_np = inp if isinstance(inp, np.ndarray) else inp.cpu().numpy()
            return calculate_wt_lm_head_original(wts_np, inp_np, w)
    else:
        raise ValueError(f"Unknown version for LM head: {version}")

def launch_gpt_oss_self_attention(version, wts, inp, w, config, attn_type="full", sliding_window=None):
    if version == 'original':
        # CPU mode: use original implementation
        return calculate_wt_gpt_oss_self_attention_parallel_original(wts, inp, w, config, attn_type=attn_type, sliding_window=sliding_window)
    elif version == 'cuda':
        # CUDA mode: use PyTorch implementation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Convert numpy arrays to tensors
            wts_t = torch.tensor(wts, dtype=torch.float32, device=device) if isinstance(wts, np.ndarray) else wts
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device) if isinstance(inp, np.ndarray) else inp
            
            # Convert weight dictionary to tensors on correct device
            w_torch = {}
            for k, v in w.items():
                if isinstance(v, dict):
                    w_torch[k] = {sub_k: torch.tensor(sub_v, dtype=torch.float32, device=device) if isinstance(sub_v, np.ndarray) else sub_v.to(device) if torch.is_tensor(sub_v) else sub_v
                                 for sub_k, sub_v in v.items()}
                elif isinstance(v, np.ndarray):
                    w_torch[k] = torch.tensor(v, dtype=torch.float32, device=device)
                elif torch.is_tensor(v):
                    w_torch[k] = v.to(device)
                else:
                    w_torch[k] = v
            
            result = calculate_wt_gpt_oss_self_attention_parallel_pytorch(wts_t, inp_t, w_torch, config, attn_type, sliding_window)
            if result is None:
                print(f"⚠️  CUDA GPT-OSS self attention implementation returned None, falling back to original")
                wts_np = wts if isinstance(wts, np.ndarray) else wts.cpu().numpy()
                inp_np = inp if isinstance(inp, np.ndarray) else inp.cpu().numpy()
                return calculate_wt_gpt_oss_self_attention_parallel_original(wts_np, inp_np, w, config, attn_type=attn_type, sliding_window=sliding_window)
            # Convert result back to numpy if it's a tensor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            print(f"⚠️  CUDA GPT-OSS self attention implementation failed: {e}")
            print(f"   Falling back to original implementation")
            wts_np = wts if isinstance(wts, np.ndarray) else wts.cpu().numpy()
            inp_np = inp if isinstance(inp, np.ndarray) else inp.cpu().numpy()
            return calculate_wt_gpt_oss_self_attention_parallel_original(wts_np, inp_np, w, config, attn_type=attn_type, sliding_window=sliding_window)
    else:
        raise ValueError(f"Unknown version for GPT-OSS self attention: {version}")

def launch_gpt_oss_feed_forward(version, wts, inp, w, config):
    if version == 'original':
        # CPU mode: use original implementation
        return calculate_wt_gpt_oss_feed_forward_parallel_original(wts, inp, w, config)
    elif version == 'cuda':
        # CUDA mode: use PyTorch implementation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Convert numpy arrays to tensors
            wts_t = torch.tensor(wts, dtype=torch.float32, device=device) if isinstance(wts, np.ndarray) else wts
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device) if isinstance(inp, np.ndarray) else inp
            
            # Convert weight dictionary to tensors on correct device
            w_torch = {}
            for k, v in w.items():
                if isinstance(v, dict):
                    w_torch[k] = {sub_k: torch.tensor(sub_v, dtype=torch.float32, device=device) if isinstance(sub_v, np.ndarray) else sub_v.to(device) if torch.is_tensor(sub_v) else sub_v
                                 for sub_k, sub_v in v.items()}
                elif isinstance(v, np.ndarray):
                    w_torch[k] = torch.tensor(v, dtype=torch.float32, device=device)
                elif torch.is_tensor(v):
                    w_torch[k] = v.to(device)
                else:
                    w_torch[k] = v
            
            result = calculate_wt_gpt_oss_feed_forward_parallel_pytorch(wts_t, inp_t, w_torch, config)
            if result is None:
                print(f"⚠️  CUDA GPT-OSS feed forward implementation returned None, falling back to original")
                wts_np = wts if isinstance(wts, np.ndarray) else wts.cpu().numpy()
                inp_np = inp if isinstance(inp, np.ndarray) else inp.cpu().numpy()
                return calculate_wt_gpt_oss_feed_forward_parallel_original(wts_np, inp_np, w, config)
            # Convert result back to numpy if it's a tensor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            print(f"⚠️  CUDA GPT-OSS feed forward implementation failed: {e}")
            print(f"   Falling back to original implementation")
            wts_np = wts if isinstance(wts, np.ndarray) else wts.cpu().numpy()
            inp_np = inp if isinstance(inp, np.ndarray) else inp.cpu().numpy()
            return calculate_wt_gpt_oss_feed_forward_parallel_original(wts_np, inp_np, w, config)
    else:
        raise ValueError(f"Unknown version for GPT-OSS feed forward: {version}")

def launch_qwen3_moe_self_attention(version, wts, inp, w, config):
    if version == 'original':
        # CPU mode: use original implementation
        return calculate_wt_qwen3_moe_self_attention_parallel_original(wts, inp, w, config)
    elif version == 'cuda':
        # CUDA mode: use PyTorch implementation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Convert numpy arrays to tensors
            wts_t = torch.tensor(wts, dtype=torch.float32, device=device) if isinstance(wts, np.ndarray) else wts
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device) if isinstance(inp, np.ndarray) else inp
            
            # Convert weight dictionary to tensors on correct device
            w_torch = {}
            for k, v in w.items():
                if isinstance(v, dict):
                    w_torch[k] = {sub_k: torch.tensor(sub_v, dtype=torch.float32, device=device) if isinstance(sub_v, np.ndarray) else sub_v.to(device) if torch.is_tensor(sub_v) else sub_v
                                 for sub_k, sub_v in v.items()}
                elif isinstance(v, np.ndarray):
                    w_torch[k] = torch.tensor(v, dtype=torch.float32, device=device)
                elif torch.is_tensor(v):
                    w_torch[k] = v.to(device)
                else:
                    w_torch[k] = v
            
            result = calculate_wt_qwen3_moe_self_attention_pytorch(wts_t, inp_t, w_torch, config)
            if result is None:
                print(f"⚠️  CUDA Qwen3-MoE self attention implementation returned None, falling back to original")
                wts_np = wts if isinstance(wts, np.ndarray) else wts.cpu().numpy()
                inp_np = inp if isinstance(inp, np.ndarray) else inp.cpu().numpy()
                return calculate_wt_qwen3_moe_self_attention_parallel_original(wts_np, inp_np, w, config)
            # Convert result back to numpy if it's a tensor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            print(f"⚠️  CUDA Qwen3-MoE self attention implementation failed: {e}")
            print(f"   Falling back to original implementation")
            wts_np = wts if isinstance(wts, np.ndarray) else wts.cpu().numpy()
            inp_np = inp if isinstance(inp, np.ndarray) else inp.cpu().numpy()
            return calculate_wt_qwen3_moe_self_attention_parallel_original(wts_np, inp_np, w, config)
    else:
        raise ValueError(f"Unknown version for Qwen3-MoE self attention: {version}")

def launch_qwen3_moe_feed_forward(version, wts, inp, w, config):
    if version == 'original':
        # CPU mode: use original implementation
        return calculate_wt_qwen3_moe_feed_forward_original(wts, inp, w, config)
    elif version == 'cuda':
        # CUDA mode: use PyTorch implementation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Convert numpy arrays to tensors
            wts_t = torch.tensor(wts, dtype=torch.float32, device=device) if isinstance(wts, np.ndarray) else wts
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device) if isinstance(inp, np.ndarray) else inp
            
            # Convert weight dictionary to tensors on correct device
            w_torch = {}
            for k, v in w.items():
                if isinstance(v, dict):
                    w_torch[k] = {sub_k: torch.tensor(sub_v, dtype=torch.float32, device=device) if isinstance(sub_v, np.ndarray) else sub_v.to(device) if torch.is_tensor(sub_v) else sub_v
                                 for sub_k, sub_v in v.items()}
                elif isinstance(v, np.ndarray):
                    w_torch[k] = torch.tensor(v, dtype=torch.float32, device=device)
                elif torch.is_tensor(v):
                    w_torch[k] = v.to(device)
                else:
                    w_torch[k] = v
            
            result = calculate_wt_qwen3_moe_feed_forward_pytorch(wts_t, inp_t, w_torch, config)
            if result is None:
                print(f"⚠️  CUDA Qwen3-MoE feed forward implementation returned None, falling back to original")
                wts_np = wts if isinstance(wts, np.ndarray) else wts.cpu().numpy()
                inp_np = inp if isinstance(inp, np.ndarray) else inp.cpu().numpy()
                return calculate_wt_qwen3_moe_feed_forward_original(wts_np, inp_np, w, config)
            # Convert result back to numpy if it's a tensor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            print(f"⚠️  CUDA Qwen3-MoE feed forward implementation failed: {e}")
            print(f"   Falling back to original implementation")
            wts_np = wts if isinstance(wts, np.ndarray) else wts.cpu().numpy()
            inp_np = inp if isinstance(inp, np.ndarray) else inp.cpu().numpy()
            return calculate_wt_qwen3_moe_feed_forward_original(wts_np, inp_np, w, config)
    else:
        raise ValueError(f"Unknown version for Qwen3-MoE feed forward: {version}")

def launch_olmoe_feed_forward(version, wts, inp, w, model):
    """
    Squeeze leading batch dim (=1) for BOTH `inp` and `wts` before calling impl,
    then unsqueeze the main result back to (1, ...). Works for original/cuda/fallback.
    Also supports backends that return either `arr` or `(arr, meta)`.
    """

    def _np(x):
        if isinstance(x, np.ndarray): return x
        if torch.is_tensor(x): return x.detach().cpu().numpy()
        return np.asarray(x)

    def _np_f32(x):
        a = _np(x)
        return a.astype(np.float32, copy=False)

    def _maybe_squeeze_pair_np(wts_np, inp_np):
        need_unsq = (inp_np.ndim >= 1 and inp_np.shape[0] == 1)
        inp_call = inp_np[0] if need_unsq else inp_np
        # If wts has an extra leading dim=1 relative to inp_call, squeeze it too
        if wts_np.ndim == inp_call.ndim + 1 and wts_np.shape[0] == 1:
            wts_call = wts_np[0]
        else:
            wts_call = wts_np
        return need_unsq, wts_call, inp_call

    def _unsqueeze_first(result, need_unsq):
        if not need_unsq:
            return result
        if isinstance(result, tuple):
            main, *rest = result
            main = np.expand_dims(main, 0) if isinstance(main, np.ndarray) else main.unsqueeze(0)
            return (main, *rest)
        else:
            return np.expand_dims(result, 0) if isinstance(result, np.ndarray) else result.unsqueeze(0)

    # ---------------- ORIGINAL (NumPy) ----------------
    if version == 'original':
        wts_np = _np(wts)
        inp_np = _np_f32(inp) 

        need_unsq, wts_call, inp_call = _maybe_squeeze_pair_np(wts_np, inp_np) 

        out = calculate_wt_olmoe_feed_forward_original(wts_call, inp_call, w, model) 
        out = _unsqueeze_first(out, need_unsq)

        # Convert torch->np if backend returned tensors
        if isinstance(out, tuple):
            main, *rest = out
            main_np = _np(main)
            rest_np = tuple(_np(x) if torch.is_tensor(x) else x for x in rest)
            return (main_np, *rest_np)
        return _np(out)

    # ---------------- CUDA (PyTorch) ----------------
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _tt(x, dtype=torch.float32):
            if torch.is_tensor(x): return x.to(device=device, dtype=dtype)
            return torch.tensor(_np(x), device=device, dtype=dtype)

        try:
            wts_t = _tt(wts, dtype=torch.float32) if isinstance(wts, np.ndarray) or not torch.is_tensor(wts) else wts.to(device)
            inp_t = _tt(inp, dtype=torch.float32) if isinstance(inp, np.ndarray) or not torch.is_tensor(inp) else inp.to(device)

            # Convert weight dictionary to tensors on correct device
            w_torch = {}
            for k, v in w.items():
                if isinstance(v, dict):
                    w_torch[k] = {sub_k: _tt(sub_v) if isinstance(sub_v, np.ndarray) or not torch.is_tensor(sub_v) else sub_v.to(device) 
                                 for sub_k, sub_v in v.items()}
                elif isinstance(v, np.ndarray):
                    w_torch[k] = _tt(v)
                elif torch.is_tensor(v):
                    w_torch[k] = v.to(device)
                else:
                    w_torch[k] = v

            # Mirror the squeeze rule applied to NumPy path
            need_unsq = (inp_t.dim() >= 1 and inp_t.size(0) == 1)
            inp_call = inp_t[0] if need_unsq else inp_t
            if wts_t.dim() == inp_call.dim() + 1 and wts_t.size(0) == 1:
                wts_call = wts_t[0]
            else:
                wts_call = wts_t 

            out = calculate_wt_olmoe_feed_forward_pytorch(wts_call, inp_call, w_torch, model)

            if out is None:
                print("[DEBUG olmoe][cuda] impl returned None -> fallback to original")
                wts_np = _np(wts_call)
                inp_np = _np(inp_call)
                out = calculate_wt_olmoe_feed_forward_original(wts_np, inp_np, w, model)

            # Unsqueeze first element if needed
            out = _unsqueeze_first(out, need_unsq)

            # Return NumPy for parity with original path
            if isinstance(out, tuple):
                main, *rest = out
                main_np = _np(main)
                rest_np = tuple(_np(x) if torch.is_tensor(x) else x for x in rest)
                return (main_np, *rest_np)
            return _np(out)

        except Exception as e:
            print(f"[DEBUG olmoe][cuda] exception: {e} -> fallback to original")
            # Fallback mirrors original path
            wts_np = _np(wts)
            inp_np = _np_f32(inp)
            need_unsq, wts_call, inp_call = _maybe_squeeze_pair_np(wts_np, inp_np)
            out = calculate_wt_olmoe_feed_forward_original(wts_call, inp_call, w, model)
            out = _unsqueeze_first(out, need_unsq)
            if isinstance(out, tuple):
                main, *rest = out
                return (_np(main), *rest)
            return _np(out)

    else:
        raise ValueError(f"Unknown version for OLMoE feed forward: {version}")


def launch_olmoe_self_attention(version, wts, inp, w, model):
    """
    Squeeze leading batch dim (=1) for BOTH `inp` and `wts` before calling impl,
    then unsqueeze the main result back to (1, ...). Works for original/cuda/fallback.
    Also supports backends that return either `arr` or `(arr, meta)`.
    """

    def _np(x):
        if isinstance(x, np.ndarray): return x
        if torch.is_tensor(x): return x.detach().cpu().numpy()
        return np.asarray(x)

    def _np_f32(x):
        a = _np(x)
        return a.astype(np.float32, copy=False)

    def _maybe_squeeze_pair_np(wts_np, inp_np):
        need_unsq = (inp_np.ndim >= 1 and inp_np.shape[0] == 1)
        inp_call = inp_np[0] if need_unsq else inp_np
        # If wts has an extra leading dim=1 relative to inp_call, squeeze it too
        if wts_np.ndim == inp_call.ndim + 1 and wts_np.shape[0] == 1:
            wts_call = wts_np[0]
        else:
            wts_call = wts_np
        return need_unsq, wts_call, inp_call

    def _unsqueeze_first(result, need_unsq):
        if not need_unsq:
            return result
        if isinstance(result, tuple):
            main, *rest = result
            main = np.expand_dims(main, 0) if isinstance(main, np.ndarray) else main.unsqueeze(0)
            return (main, *rest)
        else:
            return np.expand_dims(result, 0) if isinstance(result, np.ndarray) else result.unsqueeze(0)

    # ---------------- ORIGINAL (NumPy) ----------------
    if version == 'original':
        wts_np = _np(wts)
        inp_np = _np_f32(inp)

        need_unsq, wts_call, inp_call = _maybe_squeeze_pair_np(wts_np, inp_np) 

        out = calculate_wt_olmoe_self_attention_parallel_original(wts_call, inp_call, w, model) 
        out = _unsqueeze_first(out, need_unsq)

        # Convert torch->np if backend returned tensors
        if isinstance(out, tuple):
            main, *rest = out
            main_np = _np(main)
            rest_np = tuple(_np(x) if torch.is_tensor(x) else x for x in rest)
            return (main_np, *rest_np)
        return _np(out)

    # ---------------- CUDA (PyTorch) ----------------
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _tt(x, dtype=torch.float32):
            if torch.is_tensor(x): return x.to(device=device, dtype=dtype)
            return torch.tensor(_np(x), device=device, dtype=dtype)

        try:
            wts_t = _tt(wts, dtype=torch.float32) if isinstance(wts, np.ndarray) or not torch.is_tensor(wts) else wts.to(device)
            inp_t = _tt(inp, dtype=torch.float32) if isinstance(inp, np.ndarray) or not torch.is_tensor(inp) else inp.to(device)

            # Convert weight dictionary to tensors on correct device
            w_torch = {}
            for k, v in w.items():
                if isinstance(v, dict):
                    w_torch[k] = {sub_k: _tt(sub_v) if isinstance(sub_v, np.ndarray) or not torch.is_tensor(sub_v) else sub_v.to(device) 
                                 for sub_k, sub_v in v.items()}
                elif isinstance(v, np.ndarray):
                    w_torch[k] = _tt(v)
                elif torch.is_tensor(v):
                    w_torch[k] = v.to(device)
                else:
                    w_torch[k] = v

            # Mirror the squeeze rule applied to NumPy path
            need_unsq = (inp_t.dim() >= 1 and inp_t.size(0) == 1)
            inp_call = inp_t[0] if need_unsq else inp_t
            if wts_t.dim() == inp_call.dim() + 1 and wts_t.size(0) == 1:
                wts_call = wts_t[0]
            else:
                wts_call = wts_t

            out =  calculate_wt_olmoe_self_attention_parallel_pytorch(wts_call, inp_call, w_torch, model)

            if out is None:
                print("[DEBUG olmoe_sa][cuda] impl returned None -> fallback to original")
                wts_np = _np(wts_call)
                inp_np = _np(inp_call)
                out = calculate_wt_olmoe_self_attention_parallel_original(wts_np, inp_np, w, model)

            # Unsqueeze first element if needed
            out = _unsqueeze_first(out, need_unsq)

            # Return NumPy for parity with original path
            if isinstance(out, tuple):
                main, *rest = out
                main_np = _np(main)
                rest_np = tuple(_np(x) if torch.is_tensor(x) else x for x in rest)
                return (main_np, *rest_np)
            return _np(out)

        except Exception as e:
            print(f"[DEBUG olmoe_sa][cuda] exception: {e} -> fallback to original")
            # Fallback mirrors original path
            wts_np = _np(wts)
            inp_np = _np_f32(inp)
            need_unsq, wts_call, inp_call = _maybe_squeeze_pair_np(wts_np, inp_np)
            out = calculate_wt_olmoe_self_attention_parallel_original(wts_call, inp_call, w, model)
            out = _unsqueeze_first(out, need_unsq)
            if isinstance(out, tuple):
                main, *rest = out
                return (_np(main), *rest)
            return _np(out)

    else:
        raise ValueError(f"Unknown version for OLMoE self attention: {version}") 


def launch_jetmoe_self_attention(version, wts, inp, w, model):
    """
    Squeeze leading batch dim (=1) for BOTH `inp` and `wts` before calling impl,
    then unsqueeze the main result back to (1, ...). Works for original/cuda/fallback.
    Also supports backends that return either `arr` or `(arr, meta)`.
    """

    def _np(x):
        if isinstance(x, np.ndarray): return x
        if torch.is_tensor(x): return x.detach().cpu().numpy()
        return np.asarray(x)

    def _np_f32(x):
        a = _np(x)
        return a.astype(np.float32, copy=False)

    def _maybe_squeeze_pair_np(wts_np, inp_np):
        need_unsq = (inp_np.ndim >= 1 and inp_np.shape[0] == 1)
        inp_call = inp_np[0] if need_unsq else inp_np
        # If wts has an extra leading dim=1 relative to inp_call, squeeze it too
        if wts_np.ndim == inp_call.ndim + 1 and wts_np.shape[0] == 1:
            wts_call = wts_np[0]
        else:
            wts_call = wts_np
        return need_unsq, wts_call, inp_call

    def _unsqueeze_first(result, need_unsq):
        if not need_unsq:
            return result
        if isinstance(result, tuple):
            main, *rest = result
            main = np.expand_dims(main, 0) if isinstance(main, np.ndarray) else main.unsqueeze(0)
            return (main, *rest)
        else:
            return np.expand_dims(result, 0) if isinstance(result, np.ndarray) else result.unsqueeze(0)

    # ---------------- ORIGINAL (NumPy) ----------------
    if version == 'original':
        wts_np = _np(wts)
        inp_np = _np_f32(inp)

        need_unsq, wts_call, inp_call = _maybe_squeeze_pair_np(wts_np, inp_np)

        out = calculate_wt_jetmoe_self_attention_parallel_original(wts_call, inp_call, w, model)
        out = _unsqueeze_first(out, need_unsq)

        # Convert torch->np if backend returned tensors
        if isinstance(out, tuple):
            main, *rest = out
            main_np = _np(main)
            rest_np = tuple(_np(x) if torch.is_tensor(x) else x for x in rest)
            return (main_np, *rest_np)
        return _np(out)

    # ---------------- CUDA (PyTorch) ----------------
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _tt(x, dtype=torch.float32):
            if torch.is_tensor(x): return x.to(device=device, dtype=dtype)
            return torch.tensor(_np(x), device=device, dtype=dtype)

        try:
            wts_t = _tt(wts, dtype=torch.float32) if isinstance(wts, np.ndarray) or not torch.is_tensor(wts) else wts.to(device)
            inp_t = _tt(inp, dtype=torch.float32) if isinstance(inp, np.ndarray) or not torch.is_tensor(inp) else inp.to(device)

            # Convert weight dictionary to tensors on correct device
            w_torch = {}
            for k, v in w.items():
                if isinstance(v, dict):
                    w_torch[k] = {sub_k: _tt(sub_v) if isinstance(sub_v, np.ndarray) or not torch.is_tensor(sub_v) else sub_v.to(device) 
                                 for sub_k, sub_v in v.items()}
                elif isinstance(v, np.ndarray):
                    w_torch[k] = _tt(v)
                elif torch.is_tensor(v):
                    w_torch[k] = v.to(device)
                else:
                    w_torch[k] = v 

            # Mirror the squeeze rule applied to NumPy path
            need_unsq = (inp_t.dim() >= 1 and inp_t.size(0) == 1)
            inp_call = inp_t[0] if need_unsq else inp_t
            if wts_t.dim() == inp_call.dim() + 1 and wts_t.size(0) == 1:
                wts_call = wts_t[0]
            else:
                wts_call = wts_t

            out = calculate_wt_jetmoe_self_attention_parallel_pytorch(wts_call, inp_call, w_torch, model)

            if out is None:
                print("[DEBUG jetmoe_sa][cuda] impl returned None -> fallback to original")
                wts_np = _np(wts_call)
                inp_np = _np(inp_call)
                out = calculate_wt_jetmoe_self_attention_parallel_original(wts_np, inp_np, w, model)

            # Unsqueeze first element if needed
            out = _unsqueeze_first(out, need_unsq)

            # Return NumPy for parity with original path
            if isinstance(out, tuple):
                main, *rest = out
                main_np = _np(main)
                rest_np = tuple(_np(x) if torch.is_tensor(x) else x for x in rest)
                return (main_np, *rest_np)
            return _np(out)

        except Exception as e:
            import traceback
            print(f"[DEBUG jetmoe_sa][cuda] exception: {e} -> fallback to original")
            print(f"[DEBUG jetmoe_sa][cuda] traceback: {traceback.format_exc()}")
            # Fallback mirrors original path
            wts_np = _np(wts)
            inp_np = _np_f32(inp)
            need_unsq, wts_call, inp_call = _maybe_squeeze_pair_np(wts_np, inp_np)
            out = calculate_wt_jetmoe_self_attention_parallel_original(wts_call, inp_call, w, model)
            out = _unsqueeze_first(out, need_unsq)
            if isinstance(out, tuple):
                main, *rest = out
                return (_np(main), *rest)
            return _np(out)

    else:
        raise ValueError(f"Unknown version for JetMoE self attention: {version}")


def launch_jetmoe_feed_forward(version, wts, inp, w, model):
    """
    Squeeze leading batch dim (=1) for BOTH `inp` and `wts` before calling impl,
    then unsqueeze the main result back to (1, ...). Works for original/cuda/fallback.
    Also supports backends that return either `arr` or `(arr, meta)`.
    """

    def _np(x):
        if isinstance(x, np.ndarray): return x
        if torch.is_tensor(x): return x.detach().cpu().numpy()
        return np.asarray(x)

    def _np_f32(x):
        a = _np(x)
        return a.astype(np.float32, copy=False)

    def _maybe_squeeze_pair_np(wts_np, inp_np):
        need_unsq = (inp_np.ndim >= 1 and inp_np.shape[0] == 1)
        inp_call = inp_np[0] if need_unsq else inp_np
        # If wts has an extra leading dim=1 relative to inp_call, squeeze it too
        if wts_np.ndim == inp_call.ndim + 1 and wts_np.shape[0] == 1:
            wts_call = wts_np[0]
        else:
            wts_call = wts_np
        return need_unsq, wts_call, inp_call

    def _unsqueeze_first(result, need_unsq):
        if not need_unsq:
            return result
        if isinstance(result, tuple):
            main, *rest = result
            main = np.expand_dims(main, 0) if isinstance(main, np.ndarray) else main.unsqueeze(0)
            return (main, *rest)
        else:
            return np.expand_dims(result, 0) if isinstance(result, np.ndarray) else result.unsqueeze(0)

    # ---------------- ORIGINAL (NumPy) ----------------
    if version == 'original':
        wts_np = _np(wts)
        inp_np = _np_f32(inp)

        need_unsq, wts_call, inp_call = _maybe_squeeze_pair_np(wts_np, inp_np)

        out = calculate_wt_jetmoe_feed_forward_original(wts_call, inp_call, w, model)
        out = _unsqueeze_first(out, need_unsq)

        # Convert torch->np if backend returned tensors
        if isinstance(out, tuple):
            main, *rest = out
            main_np = _np(main)
            rest_np = tuple(_np(x) if torch.is_tensor(x) else x for x in rest)
            return (main_np, *rest_np)
        return _np(out)

    # ---------------- CUDA (PyTorch) ----------------
    elif version == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _tt(x, dtype=torch.float32):
            if torch.is_tensor(x): return x.to(device=device, dtype=dtype)
            return torch.tensor(_np(x), device=device, dtype=dtype)

        try:
            wts_t = _tt(wts, dtype=torch.float32) if isinstance(wts, np.ndarray) or not torch.is_tensor(wts) else wts.to(device)
            inp_t = _tt(inp, dtype=torch.float32) if isinstance(inp, np.ndarray) or not torch.is_tensor(inp) else inp.to(device)

            # Convert weight dictionary to tensors on correct device
            w_torch = {}
            for k, v in w.items():
                if isinstance(v, dict):
                    w_torch[k] = {sub_k: _tt(sub_v) if isinstance(sub_v, np.ndarray) or not torch.is_tensor(sub_v) else sub_v.to(device) 
                                 for sub_k, sub_v in v.items()}
                elif isinstance(v, np.ndarray):
                    w_torch[k] = _tt(v)
                elif torch.is_tensor(v):
                    w_torch[k] = v.to(device)
                else:
                    w_torch[k] = v

            # Mirror the squeeze rule applied to NumPy path
            need_unsq = (inp_t.dim() >= 1 and inp_t.size(0) == 1)
            inp_call = inp_t[0] if need_unsq else inp_t
            if wts_t.dim() == inp_call.dim() + 1 and wts_t.size(0) == 1:
                wts_call = wts_t[0]
            else:
                wts_call = wts_t

            out = calculate_wt_jetmoe_feed_forward_pytorch(wts_call, inp_call, w_torch, model)

            if out is None:
                print("[DEBUG jetmoe][cuda] impl returned None -> fallback to original")
                wts_np = _np(wts_call)
                inp_np = _np(inp_call)
                out = calculate_wt_jetmoe_feed_forward_original(wts_np, inp_np, w, model)

            # Unsqueeze first element if needed
            out = _unsqueeze_first(out, need_unsq)

            # Return NumPy for parity with original path
            if isinstance(out, tuple):
                main, *rest = out
                main_np = _np(main)
                rest_np = tuple(_np(x) if torch.is_tensor(x) else x for x in rest)
                return (main_np, *rest_np)
            return _np(out)

        except Exception as e:
            import traceback
            print(f"[DEBUG jetmoe][cuda] exception: {e} -> fallback to original")
            print(f"[DEBUG jetmoe][cuda] traceback: {traceback.format_exc()}")
            # Fallback mirrors original path
            wts_np = _np(wts)
            inp_np = _np_f32(inp)
            need_unsq, wts_call, inp_call = _maybe_squeeze_pair_np(wts_np, inp_np)
            out = calculate_wt_jetmoe_feed_forward_original(wts_call, inp_call, w, model)
            out = _unsqueeze_first(out, need_unsq)
            if isinstance(out, tuple):
                main, *rest = out
                return (_np(main), *rest)
            return _np(out)

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
    task = kwargs.get('task', None)  # Access 'task' from kwargs, default to None if not provided
    
    if task == "binary-classification":
        # x = np.argmax(arg, axis=2)  # max along features
        # m = np.max(arg, axis=2)
        # y = np.zeros_like(arg)

        # batch_size, seq_len, _ = arg.shape
        # for i in range(batch_size):
        #     for j in range(seq_len):
        #         if scaler:
        #             y[i, j, x[i, j]] = scaler
        #         else:
        #             y[i, j, x[i, j]] = m[i, j]
        # log(y.shape,"y")
        predicted_class = np.argmax(arg, axis=-1, keepdims=True)  # [B, 1] 
        print(f"predicted_class: {predicted_class}")

        # Create sparse relevance: only 1 class matters
        target_relevance = np.zeros_like(arg, dtype=np.float32)

        # Set the 1.0 at predicted indices
        for b, t in enumerate(predicted_class):
            print(f"batch: {b}, predicted_class: {t.item()}") 
            target_relevance[b, t] = 1.0
        
        print(f"target_relevance --- original array: {target_relevance}, value: {np.sum(target_relevance):.4f}, shape: {target_relevance.shape}")

    elif task == "generation":
        # code here
        print("======arg.shape=====",arg.shape)
        # x = np.argmax(arg, axis=2)
        # print("===x.shape============",x.shape)
        # y = np.zeros_like(arg)
        # value = 1 / arg.shape[1]

        # batch_size, seq_len, _ = arg.shape
        # for i in range(batch_size):
        #     for j in range(seq_len):
        #         y[i, j, x[i, j]] = value 

        # print("====y.shape=======",y.shape)

        next_token_logit = arg[:, -1, :]
        print(f"next_token_logit: {next_token_logit.shape}")
        predicted_token = np.argmax(next_token_logit, axis=-1, keepdims=True)  # [B, 1]
        print(f"predicted_token: {predicted_token}")

        # Create target relevance
        target_relevance = np.zeros_like(arg, dtype=np.float32)

        # Set the 1.0 at predicted indices
        for b, t in enumerate(predicted_token):
            print(f"batch: {b}, token: {t}")
            target_relevance[b, -1, t] = 1.0
        
        print(f"target_relevance --- value: {np.sum(target_relevance):.4f}, shape: {target_relevance.shape}")

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
