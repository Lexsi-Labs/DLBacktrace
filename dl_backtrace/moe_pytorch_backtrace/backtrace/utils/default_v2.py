import torch
import numpy as np


# ------------- Lazy import system for CUDA modules ----------------
# This prevents CUDA compilation when only using CPU

_cuda_modules_cache = {}

def _lazy_import_cuda_module(module_path, func_name, cuda_only=False):
    """
    Lazy import a function from a CUDA module.
    Only imports and compiles when actually called.
    
    Args:
        module_path: Python import path (e.g., '.cuda_utils.GPT_oss.pytorch_version')
        func_name: Function name to import
        cuda_only: If True, only import when CUDA is available
    
    Returns:
        The imported function, or None if CUDA not available and cuda_only=True
    """
    cache_key = f"{module_path}.{func_name}"
    
    if cache_key in _cuda_modules_cache:
        return _cuda_modules_cache[cache_key]
    
    # Check if CUDA is available for cuda_only modules
    if cuda_only and not torch.cuda.is_available():
        print(f"⚠️  CUDA not available, skipping CUDA-only module: {module_path}")
        _cuda_modules_cache[cache_key] = None
        return None
    
    try:
        # Dynamically import the module
        from importlib import import_module
        module = import_module(module_path, package=__package__)
        func = getattr(module, func_name)
        _cuda_modules_cache[cache_key] = func
        return func
    except Exception as e:
        print(f"⚠️  Failed to import {module_path}.{func_name}: {e}")
        _cuda_modules_cache[cache_key] = None
        return None

# Helper functions to get implementations with lazy loading
def _get_gpt_oss_impl(version, impl_type):
    """Get GPT-OSS implementation with lazy loading."""
    if version == 'original':
        if impl_type == 'lm_head':
            return _lazy_import_cuda_module('.cuda_utils.GPT_oss.original_version', 'calculate_wt_lm_head')
        elif impl_type == 'feed_forward':
            return _lazy_import_cuda_module('.cuda_utils.GPT_oss.original_version', 'calculate_wt_gpt_oss_feed_forward_parallel')
        elif impl_type == 'self_attention':
            return _lazy_import_cuda_module('.cuda_utils.GPT_oss.original_version', 'calculate_wt_self_attention_parallel')
    elif version == 'cuda':
        if impl_type == 'lm_head':
            return _lazy_import_cuda_module('.cuda_utils.GPT_oss.pytorch_version', 'calculate_wt_lm_head')
        elif impl_type == 'feed_forward':
            return _lazy_import_cuda_module('.cuda_utils.GPT_oss.pytorch_version', 'calculate_wt_gpt_oss_feed_forward_parallel')
        elif impl_type == 'self_attention':
            return _lazy_import_cuda_module('.cuda_utils.GPT_oss.pytorch_version', 'calculate_wt_self_attention_parallel_torch')
    return None

def _get_qwen_moe_impl(version, impl_type):
    """Get Qwen3-MoE implementation with lazy loading."""
    if version == 'original':
        if impl_type == 'feed_forward':
            return _lazy_import_cuda_module('.cuda_utils.Qwen_MoE.original_version', 'calculate_wt_feed_forward')
        elif impl_type == 'self_attention':
            return _lazy_import_cuda_module('.cuda_utils.Qwen_MoE.original_version', 'calculate_wt_self_attention_parallel')
    elif version == 'cuda':
        if impl_type == 'feed_forward':
            return _lazy_import_cuda_module('.cuda_utils.Qwen_MoE.pytorch_version', 'calculate_wt_feed_forward')
        elif impl_type == 'self_attention':
            return _lazy_import_cuda_module('.cuda_utils.Qwen_MoE.pytorch_version', 'calculate_wt_self_attention')
    return None

def _get_olmoe_impl(version):
    """Get OLMoE implementation with lazy loading."""
    if version == 'original':
        return _lazy_import_cuda_module('.cuda_utils.OLMoE.original_version', 'calculate_wt_olmoe_feed_forward_parallel')
    elif version == 'cuda':
        return _lazy_import_cuda_module('.cuda_utils.OLMoE.pytorch_version', 'calculate_wt_olmoe_feed_forward_parallel')
    return None

def _get_jetmoe_impl(version, impl_type):
    """Get JetMoE implementation with lazy loading."""
    if version == 'original':
        if impl_type == 'feed_forward':
            return _lazy_import_cuda_module('.cuda_utils.JetMoE.original_version', 'calculate_wt_jetmoe_feed_forward')
        elif impl_type == 'self_attention':
            return _lazy_import_cuda_module('.cuda_utils.JetMoE.original_version', 'calculate_wt_jetmoe_self_attention_parallel')
    elif version == 'cuda':
        if impl_type == 'feed_forward':
            return _lazy_import_cuda_module('.cuda_utils.JetMoE.pytorch_version', 'calculate_wt_jetmoe_feed_forward')
        elif impl_type == 'self_attention':
            return _lazy_import_cuda_module('.cuda_utils.JetMoE.pytorch_version', 'calculate_wt_jetmoe_self_attention_parallel')
    return None

def launch_lm_head(version, wts, inp, w, b, act):
    if version == 'original':
        # CPU mode: use original implementation
        impl_func = _get_gpt_oss_impl('original', 'lm_head')
        if impl_func is None:
            raise RuntimeError("Failed to load original LM head implementation")
        return impl_func(wts, inp, w, b, act)
    elif version == 'cuda':
        # CUDA mode: use PyTorch implementation
        impl_func = _get_gpt_oss_impl('cuda', 'lm_head')
        if impl_func is None:
            print(f"⚠️  CUDA LM head implementation not available, falling back to original")
            return launch_lm_head('original', wts, inp, w, b, act)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Convert numpy arrays to tensors
            wts_t = torch.tensor(wts, dtype=torch.float32, device=device) if isinstance(wts, np.ndarray) else wts
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device) if isinstance(inp, np.ndarray) else inp
            w_t = torch.tensor(w, dtype=torch.float32, device=device) if isinstance(w, np.ndarray) else w
            b_t = torch.tensor(b, dtype=torch.float32, device=device) if isinstance(b, np.ndarray) else b
            # w is a dict, no conversion needed
            result = impl_func(wts_t, inp_t, w_t, b_t, act)
            if result is None:
                print(f"⚠️  CUDA LM head implementation returned None, falling back to original")
                return launch_lm_head('original', wts, inp, w, b, act)
            # Convert result back to numpy if it's a tensor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            print(f"⚠️  CUDA LM head implementation failed: {e}")
            print(f"   Falling back to original implementation")
            return launch_lm_head('original', wts, inp, w, b, act)
    else:
        raise ValueError(f"Unknown version for LM head: {version}")

def launch_gpt_oss_self_attention(version, wts, inp, w, config, attn_type="full", sliding_window=None):
    if version == 'original':
        # CPU mode: use original implementation
        impl_func = _get_gpt_oss_impl('original', 'self_attention')
        if impl_func is None:
            raise RuntimeError("Failed to load original GPT-OSS self attention implementation")
        return impl_func(wts, inp, w, config, attn_type, sliding_window)
    elif version == 'cuda':
        # CUDA mode: use PyTorch implementation
        impl_func = _get_gpt_oss_impl('cuda', 'self_attention')
        if impl_func is None:
            print(f"⚠️  CUDA GPT-OSS self attention implementation not available, falling back to original")
            return launch_gpt_oss_self_attention('original', wts, inp, w, config, attn_type, sliding_window)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Convert numpy arrays to tensors
            wts_t = torch.tensor(wts, dtype=torch.float32, device=device) if isinstance(wts, np.ndarray) else wts
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device) if isinstance(inp, np.ndarray) else inp
            w_t = torch.tensor(w, dtype=torch.float32, device=device) if isinstance(w, np.ndarray) else w
            result = impl_func(wts_t, inp_t, w_t, config, attn_type, sliding_window)
            if result is None:
                print(f"⚠️  CUDA GPT-OSS self attention implementation returned None, falling back to original")
                return launch_gpt_oss_self_attention('original', wts, inp, w, config, attn_type, sliding_window)
            # Convert result back to numpy if it's a tensor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            print(f"⚠️  CUDA GPT-OSS self attention implementation failed: {e}")
            print(f"   Falling back to original implementation")
            return launch_gpt_oss_self_attention('original', wts, inp, w, config, attn_type, sliding_window)
    else:
        raise ValueError(f"Unknown version for GPT-OSS self attention: {version}")

def launch_gpt_oss_feed_forward(version, wts, inp, w, config):
    if version == 'original':
        # CPU mode: use original implementation
        impl_func = _get_gpt_oss_impl('original', 'feed_forward')
        if impl_func is None:
            raise RuntimeError("Failed to load original GPT-OSS feed forward implementation")
        return impl_func(wts, inp, w, config)
    elif version == 'cuda':
        # CUDA mode: use PyTorch implementation
        impl_func = _get_gpt_oss_impl('cuda', 'feed_forward')
        if impl_func is None:
            print(f"⚠️  CUDA GPT-OSS feed forward implementation not available, falling back to original")
            return launch_gpt_oss_feed_forward('original', wts, inp, w, config)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Convert numpy arrays to tensors
            wts_t = torch.tensor(wts, dtype=torch.float32, device=device) if isinstance(wts, np.ndarray) else wts
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device) if isinstance(inp, np.ndarray) else inp   
            w_t = torch.tensor(w, dtype=torch.float32, device=device) if isinstance(w, np.ndarray) else w
            result = impl_func(wts_t, inp_t, w_t, config)
            if result is None:
                print(f"⚠️  CUDA GPT-OSS feed forward implementation returned None, falling back to original")
                return launch_gpt_oss_feed_forward('original', wts, inp, w, config)
            # Convert result back to numpy if it's a tensor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            print(f"⚠️  CUDA GPT-OSS feed forward implementation failed: {e}")
            print(f"   Falling back to original implementation")
            return launch_gpt_oss_feed_forward('original', wts, inp, w, config)
    else:
        raise ValueError(f"Unknown version for GPT-OSS feed forward: {version}")

def launch_qwen3_moe_self_attention(version, wts, inp, w, config):
    if version == 'original':
        # CPU mode: use original implementation
        impl_func = _get_qwen_moe_impl('original', 'self_attention')
        if impl_func is None:
            raise RuntimeError("Failed to load original Qwen3-MoE self attention implementation")
        return impl_func(wts, inp, w, config)
    elif version == 'cuda':
        # CUDA mode: use PyTorch implementation
        impl_func = _get_qwen_moe_impl('cuda', 'self_attention')
        if impl_func is None:
            print(f"⚠️  CUDA Qwen3-MoE self attention implementation not available, falling back to original")
            return launch_qwen3_moe_self_attention('original', wts, inp, w, config)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Convert numpy arrays to tensors
            wts_t = torch.tensor(wts, dtype=torch.float32, device=device) if isinstance(wts, np.ndarray) else wts
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device) if isinstance(inp, np.ndarray) else inp
            w_t = torch.tensor(w, dtype=torch.float32, device=device) if isinstance(w, np.ndarray) else w
            result = impl_func(wts_t, inp_t, w_t, config)
            if result is None:
                print(f"⚠️  CUDA Qwen3-MoE self attention implementation returned None, falling back to original")
                return launch_qwen3_moe_self_attention('original', wts, inp, w, config)
            # Convert result back to numpy if it's a tensor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            print(f"⚠️  CUDA Qwen3-MoE self attention implementation failed: {e}")
            print(f"   Falling back to original implementation")
            return launch_qwen3_moe_self_attention('original', wts, inp, w, config)
    else:
        raise ValueError(f"Unknown version for Qwen3-MoE self attention: {version}")

def launch_qwen3_moe_feed_forward(version, wts, inp, w, config):
    if version == 'original':
        # CPU mode: use original implementation
        impl_func = _get_qwen_moe_impl('original', 'feed_forward')
        if impl_func is None:
            raise RuntimeError("Failed to load original Qwen3-MoE feed forward implementation")
        return impl_func(wts, inp, w, config)
    elif version == 'cuda':
        # CUDA mode: use PyTorch implementation
        impl_func = _get_qwen_moe_impl('cuda', 'feed_forward')
        if impl_func is None:
            print(f"⚠️  CUDA Qwen3-MoE feed forward implementation not available, falling back to original")
            return launch_qwen3_moe_feed_forward('original', wts, inp, w, config)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Convert numpy arrays to tensors
            wts_t = torch.tensor(wts, dtype=torch.float32, device=device) if isinstance(wts, np.ndarray) else wts
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device) if isinstance(inp, np.ndarray) else inp
            w_t = torch.tensor(w, dtype=torch.float32, device=device) if isinstance(w, np.ndarray) else w
            result = impl_func(wts_t, inp_t, w_t, config)
            if result is None:
                print(f"⚠️  CUDA Qwen3-MoE feed forward implementation returned None, falling back to original")
                return launch_qwen3_moe_feed_forward('original', wts, inp, w, config)
            # Convert result back to numpy if it's a tensor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            print(f"⚠️  CUDA Qwen3-MoE feed forward implementation failed: {e}")
            print(f"   Falling back to original implementation")
            return launch_qwen3_moe_feed_forward('original', wts, inp, w, config)
    else:
        raise ValueError(f"Unknown version for Qwen3-MoE feed forward: {version}")

def launch_olmoe_feed_forward(version, wts, inp, w, model):
    if version == 'original':
        # CPU mode: use original implementation
        impl_func = _get_olmoe_impl('original')
        if impl_func is None:
            raise RuntimeError("Failed to load original OLMoE feed forward implementation")
        return impl_func(wts, inp, w, model)
    elif version == 'cuda':
        # CUDA mode: use PyTorch implementation
        impl_func = _get_olmoe_impl('cuda')
        if impl_func is None:
            print(f"⚠️  CUDA OLMoE feed forward implementation not available, falling back to original")
            return launch_olmoe_feed_forward('original', wts, inp, w, model)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Convert numpy arrays to tensors
            wts_t = torch.tensor(wts, dtype=torch.float32, device=device) if isinstance(wts, np.ndarray) else wts
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device) if isinstance(inp, np.ndarray) else inp
            w_t = torch.tensor(w, dtype=torch.float32, device=device) if isinstance(w, np.ndarray) else w
            result = impl_func(wts_t, inp_t, w_t, model)
            if result is None:
                print(f"⚠️  CUDA OLMoE feed forward implementation returned None, falling back to original")
                return launch_olmoe_feed_forward('original', wts, inp, w, model)
            # Convert result back to numpy if it's a tensor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            print(f"⚠️  CUDA OLMoE feed forward implementation failed: {e}")
            print(f"   Falling back to original implementation")
            return launch_olmoe_feed_forward('original', wts, inp, w, model)
    else:
        raise ValueError(f"Unknown version for OLMoE feed forward: {version}")

def launch_jetmoe_self_attention(version, wts, inp, w, model):
    if version == 'original':
        # CPU mode: use original implementation
        impl_func = _get_jetmoe_impl('original', 'self_attention')
        if impl_func is None:
            raise RuntimeError("Failed to load original JetMoE self attention implementation")
        return impl_func(wts, inp, w, model)
    elif version == 'cuda':
        # CUDA mode: use PyTorch implementation
        impl_func = _get_jetmoe_impl('cuda', 'self_attention')
        if impl_func is None:
            print(f"⚠️  CUDA JetMoE self attention implementation not available, falling back to original")
            return launch_jetmoe_self_attention('original', wts, inp, w, model)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Convert numpy arrays to tensors
            wts_t = torch.tensor(wts, dtype=torch.float32, device=device) if isinstance(wts, np.ndarray) else wts
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device) if isinstance(inp, np.ndarray) else inp
            w_t = torch.tensor(w, dtype=torch.float32, device=device) if isinstance(w, np.ndarray) else w
            result = impl_func(wts_t, inp_t, w_t, model)
            if result is None:
                print(f"⚠️  CUDA JetMoE self attention implementation returned None, falling back to original")
                return launch_jetmoe_self_attention('original', wts, inp, w, model)
            # Convert result back to numpy if it's a tensor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            print(f"⚠️  CUDA JetMoE self attention implementation failed: {e}")
            print(f"   Falling back to original implementation")
            return launch_jetmoe_self_attention('original', wts, inp, w, model)
    else:
        raise ValueError(f"Unknown version for JetMoE self attention: {version}")

def launch_jetmoe_feed_forward(version, wts, inp, w, model):
    if version == 'original':
        # CPU mode: use original implementation
        impl_func = _get_jetmoe_impl('original', 'feed_forward')
        if impl_func is None:
            raise RuntimeError("Failed to load original JetMoE feed forward implementation")
        return impl_func(wts, inp, w, model)
    elif version == 'cuda':
        # CUDA mode: use PyTorch implementation
        impl_func = _get_jetmoe_impl('cuda', 'feed_forward')
        if impl_func is None:
            print(f"⚠️  CUDA JetMoE feed forward implementation not available, falling back to original")
            return launch_jetmoe_feed_forward('original', wts, inp, w, model)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Convert numpy arrays to tensors
            wts_t = torch.tensor(wts, dtype=torch.float32, device=device) if isinstance(wts, np.ndarray) else wts
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device) if isinstance(inp, np.ndarray) else inp
            w_t = torch.tensor(w, dtype=torch.float32, device=device) if isinstance(w, np.ndarray) else w
            result = impl_func(wts_t, inp_t, w_t, model)
            if result is None:
                print(f"⚠️  CUDA JetMoE feed forward implementation returned None, falling back to original")
                return launch_jetmoe_feed_forward('original', wts, inp, w, model)
            # Convert result back to numpy if it's a tensor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            print(f"⚠️  CUDA JetMoE feed forward implementation failed: {e}")
            print(f"   Falling back to original implementation")
            return launch_jetmoe_feed_forward('original', wts, inp, w, model)
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
