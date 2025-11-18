import numpy as np
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from dl_backtrace.moe_pytorch_backtrace.backtrace.utils import contrast as UC
from dl_backtrace.moe_pytorch_backtrace.backtrace.utils import prop as UP
from dl_backtrace.moe_pytorch_backtrace.backtrace.config import activation_master
from dl_backtrace.moe_pytorch_backtrace.backtrace.core import (
    jetmoe as jetmoe,
    olmoe as olmoe,
    qwen3_moe as qwen3_moe,
    gpt_oss as gpt_oss,
    helper as helper,
)
from dl_backtrace.moe_pytorch_backtrace.backtrace.utils import default_v2 as UD2


def t2np32(t):
    # Cast any torch tensor to float32 on CPU before numpy() to avoid bf16 errors
    if isinstance(t, torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()
    return t


def _layer_idx(name: str):
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else None


def build_attention_plan(layer_stack, config):
    plan = {}
    for name in layer_stack:
        if not name.startswith("decoder_self_attention_"):
            continue
        idx = _layer_idx(name)
        if idx is None:
            continue
        is_sliding = (config.layer_types[idx] == "sliding_attention")
        plan[name] = {
            "attn_type": "sliding" if is_sliding else "full",
            "window": config.sliding_window if is_sliding else None,
        }
    return plan


def get_model_config(model):
    """
    Extracts the configuration object from a model, handling wrapped models.
    """
    if hasattr(model, "model"):
        inner_model = model.model
        if hasattr(inner_model, "config"):
            return inner_model.config
    elif hasattr(model, "config"):
        return model.config
    raise AttributeError("The provided object does not have a 'config' attribute.")


def get_tensor_or_raise(all_in, all_out, key, where="all_out"):
    x = all_out.get(key, None)
    if x is None:
        x = all_in.get(key, None)
        where = "all_in" if x is not None else where
    if x is None:
        raise KeyError(
            f"Missing tensor for node '{key}'. "
            f"Have keys(all_out)={list(all_out.keys())} "
            f"keys(all_in)={list(all_in.keys())}"
        )
    # unwrap tuples/lists from hooks
    if isinstance(x, (tuple, list)):
        x = x[0]
    if not torch.is_tensor(x):
        raise TypeError(f"Node '{key}' is not a tensor (got {type(x)}) from {where}.")
    return x


def tensor_to_numpy(x):
    """Convert Tensor, scalar, list/tuple, or ndarray to a NumPy array with memory-efficient float32."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32) if x.dtype == np.float64 else x
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    if isinstance(x, (int, float)):
        return np.array(x, dtype=np.float32)
    if isinstance(x, (list, tuple)):
        arrs = []
        for xi in x:
            converted = tensor_to_numpy(xi) if not isinstance(xi, np.ndarray) else xi
            if hasattr(converted, "dtype") and converted.dtype == np.float64:
                converted = converted.astype(np.float32)
            arrs.append(converted)
        try:
            return np.stack(arrs)
        except Exception:
            return np.array(arrs, dtype=object)
    raise TypeError(f"Cannot convert type {type(x)} to numpy")


class Backtrace(object):
    """
    Build graph + extract weights in __init__. Compute outputs only when asked.
    Device is set at construction and used everywhere (no device arg in eval()).
    """

    def __init__(self, model=None, activation_dict={}, model_type=None, device="cpu"):
        if model_type not in {"gpt_oss", "qwen3_moe", "jetmoe", "olmoe"}:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # ---- device handling ----
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda'.")
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = device
        self.impl = "cuda" if device == "cuda" else "original"

        self.model = model.to(device) if model is not None else None
        self.model_type = model_type
        self.activation_dict = None
        self.all_layer_expert_relevance = {}  # expert relevance filled during proportional_eval
        self.all_out_model = None             # filled by compute_outputs()
        self.all_wt = {}                      # token relevance (like PyTorch Backtrace)

        # ---- Step 1: build tree / layer_stack
        if model_type == "gpt_oss":
            self.model_resource, self.layer_stack = gpt_oss.build_gpt_oss_tree(model)
        elif model_type == "qwen3_moe":
            self.model_resource, self.layer_stack = qwen3_moe.build_qwen3_moe_tree(model)
        elif model_type == "jetmoe":
            self.model_resource, self.layer_stack = jetmoe.build_jetmoe_tree(model)
        elif model_type == "olmoe":
            self.model_resource, self.layer_stack = olmoe.build_olmoe_tree(model)

        # ---- Step 2: extract weights
        if model_type == "gpt_oss":
            self.model_weights = gpt_oss.extract_gpt_oss_weights(model)
        elif model_type == "qwen3_moe":
            self.model_weights = qwen3_moe.extract_qwen3_moe_weights(model)
        elif model_type == "jetmoe":
            self.model_weights = jetmoe.extract_jetmoe_weights(model)
        elif model_type == "olmoe":
            self.model_weights = olmoe.extract_olmoe_weights(model)

        # Registry for output creators (used by compute_outputs)
        self._create_output_fn = {
            "gpt_oss": gpt_oss.create_gpt_oss_output,
            "qwen3_moe": qwen3_moe.create_qwen3_moe_output,
            "jetmoe": jetmoe.create_jetmoe_output,
            "olmoe": olmoe.create_olmoe_output,
        }[model_type]

    # ---- Step 3 moved out of __init__
    def compute_outputs(self, *, input_text, tokenizer, max_length=None):
        """
        Compute and cache per-submodule outputs for the current model_type.
        Uses the instance device set at initialization.
        """
        self.all_out_model = self._create_output_fn(
            input_text, self.model, tokenizer, max_length, self.device
        )
        return self.all_out_model

    def _require_outputs(self):
        if self.all_out_model is None:
            raise RuntimeError(
                "Outputs have not been computed. Call `compute_outputs(...)` first "
                "or pass `all_in/all_out` explicitly to evaluation methods."
            )

    def _parse_device_to_implementation(self, device):
        """
        Parse device configuration to determine implementation version.
        """
        if device == "cpu":
            return "original"
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA device requested but CUDA not available. Falling back to CPU mode.")
            return "original"
        raise ValueError(f"device must be 'cpu' or 'cuda', got: {device}")

    def eval(
        self,
        all_in,
        all_out,
        mode="default",
        start_wt=[],
        multiplier=100.0,
        scaler=0,
        max_unit=0,
        predicted_token=None,
        thresholding=0.5,
        task="binary-classification",
    ):
        """
        Evaluate layer-wise relevance based on different modes.
        Device is taken from self.device.
        """
        if mode == "default":
            return self.proportional_eval(
                all_in=all_in,
                all_out=all_out,
                start_wt=start_wt,
                multiplier=multiplier,
                scaler=scaler,
                max_unit=max_unit,
                predicted_token=predicted_token,
                thresholding=thresholding,
                task=task,
            )

    def proportional_eval(
        self,
        all_in,
        all_out,
        start_wt=[],
        multiplier=100.0,
        scaler=0,
        max_unit=0,
        predicted_token=None,
        thresholding=0.5,
        task="binary-classification",
    ):
        # Parse device configuration to determine implementation
        device = self.device
        impl = self._parse_device_to_implementation(device)
        print(f"device: {device}, implementation: {impl}")
        
        # ---- helpers for device-aware I/O ----
        def arr_from_key(key_or_val):
            """
            Accepts:
            • str key  -> looks up in (all_in/all_out)
            • tensor/ndarray/value -> uses directly
            Returns:
            • impl=='cuda'     -> torch.Tensor on self.device
            • impl=='original' -> CPU float32 numpy array
            """
            # Resolve value
            if isinstance(key_or_val, str):
                x = get_tensor_or_raise(all_in, all_out, key_or_val)
            else:
                x = key_or_val  # already an activation/value

            # Unwrap hooks that return tuples/lists
            if isinstance(x, (tuple, list)):
                x = x[0]

            # Normalize by implementation
            if impl == "cuda":
                if torch.is_tensor(x):
                    return x.to(self.device)
                return torch.tensor(t2np32(x), dtype=torch.float32, device=self.device)
            else:  # original/CPU
                if torch.is_tensor(x):
                    return t2np32(x)
                if isinstance(x, np.ndarray):
                    return x.astype(np.float32, copy=False)
                return np.asarray(x, dtype=np.float32)

        def arr_list_from_keys(keys):
            return [arr_from_key(k) for k in keys]

        def to_np64(x):
            """
            Ensure value is a NumPy float64 array for accumulation into all_wt[].
            Accepts torch, numpy, lists/tuples of arrays.
            """
            if torch.is_tensor(x):
                return x.detach().to(torch.float64).cpu().numpy()
            if isinstance(x, np.ndarray):
                return x.astype(np.float64, copy=False)
            if isinstance(x, (list, tuple)):
                return [to_np64(xx) for xx in x]
            return np.array(x, dtype=np.float64)

        model_resource = self.model_resource
        layer_stack = self.layer_stack
        all_wts = self.model_weights
        all_wt = {}
        
        # Reset expert relevance for this step
        self.all_layer_expert_relevance = {}

        # ---- seed relevance from model output ----
        out_layer = model_resource["outputs"][0]
        out_arr = arr_from_key(out_layer)  # torch or numpy, depending on impl
        if len(start_wt) == 0:
            # UD2.calculate_start_wt expects numpy
            sw_src = out_arr if isinstance(out_arr, np.ndarray) else t2np32(out_arr)
            start_wt = UD2.calculate_start_wt(sw_src, scaler=scaler, task="generation")
        all_wt[out_layer] = start_wt * multiplier

        # ---- propagate relevance ----
        for start_layer in tqdm(layer_stack):
            if model_resource["graph"][start_layer]["child"]:
                child_nodes = model_resource["graph"][start_layer]["child"]
                for ch in child_nodes:
                    if ch not in all_wt:
                        x = get_tensor_or_raise(all_in, all_out, ch)
                        all_wt[ch] = np.zeros_like(t2np32(x), dtype=np.float64)

                node_class = model_resource["graph"][start_layer]["class"]

                if node_class == "LM_Head":
                    weights = all_wts[start_layer]
                    lm_head_weights = helper.rename_decoder_lm_head(weights)
                    x = arr_from_key(child_nodes[0])
                    temp_wt = UD2.launch_lm_head(
                        impl, all_wt[start_layer], x, lm_head_weights
                    )
                    all_wt[child_nodes[0]] += to_np64(temp_wt)

                elif node_class == "Layer_Norm":
                    all_wt[child_nodes[0]] += all_wt[start_layer]

                elif node_class == "Residual":
                    xs = arr_list_from_keys(child_nodes)
                    if impl == "cuda" and hasattr(UP, "calculate_wt_residual_cuda"):
                        temp_wt = UP.calculate_wt_residual_cuda(all_wt[start_layer], xs)
                    else:
                        xs_np = [t2np32(xx) if torch.is_tensor(xx) else xx for xx in xs]
                        temp_wt = UP.calculate_wt_residual(all_wt[start_layer], xs_np)
                    for ind, ch in enumerate(child_nodes):
                        all_wt[ch] += to_np64(temp_wt[ind])

                # -------------------- For JetMoE ---------------------
                elif node_class == "JetMoE_Feed_Forward":
                    weights = all_wts[start_layer]
                    ff_w = helper.rename_jetmoe_feed_forward_keys(weights)
                    x = arr_from_key(child_nodes[0])
                    temp_wt, ff_expert = UD2.launch_jetmoe_feed_forward(
                        impl, all_wt[start_layer], x, ff_w, self.model
                    )
                    all_wt[child_nodes[0]] += to_np64(temp_wt)
                    self.all_layer_expert_relevance[f"{start_layer}_ff_expert"] = ff_expert

                elif node_class == "JetMoE_Self_Attention":
                    weights = all_wts[start_layer]
                    sa_w = helper.rename_jetmoe_self_attention_keys(weights)
                    x = arr_from_key(child_nodes[0])
                    temp_wt, attn_expert = UD2.launch_jetmoe_self_attention(
                        impl, all_wt[start_layer], x, sa_w, self.model
                    )
                    all_wt[child_nodes[0]] += to_np64(temp_wt)
                    self.all_layer_expert_relevance[f"{start_layer}_attention_expert"] = attn_expert

                # -------------------- For OLMoE ---------------------
                elif node_class == "OLMoE_Feed_Forward":
                    weights = all_wts[start_layer]
                    ff_w = helper.rename_olmoe_feed_forward_keys(weights)
                    x = arr_from_key(all_in[child_nodes[0]])
                    temp_wt, ff_expert = UD2.launch_olmoe_feed_forward(
                        impl, all_wt[start_layer], x, ff_w, self.model
                    )
                    all_wt[child_nodes[0]] += to_np64(temp_wt)
                    self.all_layer_expert_relevance[f"{start_layer}_ff_expert"] = ff_expert

                elif node_class == "Self_Attention":
                    weights = all_wts[start_layer]
                    sa_w = helper.rename_self_attention_keys(weights)
                    config = get_model_config(self.model)
                    x = arr_from_key(all_in[child_nodes[0]])
                    temp_wt = UD2.launch_olmoe_self_attention(
                        impl, all_wt[start_layer], x, sa_w, self.model
                    )
                    all_wt[child_nodes[0]] += to_np64(temp_wt)

                # -------------------- For Qwen3-MoE ---------------------
                elif node_class == "Qwen_Feed_Forward":
                    weights = all_wts[start_layer]
                    ff_w = helper.rename_qwenmoe_feed_forward_keys(weights)
                    config = get_model_config(self.model)
                    x = arr_from_key(child_nodes[0])
                    temp_wt, ff_expert = UD2.launch_qwen3_moe_feed_forward(
                        impl, all_wt[start_layer], x, ff_w, config
                    )
                    all_wt[child_nodes[0]] += to_np64(temp_wt)
                    self.all_layer_expert_relevance[f"{start_layer}_ff_expert"] = ff_expert

                elif node_class == "Grouped_Query_Attention":
                    weights = all_wts[start_layer]
                    sa_w = helper.rename_self_attention_keys(weights)
                    config = get_model_config(self.model)
                    x = arr_from_key(child_nodes[0])
                    temp_wt = UD2.launch_qwen3_moe_self_attention(
                        impl, all_wt[start_layer], x, sa_w, config
                    )
                    all_wt[child_nodes[0]] += to_np64(temp_wt)

                # -------------------- For GPT-OSS MoE ---------------------
                elif node_class == "GPT_OSS_Feed_Forward":
                    weights = all_wts[start_layer]
                    ff_w = helper.rename_gptoss_feed_forward_keys(weights)
                    config = get_model_config(self.model)
                    x = arr_from_key(child_nodes[0])
                    temp_wt, ff_expert = UD2.launch_gpt_oss_feed_forward(
                        impl, all_wt[start_layer], x, ff_w, config
                    )
                    all_wt[child_nodes[0]] += to_np64(temp_wt)
                    self.all_layer_expert_relevance[f"{start_layer}_ff_expert"] = ff_expert

                elif node_class == "GPT_OSS_Self_Attention":
                    weights = all_wts[start_layer]
                    sa_w = helper.rename_self_attention_keys(weights)
                    config = get_model_config(self.model)
                    ATTN_PLAN = build_attention_plan(layer_stack, config)
                    attn_info = ATTN_PLAN.get(start_layer, {"attn_type": "full", "window": None})
                    x = arr_from_key(child_nodes[0])
                    temp_wt = UD2.launch_gpt_oss_self_attention(
                        impl,
                        all_wt[start_layer],
                        x,
                        sa_w,
                        config,
                        attn_type=attn_info["attn_type"],
                        sliding_window=attn_info["window"],
                    )
                    all_wt[child_nodes[0]] += to_np64(temp_wt)

                # Default passthrough
                else:
                    all_wt[child_nodes[0]] += all_wt[start_layer]

        # ---- post-scale/normalize (works on numpy) ----
        if max_unit > 0 and scaler == 0:
            temp_dict = {}
            for k in all_wt.keys():
                temp_dict[k] = UC.weight_normalize(all_wt[k], max_val=max_unit)
            all_wt = temp_dict
        elif scaler > 0:
            temp_dict = {}
            for k in all_wt.keys():
                temp_dict[k] = UC.weight_scaler(all_wt[k], scaler=scaler)
            all_wt = temp_dict

        # Store in instance variable (like PyTorch Backtrace)
        self.all_wt = all_wt

        return all_wt

    def run_task(
        self,
        task="generation",
        inputs=None,
        tokenizer=None,
        mode="default",
        multiplier=100.0,
        scaler=1.0,
        thresholding=0.5,
        temperature=1.0,
        return_scores=False,
        return_relevance=False,
        return_layerwise_output=False,
        debug=False,
        **generation_kwargs
    ):
        """
        Unified method for running MoE DL-Backtrace on generation tasks.
        
        Args:
            task (str): Task type - only "generation" is supported for MoE models
            
            inputs: Input data for the model
                - dict with 'input_ids' and 'attention_mask' or tuple of tensors
            
            tokenizer: Required for generation tasks (HuggingFace tokenizer)
            
            mode (str): Relevance propagation mode (default: "default")
            multiplier (float): Starting relevance value (default: 100.0)
            scaler (float): Relevance scaling factor (default: 1.0)
            thresholding (float): Relevance threshold (default: 0.5)
            temperature (float): Temperature for generation (default: 1.0)
            return_scores (bool): Return scores trace (default: False)
            return_relevance (bool): Return relevance trace (default: False)
            return_layerwise_output (bool): Return layer-wise output trace (default: False)
            debug (bool): Enable debug logging (default: False)
            
            **generation_kwargs: Additional kwargs for generation task
                - max_new_tokens, top_k, top_p, num_beams, etc.
        
        Returns:
            dict: Results containing:
                - 'task': Task type ("generation")
                - 'generated_ids': Generated token IDs
                - 'scores_trace': (if return_scores=True) Scores trace
                - 'relevance_trace': (if return_relevance=True) List of dicts with 'all_wt' and 'expert_relevance'
                - 'layerwise_output_trace': (if return_layerwise_output=True) Layer-wise output trace
        
        Note:
            relevance_trace structure:
            [
                {
                    'all_wt': dict,              # Token relevance for all nodes
                    'expert_relevance': dict     # Expert-specific relevance
                },
                ...  # One dict per generation step
            ]
        
        Examples:
            # Text generation with traces
            results = bt.run_task(
                task="generation",
                inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
                tokenizer=tokenizer,
                max_new_tokens=50,
                temperature=0.7,
                return_relevance=True,
                return_scores=True
            )
            
            # Access token relevance and expert relevance separately
            for step_idx, step_data in enumerate(results['relevance_trace']):
                all_wt = step_data['all_wt']
                expert_rel = step_data['expert_relevance']
                
                print(f"Step {step_idx}:")
                print(f"  Token relevance keys: {list(all_wt.keys())[:3]}")
                print(f"  Expert relevance keys: {list(expert_rel.keys())}")
        """
        
        # Validate task type
        if task != "generation":
            raise ValueError(f"MoE models only support 'generation' task, got: {task}")
        
        # Validate tokenizer
        if tokenizer is None:
            raise ValueError("tokenizer is required for generation task")
        
        # Extract input_ids and attention_mask
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
        elif isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else None
        else:
            raise ValueError("For generation, inputs must be dict or tuple of (input_ids, attention_mask)")
        
        if debug:
            print(f"🚀 Running generation task with sample_auto...")
        
        # Call sample_auto with generation kwargs and trace flags
        generated_output = self.sample_auto(
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_scores=return_scores,
            return_relevance=return_relevance,
            return_layerwise_output=return_layerwise_output,
            debug=debug,
            **generation_kwargs
        )
        
        # Parse output (can be tensor or tuple)
        result = {
            'task': task,
        }
        
        if isinstance(generated_output, tuple):
            # (generated_ids, info_dict)
            result['generated_ids'] = generated_output[0]
            info_dict = generated_output[1]
            if 'scores_trace' in info_dict:
                result['scores_trace'] = info_dict['scores_trace']
            if 'relevance_trace' in info_dict:
                result['relevance_trace'] = info_dict['relevance_trace']
            if 'layerwise_output_trace' in info_dict:
                result['layerwise_output_trace'] = info_dict['layerwise_output_trace']
        else:
            # Just generated_ids
            result['generated_ids'] = generated_output
        
        return result

    def sample_auto(self, tokenizer, input_ids, attention_mask=None, **kwargs):
        """
        Wrapper for MoE-based generation (greedy / sampling / beam).
        Accepts kwargs: temperature, top_k, top_p, max_new_tokens, min_new_tokens, max_time,
                        early_stopping, repetition_penalty, no_repeat_ngram_size,
                        bad_words_ids, bos_token_id, eos_token_id, pad_token_id,
                        num_beams, num_return_sequences, length_penalty, return_scores, 
                        return_relevance, return_layerwise_output, debug

        Returns:
            - Always a single sequence with shape [1, T_total]
            (If return_scores=True or return_relevance=True, returns (sequence, info_dict))
        """

        # --- helpers ---
        def _pick_device():
            mdl = getattr(self, "model", None)
            dev = getattr(mdl, "device", None)
            if isinstance(dev, torch.device):
                return dev
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _to_long_tensor(x, device):
            if isinstance(x, torch.Tensor):
                t = x
            elif isinstance(x, (list, tuple)):
                t = torch.tensor(x)
            else:
                try:
                    import numpy as np  # noqa: F401
                    if isinstance(x, np.ndarray):
                        t = torch.from_numpy(x)
                    else:
                        raise TypeError
                except Exception:
                    raise TypeError(
                        f"input must be Tensor/list/tuple/ndarray; got {type(x).__name__}"
                    )
            if t.dim() == 1:
                t = t.unsqueeze(0)  # ensure [B=1, T]
            return t.to(device=device, dtype=torch.long, non_blocking=True)

        def _ensure_mask(mask, ids, pad_id, device):
            if mask is None:
                if pad_id is not None:
                    m = (ids != int(pad_id)).long()
                else:
                    m = torch.ones_like(ids, dtype=torch.long, device=device)
            else:
                m = _to_long_tensor(mask, device)
                if m.shape != ids.shape:
                    raise ValueError(
                        f"attention_mask shape {tuple(m.shape)} does not match input_ids shape {tuple(ids.shape)}"
                    )
            return m

        # --- device & core tensors ---
        device = _pick_device()
        input_ids = _to_long_tensor(input_ids, device)

        # Enforce B=1 (MoE export/engine assumes batch-static B=1)
        if input_ids.size(0) != 1:
            raise AssertionError("MoE AutoSampler currently assumes batch size = 1. Provide a single prompt.")

        # --- auto-fill token IDs from tokenizer if absent ---
        for kid in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if kwargs.get(kid) is None and hasattr(tokenizer, kid):
                v = getattr(tokenizer, kid)
                if v is not None:
                    kwargs[kid] = v

        pad_id = kwargs.get("pad_token_id", None)
        attention_mask = _ensure_mask(attention_mask, input_ids, pad_id, device)

        # --- sanitize knobs / lengths ---
        temperature = kwargs.get("temperature", None)
        if temperature is not None and float(temperature) <= 0.0:
            raise ValueError("temperature must be > 0 when provided")

        top_k = kwargs.get("top_k", None)
        if top_k is not None and int(top_k) < 0:
            raise ValueError("top_k must be >= 0")

        top_p = kwargs.get("top_p", None)
        if top_p is not None and not (0.0 < float(top_p) <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

        max_new_tokens = kwargs.get("max_new_tokens", 50)
        if int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        kwargs["max_new_tokens"] = int(max_new_tokens)

        mnt = kwargs.get("min_new_tokens", None)
        if mnt is not None:
            mnt = int(mnt)
            if mnt < 0:
                raise ValueError("min_new_tokens must be >= 0")
            if mnt > max_new_tokens:
                mnt = max_new_tokens
            kwargs["min_new_tokens"] = mnt

        # --- sanitize beams ---
        num_beams = int(kwargs.get("num_beams", 1))
        if num_beams < 1:
            raise ValueError("num_beams must be >= 1")
        kwargs["num_beams"] = num_beams

        # even though the generator returns top-1 for beams, keep HF internals happy:
        nret = int(kwargs.get("num_return_sequences", 1))
        if nret < 1:
            nret = 1
        if nret > num_beams:
            nret = num_beams
        kwargs["num_return_sequences"] = nret

        # --- normalize bad_words_ids to List[List[int]] ---
        bwi = kwargs.get("bad_words_ids", None)
        if bwi is not None:
            norm = []
            if isinstance(bwi, (list, tuple)):
                for seq in bwi:
                    if isinstance(seq, (list, tuple)):
                        norm.append([int(t) for t in seq])
                    else:
                        norm.append([int(seq)])
            else:
                norm.append([int(bwi)])
            kwargs["bad_words_ids"] = norm

        # --- create engine & dispatch ---
        from dl_backtrace.moe_pytorch_backtrace.backtrace.core.moe_auto_sampler import MoEAutoSampler
        eng = MoEAutoSampler(self, tokenizer)  # `self` is the MoE Backtrace engine

        return eng.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )