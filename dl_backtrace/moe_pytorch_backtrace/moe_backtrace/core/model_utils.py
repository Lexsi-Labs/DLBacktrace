"""
Shared utilities for MoE model unwrapping, hook setup, and tree building.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Model unwrapping
# ---------------------------------------------------------------------------

def unwrap_model(model, max_depth=8):
    """
    Unwrap a HuggingFace model to its core transformer + lm_head.

    Returns:
        tuple: (core_model, lm_head)
            - core_model: transformer body with .layers, .embed_tokens, .norm
            - lm_head: language model head, or None
    """
    obj = model
    seen = set()
    for depth in range(max_depth):
        if _is_transformer_core(obj):
            lm_head = _find_lm_head(model, obj)
            return obj, lm_head
        obj_id = id(obj)
        if obj_id in seen:
            break
        seen.add(obj_id)
        if hasattr(obj, "model"):
            obj = obj.model
        else:
            break
    raise AttributeError(
        f"Could not find HuggingFace transformer structure in {type(model).__name__}. "
        f"Expected attributes: .layers, .embed_tokens, .norm. "
        f"Searched {depth + 1} levels deep."
    )


def _is_transformer_core(obj):
    return (
        hasattr(obj, "layers") and
        hasattr(obj, "embed_tokens") and
        hasattr(obj, "norm")
    )


def _find_lm_head(root_model, core_model):
    if hasattr(root_model, "lm_head"):
        return root_model.lm_head
    obj = root_model
    seen = set()
    for _ in range(8):
        if id(obj) in seen:
            break
        seen.add(id(obj))
        if hasattr(obj, "lm_head"):
            return obj.lm_head
        if obj is core_model:
            break
        if hasattr(obj, "model"):
            obj = obj.model
        else:
            break
    for attr_name in ["output", "lm_head", "head", "classifier"]:
        if hasattr(root_model, attr_name):
            return getattr(root_model, attr_name)
    return None


def get_model_structure_info(model):
    """Return a dict describing model structure (for debugging)."""
    try:
        core, lm_head = unwrap_model(model)
        return {
            "success": True,
            "core_type": type(core).__name__,
            "lm_head_type": type(lm_head).__name__ if lm_head else None,
            "num_layers": len(core.layers) if hasattr(core, "layers") else None,
            "has_embed_tokens": hasattr(core, "embed_tokens"),
            "has_norm": hasattr(core, "norm"),
        }
    except AttributeError as e:
        return {
            "success": False,
            "error": str(e),
            "model_type": type(model).__name__,
            "has_model_attr": hasattr(model, "model"),
        }


# ---------------------------------------------------------------------------
# Hook helpers (shared across all model-specific output-capture functions)
# ---------------------------------------------------------------------------

def clear_all_hooks(mod):
    """Hard-reset all forward hooks on a module and its submodules."""
    for m in mod.modules():
        if hasattr(m, "_forward_hooks"):
            m._forward_hooks.clear()
        if hasattr(m, "_forward_pre_hooks"):
            m._forward_pre_hooks.clear()
        if hasattr(m, "_forward_hooks_with_kwargs"):
            m._forward_hooks_with_kwargs.clear()
        if hasattr(m, "_forward_pre_hooks_with_kwargs"):
            m._forward_pre_hooks_with_kwargs.clear()


def first_tensor(args, kwargs, prefer_key=None):
    """Return the first tensor found in args/kwargs, with optional key preference."""
    if prefer_key is not None and isinstance(kwargs, dict):
        v = kwargs.get(prefer_key)
        if torch.is_tensor(v):
            return v
    if isinstance(args, (tuple, list)) and args and torch.is_tensor(args[0]):
        return args[0]
    if isinstance(kwargs, dict):
        for v in kwargs.values():
            if torch.is_tensor(v):
                return v
            if isinstance(v, (tuple, list)):
                for it in v:
                    if torch.is_tensor(it):
                        return it
    return None


def clone(x):
    """Detach-clone a tensor, or pass through non-tensors."""
    return x.detach().clone() if torch.is_tensor(x) else x


def get_tensor(d, key):
    """Safely get a tensor from dict d at key; unwrap single-element tuples/lists."""
    if d is None or key is None:
        return None
    x = d.get(key)
    if isinstance(x, (tuple, list)):
        x = x[0]
    if torch.is_tensor(x) or isinstance(x, np.ndarray):
        return x
    return None


def to_torch_like(x, ref):
    """Cast x to a tensor with the same dtype/device as ref."""
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
    return None


# ---------------------------------------------------------------------------
# Tree builder (shared across all model-specific build_*_tree functions)
# ---------------------------------------------------------------------------

def build_decoder_tree(core, lm_head, *, attn_class: str, ff_class: str):
    """
    Build the relevance-propagation graph for any standard decoder-only MoE model.

    Args:
        core:       Transformer body (has .layers, .embed_tokens, .norm)
        lm_head:    LM head module (or None)
        attn_class: Node class string for the self-attention node
                    e.g. "Self_Attention", "GPT_OSS_Self_Attention", "Grouped_Query_Attention"
        ff_class:   Node class string for the feed-forward node
                    e.g. "OLMoE_Feed_Forward", "GPT_OSS_Feed_Forward", "Qwen_Feed_Forward"

    Returns:
        (model_resource dict, layer_stack list)
    """
    ltree = {}
    layer_tree = {}
    inputs = []
    outputs = []
    intermediates = []
    layer_stack = []

    def add_component(ltree, name, component, child=None):
        ltree[name] = {
            "name": name,
            "class": component if isinstance(component, str) else type(component).__name__,
            "type": str(type(component)),
            "parent": None,
            "child": None,
        }
        if isinstance(child, list):
            ltree[name]["child"] = child
        elif isinstance(child, str):
            ltree[name]["child"] = [child]

        if ltree[name]["class"] == "list":
            ltree[name]["class"] = [type(item).__name__ for item in component]
            ltree[name]["type"] = [str(type(item)) for item in component]

        layer_tree[name] = component if isinstance(component, str) else ltree[name]["type"]
        layer_stack.append(name)

        if isinstance(child, list):
            for ch in child:
                if ch in ltree:
                    ltree[ch]["parent"] = [name]
        elif isinstance(child, str):
            if child in ltree:
                ltree[child]["parent"] = [name]

    decoder_embeddings = add_component(ltree, 'decoder_embeddings', 'Embeddings', child=None)

    # Add jet_moe layers dynamically
    current_child = 'decoder_embeddings'
    for i, layer in enumerate(core.layers):
        decoder_layer_norm_0 = add_component(ltree, f'decoder_layer_norm_{i}_0', 'Layer_Norm', child=current_child)
        decoder_self_attention = add_component(ltree, f'decoder_self_attention_{i}', 'Self_Attention', child=f'decoder_layer_norm_{i}_0')
        decoder_residual_self_attention = add_component(ltree, f'decoder_residual_self_attention_{i}', 'Residual', child=[current_child, f'decoder_self_attention_{i}'])

        decoder_layer_norm_1 = add_component(ltree, f'decoder_layer_norm_{i}_1', 'Layer_Norm', child=f'decoder_self_attention_{i}')
        decoder_feed_forward = add_component(ltree, f'decoder_feed_forward_{i}', 'OLMoE_Feed_Forward', child=f'decoder_layer_norm_{i}_1')
        decoder_residual_feed_forward = add_component(ltree, f'decoder_residual_feed_forward_{i}', 'Residual', child=[f'decoder_residual_self_attention_{i}', f'decoder_feed_forward_{i}'])

        current_child = f'decoder_residual_feed_forward_{i}'

    if hasattr(core, 'norm'):
        decoder_final_layer_norm = add_component(ltree, 'decoder_layer_norm', 'Layer_Norm', child=current_child)
        current_child = 'decoder_layer_norm'

    # Decoder LM-Head
    if lm_head is not None:
        decoder_lm_head = add_component(ltree, 'decoder_lm_head', 'LM_Head', child=current_child)
        current_child = 'decoder_lm_head'

    # Classify components
    for name, component in ltree.items():
        if component['parent'] is None:
            outputs.append(component['name'])
        elif component['child'] is None:
            inputs.append(component['name'])
        else:
            intermediates.append(component['name'])

    # reverse the layer_stack
    layer_stack = list(reversed(layer_stack))

    # model_resource = (layer_tree, ltree, outputs, inputs)
    model_resource = {
        "layers": layer_tree,
        "graph": ltree,
        "outputs": outputs,
        "inputs": inputs
    } 

    return model_resource, layer_stack


# ---------------------------------------------------------------------------
# Shared output-capture generation loop
# ---------------------------------------------------------------------------

def run_hook_generation(model, core, lm_head, tokenizer, input_text, max_length, device,
                        decoder_outputs, decoder_inputs, decoder_hooks):
    """
    Run a single-step-per-forward-pass generation loop with hooks already registered.
    Removes all hooks in a finally block even on error.

    Args:
        model:           The full model (used for forward pass)
        core:            Unwrapped transformer core (for config fallback)
        lm_head:         LM head module (unused here, kept for symmetry)
        tokenizer:       HF tokenizer
        input_text:      Prompt text string
        max_length:      Number of tokens to generate (1 = single step)
        device:          Torch device string
        decoder_outputs: defaultdict(dict) to write output activations into
        decoder_inputs:  defaultdict(dict) to write input activations into
        decoder_hooks:   List of registered hook handles to remove on exit

    Returns:
        (decoder_outputs, decoder_inputs, generated_tokens)
    """
    enc = tokenizer(input_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    model = model.to(device)

    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings", 2048)

    B = input_ids.shape[0]
    generated_tokens = [[] for _ in range(B)]

    try:
        for _ in range(max_length):
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            for b in range(B):
                generated_tokens[b].append(next_token_id[b].item())
                eos = getattr(model.config, "eos_token_id", None)
                if eos is not None and next_token_id[b].item() == eos:
                    break
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    finally:
        for h in decoder_hooks:
            try:
                h.remove()
            except Exception:
                pass

    return decoder_outputs, decoder_inputs, generated_tokens