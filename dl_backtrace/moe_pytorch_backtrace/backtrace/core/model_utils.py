"""
Shared utilities for MoE model unwrapping, hook setup, tree building,
and output capture.

All model-specific files (olmoe, jetmoe, gpt_oss, qwen3_moe) delegate
to these shared functions, passing only the model-specific differences
(class names, attribute names, weight routing).
"""

import numpy as np
import torch
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════
#  Model unwrapping
# ═══════════════════════════════════════════════════════════════════════

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
    """Check if an object has the standard HF transformer core structure."""
    return (
        hasattr(obj, "layers") and
        hasattr(obj, "embed_tokens") and
        hasattr(obj, "norm")
    )


def _find_lm_head(root_model, core_model):
    """Find the lm_head by searching from root_model down to core_model."""
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


# ═══════════════════════════════════════════════════════════════════════
#  Hook helpers
# ═══════════════════════════════════════════════════════════════════════

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
    """Detach a tensor without copying (zero-copy view).
    
    Under torch.no_grad() generation, in-place modifications don't happen,
    so a full clone is unnecessary. detach() is sufficient and saves memory.
    """
    return x.detach() if torch.is_tensor(x) else x


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


def to_numpy(param):
    """Return a lightweight detached reference to the parameter tensor.
    
    Previously converted to float32 numpy eagerly, duplicating the entire model.
    Now returns a detached tensor view (no memory copy). Downstream consumers
    call t2np32() or .numpy() only when they actually need the data.
    """
    if isinstance(param, torch.Tensor):
        return param.detach()
    return param


# ═══════════════════════════════════════════════════════════════════════
#  Shared tree builder
# ═══════════════════════════════════════════════════════════════════════

def build_decoder_tree(core, lm_head, *, attn_class, ff_class):
    """
    Build the relevance-propagation graph for any standard decoder-only MoE model.

    Args:
        core:        Transformer body (has .layers, .embed_tokens, .norm)
        lm_head:     LM head module (or None)
        attn_class:  Node class string for the self-attention node
                     e.g. "Self_Attention", "JetMoE_Self_Attention",
                     "GPT_OSS_Self_Attention", "Grouped_Query_Attention"
        ff_class:    Node class string for the feed-forward node
                     e.g. "OLMoE_Feed_Forward", "JetMoE_Feed_Forward",
                     "GPT_OSS_Feed_Forward", "Qwen_Feed_Forward"

    Returns:
        (model_resource dict, layer_stack list)
    """
    ltree = {}
    layer_tree = {}
    inputs = []
    outputs = []
    intermediates = []
    layer_stack = []

    def add_component(tree, name, component, child=None):
        tree[name] = {
            'name': name,
            'class': component if isinstance(component, str) else type(component).__name__,
            'type': str(type(component)),
            'parent': None,
            'child': None,
        }

        if isinstance(child, list):
            tree[name]['child'] = child
        elif isinstance(child, str):
            tree[name]['child'] = [child]

        if tree[name]['class'] == 'list':
            tree[name]['class'] = [type(item).__name__ for item in component]
            tree[name]['type'] = [str(type(item)) for item in component]

        layer_tree[name] = component if isinstance(component, str) else tree[name]['type']
        layer_stack.append(name)

        if isinstance(child, list):
            for ch in child:
                if ch in tree:
                    tree[ch]['parent'] = [name]
        elif isinstance(child, str):
            if child in tree:
                tree[child]['parent'] = [name]

        return tree[name]

    # Embeddings
    add_component(ltree, 'decoder_embeddings', 'Embeddings', child=None)

    # Transformer layers
    current_child = 'decoder_embeddings'
    for i, layer in enumerate(core.layers):
        add_component(ltree, f'decoder_layer_norm_{i}_0', 'Layer_Norm', child=current_child)
        add_component(ltree, f'decoder_self_attention_{i}', attn_class, child=f'decoder_layer_norm_{i}_0')
        add_component(ltree, f'decoder_residual_self_attention_{i}', 'Residual',
                       child=[current_child, f'decoder_self_attention_{i}'])

        add_component(ltree, f'decoder_layer_norm_{i}_1', 'Layer_Norm', child=f'decoder_self_attention_{i}')
        add_component(ltree, f'decoder_feed_forward_{i}', ff_class, child=f'decoder_layer_norm_{i}_1')
        add_component(ltree, f'decoder_residual_feed_forward_{i}', 'Residual',
                       child=[f'decoder_residual_self_attention_{i}', f'decoder_feed_forward_{i}'])

        current_child = f'decoder_residual_feed_forward_{i}'

    # Final layer norm
    if hasattr(core, 'norm'):
        add_component(ltree, 'decoder_layer_norm', 'Layer_Norm', child=current_child)
        current_child = 'decoder_layer_norm'

    # LM Head
    if lm_head is not None:
        add_component(ltree, 'decoder_lm_head', 'LM_Head', child=current_child)

    # Classify components
    for name, component in ltree.items():
        if component['parent'] is None:
            outputs.append(component['name'])
        elif component['child'] is None:
            inputs.append(component['name'])
        else:
            intermediates.append(component['name'])

    layer_stack_reversed = list(reversed(layer_stack))

    model_resource = {
        "layers": layer_tree,
        "graph": ltree,
        "outputs": outputs,
        "inputs": inputs,
    }

    return model_resource, layer_stack_reversed


# ═══════════════════════════════════════════════════════════════════════
#  Shared output-capture (hooks + generation loop)
# ═══════════════════════════════════════════════════════════════════════

def create_decoder_output(input_text, model, tokenizer, max_length, device,
                          *, attn_attr='self_attn', input_ids=None,
                          attention_mask=None):
    """
    Run greedy decoding while capturing inputs and outputs for key submodules
    at each decoding step. Works for any decoder-only MoE architecture.

    Args:
        input_text:  Prompt string
        model:       Full HF model (e.g. OlmoeForCausalLM)
        tokenizer:   HF tokenizer
        max_length:  Number of decoding steps (None = use model default)
        device:      Torch device string ("cpu" or "cuda")
        attn_attr:   Name of the attention submodule on each layer.
                     "self_attn" for OLMoE/GPT-OSS/Qwen3, "self_attention" for JetMoE.

    Returns:
        (decoder_outputs, decoder_inputs, generated_tokens)
    """
    core, lm_head = unwrap_model(model)

    # Clear stale hooks
    clear_all_hooks(core)
    if lm_head is not None:
        clear_all_hooks(lm_head)

    token_idx = 0
    decoder_outputs = defaultdict(dict)
    decoder_inputs = defaultdict(dict)
    decoder_hooks = []

    def ts():
        return str(token_idx)

    # ── Hook factories ─────────────────────────────────────────────────

    def pre_emb(m, a, kw):
        x = first_tensor(a, kw, prefer_key='input_ids')
        if x is not None:
            decoder_inputs[ts()]['decoder_embeddings'] = clone(x)

    def post_emb(m, a, kw, out):
        decoder_outputs[ts()]['decoder_embeddings'] = clone(out)

    def pre_ln0(i):
        key = f'decoder_layer_norm_{i}_0'
        def _h(m, a, kw):
            x = first_tensor(a, kw, prefer_key='hidden_states')
            if x is not None:
                decoder_inputs[ts()][key] = clone(x)
        return _h

    def post_ln0(i):
        key = f'decoder_layer_norm_{i}_0'
        def _h(m, a, kw, out):
            decoder_outputs[ts()][key] = clone(out)
        return _h

    def pre_attn(i):
        key = f'decoder_self_attention_{i}'
        def _h(m, a, kw):
            x = first_tensor(a, kw, prefer_key='hidden_states')
            if x is not None:
                decoder_inputs[ts()][key] = clone(x)
        return _h

    def post_attn(i):
        key = f'decoder_self_attention_{i}'
        def _h(m, a, kw, out):
            out0 = out[0] if isinstance(out, (tuple, list)) else out
            decoder_outputs[ts()][key] = clone(out0)
        return _h

    def pre_ln1(i):
        key = f'decoder_layer_norm_{i}_1'
        def _h(m, a, kw):
            x = first_tensor(a, kw, prefer_key='hidden_states')
            if x is not None:
                decoder_inputs[ts()][key] = clone(x)
        return _h

    def post_ln1(i):
        key = f'decoder_layer_norm_{i}_1'
        def _h(m, a, kw, out):
            decoder_outputs[ts()][key] = clone(out)
        return _h

    def pre_mlp(i):
        key = f'decoder_feed_forward_{i}'
        def _h(m, a, kw):
            x = first_tensor(a, kw, prefer_key='hidden_states')
            if x is not None:
                decoder_inputs[ts()][key] = clone(x)
        return _h

    def post_mlp(i):
        key = f'decoder_feed_forward_{i}'
        def _h(m, a, kw, out):
            out0 = out[0] if isinstance(out, (tuple, list)) else out
            decoder_outputs[ts()][key] = clone(out0)
        return _h

    def post_residual_attn(i):
        key = f"decoder_residual_self_attention_{i}"
        ln0_key = f"decoder_layer_norm_{i}_0"
        att_key = f"decoder_self_attention_{i}"
        def _h(m, a, kw, out):
            t = ts()
            branch = out[0] if isinstance(out, (tuple, list)) else out
            skip = get_tensor(decoder_outputs[t], ln0_key)
            if skip is None:
                skip = get_tensor(decoder_inputs[t], att_key)
            if skip is None:
                skip = torch.zeros_like(branch)
            skip = to_torch_like(skip, branch)
            decoder_outputs[t][key] = clone(skip + branch)
        return _h

    def post_residual_mlp(i):
        key = f"decoder_residual_feed_forward_{i}"
        ln1_key = f"decoder_layer_norm_{i}_1"
        mlp_key = f"decoder_feed_forward_{i}"
        def _h(m, a, kw, out):
            t = ts()
            branch = out[0] if isinstance(out, (tuple, list)) else out
            skip = get_tensor(decoder_outputs[t], ln1_key)
            if skip is None:
                skip = get_tensor(decoder_inputs[t], mlp_key)
            if skip is None:
                skip = torch.zeros_like(branch)
            skip = to_torch_like(skip, branch)
            decoder_outputs[t][key] = clone(skip + branch)
        return _h

    def pre_final_ln(m, a, kw):
        x = first_tensor(a, kw, prefer_key='hidden_states')
        if x is not None:
            decoder_inputs[ts()]['decoder_layer_norm'] = clone(x)

    def post_final_ln(m, a, kw, out):
        decoder_outputs[ts()]['decoder_layer_norm'] = clone(out)

    def pre_lm(m, a, kw):
        x = first_tensor(a, kw, prefer_key='hidden_states')
        if x is None and isinstance(a, (tuple, list)) and a and torch.is_tensor(a[0]):
            x = a[0]
        if x is not None:
            decoder_inputs[ts()]['decoder_lm_head'] = clone(x)

    def post_lm(m, a, kw, out):
        decoder_outputs[ts()]['decoder_lm_head'] = clone(out)

    # ── Register hooks ────────────────────────────────────────────────

    decoder_hooks.append(core.embed_tokens.register_forward_pre_hook(pre_emb, with_kwargs=True))
    decoder_hooks.append(core.embed_tokens.register_forward_hook(post_emb, with_kwargs=True))

    for i, layer in enumerate(core.layers):
        decoder_hooks.append(layer.input_layernorm.register_forward_pre_hook(pre_ln0(i), with_kwargs=True))
        decoder_hooks.append(layer.input_layernorm.register_forward_hook(post_ln0(i), with_kwargs=True))

        # Use the model-specific attention attribute name
        attn_module = getattr(layer, attn_attr)
        decoder_hooks.append(attn_module.register_forward_pre_hook(pre_attn(i), with_kwargs=True))
        decoder_hooks.append(attn_module.register_forward_hook(post_attn(i), with_kwargs=True))
        decoder_hooks.append(attn_module.register_forward_hook(post_residual_attn(i), with_kwargs=True))

        decoder_hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(pre_ln1(i), with_kwargs=True))
        decoder_hooks.append(layer.post_attention_layernorm.register_forward_hook(post_ln1(i), with_kwargs=True))

        decoder_hooks.append(layer.mlp.register_forward_pre_hook(pre_mlp(i), with_kwargs=True))
        decoder_hooks.append(layer.mlp.register_forward_hook(post_mlp(i), with_kwargs=True))
        decoder_hooks.append(layer.mlp.register_forward_hook(post_residual_mlp(i), with_kwargs=True))

    decoder_hooks.append(core.norm.register_forward_pre_hook(pre_final_ln, with_kwargs=True))
    decoder_hooks.append(core.norm.register_forward_hook(post_final_ln, with_kwargs=True))
    if lm_head is not None:
        decoder_hooks.append(lm_head.register_forward_pre_hook(pre_lm, with_kwargs=True))
        decoder_hooks.append(lm_head.register_forward_hook(post_lm, with_kwargs=True))

    # ── Generation loop ───────────────────────────────────────────────

    def tick():
        nonlocal token_idx
        token_idx += 1

    if input_ids is None:
        enc = tokenizer(input_text, return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", attention_mask)

    input_ids = input_ids.detach().to(device=device, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    if attention_mask is not None:
        attention_mask = attention_mask.detach().to(device=device, dtype=torch.long)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
    model = model.to(device)

    token_idx = 0
    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings", 2048)

    B = input_ids.shape[0]
    generated_tokens = [[] for _ in range(B)]
    eos_id = getattr(model.config, "eos_token_id", None)

    try:
        with torch.no_grad():
            for _ in range(max_length):
                model_kwargs = {"input_ids": input_ids}
                if attention_mask is not None:
                    model_kwargs["attention_mask"] = attention_mask
                outputs = model(**model_kwargs)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
                # Free model intermediates immediately
                del outputs
                for b in range(B):
                    generated_tokens[b].append(next_token_id[b].item())
                    if eos_id is not None and next_token_id[b].item() == eos_id:
                        break
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (attention_mask.shape[0], 1),
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                        ],
                        dim=-1,
                    )
                tick()
    finally:
        for h in decoder_hooks:
            try:
                h.remove()
            except Exception:
                pass

    return decoder_outputs, decoder_inputs, generated_tokens
