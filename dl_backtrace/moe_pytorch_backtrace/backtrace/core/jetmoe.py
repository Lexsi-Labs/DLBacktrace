import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from .model_utils import unwrap_model


def build_jetmoe_tree(model, root='jet_moe'):
    """Build tree for JetMoE model (supports wrapped models)."""
    
    # Unwrap the model to find the core transformer
    core, lm_head = unwrap_model(model)
    
    # Initialize the tree structure
    ltree = {}
    layer_tree = {}
    inputs = []
    outputs = []
    intermediates = []
    layer_stack = []

    # Base component setup
    def add_component(tree, name, component, child=None):
        tree[name] = {
            'name': name,
            'class': component if type(component).__name__ == 'str' else type(component).__name__,
            'type': str(type(component)),
            'parent': None,
            'child': None
        }

        if isinstance(child, list):
            tree[name]['child'] = child
        elif isinstance(child, str):
            tree[name]['child'] = [child]

        if tree[name]['class'] == 'list':
            tree[name]['class'] = [type(item).__name__ for item in component]
            tree[name]['type'] = [str(type(item)) for item in component]

        # # Keep track of component type in a separate dictionary
        layer_tree[name] = component if type(component).__name__ == 'str' else tree[name]['type']

        # keep track of layer stack
        layer_stack.append(name)

        # Link the parent to its children
        if isinstance(child, list):
            for ch in child:
                if ch in tree:
                    tree[ch]['parent'] = [name]

        elif isinstance(child, str):
            if child in tree:
                tree[child]['parent'] = [name]

        return tree[name]

    # Add root and embeddings component
    # add_component(ltree, root, model, parent=None)
    decoder_embeddings = add_component(ltree, 'decoder_embeddings', 'Embeddings', child=None)

    # Add jet_moe layers dynamically
    current_child = 'decoder_embeddings'
    for i, layer in enumerate(core.layers):
        decoder_layer_norm_0 = add_component(ltree, f'decoder_layer_norm_{i}_0', 'Layer_Norm', child=current_child)
        decoder_self_attention = add_component(ltree, f'decoder_self_attention_{i}', 'JetMoE_Self_Attention', child=f'decoder_layer_norm_{i}_0')
        decoder_residual_self_attention = add_component(ltree, f'decoder_residual_self_attention_{i}', 'Residual', child=[current_child, f'decoder_self_attention_{i}'])

        decoder_layer_norm_1 = add_component(ltree, f'decoder_layer_norm_{i}_1', 'Layer_Norm', child=f'decoder_self_attention_{i}')
        decoder_feed_forward = add_component(ltree, f'decoder_feed_forward_{i}', 'JetMoE_Feed_Forward', child=f'decoder_layer_norm_{i}_1')
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


def extract_jetmoe_weights(model):
    """Extract weights from JetMoE model (supports wrapped models)."""
    
    # Unwrap the model
    core, lm_head = unwrap_model(model)
    
    # Get config from the core model
    config = core.config if hasattr(core, 'config') else model.config
    
    # Initialize a dictionary to hold the weights
    weights_dict = {
        'decoder_embeddings': {},
        'decoder_layer_norm': {},
        'decoder_lm_head': {}
    }

    # Init per-layer dicts
    for i in range(config.num_hidden_layers):
        weights_dict[f'decoder_layer_norm_{i}_0'] = {}
        weights_dict[f'decoder_self_attention_{i}'] = {}
        weights_dict[f'decoder_layer_norm_{i}_1'] = {}
        weights_dict[f'decoder_feed_forward_{i}'] = {}

    # helper: safe tensor → numpy
    def to_np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else x

    # Extract parameters
    for name, param in model.named_parameters():
        param_np = to_np(param)

        if 'embed_tokens' in name:
            weights_dict['decoder_embeddings'][name] = param_np

        elif 'layers' in name:
            layer = name.split('.')[2]
            if 'input_layernorm' in name:
                weights_dict[f'decoder_layer_norm_{layer}_0'][name] = param_np
            elif 'self_attention' in name:
                weights_dict[f'decoder_self_attention_{layer}'][name] = param_np
            elif 'post_attention_layernorm' in name:
                weights_dict[f'decoder_layer_norm_{layer}_1'][name] = param_np
            elif 'mlp' in name:
                weights_dict[f'decoder_feed_forward_{layer}'][name] = param_np

        elif 'norm.weight' in name:
            weights_dict['decoder_layer_norm']['norm.weight'] = param_np

    if lm_head is not None and hasattr(lm_head, 'weight'):
        weights_dict['decoder_lm_head']['lm_head.weight'] = to_np(lm_head.weight.data)

    return weights_dict


# ----------------------- Public API: patch installer ----------------------- #
def install_jetmoe_rope_patch(force: bool = False) -> None:
    """
    Idempotently installs a RoPE monkey-patch for JetMoE on transformers==4.52.x.

    Rotates only the first `rotary_dim` dims when head_dim > rotary_dim,
    preventing shape mismatches in attention.

    Args:
        force: re-apply the patch even if it appears installed already.
    """
    from transformers.models.jetmoe import modeling_jetmoe as jtm

    # If already patched, skip unless forced
    if getattr(jtm, "_rope_partial_patch_installed", False) and not force:
        return

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb_partial(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rotary_dim = cos.size(-1)

        # Full-rotation case
        if q.size(-1) == rotary_dim:
            q_rot = (q * cos) + (_rotate_half(q) * sin)
            k_rot = (k * cos) + (_rotate_half(k) * sin)
            return q_rot, k_rot

        # Partial-rotation case
        q_head, q_tail = q[..., :rotary_dim], q[..., rotary_dim:]
        k_head, k_tail = k[..., :rotary_dim], k[..., rotary_dim:]
        q_head = (q_head * cos) + (_rotate_half(q_head) * sin)
        k_head = (k_head * cos) + (_rotate_half(k_head) * sin)
        return torch.cat([q_head, q_tail], dim=-1), torch.cat([k_head, k_tail], dim=-1)

    # Monkey-patch
    jtm.apply_rotary_pos_emb = apply_rotary_pos_emb_partial
    jtm._rope_partial_patch_installed = True


def create_jetmoe_output(
    input_text: str,
    model: Any,
    tokenizer: Any,
    max_length: Optional[int],
    device: str,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]], List[List[int]]]:
    """
    Run greedy decoding on JetMoE while capturing BOTH inputs and outputs
    for key submodules at each decoding step (supports wrapped models).

    Returns:
        decoder_outputs: {token_idx(str): {name: tensor}}
        decoder_inputs:  {token_idx(str): {name: tensor}}
        generated_tokens: List[List[int]]  (per-batch token ids)
    """
    # Unwrap the model
    core, lm_head = unwrap_model(model)

    # ---- hard-reset any existing hooks so stale ones don't fire ----
    def _clear_all_hooks(mod):
        for m in mod.modules():
            if hasattr(m, "_forward_hooks"): m._forward_hooks.clear()
            if hasattr(m, "_forward_pre_hooks"): m._forward_pre_hooks.clear()
            if hasattr(m, "_forward_hooks_with_kwargs"): m._forward_hooks_with_kwargs.clear()
            if hasattr(m, "_forward_pre_hooks_with_kwargs"): m._forward_pre_hooks_with_kwargs.clear()

    _clear_all_hooks(core)
    if lm_head is not None:
        _clear_all_hooks(lm_head)

    token_idx = 0
    decoder_outputs: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    decoder_inputs:  Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    decoder_hooks = []

    def ts() -> str:
        return str(token_idx)

    # ---- helpers ----
    def first_tensor(args, kwargs, prefer_key=None):
        if prefer_key is not None and isinstance(kwargs, dict):
            v = kwargs.get(prefer_key, None)
            if torch.is_tensor(v): return v
        if isinstance(args, (tuple, list)) and len(args) and torch.is_tensor(args[0]):
            return args[0]
        if isinstance(kwargs, dict):
            for v in kwargs.values():
                if torch.is_tensor(v): return v
                if isinstance(v, (tuple, list)):
                    for it in v:
                        if torch.is_tensor(it): return it
        return None

    def clone(x):
        return x.detach().clone() if torch.is_tensor(x) else x

    def get_tensor(d, key):
        if d is None or key is None: return None
        x = d.get(key, None)
        if isinstance(x, (tuple, list)): x = x[0]
        if torch.is_tensor(x) or isinstance(x, np.ndarray): return x
        return None

    def to_torch_like(x, ref):
        if torch.is_tensor(x): return x
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
        return None

    # ---- hooks (JetMoE names: embed_tokens, layers[i].self_attention, etc.) ----
    def pre_emb(m, a, kw):
        x = first_tensor(a, kw, prefer_key="input_ids")
        if x is not None: decoder_inputs[ts()]["decoder_embeddings"] = clone(x)

    def post_emb(m, a, kw, out):
        decoder_outputs[ts()]["decoder_embeddings"] = clone(out)

    def pre_ln0(i):
        key = f"decoder_layer_norm_{i}_0"
        def _h(m, a, kw):
            x = first_tensor(a, kw, prefer_key="hidden_states")
            if x is not None: decoder_inputs[ts()][key] = clone(x)
        return _h

    def post_ln0(i):
        key = f"decoder_layer_norm_{i}_0"
        def _h(m, a, kw, out):
            decoder_outputs[ts()][key] = clone(out)
        return _h

    def pre_attn(i):
        key = f"decoder_self_attention_{i}"
        def _h(m, a, kw):
            x = first_tensor(a, kw, prefer_key="hidden_states")
            if x is not None: decoder_inputs[ts()][key] = clone(x)
        return _h

    def post_attn(i):
        key = f"decoder_self_attention_{i}"
        def _h(m, a, kw, out):
            out0 = out[0] if isinstance(out, (tuple, list)) else out
            decoder_outputs[ts()][key] = clone(out0)
        return _h

    def pre_ln1(i):
        key = f"decoder_layer_norm_{i}_1"
        def _h(m, a, kw):
            x = first_tensor(a, kw, prefer_key="hidden_states")
            if x is not None: decoder_inputs[ts()][key] = clone(x)
        return _h

    def post_ln1(i):
        key = f"decoder_layer_norm_{i}_1"
        def _h(m, a, kw, out):
            decoder_outputs[ts()][key] = clone(out)
        return _h

    def pre_mlp(i):
        key = f"decoder_feed_forward_{i}"
        def _h(m, a, kw):
            x = first_tensor(a, kw, prefer_key="hidden_states")
            if x is not None: decoder_inputs[ts()][key] = clone(x)
        return _h

    def post_mlp(i):
        key = f"decoder_feed_forward_{i}"
        def _h(m, a, kw, out):
            out0 = out[0] if isinstance(out, (tuple, list)) else out
            decoder_outputs[ts()][key] = clone(out0)
        return _h

    # Residual captures
    def post_residual_attn(i):
        key     = f"decoder_residual_self_attention_{i}"
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
        key     = f"decoder_residual_feed_forward_{i}"
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
        x = first_tensor(a, kw, prefer_key="hidden_states")
        if x is not None: decoder_inputs[ts()]["decoder_layer_norm"] = clone(x)

    def post_final_ln(m, a, kw, out):
        decoder_outputs[ts()]["decoder_layer_norm"] = clone(out)

    def pre_lm(m, a, kw):
        x = first_tensor(a, kw, prefer_key="hidden_states")
        if x is None and isinstance(a, (tuple, list)) and len(a) and torch.is_tensor(a[0]):
            x = a[0]
        if x is not None: decoder_inputs[ts()]["decoder_lm_head"] = clone(x)

    def post_lm(m, a, kw, out):
        decoder_outputs[ts()]["decoder_lm_head"] = clone(out)

    # ---- register hooks ----
    decoder_hooks.append(core.embed_tokens.register_forward_pre_hook(pre_emb, with_kwargs=True))
    decoder_hooks.append(core.embed_tokens.register_forward_hook(post_emb, with_kwargs=True))

    for i, layer in enumerate(core.layers):
        decoder_hooks.append(layer.input_layernorm.register_forward_pre_hook(pre_ln0(i), with_kwargs=True))
        decoder_hooks.append(layer.input_layernorm.register_forward_hook(post_ln0(i), with_kwargs=True))

        # JetMoE uses 'self_attention'
        decoder_hooks.append(layer.self_attention.register_forward_pre_hook(pre_attn(i), with_kwargs=True))
        decoder_hooks.append(layer.self_attention.register_forward_hook(post_attn(i), with_kwargs=True))
        decoder_hooks.append(layer.self_attention.register_forward_hook(post_residual_attn(i), with_kwargs=True))

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

    # ---- generation (greedy) ----
    def tick():
        nonlocal token_idx
        token_idx += 1

    enc = tokenizer(input_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    model = model.to(device).eval()

    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings", 2048)

    B = input_ids.shape[0]
    generated_tokens: List[List[int]] = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    eos_id = getattr(model.config, "eos_token_id", None)

    try:
        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(input_ids=input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

                for b in range(B):
                    if not done[b]:
                        tok = next_token_id[b].item()
                        generated_tokens[b].append(tok)
                        if (eos_id is not None) and (tok == eos_id):
                            done[b] = True

                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                tick()

                if bool(done.all()):
                    break
    finally:
        for h in decoder_hooks:
            try:
                h.remove()
            except Exception:
                pass

    return decoder_outputs, decoder_inputs, generated_tokens