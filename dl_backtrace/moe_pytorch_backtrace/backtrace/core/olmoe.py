from collections import defaultdict
import torch
import numpy as np
from .model_utils import unwrap_model


def build_olmoe_tree(model, root='jet_moe'):
    """Build tree for OLMoE model (supports wrapped models)."""
    
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


def extract_olmoe_weights(model):
    """Extract weights from OLMoE model (supports wrapped models)."""
    
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

    # Loop through the layers to initialize the weight dictionaries for each
    for i in range(config.num_hidden_layers):
        weights_dict[f'decoder_layer_norm_{i}_0'] = {}
        weights_dict[f'decoder_self_attention_{i}'] = {}
        weights_dict[f'decoder_layer_norm_{i}_1'] = {}
        weights_dict[f'decoder_feed_forward_{i}'] = {}

    for layer in range(config.num_hidden_layers):
        for expert_id in range(config.num_experts):
            weights_dict[f'decoder_feed_forward_{layer}'][f'{expert_id}'] = {}

    # Helper to safely convert params to numpy
    def to_np(p):
        if isinstance(p, torch.Tensor):
            return p.detach().cpu().numpy()
        return p

    # Extract parameters
    for name, param in model.named_parameters():
        param_np = to_np(param)

        if 'embed_tokens' in name:
            weights_dict['decoder_embeddings'][name] = param_np

        elif 'layers' in name:
            layer = name.split('.')[2]

            if 'input_layernorm' in name:
                weights_dict[f'decoder_layer_norm_{layer}_0'][name] = param_np

            elif 'post_attention_layernorm' in name:
                weights_dict[f'decoder_layer_norm_{layer}_1'][name] = param_np

            elif any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm']):
                weights_dict[f'decoder_self_attention_{layer}'][name] = param_np

            elif 'gate' in name and 'gate_proj' not in name:
                weights_dict[f'decoder_feed_forward_{layer}'][name] = param_np

            elif 'gate_proj' in name or 'up_proj' in name or 'down_proj' in name:
                expert_id = name.split('.')[5]
                weights_dict[f'decoder_feed_forward_{layer}'][f'{expert_id}'][name] = param_np

        elif 'norm.weight' in name:
            weights_dict['decoder_layer_norm'][name] = param_np

    if lm_head is not None and hasattr(lm_head, 'weight'):
        weights_dict['decoder_lm_head']['lm_head.weight'] = to_np(lm_head.weight.data)

    return weights_dict


def create_olmoe_output(input_text, model, tokenizer, max_length, device):
    """Create outputs for OLMoE model (supports wrapped models)."""
    
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
    decoder_outputs = defaultdict(dict)
    decoder_inputs  = defaultdict(dict)
    decoder_hooks = []

    def ts():
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

    # ---- hooks ----
    def pre_emb(m,a,kw):
        x = first_tensor(a, kw, prefer_key='input_ids')
        if x is not None: decoder_inputs[ts()]['decoder_embeddings'] = clone(x)
    def post_emb(m,a,kw,out):
        decoder_outputs[ts()]['decoder_embeddings'] = clone(out)

    def pre_ln0(i):
        key = f'decoder_layer_norm_{i}_0'
        def _h(m,a,kw):
            x = first_tensor(a, kw, prefer_key='hidden_states')
            if x is not None: decoder_inputs[ts()][key] = clone(x)
        return _h
    def post_ln0(i):
        key = f'decoder_layer_norm_{i}_0'
        def _h(m,a,kw,out):
            decoder_outputs[ts()][key] = clone(out)
        return _h

    def pre_attn(i):
        key = f'decoder_self_attention_{i}'
        def _h(m,a,kw):
            x = first_tensor(a, kw, prefer_key='hidden_states')
            if x is not None: decoder_inputs[ts()][key] = clone(x)
        return _h
    def post_attn(i):
        key = f'decoder_self_attention_{i}'
        def _h(m,a,kw,out):
            out0 = out[0] if isinstance(out, (tuple, list)) else out
            decoder_outputs[ts()][key] = clone(out0)
        return _h

    def pre_ln1(i):
        key = f'decoder_layer_norm_{i}_1'
        def _h(m,a,kw):
            x = first_tensor(a, kw, prefer_key='hidden_states')
            if x is not None: decoder_inputs[ts()][key] = clone(x)
        return _h
    def post_ln1(i):
        key = f'decoder_layer_norm_{i}_1'
        def _h(m,a,kw,out):
            decoder_outputs[ts()][key] = clone(out)
        return _h

    def pre_mlp(i):
        key = f'decoder_feed_forward_{i}'
        def _h(m,a,kw):
            # OlmoeSparseMoeBlock usually gets hidden_states; capture it
            x = first_tensor(a, kw, prefer_key='hidden_states')
            if x is not None: decoder_inputs[ts()][key] = clone(x)
        return _h
    def post_mlp(i):
        key = f'decoder_feed_forward_{i}'
        def _h(m,a,kw,out):
            out0 = out[0] if isinstance(out, (tuple, list)) else out
            decoder_outputs[ts()][key] = clone(out0)
        return _h

    # Residual captures
    def post_residual_attn(i):
        key     = f"decoder_residual_self_attention_{i}"
        ln0_key = f"decoder_layer_norm_{i}_0"
        att_key = f"decoder_self_attention_{i}"
        def _h(m,a,kw,out):
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
        def _h(m,a,kw,out):
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

    def pre_final_ln(m,a,kw):
        x = first_tensor(a, kw, prefer_key='hidden_states')
        if x is not None: decoder_inputs[ts()]['decoder_layer_norm'] = clone(x)
    def post_final_ln(m,a,kw,out):
        decoder_outputs[ts()]['decoder_layer_norm'] = clone(out)

    def pre_lm(m,a,kw):
        x = first_tensor(a, kw, prefer_key='hidden_states')
        if x is None and isinstance(a, (tuple, list)) and len(a) and torch.is_tensor(a[0]): x = a[0]
        if x is not None: decoder_inputs[ts()]['decoder_lm_head'] = clone(x)
    def post_lm(m,a,kw,out):
        decoder_outputs[ts()]['decoder_lm_head'] = clone(out)

    # ---- register hooks (Olmoe names) ----
    decoder_hooks.append(core.embed_tokens.register_forward_pre_hook(pre_emb, with_kwargs=True))
    decoder_hooks.append(core.embed_tokens.register_forward_hook(post_emb, with_kwargs=True))

    for i, layer in enumerate(core.layers):
        decoder_hooks.append(layer.input_layernorm.register_forward_pre_hook(pre_ln0(i), with_kwargs=True))
        decoder_hooks.append(layer.input_layernorm.register_forward_hook(post_ln0(i), with_kwargs=True))

        attn = layer.self_attn
        decoder_hooks.append(attn.register_forward_pre_hook(pre_attn(i), with_kwargs=True))
        decoder_hooks.append(attn.register_forward_hook(post_attn(i), with_kwargs=True))
        decoder_hooks.append(attn.register_forward_hook(post_residual_attn(i), with_kwargs=True))

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

    # ---- generation (ensure hooks are removed even on error) ----
    def tick():
        nonlocal token_idx
        token_idx += 1

    enc = tokenizer(input_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    model = model.to(device)

    token_idx = 0
    if max_length is None:
        max_length = model.config.max_position_embeddings

    B = input_ids.shape[0]
    generated_tokens = [[] for _ in range(B)]

    try:
        for _ in range(max_length):
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            for b in range(B):
                generated_tokens[b].append(next_token_id[b].item())
                if next_token_id[b].item() == model.config.eos_token_id:
                    break
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            tick()
    finally:
        for h in decoder_hooks:
            try: h.remove()
            except Exception: pass

    return decoder_outputs, decoder_inputs, generated_tokens