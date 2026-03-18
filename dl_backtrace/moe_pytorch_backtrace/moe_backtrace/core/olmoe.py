from collections import defaultdict
import torch
from .model_utils import (
    unwrap_model, build_decoder_tree, run_hook_generation,
    clear_all_hooks, first_tensor, clone, get_tensor, to_torch_like,
)


def build_olmoe_tree(model, root='olmoe'):
    """Build the relevance-propagation graph for OLMoE (supports wrapped models)."""
    core, lm_head = unwrap_model(model)
    return build_decoder_tree(
        core, lm_head,
        attn_class='Self_Attention',
        ff_class='OLMoE_Feed_Forward',
    )


def _to_torch(t):
    """Detach and cast to float32 CPU tensor. Zero-copy when already fp32 CPU."""
    if not isinstance(t, torch.Tensor):
        return torch.as_tensor(t, dtype=torch.float32)
    return t.detach().to(dtype=torch.float32, device='cpu')


def extract_olmoe_weights(model):
    """Extract weights from an OLMoE model (supports wrapped models).
    
    Returns weight dicts containing torch.Tensor (fp32, CPU) instead of numpy.
    """
    core, lm_head = unwrap_model(model)
    config = core.config if hasattr(core, 'config') else model.config

    weights_dict = {
        'decoder_embeddings': {},
        'decoder_layer_norm': {},
        'decoder_lm_head': {},
    }
    for i in range(config.num_hidden_layers):
        weights_dict[f'decoder_layer_norm_{i}_0'] = {}
        weights_dict[f'decoder_self_attention_{i}'] = {}
        weights_dict[f'decoder_layer_norm_{i}_1'] = {}
        weights_dict[f'decoder_feed_forward_{i}'] = {}
        
    for layer in range(config.num_hidden_layers):
        for expert_id in range(config.num_experts):
            weights_dict[f'decoder_feed_forward_{layer}'][f'{expert_id}'] = {}

    for name, param in model.named_parameters():
        param_t = _to_torch(param)
        if 'embed_tokens' in name:
            weights_dict['decoder_embeddings'][name] = param_t
        elif 'layers' in name:
            layer = name.split('.')[2]
            if 'input_layernorm' in name:
                weights_dict[f'decoder_layer_norm_{layer}_0'][name] = param_t
            elif 'post_attention_layernorm' in name:
                weights_dict[f'decoder_layer_norm_{layer}_1'][name] = param_t
            elif any(p in name for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm']):
                weights_dict[f'decoder_self_attention_{layer}'][name] = param_t
            elif 'gate' in name and 'gate_proj' not in name:
                weights_dict[f'decoder_feed_forward_{layer}'][name] = param_t
            elif 'gate_proj' in name or 'up_proj' in name or 'down_proj' in name:
                expert_id = name.split('.')[5]
                weights_dict[f'decoder_feed_forward_{layer}'][f'{expert_id}'][name] = param_t
        elif 'norm.weight' in name:
            weights_dict['decoder_layer_norm'][name] = param_t

    if lm_head is not None and hasattr(lm_head, 'weight'):
        weights_dict['decoder_lm_head']['lm_head.weight'] = _to_torch(lm_head.weight.data)

    return weights_dict


def create_olmoe_output(input_text, model, tokenizer, max_length, device):
    """Capture per-step activations for OLMoE (supports wrapped models)."""
    core, lm_head = unwrap_model(model)
    clear_all_hooks(core)
    if lm_head is not None:
        clear_all_hooks(lm_head)

    token_idx = 0
    decoder_outputs = defaultdict(dict)
    decoder_inputs = defaultdict(dict)
    decoder_hooks = []

    def ts():
        return str(token_idx)

    # ---- embedding hooks ----
    def pre_emb(m, a, kw):
        x = first_tensor(a, kw, prefer_key='input_ids')
        if x is not None:
            decoder_inputs[ts()]['decoder_embeddings'] = clone(x)

    def post_emb(m, a, kw, out):
        decoder_outputs[ts()]['decoder_embeddings'] = clone(out)

    # ---- per-layer hooks ----
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

    def post_residual_attn(i):
        key = f'decoder_residual_self_attention_{i}'
        ln0_key = f'decoder_layer_norm_{i}_0'
        att_key = f'decoder_self_attention_{i}'
        def _h(m, a, kw, out):
            t = ts()
            branch = out[0] if isinstance(out, (tuple, list)) else out
            skip = get_tensor(decoder_outputs[t], ln0_key)
            if skip is None:
                skip = get_tensor(decoder_inputs[t], att_key)
            if skip is None:
                skip = torch.zeros_like(branch)
            decoder_outputs[t][key] = clone(to_torch_like(skip, branch) + branch)
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

    def post_residual_mlp(i):
        key = f'decoder_residual_feed_forward_{i}'
        ln1_key = f'decoder_layer_norm_{i}_1'
        mlp_key = f'decoder_feed_forward_{i}'
        def _h(m, a, kw, out):
            t = ts()
            branch = out[0] if isinstance(out, (tuple, list)) else out
            skip = get_tensor(decoder_outputs[t], ln1_key)
            if skip is None:
                skip = get_tensor(decoder_inputs[t], mlp_key)
            if skip is None:
                skip = torch.zeros_like(branch)
            decoder_outputs[t][key] = clone(to_torch_like(skip, branch) + branch)
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

    # ---- register hooks ----
    decoder_hooks.append(core.embed_tokens.register_forward_pre_hook(pre_emb, with_kwargs=True))
    decoder_hooks.append(core.embed_tokens.register_forward_hook(post_emb, with_kwargs=True))

    for i, layer in enumerate(core.layers):
        decoder_hooks.append(layer.input_layernorm.register_forward_pre_hook(pre_ln0(i), with_kwargs=True))
        decoder_hooks.append(layer.input_layernorm.register_forward_hook(post_ln0(i), with_kwargs=True))
        decoder_hooks.append(layer.self_attn.register_forward_pre_hook(pre_attn(i), with_kwargs=True))
        decoder_hooks.append(layer.self_attn.register_forward_hook(post_attn(i), with_kwargs=True))
        decoder_hooks.append(layer.self_attn.register_forward_hook(post_residual_attn(i), with_kwargs=True))
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

    # ---- generation (hooks removed in finally even on error) ----
    def tick():
        nonlocal token_idx
        token_idx += 1

    enc = tokenizer(input_text, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    model = model.to(device)
    token_idx = 0
    if max_length is None:
        max_length = getattr(model.config, 'max_position_embeddings', 2048)

    B = input_ids.shape[0]
    generated_tokens = [[] for _ in range(B)]

    try:
        for _ in range(max_length):
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            for b in range(B):
                generated_tokens[b].append(next_token_id[b].item())
                eos = getattr(model.config, 'eos_token_id', None)
                if eos is not None and next_token_id[b].item() == eos:
                    break
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            tick()
    finally:
        for h in decoder_hooks:
            try:
                h.remove()
            except Exception:
                pass

    return decoder_outputs, decoder_inputs, generated_tokens