import torch
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .model_utils import (
    unwrap_model, build_decoder_tree, clear_all_hooks,
    first_tensor, clone, get_tensor, to_torch_like,
)


def build_jetmoe_tree(model, root='jet_moe'):
    """Build the relevance-propagation graph for JetMoE (supports wrapped models)."""
    core, lm_head = unwrap_model(model)
    return build_decoder_tree(
        core, lm_head,
        attn_class='JetMoE_Self_Attention',
        ff_class='JetMoE_Feed_Forward',
    )


def extract_jetmoe_weights(model):
    """Extract weights from a JetMoE model (supports wrapped models)."""
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

    def to_np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else x

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


# ---- Public API: RoPE patch for JetMoE on transformers 4.52.x ---- #

def install_jetmoe_rope_patch(force: bool = False) -> None:
    """
    Idempotently install a RoPE monkey-patch for JetMoE on transformers==4.52.x.

    Rotates only the first `rotary_dim` dims when head_dim > rotary_dim,
    preventing shape mismatches in attention.

    Args:
        force: Re-apply the patch even if it appears installed already.
    """
    from transformers.models.jetmoe import modeling_jetmoe as jtm

    if getattr(jtm, '_rope_partial_patch_installed', False) and not force:
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
        if q.size(-1) == rotary_dim:
            return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)
        # Partial rotation
        q_head, q_tail = q[..., :rotary_dim], q[..., rotary_dim:]
        k_head, k_tail = k[..., :rotary_dim], k[..., rotary_dim:]
        q_head = (q_head * cos) + (_rotate_half(q_head) * sin)
        k_head = (k_head * cos) + (_rotate_half(k_head) * sin)
        return torch.cat([q_head, q_tail], dim=-1), torch.cat([k_head, k_tail], dim=-1)

    jtm.apply_rotary_pos_emb = apply_rotary_pos_emb_partial
    jtm._rope_partial_patch_installed = True


def create_jetmoe_output(
    input_text: str,
    model: Any,
    tokenizer: Any,
    max_length: Optional[int],
    device: str,
) -> Tuple[Dict, Dict, List[List[int]]]:
    """
    Run greedy decoding on JetMoE while capturing inputs and outputs
    for key submodules at each decoding step (supports wrapped models).

    Returns:
        decoder_outputs: {token_idx(str): {name: tensor}}
        decoder_inputs:  {token_idx(str): {name: tensor}}
        generated_tokens: List[List[int]]
    """
    core, lm_head = unwrap_model(model)
    clear_all_hooks(core)
    if lm_head is not None:
        clear_all_hooks(lm_head)

    token_idx = 0
    decoder_outputs: Dict[str, Dict] = defaultdict(dict)
    decoder_inputs: Dict[str, Dict] = defaultdict(dict)
    decoder_hooks = []

    def ts() -> str:
        return str(token_idx)

    # ---- hooks (JetMoE uses 'self_attention' instead of 'self_attn') ----
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
        # JetMoE uses 'self_attention' (not 'self_attn')
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

    # ---- generation (greedy, with batch-done tracking) ----
    def tick():
        nonlocal token_idx
        token_idx += 1

    enc = tokenizer(input_text, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    model = model.to(device).eval()
    token_idx = 0

    if max_length is None:
        max_length = getattr(model.config, 'max_position_embeddings', 2048)

    B = input_ids.shape[0]
    generated_tokens: List[List[int]] = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    eos_id = getattr(model.config, 'eos_token_id', None)

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
                        if eos_id is not None and tok == eos_id:
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