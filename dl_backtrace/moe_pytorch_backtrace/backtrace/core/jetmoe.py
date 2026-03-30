"""
JetMoE model support – thin wrappers over shared model_utils,
plus the JetMoE-specific RoPE patch.
"""
import torch
from typing import Tuple
from .model_utils import unwrap_model, build_decoder_tree, create_decoder_output, to_numpy


def build_jetmoe_tree(model, root='jet_moe'):
    """Build the relevance-propagation graph for JetMoE."""
    core, lm_head = unwrap_model(model)
    return build_decoder_tree(
        core, lm_head,
        attn_class='JetMoE_Self_Attention',
        ff_class='JetMoE_Feed_Forward',
    )


def extract_jetmoe_weights(model):
    """Extract weights from JetMoE model (supports wrapped models)."""
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

    for name, param in model.named_parameters():
        param_np = to_numpy(param)

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
        weights_dict['decoder_lm_head']['lm_head.weight'] = to_numpy(lm_head.weight.data)

    return weights_dict


def create_jetmoe_output(input_text, model, tokenizer, max_length, device):
    """Capture per-step activations for JetMoE."""
    return create_decoder_output(
        input_text, model, tokenizer, max_length, device,
        attn_attr='self_attention',
    )


# ─────────────────────────────────────────────────────────────────────
#  JetMoE-specific: RoPE monkey-patch for transformers==4.52.x
# ─────────────────────────────────────────────────────────────────────

def install_jetmoe_rope_patch(force: bool = False) -> None:
    """
    Idempotently installs a RoPE monkey-patch for JetMoE on transformers==4.52.x.

    Rotates only the first `rotary_dim` dims when head_dim > rotary_dim,
    preventing shape mismatches in attention.

    Args:
        force: re-apply the patch even if it appears installed already.
    """
    from transformers.models.jetmoe import modeling_jetmoe as jtm

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

        if q.size(-1) == rotary_dim:
            q_rot = (q * cos) + (_rotate_half(q) * sin)
            k_rot = (k * cos) + (_rotate_half(k) * sin)
            return q_rot, k_rot

        q_head, q_tail = q[..., :rotary_dim], q[..., rotary_dim:]
        k_head, k_tail = k[..., :rotary_dim], k[..., rotary_dim:]
        q_head = (q_head * cos) + (_rotate_half(q_head) * sin)
        k_head = (k_head * cos) + (_rotate_half(k_head) * sin)
        return torch.cat([q_head, q_tail], dim=-1), torch.cat([k_head, k_tail], dim=-1)

    jtm.apply_rotary_pos_emb = apply_rotary_pos_emb_partial
    jtm._rope_partial_patch_installed = True