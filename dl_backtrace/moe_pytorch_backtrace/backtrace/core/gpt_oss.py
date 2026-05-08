"""
GPT-OSS model support – thin wrappers over shared model_utils.
"""
import torch
from .model_utils import unwrap_model, build_decoder_tree, create_decoder_output, to_numpy


def build_gpt_oss_tree(model, root='gpt_oss'):
    """Build the relevance-propagation graph for GPT-OSS."""
    core, lm_head = unwrap_model(model)
    return build_decoder_tree(
        core, lm_head,
        attn_class='GPT_OSS_Self_Attention',
        ff_class='GPT_OSS_Feed_Forward',
    )


def extract_gpt_oss_weights(model):
    """Extract weights from GPT-OSS model (supports wrapped models)."""
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
            elif 'post_attention_layernorm' in name:
                weights_dict[f'decoder_layer_norm_{layer}_1'][name] = param_np
            elif any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'sinks']):
                weights_dict[f'decoder_self_attention_{layer}'][name] = param_np
            elif 'router' in name or 'gate_up_proj' in name or 'down_proj' in name:
                weights_dict[f'decoder_feed_forward_{layer}'][name] = param_np
        elif 'norm.weight' in name:
            weights_dict['decoder_layer_norm'][name] = param_np

    if lm_head is not None and hasattr(lm_head, 'weight'):
        weights_dict['decoder_lm_head']['lm_head.weight'] = to_numpy(lm_head.weight)

    return weights_dict


def create_gpt_oss_output(input_text, model, tokenizer, max_length, device, **kwargs):
    """Capture per-step activations for GPT-OSS."""
    return create_decoder_output(
        input_text, model, tokenizer, max_length, device,
        attn_attr='self_attn',
        **kwargs,
    )
