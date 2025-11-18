import torch
from collections import defaultdict


# Function to rename the dictionary keys
def rename_self_attention_keys(attention_weights):
    renamed_weights = {}
    for key, value in attention_weights.items():
        if 'query.weight' in key or 'SelfAttention.q.weight' in key or 'self_attn.q_proj.weight' in key:
            new_key = key.replace(key, 'W_q')
            renamed_weights[new_key] = value
        elif 'query.bias' in key or 'SelfAttention.q.bias' in key or 'self_attn.q_proj.bias' in key:
            new_key = key.replace(key, 'b_q')
            renamed_weights[new_key] = value
        elif 'key.weight' in key or 'SelfAttention.k.weight' in key or 'self_attn.k_proj.weight' in key:
            new_key = key.replace(key, 'W_k')
            renamed_weights[new_key] = value
        elif 'key.bias' in key or 'SelfAttention.k.bias' in key or 'self_attn.k_proj.bias' in key:
            new_key = key.replace(key, 'b_k')
            renamed_weights[new_key] = value
        elif 'value.weight' in key or 'SelfAttention.v.weight' in key or 'self_attn.v_proj.weight' in key:
            new_key = key.replace(key, 'W_v')
            renamed_weights[new_key] = value
        elif 'value.bias' in key or 'SelfAttention.v.bias' in key or 'self_attn.v_proj.bias' in key:
            new_key = key.replace(key, 'b_v')
            renamed_weights[new_key] = value
        elif 'output.dense.weight' in key or 'SelfAttention.o.weight' in key or 'self_attn.o_proj.weight' in key:
            new_key = key.replace(key, 'W_d')
            renamed_weights[new_key] = value
        elif 'output.dense.bias' in key or 'SelfAttention.o.bias' in key or 'self_attn.o_proj.bias' in key:
            new_key = key.replace(key, 'b_d')
            renamed_weights[new_key] = value
        elif 'self_attn.q_norm' in key:
            new_key = key.replace(key, 'q_norm')
            renamed_weights[new_key] = value
        elif 'self_attn.k_norm' in key:
            new_key = key.replace(key, 'k_norm')
            renamed_weights[new_key] = value
        elif 'self_attn.sinks' in key:
            new_key = key.replace(key, 'W_sinks')
            renamed_weights[new_key] = value

    return renamed_weights


# def rename_cross_attention_keys(cross_attention_weights):
#     renamed_weights = {}

#     for key, value in cross_attention_weights.items():
#         if 'EncDecAttention.q.weight' in key:
#             new_key = key.replace(key, 'W_q')
#         elif 'EncDecAttention.k.weight' in key:
#             new_key = key.replace(key, 'W_k')
#         elif 'EncDecAttention.v.weight' in key:
#             new_key = key.replace(key, 'W_v')
#         elif 'EncDecAttention.o.weight' in key:
#             new_key = key.replace(key, 'W_d')

#         renamed_weights[new_key] = value
#     return renamed_weights


# def rename_feed_forward_keys(feed_forward_weights):
#     renamed_weights = {}

#     for key, value in feed_forward_weights.items():
#         if 'intermediate.dense.weight' in key or 'DenseReluDense.wi.weight' in key:
#             new_key = key.replace(key, 'W_int')
#         elif 'intermediate.dense.bias' in key or 'DenseReluDense.wi.bias' in key:
#             new_key = key.replace(key, 'b_int')
#         elif 'output.dense.weight' in key or 'DenseReluDense.wo.weight' in key:
#             new_key = key.replace(key, 'W_out')
#         elif 'output.dense.bias' in key or 'DenseReluDense.wo.bias' in key:
#             new_key = key.replace(key, 'b_out')

#         renamed_weights[new_key] = value
#     return renamed_weights


# def rename_pooler_keys(pooler_weights):
#     renamed_weights = {}
#     for key, value in pooler_weights.items():
#         if 'pooler.dense.weight' in key:
#             new_key = key.replace(key, 'W_p')
#         elif 'pooler.dense.bias' in key:
#             new_key = key.replace(key, 'b_p')

#         renamed_weights[new_key] = value
#     return renamed_weights


# def rename_classifier_keys(classifier_weights):
#     renamed_weights = {}
#     for key, value in classifier_weights.items():
#         if 'classifier.weight' in key:
#             new_key = key.replace(key, 'W_cls')
#         elif 'classifier.bias' in key:
#             new_key = key.replace(key, 'b_cls')

#         renamed_weights[new_key] = value
#     return renamed_weights


def rename_decoder_lm_head(lm_head_weights):
    renamed_weights = {}

    for key, value in lm_head_weights.items():
        if 'shared.weight' in key or 'lm_head.weight' in key:
            new_key = key.replace(key, 'W_lm_head')

        renamed_weights[new_key] = value
    return renamed_weights


# def rename_llama_feed_forward_keys(feed_forward_weights):
#     renamed_weights = {}

#     for key, value in feed_forward_weights.items():
#         if 'mlp.gate_proj.weight' in key:
#             new_key = key.replace(key, 'W_g')
#         elif 'mlp.up_proj.weight' in key:
#             new_key = key.replace(key, 'W_u')
#         elif 'mlp.down_proj.weight' in key:
#             new_key = key.replace(key, 'W_d')

#         renamed_weights[new_key] = value
#     return renamed_weights


def rename_jetmoe_feed_forward_keys(feed_forward_weights):
    renamed_weights = {}

    for key, value in feed_forward_weights.items():
        if 'mlp.bias' in key:
            new_key = key.replace(key, 'bias')
            renamed_weights[new_key] = value

        elif 'input_linear.weight' in key:
            new_key = key.replace(key, 'W_in')
            renamed_weights[new_key] = value

        elif 'output_linear.weight' in key:
            new_key = key.replace(key, 'W_out')
            renamed_weights[new_key] = value

        elif 'router.layer.weight' in key:
            new_key = key.replace(key, 'W_router')
            renamed_weights[new_key] = value

    return renamed_weights 


def rename_jetmoe_self_attention_keys(self_attention_weights):
    renamed_weights = {}

    for key, value in self_attention_weights.items():
        if 'experts.bias' in key:
            new_key = key.replace(key, 'bias')
            renamed_weights[new_key] = value
        elif 'experts.input_linear.weight' in key:
            new_key = key.replace(key, 'W_in')
            renamed_weights[new_key] = value
        elif 'experts.output_linear.weight' in key:
            new_key = key.replace(key, 'W_out')
            renamed_weights[new_key] = value
        elif 'experts.router.layer.weight' in key:
            new_key = key.replace(key, 'W_router')
            renamed_weights[new_key] = value
        elif 'kv_proj.weight' in key:
            new_key = key.replace(key, 'W_kv')
            renamed_weights[new_key] = value

    return renamed_weights


def rename_olmoe_feed_forward_keys(feed_forward_weights):
    renamed_weights = {}

    for key, value in feed_forward_weights.items():
        if 'mlp.gate' in key:
            new_key = key.replace(key, 'W_gate')
            renamed_weights[new_key] = value

        else:
            renamed_weights[key] = {}
            for k, v in value.items():
                if 'gate_proj' in k:
                    new_key = k.replace(k, 'W_gate_proj')
                    renamed_weights[key][new_key] = v
                elif 'up_proj' in k:
                    new_key = k.replace(k, 'W_up_proj')
                    renamed_weights[key][new_key] = v
                elif 'down_proj' in k:
                    new_key = k.replace(k, 'W_down_proj')
                    renamed_weights[key][new_key] = v

    return renamed_weights 


def rename_qwenmoe_feed_forward_keys(feed_forward_weights):
    renamed_weights = {}

    for key, value in feed_forward_weights.items():
        if 'mlp.gate' in key:
            new_key = key.replace(key, 'W_gate')
            renamed_weights[new_key] = value

        else:
            renamed_weights[key] = {}
            for k, v in value.items():
                if 'gate_proj' in k:
                    new_key = k.replace(k, 'W_gate_proj')
                    renamed_weights[key][new_key] = v
                elif 'up_proj' in k:
                    new_key = k.replace(k, 'W_up_proj')
                    renamed_weights[key][new_key] = v
                elif 'down_proj' in k:
                    new_key = k.replace(k, 'W_down_proj')
                    renamed_weights[key][new_key] = v

    return renamed_weights


def rename_gptoss_feed_forward_keys(feed_forward_weights):
    renamed_weights = {}

    for key, value in feed_forward_weights.items():
        if key.endswith('.mlp.router.weight'):
            new_key = key.replace(key, 'W_router')
            renamed_weights[new_key] = value
        elif key.endswith('.mlp.router.bias'):
            new_key = key.replace(key, 'b_router')
            renamed_weights[new_key] = value
        elif key.endswith('.mlp.experts.gate_up_proj'):
            new_key = key.replace(key, 'W_gate_up_proj')
            renamed_weights[new_key] = value
        elif key.endswith('mlp.experts.gate_up_proj_bias'):
            new_key = key.replace(key, 'b_gate_up_proj')
            renamed_weights[new_key] = value
        elif key.endswith('mlp.experts.down_proj'):
            new_key = key.replace(key, 'W_down_proj')
            renamed_weights[new_key] = value
        elif key.endswith('mlp.experts.down_proj_bias'):
            new_key = key.replace(key, 'b_down_proj')
            renamed_weights[new_key] = value

    return renamed_weights