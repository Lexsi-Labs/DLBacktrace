import torch


def rename_self_attention_keys(attention_weights):
    renamed_weights = {}
    for key, value in attention_weights.items():
        if 'query.weight' in key or 'SelfAttention.q.weight' in key or 'self_attn.q_proj.weight' in key:
            renamed_weights['W_q'] = value
        elif 'query.bias' in key or 'SelfAttention.q.bias' in key or 'self_attn.q_proj.bias' in key:
            renamed_weights['b_q'] = value
        elif 'key.weight' in key or 'SelfAttention.k.weight' in key or 'self_attn.k_proj.weight' in key:
            renamed_weights['W_k'] = value
        elif 'key.bias' in key or 'SelfAttention.k.bias' in key or 'self_attn.k_proj.bias' in key:
            renamed_weights['b_k'] = value
        elif 'value.weight' in key or 'SelfAttention.v.weight' in key or 'self_attn.v_proj.weight' in key:
            renamed_weights['W_v'] = value
        elif 'value.bias' in key or 'SelfAttention.v.bias' in key or 'self_attn.v_proj.bias' in key:
            renamed_weights['b_v'] = value
        elif 'output.dense.weight' in key or 'SelfAttention.o.weight' in key or 'self_attn.o_proj.weight' in key:
            renamed_weights['W_d'] = value
        elif 'output.dense.bias' in key or 'SelfAttention.o.bias' in key or 'self_attn.o_proj.bias' in key:
            renamed_weights['b_d'] = value
        elif 'self_attn.q_norm' in key:
            renamed_weights['q_norm'] = value
        elif 'self_attn.k_norm' in key:
            renamed_weights['k_norm'] = value
        elif 'self_attn.sinks' in key:
            renamed_weights['W_sinks'] = value
    return renamed_weights


def rename_decoder_lm_head(lm_head_weights):
    renamed_weights = {}
    for key, value in lm_head_weights.items():
        if 'shared.weight' in key or 'lm_head.weight' in key:
            renamed_weights['W_lm_head'] = value
        else:
            renamed_weights[key] = value
    return renamed_weights


def rename_jetmoe_feed_forward_keys(feed_forward_weights):
    renamed_weights = {}
    for key, value in feed_forward_weights.items():
        if 'mlp.bias' in key:
            renamed_weights['bias'] = value
        elif 'input_linear.weight' in key:
            renamed_weights['W_in'] = value
        elif 'output_linear.weight' in key:
            renamed_weights['W_out'] = value
        elif 'router.layer.weight' in key:
            renamed_weights['W_router'] = value
    return renamed_weights


def rename_jetmoe_self_attention_keys(self_attention_weights):
    renamed_weights = {}
    for key, value in self_attention_weights.items():
        if 'experts.bias' in key:
            renamed_weights['bias'] = value
        elif 'experts.input_linear.weight' in key:
            renamed_weights['W_in'] = value
        elif 'experts.output_linear.weight' in key:
            renamed_weights['W_out'] = value
        elif 'experts.router.layer.weight' in key:
            renamed_weights['W_router'] = value
        elif 'kv_proj.weight' in key:
            renamed_weights['W_kv'] = value
    return renamed_weights


def rename_olmoe_feed_forward_keys(feed_forward_weights):
    renamed_weights = {}
    for key, value in feed_forward_weights.items():
        if 'mlp.gate' in key:
            renamed_weights['W_gate'] = value
        else:
            renamed_weights[key] = {}
            for k, v in value.items():
                if 'gate_proj' in k:
                    renamed_weights[key]['W_gate_proj'] = v
                elif 'up_proj' in k:
                    renamed_weights[key]['W_up_proj'] = v
                elif 'down_proj' in k:
                    renamed_weights[key]['W_down_proj'] = v
    return renamed_weights


def rename_qwenmoe_feed_forward_keys(feed_forward_weights):
    renamed_weights = {}
    for key, value in feed_forward_weights.items():
        if 'mlp.gate' in key:
            renamed_weights['W_gate'] = value
        else:
            renamed_weights[key] = {}
            for k, v in value.items():
                if 'gate_proj' in k:
                    renamed_weights[key]['W_gate_proj'] = v
                elif 'up_proj' in k:
                    renamed_weights[key]['W_up_proj'] = v
                elif 'down_proj' in k:
                    renamed_weights[key]['W_down_proj'] = v
    return renamed_weights


def rename_gptoss_feed_forward_keys(feed_forward_weights):
    renamed_weights = {}
    for key, value in feed_forward_weights.items():
        if key.endswith('.mlp.router.weight'):
            renamed_weights['W_router'] = value
        elif key.endswith('.mlp.router.bias'):
            renamed_weights['b_router'] = value
        elif key.endswith('.mlp.experts.gate_up_proj'):
            renamed_weights['W_gate_up_proj'] = value
        elif key.endswith('mlp.experts.gate_up_proj_bias'):
            renamed_weights['b_gate_up_proj'] = value
        elif key.endswith('mlp.experts.down_proj'):
            renamed_weights['W_down_proj'] = value
        elif key.endswith('mlp.experts.down_proj_bias'):
            renamed_weights['b_down_proj'] = value
    return renamed_weights