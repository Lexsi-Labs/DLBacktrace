import torch
from collections import defaultdict


# Function to rename the dictionary keys
def rename_self_attention_keys(attention_weights):
    renamed_weights = {}
    for key, value in attention_weights.items():
        if 'query.weight' in key or 'SelfAttention.q.weight' in key:
            new_key = key.replace(key, 'W_q')
        elif 'query.bias' in key or 'SelfAttention.q.bias' in key:
            new_key = key.replace(key, 'b_q')
        elif 'key.weight' in key or 'SelfAttention.k.weight' in key:
            new_key = key.replace(key, 'W_k')
        elif 'key.bias' in key or 'SelfAttention.k.bias' in key:
            new_key = key.replace(key, 'b_k')
        elif 'value.weight' in key or 'SelfAttention.v.weight' in key:
            new_key = key.replace(key, 'W_v')
        elif 'value.bias' in key or 'SelfAttention.v.bias' in key:
            new_key = key.replace(key, 'b_v')
        elif 'output.dense.weight' in key or 'SelfAttention.o.weight' in key:
            new_key = key.replace(key, 'W_d')
        elif 'output.dense.bias' in key or 'SelfAttention.o.bias' in key:
            new_key = key.replace(key, 'b_d')

        renamed_weights[new_key] = value
    return renamed_weights


def rename_cross_attention_keys(cross_attention_weights):
    renamed_weights = {}

    for key, value in cross_attention_weights.items():
        if 'EncDecAttention.q.weight' in key:
            new_key = key.replace(key, 'W_q')
        elif 'EncDecAttention.k.weight' in key:
            new_key = key.replace(key, 'W_k')
        elif 'EncDecAttention.v.weight' in key:
            new_key = key.replace(key, 'W_v')
        elif 'EncDecAttention.o.weight' in key:
            new_key = key.replace(key, 'W_o')

        renamed_weights[new_key] = value
    return renamed_weights


def rename_feed_forward_keys(feed_forward_weights):
    renamed_weights = {}

    for key, value in feed_forward_weights.items():
        if 'intermediate.dense.weight' in key or 'DenseReluDense.wi.weight' in key:
            new_key = key.replace(key, 'W_int')
        elif 'intermediate.dense.bias' in key or 'DenseReluDense.wi.bias' in key:
            new_key = key.replace(key, 'b_int')
        elif 'output.dense.weight' in key or 'DenseReluDense.wo.weight' in key:
            new_key = key.replace(key, 'W_out')
        elif 'output.dense.bias' in key or 'DenseReluDense.wo.bias' in key:
            new_key = key.replace(key, 'b_out')

        renamed_weights[new_key] = value
    return renamed_weights


def rename_pooler_keys(pooler_weights):
    renamed_weights = {}
    for key, value in pooler_weights.items():
        if 'pooler.dense.weight' in key:
            new_key = key.replace(key, 'W_p')
        elif 'pooler.dense.bias' in key:
            new_key = key.replace(key, 'b_p')

        renamed_weights[new_key] = value
    return renamed_weights


def rename_classifier_keys(classifier_weights):
    renamed_weights = {}
    for key, value in classifier_weights.items():
        if 'classifier.weight' in key:
            new_key = key.replace(key, 'W_cls')
        elif 'classifier.bias' in key:
            new_key = key.replace(key, 'b_cls')

        renamed_weights[new_key] = value
    return renamed_weights

def rename_decoder_lm_head(lm_head_weights):
    renamed_weights = {}

    for key, value in lm_head_weights.items():
        if 'shared.weight' in key:
            new_key = key.replace(key, 'W_lm_head')

        renamed_weights[new_key] = value
    return renamed_weights
