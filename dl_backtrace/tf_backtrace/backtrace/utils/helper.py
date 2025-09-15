import tensorflow as tf


# Function to rename the dictionary keys
def rename_self_attention_keys(attention_weights):
    renamed_weights = {}
    for key, value in attention_weights.items():
        if 'query/kernel' in key or 'SelfAttention/q' in key:
            new_key = key.replace(key, 'W_q')
        elif 'query/bias' in key:
            new_key = key.replace(key, 'b_q')
        elif 'key/kernel' in key or 'SelfAttention/k' in key:
            new_key = key.replace(key, 'W_k')
        elif 'key/bias' in key:
            new_key = key.replace(key, 'b_k')
        elif 'value/kernel' in key or 'SelfAttention/v' in key:
            new_key = key.replace(key, 'W_v')
        elif 'value/bias' in key:
            new_key = key.replace(key, 'b_v')
        elif 'output/dense/kernel' in key or 'SelfAttention/o' in key:
            new_key = key.replace(key, 'W_d')
        elif 'output/dense/bias' in key:
            new_key = key.replace(key, 'b_d')
        elif 'SelfAttention/relative_attention_bias' in key:
            new_key = key.replace(key, 'relative_attn_bias')

        renamed_weights[new_key] = value
    return renamed_weights


def rename_cross_attention_keys(cross_attention_weights):
    renamed_weights = {}

    for key, value in cross_attention_weights.items():
        if 'EncDecAttention/q' in key:
            new_key = key.replace(key, 'W_q')
        elif 'EncDecAttention/k' in key:
            new_key = key.replace(key, 'W_k')
        elif 'EncDecAttention/v' in key:
            new_key = key.replace(key, 'W_v')
        elif 'EncDecAttention/o' in key:
            new_key = key.replace(key, 'W_o')

        renamed_weights[new_key] = value
    return renamed_weights


def rename_feed_forward_keys(feed_forward_weights):
    renamed_weights = {}

    for key, value in feed_forward_weights.items():
        if 'intermediate/dense/kernel' in key or 'DenseReluDense/wi' in key:
            new_key = key.replace(key, 'W_int')
        elif 'intermediate/dense/bias' in key or 'DenseReluDense/bi' in key:
            new_key = key.replace(key, 'b_int')
        elif 'output/dense/kernel' in key or 'DenseReluDense/wo' in key:
            new_key = key.replace(key, 'W_out')
        elif 'output/dense/bias' in key or 'DenseReluDense/bo' in key:
            new_key = key.replace(key, 'b_out')

        renamed_weights[new_key] = value
    return renamed_weights


def rename_pooler_keys(pooler_weights):
    renamed_weights = {}
    
    for key, value in pooler_weights.items():
        if 'pooler/dense/kernel' in key:
            new_key = key.replace(key, 'W_p')
        elif 'pooler/dense/bias' in key:
            new_key = key.replace(key, 'b_p')

        renamed_weights[new_key] = value
    return renamed_weights


def rename_classifier_keys(classifier_weights):
    renamed_weights = {}
    
    for key, value in classifier_weights.items():
        if 'classifier/kernel' in key:
            new_key = key.replace(key, 'W_cls')
        elif 'classifier/bias' in key:
            new_key = key.replace(key, 'b_cls')

        renamed_weights[new_key] = value
    return renamed_weights


def rename_decoder_lm_head(lm_head_weights):
    renamed_weights = {}

    for key, value in lm_head_weights.items():
        if 'shared/shared/embeddings' in key:
            new_key = key.replace(key, 'W_lm_head')

        renamed_weights[new_key] = value
    return renamed_weights
