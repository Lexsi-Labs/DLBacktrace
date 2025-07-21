import tensorflow as tf
from collections import defaultdict


def build_enc_dec_tree(model, root='enc-dec'):
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
            'class': component if isinstance(component, str) else type(component).__name__,
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

        # Keep track of component type in a separate dictionary
        layer_tree[name] = component if isinstance(component, str) else tree[name]['type']

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
    encoder_embeddings = add_component(ltree, 'encoder_embedding', model.encoder.embed_tokens, child=None)

    # Add encoder layers dynamically
    current_child = 'encoder_embedding'
    for i, layer in enumerate(model.encoder.block):
        encoder_layer_norm_0 = add_component(ltree, f'encoder_layer_norm_{i}_0', 'Layer_Norm', child=current_child)
        encoder_self_attention = add_component(ltree, f'encoder_self_attention_{i}', 'Self_Attention', child=f'encoder_layer_norm_{i}_0')
        encoder_residual_self_attention = add_component(ltree, f'encoder_residual_self_attention_{i}', 'Residual', child=[current_child, f'encoder_self_attention_{i}'])

        encoder_layer_norm_1 = add_component(ltree, f'encoder_layer_norm_{i}_1', 'Layer_Norm', child=f'encoder_residual_self_attention_{i}')
        encoder_feed_forward = add_component(ltree, f'encoder_feed_forward_{i}', 'Feed_Forward', child=f'encoder_layer_norm_{i}_1')
        encoder_residual_feed_forward = add_component(ltree, f'encoder_residual_feed_forward_{i}', 'Residual', child=[f'encoder_residual_self_attention_{i}', f'encoder_feed_forward_{i}'])

        current_child = f'encoder_residual_feed_forward_{i}'

    if hasattr(model.encoder, 'final_layer_norm'):
        encoder_final_layer_norm = add_component(ltree, 'encoder_layer_norm', model.encoder.final_layer_norm, child=current_child)
        current_child = 'encoder_layer_norm'

    # Add Decoder layers
    decoder_embeddings = add_component(ltree, 'decoder_embedding', model.decoder.embed_tokens, child=None)

    # Add decoder layers dynamically
    current_child = 'decoder_embedding'
    for i, layer in enumerate(model.decoder.block):
        decoder_layer_norm_0 = add_component(ltree, f'decoder_layer_norm_{i}_0', 'Layer_Norm', child=current_child)
        decoder_self_attention = add_component(ltree, f'decoder_self_attention_{i}', 'Self_Attention', child=f'decoder_layer_norm_{i}_0')
        decoder_residual_self_attention = add_component(ltree, f'decoder_residual_self_attention_{i}', 'Residual', child=[current_child, f'decoder_self_attention_{i}'])

        decoder_layer_norm_1 = add_component(ltree, f'decoder_layer_norm_{i}_1', 'Layer_Norm', child=f'decoder_residual_self_attention_{i}')
        decoder_cross_attention = add_component(ltree, f'decoder_cross_attention_{i}', 'Cross_Attention', child=['encoder_layer_norm', f'decoder_layer_norm_{i}_1'])
        decoder_residual_cross_attention = add_component(ltree, f'decoder_residual_cross_attention_{i}', 'Residual', child=[f'decoder_residual_self_attention_{i}', f'decoder_cross_attention_{i}'])

        decoder_layer_norm_2 = add_component(ltree, f'decoder_layer_norm_{i}_2', 'Layer_Norm', child=f'decoder_residual_cross_attention_{i}')
        decoder_feed_forward = add_component(ltree, f'decoder_feed_forward_{i}', 'Feed_Forward', child=f'decoder_layer_norm_{i}_2')
        decoder_residual_feed_forward = add_component(ltree, f'decoder_residual_feed_forward_{i}', 'Residual', child=[f'decoder_residual_cross_attention_{i}', f'decoder_feed_forward_{i}'])

        current_child = f'decoder_residual_feed_forward_{i}'

    if hasattr(model.decoder, 'final_layer_norm'):
        decoder_final_layer_norm = add_component(ltree, 'decoder_layer_norm', model.decoder.final_layer_norm, child=current_child)
        current_child = 'decoder_layer_norm'

    # Decoder LM-Head
    if hasattr(model, 'lm_head'):
        decoder_lm_head = add_component(ltree, 'decoder_lm_head', 'LM_Head', child=current_child)
        current_child = 'decoder_lm_head'
    else:
        if model.config.tie_word_embeddings:
            decoder_lm_head = add_component(ltree, 'decoder_lm_head', 'LM_Head', child=current_child)
        else:
            decoder_lm_head = add_component(ltree, 'decoder_lm_head', None, child=current_child)

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
    
    model_resource = {
        "layers": layer_tree,
        "graph": ltree,
        "outputs": outputs,
        "inputs": inputs
    }

    return model_resource, layer_stack


def extract_encoder_decoder_weights(model):
    # Initialize a dictionary to hold the weights
    weights_dict = {
        # 'shared_embeddings': {},
        'encoder_embedding': {},
        'encoder_layer_norm': {},
        'decoder_embedding': {},
        'decoder_layer_norm': {},
        'decoder_lm_head': {}
    }

    # Extract the model's parameters and organize them into the dictionary
    for weight in model.weights:
        name = weight.name
        value = weight.numpy()

        if 'shared' in name:
            weights_dict['encoder_embedding'][name] = value
            weights_dict['decoder_embedding'][name] = value
            weights_dict['decoder_lm_head'][name] = value

        elif 'encoder' in name:
            if 'block' in name:
                layer = name.split('/')[2].split('_')[-1]
                sub_layer = name.split('/')[3].split('_')[-1]
                submodule = name.split('/')[4]

                if 'SelfAttention' in submodule and f'encoder_self_attention_{layer}' not in weights_dict:
                    weights_dict[f'encoder_self_attention_{layer}'] = {}
                if 'layer_norm' in submodule and f'encoder_layer_norm_{layer}' not in weights_dict:
                    weights_dict[f'encoder_layer_norm_{layer}'] = {}
                if 'DenseReluDense' in submodule and f'encoder_feed_forward_{layer}' not in weights_dict:
                    weights_dict[f'encoder_feed_forward_{layer}'] = {}

                if 'SelfAttention' in submodule:
                    weights_dict[f'encoder_self_attention_{layer}'][name] = value
                elif 'layer_norm' in submodule:
                    weights_dict[f'encoder_layer_norm_{layer}'][name] = value
                elif 'DenseReluDense' in submodule:
                    weights_dict[f'encoder_feed_forward_{layer}'][name] = value

            elif 'final_layer_norm' in name:
                weights_dict['encoder_layer_norm'][name] = value

        elif 'decoder/block' in name:
            layer = name.split('/')[2].split('_')[-1]
            sub_layer = name.split('/')[3].split('_')[-1]
            submodule = name.split('/')[4]

            if 'SelfAttention' in submodule and f'decoder_self_attention_{layer}' not in weights_dict:
                weights_dict[f'decoder_self_attention_{layer}'] = {}
            if 'layer_norm' in submodule and f'decoder_layer_norm_{layer}' not in weights_dict:
                weights_dict[f'decoder_layer_norm_{layer}'] = {}
            if 'EncDecAttention' in submodule and f'decoder_cross_attention_{layer}' not in weights_dict:
                weights_dict[f'decoder_cross_attention_{layer}'] = {}
            if 'DenseReluDense' in submodule and f'decoder_feed_forward_{layer}' not in weights_dict:
                weights_dict[f'decoder_feed_forward_{layer}'] = {}

            if 'SelfAttention' in submodule:
                weights_dict[f'decoder_self_attention_{layer}'][name] = value
            elif 'layer_norm' in submodule:
                weights_dict[f'decoder_layer_norm_{layer}'][name] = value
            elif 'EncDecAttention' in submodule:
                weights_dict[f'decoder_cross_attention_{layer}'][name] = value
            elif 'DenseReluDense' in submodule:
                weights_dict[f'decoder_feed_forward_{layer}'][name] = value

        elif 'decoder/final_layer_norm' in name:
            weights_dict['decoder_layer_norm'][name] = value

    return weights_dict


def calculate_encoder_decoder_output(input_text, model, tokenizer):
    # Dictionaries to store the inputs and outputs
    encoder_inputs = {}
    encoder_outputs = defaultdict(lambda: defaultdict(dict))
    decoder_inputs = defaultdict(lambda: defaultdict(dict))
    decoder_outputs = defaultdict(lambda: defaultdict(dict))
    
    # Hook manager to store hook information
    hook_manager = []
    
    # Global variable to keep track of the token index
    token_idx = 0

    # Function to generate timestamp (token index)
    def get_timestamp():
        return str(token_idx)
    
    # Function to wrap the call method of the layer to add hooks
    def wrap_call(layer, hook_fn, layer_index=None):
        original_call = layer.call

        def hooked_call(*args, **kwargs):
            outputs = original_call(*args, **kwargs)
            if layer_index is not None:
                hook_fn(layer, args, outputs, layer_index)
            else:
                hook_fn(layer, args, outputs)
            return outputs

        layer.call = hooked_call
        hook_manager.append((layer, original_call))
    
    def capture_encoder_embeddings(model, tokenizer, input_text):
        # Tokenize the input text
        encoding = tokenizer(input_text, return_tensors='tf')
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Manually capture the embedding output
        embedding_output = model.encoder.embed_tokens(input_ids)

        # Return the captured embeddings
        return embedding_output

    ## ------------ Hook function for Self-Attention Block --------------------
    def hook_fn__encoder_normalized_hidden_states(layer, inputs, outputs, layer_index):
        encoder_inputs[f'encoder_layer_norm_{layer_index}_0'] = inputs
        encoder_outputs[f'encoder_layer_norm_{layer_index}_0'] = outputs
        encoder_outputs[f'input_to_layer_norm_{layer_index}'] = inputs[0]

    def hook_fn_encoder_self_attention_outputs(layer, inputs, outputs, layer_index):
        encoder_inputs[f'encoder_self_attention_{layer_index}'] = inputs
        encoder_outputs[f'encoder_self_attention_{layer_index}'] = outputs[0]

    def hook_fn_encoder_dropout_attention_output(layer, inputs, outputs, layer_index):
        encoder_outputs[f'encoder_dropout_attention_{layer_index}'] = outputs[0]

    ## ------------  Hook function for Feed-Forward Block --------------
    def hook_fn_encoder_normalized_forwarded_states(layer, inputs, outputs, layer_index):
        encoder_inputs[f'encoder_layer_norm_{layer_index}_1'] = inputs
        encoder_outputs[f'encoder_layer_norm_{layer_index}_1'] = outputs
        encoder_outputs[f'input_to_ff_layer_norm_{layer_index}'] = inputs[0]

    def hook_fn_encoder_forwarded_states(layer, inputs, outputs, layer_index):
        encoder_inputs[f'encoder_feed_forward_{layer_index}'] = inputs
        encoder_outputs[f'encoder_feed_forward_{layer_index}'] = outputs

    def hook_fn_encoder_dropout_forwarded_states(layer, inputs, outputs, layer_index):
        encoder_outputs[f'dropout_forwarded_states_{layer_index}'] = outputs

    # Custom hooks to calculate residuals
    def hook_fn_encoder_residual_self_attention(layer, inputs, outputs, layer_index):
        input_to_layer_norm = encoder_outputs[f'input_to_layer_norm_{layer_index}']
        encoder_outputs[f'encoder_residual_self_attention_{layer_index}'] = input_to_layer_norm + outputs

    def hook_fn_encoder_residual_feed_forward(layer, inputs, outputs, layer_index):
        input_to_ff_layer_norm = encoder_outputs[f'input_to_ff_layer_norm_{layer_index}']
        encoder_outputs[f'encoder_residual_feed_forward_{layer_index}'] = input_to_ff_layer_norm + outputs

    # Hook for Final Layer normalization and dropout for Encoder
    def hook_fn_normalized_encoder_output(layer, inputs, outputs):
        encoder_outputs['encoder_layer_norm'] = outputs

    def hook_fn_dropout_normalized_encoder_output(layer, inputs, outputs):
        encoder_outputs['dropout_normalized_encoder_output'] = outputs
        
    # Register hooks to the encoder submodules
    for i, layer in enumerate(model.encoder.block):
        layer.layer[0].layer_norm.layer_index = i
        layer.layer[0].SelfAttention.layer_index = i
        layer.layer[0].dropout.layer_index = i
        layer.layer[1].layer_norm.layer_index = i
        layer.layer[1].DenseReluDense.layer_index = i
        layer.layer[1].dropout.layer_index = i

        wrap_call(layer.layer[0].layer_norm, hook_fn__encoder_normalized_hidden_states, i)
        wrap_call(layer.layer[0].SelfAttention, hook_fn_encoder_self_attention_outputs, i)
        wrap_call(layer.layer[0].dropout, hook_fn_encoder_dropout_attention_output, i)
        wrap_call(layer.layer[0].dropout, hook_fn_encoder_residual_self_attention, i)

        wrap_call(layer.layer[1].layer_norm, hook_fn_encoder_normalized_forwarded_states, i)
        wrap_call(layer.layer[1].DenseReluDense, hook_fn_encoder_forwarded_states, i)
        wrap_call(layer.layer[1].dropout, hook_fn_encoder_dropout_forwarded_states, i)
        wrap_call(layer.layer[1].dropout, hook_fn_encoder_residual_feed_forward, i)

    wrap_call(model.encoder.final_layer_norm, hook_fn_normalized_encoder_output)
    wrap_call(model.encoder.dropout, hook_fn_dropout_normalized_encoder_output)
    
    ############################ Hook for Decoder ################################
    def hook_fn_decoder_embedding(layer, inputs, outputs):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp]['decoder_embedding'] = outputs

    def hook_fn_decoder_normalized_hidden_states(layer, inputs, outputs, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_inputs[timestamp][f'decoder_layer_norm_{layer_index}_0'] = inputs
        decoder_outputs[timestamp][f'decoder_layer_norm_{layer_index}_0'] = outputs
        decoder_outputs[timestamp][f'input_to_layer_norm_{layer_index}'] = inputs[0]

    def hook_fn_decoder_self_attention_outputs(layer, inputs, outputs, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_inputs[timestamp][f'decoder_self_attention_{layer_index}'] = inputs
        decoder_outputs[timestamp][f'decoder_self_attention_{layer_index}'] = outputs[0]

    def hook_fn_decoder_dropout_attention_output(layer, inputs, outputs, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp][f'dropout_attention_output_{layer_index}'] = outputs

    def hook_fn_decoder_normalized_cross_attn_hidden_states(layer, inputs, outputs, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_inputs[timestamp][f'decoder_layer_norm_{layer_index}_1'] = inputs
        decoder_outputs[timestamp][f'decoder_layer_norm_{layer_index}_1'] = outputs
        decoder_outputs[timestamp][f'input_to_cross_attn_layer_norm_{layer_index}'] = inputs[0]

    def hook_fn_decoder_cross_attention_outputs(layer, inputs, outputs, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_inputs[timestamp][f'decoder_cross_attention_{layer_index}'] = {'query': inputs[0], 'key': outputs, 'value': outputs}
        decoder_outputs[timestamp][f'decoder_cross_attention_{layer_index}'] = outputs[0]

    def hook_fn_decoder_dropout_cross_attn_output(layer, inputs, outputs, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp][f'dropout_cross_attn_output_{layer_index}'] = outputs

    def hook_fn_decoder_normalized_forwarded_states(layer, inputs, outputs, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_inputs[timestamp][f'decoder_layer_norm_{layer_index}_2'] = inputs
        decoder_outputs[timestamp][f'decoder_layer_norm_{layer_index}_2'] = outputs
        decoder_outputs[timestamp][f'input_to_ff_layer_norm_{layer_index}'] = inputs[0]

    def hook_fn_decoder_forwarded_states(layer, inputs, outputs, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_inputs[timestamp][f'decoder_feed_forward_{layer_index}'] = inputs[0]
        decoder_outputs[timestamp][f'decoder_feed_forward_{layer_index}'] = outputs

    def hook_fn_decoder_dropout_forwarded_states(layer, inputs, outputs, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp][f'dropout_forwarded_states_{layer_index}'] = outputs

    def hook_fn_decoder_residual_self_attention(layer, inputs, outputs, layer_index):
        global token_idx
        timestamp = get_timestamp()
        input_to_layer_norm = decoder_outputs[timestamp][f'input_to_layer_norm_{layer_index}']
        decoder_outputs[timestamp][f'decoder_residual_self_attention_{layer_index}'] = input_to_layer_norm + outputs

    def hook_fn_decoder_residual_cross_attention(layer, inputs, outputs, layer_index):
        global token_idx
        timestamp = get_timestamp()
        input_to_layer_norm = decoder_outputs[timestamp][f'input_to_cross_attn_layer_norm_{layer_index}']
        decoder_outputs[timestamp][f'decoder_residual_cross_attention_{layer_index}'] = input_to_layer_norm + outputs

    def hook_fn_decoder_residual_feed_forward(layer, inputs, outputs, layer_index):
        global token_idx
        timestamp = get_timestamp()
        input_to_ff_layer_norm = decoder_outputs[timestamp][f'input_to_ff_layer_norm_{layer_index}']
        decoder_outputs[timestamp][f'decoder_residual_feed_forward_{layer_index}'] = input_to_ff_layer_norm + outputs

    def hook_fn_normalized_decoder_output(layer, inputs, outputs):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp]['decoder_layer_norm'] = outputs

    def hook_fn_dropout_normalized_decoder_output(layer, inputs, outputs):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp]['dropout_normalized_decoder_output'] = outputs

    def hook_fn_final_logits(layer, inputs, outputs):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp]['decoder_lm_head'] = outputs.logits

    # Register hooks to the decoder submodules
    wrap_call(model.decoder.embed_tokens, hook_fn_decoder_embedding)

    for i, layer in enumerate(model.decoder.block):
        layer.layer[0].layer_norm.layer_index = i
        layer.layer[0].SelfAttention.layer_index = i
        layer.layer[0].dropout.layer_index = i
        layer.layer[1].layer_norm.layer_index = i
        layer.layer[1].EncDecAttention.layer_index = i
        layer.layer[1].dropout.layer_index = i
        layer.layer[2].layer_norm.layer_index = i
        layer.layer[2].DenseReluDense.layer_index = i
        layer.layer[2].dropout.layer_index = i

        wrap_call(layer.layer[0].layer_norm, hook_fn_decoder_normalized_hidden_states, i)
        wrap_call(layer.layer[0].SelfAttention, hook_fn_decoder_self_attention_outputs, i)
        wrap_call(layer.layer[0].dropout, hook_fn_decoder_dropout_attention_output, i)
        wrap_call(layer.layer[0].dropout, hook_fn_decoder_residual_self_attention, i)

        wrap_call(layer.layer[1].layer_norm, hook_fn_decoder_normalized_cross_attn_hidden_states, i)
        wrap_call(layer.layer[1].EncDecAttention, hook_fn_decoder_cross_attention_outputs, i)
        wrap_call(layer.layer[1].dropout, hook_fn_decoder_dropout_cross_attn_output, i)
        wrap_call(layer.layer[1].dropout, hook_fn_decoder_residual_cross_attention, i)

        wrap_call(layer.layer[2].layer_norm, hook_fn_decoder_normalized_forwarded_states, i)
        wrap_call(layer.layer[2].DenseReluDense, hook_fn_decoder_forwarded_states, i)
        wrap_call(layer.layer[2].dropout, hook_fn_decoder_dropout_forwarded_states, i)
        wrap_call(layer.layer[2].dropout, hook_fn_decoder_residual_feed_forward, i)

    wrap_call(model.decoder.final_layer_norm, hook_fn_normalized_decoder_output)
    wrap_call(model.decoder.dropout, hook_fn_dropout_normalized_decoder_output)

    # Register hook for the final logits by wrapping the call method of the model itself
    original_call = model.call
    def hooked_call(*args, **kwargs):
        outputs = original_call(*args, **kwargs)
        hook_fn_final_logits(model, args, outputs)
        return outputs

    model.call = hooked_call
    hook_manager.append((model, original_call))

    # Function to remove hooks
    def remove_hooks():
        for layer, original_call in hook_manager:
            layer.call = original_call
        hook_manager.clear()

    # Function to get shape
    def get_shape(value):
        if isinstance(value, tf.Tensor):
            return value.shape
        elif isinstance(value, tuple):
            return [get_shape(v) for v in value if v is not None]
        elif isinstance(value, list):
            return [get_shape(v) for v in value if v is not None]
        elif isinstance(value, dict):
            return {k: get_shape(v) for k, v in value.items()}
        else:
            return None

    # Function to increment token index
    def increment_token_idx():
        global token_idx
        token_idx += 1
        
    encoding = tokenizer(input_text, return_tensors='tf')
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    embedding_output = capture_encoder_embeddings(model, tokenizer, input_text)
    encoder_outputs['encoder_embedding'] = embedding_output
    
    # Initialize decoder_input_ids with the start token
    decoder_start_token_id = model.config.decoder_start_token_id
    decoder_input_ids = tf.fill((tf.shape(input_ids)[0], 1), decoder_start_token_id)
    
    # Reset token_idx before generating
    token_idx = 0
    max_length = model.config.n_positions if hasattr(model.config, 'n_positions') else model.config.d_model
    generated_tokens = []
    
    for _ in range(max_length):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)
        next_token_id = tf.expand_dims(next_token_id, axis=-1)
        generated_tokens.append(next_token_id.numpy().item())
        decoder_input_ids = tf.concat([decoder_input_ids, next_token_id], axis=-1)
        increment_token_idx()

        if next_token_id.numpy().item() == model.config.eos_token_id:
            break
    
    # Merge the encoder_outputs with timestep decoder_outputs to generate timestep wise outputs of the model
    outputs = {}

    for i in range(len(decoder_outputs)):
        outputs[f'{i}'] = {**encoder_outputs, **decoder_outputs[f'{i}']}
        
    return outputs, generated_tokens
