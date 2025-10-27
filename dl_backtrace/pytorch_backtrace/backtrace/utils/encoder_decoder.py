import torch
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

        # Keep track of component type in a separate dictionary
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
    encoder_embeddings = add_component(ltree, 'encoder_embedding', model.encoder.embed_tokens, child=None)

    # Add encoder layers dynamically
    current_child = 'encoder_embedding'
    for i, layer in enumerate(model.encoder.block):
        encoder_layer_norm_0 = add_component(ltree, f'encoder_layer_norm_{i}_0', 'Layer_Norm', child=current_child)
        encoder_self_attention = add_component(ltree, f'encoder_self_attention_{i}', 'Self_Attention', child=f'encoder_layer_norm_{i}_0')
        encoder_residual_self_attention = add_component(ltree, f'encoder_residual_self_attention_{i}', 'Residual', child=[current_child, f'encoder_self_attention_{i}'])

        encoder_layer_norm_1 = add_component(ltree, f'encoder_layer_norm_{i}_1', 'Layer_Norm', child=f'encoder_residual_self_attention_{i}')
        encoder_feed_forward = add_component(ltree, f'encoder_feed_forward_{i}',  'Feed_Forward', child=f'encoder_layer_norm_{i}_1')
        encoder_residual_feed_forward = add_component(ltree, f'encoder_residual_feed_forward_{i}',  'Residual', child=[f'encoder_residual_self_attention_{i}', f'encoder_feed_forward_{i}'])

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
        decoder_residual_self_attention = add_component(ltree, f'decoder_residual_self_attention_{i}',  'Residual', child=[current_child, f'decoder_self_attention_{i}'])

        decoder_layer_norm_1 = add_component(ltree, f'decoder_layer_norm_{i}_1', 'Layer_Norm', child=f'decoder_residual_self_attention_{i}')
        decoder_cross_attention = add_component(ltree, f'decoder_cross_attention_{i}', 'Cross_Attention', child=['encoder_layer_norm', f'decoder_layer_norm_{i}_1'])
        decoder_residual_cross_attention = add_component(ltree, f'decoder_residual_cross_attention_{i}',  'Residual', child=[f'decoder_residual_self_attention_{i}', f'decoder_cross_attention_{i}'])

        decoder_layer_norm_2 = add_component(ltree, f'decoder_layer_norm_{i}_2', 'Layer_Norm', child=f'decoder_residual_cross_attention_{i}')
        decoder_feed_forward = add_component(ltree, f'decoder_feed_forward_{i}',  'Feed_Forward', child=f'decoder_layer_norm_{i}_2')
        decoder_residual_feed_forward = add_component(ltree, f'decoder_residual_feed_forward_{i}',  'Residual', child=[f'decoder_residual_cross_attention_{i}', f'decoder_feed_forward_{i}'])

        current_child = f'decoder_residual_feed_forward_{i}'

    if hasattr(model.decoder, 'final_layer_norm'):
        decoder_final_layer_norm = add_component(ltree, 'decoder_layer_norm', model.decoder.final_layer_norm, child=current_child)
        current_child = 'decoder_layer_norm'

    # Decoder LM-Head
    if hasattr(model, 'lm_head'):
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
    model_resource = (layer_tree, ltree, outputs, inputs)
    
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
    for name, param in model.named_parameters():
        if 'shared' in name:
            weights_dict['encoder_embedding'][name] = param.data.cpu().numpy()
            weights_dict['decoder_embedding'][name] = param.data.cpu().numpy()
            weights_dict['decoder_lm_head'][name] = param.data.cpu().numpy()

        elif 'encoder.block' in name:
            layer = name.split('.')[2]
            sub_layer = name.split('.')[4]
            submodule = name.split('.')[5]

            if 'SelfAttention' in submodule and f'encoder_self_attention_{layer}' not in weights_dict:
                weights_dict[f'encoder_self_attention_{layer}'] = {}
            if 'layer_norm' in submodule and f'encoder_layer_norm_{layer}' not in weights_dict:
                weights_dict[f'encoder_layer_norm_{layer}'] = {}
            if 'DenseReluDense' in submodule and f'encoder_feed_forward_{layer}' not in weights_dict:
                weights_dict[f'encoder_feed_forward_{layer}'] = {}

            if 'SelfAttention' in submodule:
                weights_dict[f'encoder_self_attention_{layer}'][name] = param.data.cpu().numpy()
            elif 'layer_norm' in submodule:
                weights_dict[f'encoder_layer_norm_{layer}'][name] = param.data.cpu().numpy()
            elif 'DenseReluDense' in submodule:
                weights_dict[f'encoder_feed_forward_{layer}'][name] = param.data.cpu().numpy()

        elif 'encoder.final_layer_norm.weight' in name:
            weights_dict['encoder_layer_norm'][name] = param.data.cpu().numpy()

        elif 'decoder.block' in name:
            layer = name.split('.')[2]
            sub_layer = name.split('.')[4]
            submodule = name.split('.')[5]

            if 'SelfAttention' in submodule and f'decoder_self_attention_{layer}' not in weights_dict:
                weights_dict[f'decoder_self_attention_{layer}'] = {}
            if 'layer_norm' in submodule and f'decoder_layer_norm_{layer}' not in weights_dict:
                weights_dict[f'decoder_layer_norm_{layer}'] = {}
            if 'EncDecAttention' in submodule and f'decoder_cross_attention_{layer}' not in weights_dict:
                weights_dict[f'decoder_cross_attention_{layer}'] = {}
            if 'DenseReluDense' in submodule and f'decoder_feed_forward_{layer}' not in weights_dict:
                weights_dict[f'decoder_feed_forward_{layer}'] = {}

            if 'SelfAttention' in submodule:
                weights_dict[f'decoder_self_attention_{layer}'][name] = param.data.cpu().numpy()
            elif 'layer_norm' in submodule:
                weights_dict[f'decoder_layer_norm_{layer}'][name] = param.data.cpu().numpy()
            elif 'EncDecAttention' in submodule:
                weights_dict[f'decoder_cross_attention_{layer}'][name] = param.data.cpu().numpy()
            elif 'DenseReluDense' in submodule:
                weights_dict[f'decoder_feed_forward_{layer}'][name] = param.data.cpu().numpy()

        elif 'decoder.final_layer_norm.weight' in name:
            weights_dict['decoder_layer_norm'][name] = param.data.cpu().numpy()

    return weights_dict


def calculate_encoder_decoder_output(input_text, model, tokenizer):
    encoder_inputs = {}
    encoder_outputs = {}
    encoder_hooks = []
    decoder_outputs = defaultdict(lambda: defaultdict(dict))
    decoder_inputs = defaultdict(lambda: defaultdict(dict))
    decoder_hooks = []
    
    def capture_encoder_embeddings(model, tokenizer, input_text):
        # Ensure the model is in evaluation mode
        model.eval()

        # Tokenize the input text
        encoding = tokenizer(input_text, return_tensors='pt')
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Manually capture the embedding output
        with torch.no_grad():
            embedding_output = model.encoder.embed_tokens(input_ids)

        # Return the captured embeddings
        return embedding_output
    
    ## ------------ Hook function for Self-Attention Block --------------------
    def hook_fn__encoder_normalized_hidden_states(module, input, output):
        encoder_inputs[f'encoder_layer_norm_{module.layer_index}_0'] = input
        encoder_outputs[f'encoder_layer_norm_{module.layer_index}_0'] = output
        encoder_outputs[f'input_to_layer_norm_{module.layer_index}'] = input[0]

    def hook_fn_encoder_self_attention_outputs(module, input, output):
        encoder_inputs[f'encoder_self_attention_{module.layer_index}'] = input
        encoder_outputs[f'encoder_self_attention_{module.layer_index}'] = output[0]

    def hook_fn_encoder_dropout_attention_output(module, input, output):
        encoder_outputs[f'encoder_dropout_attention_{module.layer_index}'] = output[0]

    ## ------------  Hook function for Feed-Forward Block --------------
    def hook_fn_encoder_normalized_forwarded_states(module, input, output):
        encoder_inputs[f'encoder_layer_norm_{module.layer_index}_1'] = input
        encoder_outputs[f'encoder_layer_norm_{module.layer_index}_1'] = output
        encoder_outputs[f'input_to_ff_layer_norm_{module.layer_index}'] = input[0]

    def hook_fn_encoder_forwarded_states(module, input, output):
        encoder_inputs[f'encoder_feed_forward_{module.layer_index}'] = input
        encoder_outputs[f'encoder_feed_forward_{module.layer_index}'] = output

    def hook_fn_encoder_dropout_forwarded_states(module, input, output):
        encoder_outputs[f'dropout_forwarded_states_{module.layer_index}'] = output

    # Custom hooks to calculate residuals
    def hook_fn_encoder_residual_self_attention(layer_index):
        def hook(module, input, output):
            input_to_layer_norm = encoder_outputs[f'input_to_layer_norm_{layer_index}']
            encoder_outputs[f'encoder_residual_self_attention_{layer_index}'] = input_to_layer_norm + output
        return hook

    def hook_fn_encoder_residual_feed_forward(layer_index):
        def hook(module, input, output):
            input_to_ff_layer_norm = encoder_outputs[f'input_to_ff_layer_norm_{layer_index}']
            encoder_outputs[f'encoder_residual_feed_forward_{layer_index}'] = input_to_ff_layer_norm + output
        return hook

    # Hook for Final Layer normalization and dropout for Encoder
    def hook_fn_normalized_encoder_output(module, input, output):
        encoder_outputs['encoder_layer_norm'] = output

    def hook_fn_dropout_normalized_encoder_output(module, input, output):
        encoder_outputs['dropout_normalized_encoder_output'] = output
        
    # Register hooks to the encoder submodules
    for i, layer in enumerate(model.encoder.block):
        # Set layer_index attribute to all relevant submodules
        layer.layer[0].layer_norm.layer_index = i
        layer.layer[0].SelfAttention.layer_index = i
        layer.layer[0].dropout.layer_index = i
        layer.layer[1].layer_norm.layer_index = i
        layer.layer[1].DenseReluDense.layer_index = i
        layer.layer[1].dropout.layer_index = i

        encoder_hooks.append(layer.layer[0].layer_norm.register_forward_hook(hook_fn__encoder_normalized_hidden_states))
        encoder_hooks.append(layer.layer[0].SelfAttention.register_forward_hook(hook_fn_encoder_self_attention_outputs))
        encoder_hooks.append(layer.layer[0].dropout.register_forward_hook(hook_fn_encoder_dropout_attention_output))
        encoder_hooks.append(layer.layer[0].dropout.register_forward_hook(hook_fn_encoder_residual_self_attention(i)))  # Custom hook for residual self-attention

        encoder_hooks.append(layer.layer[1].layer_norm.register_forward_hook(hook_fn_encoder_normalized_forwarded_states))
        encoder_hooks.append(layer.layer[1].DenseReluDense.register_forward_hook(hook_fn_encoder_forwarded_states))
        encoder_hooks.append(layer.layer[1].dropout.register_forward_hook(hook_fn_encoder_dropout_forwarded_states))
        encoder_hooks.append(layer.layer[1].dropout.register_forward_hook(hook_fn_encoder_residual_feed_forward(i)))  # Custom hook for residual feed-forward

    # Register hook for Final Layer Normalization and dropout for Encoder
    encoder_hooks.append(model.encoder.final_layer_norm.register_forward_hook(hook_fn_normalized_encoder_output))
    encoder_hooks.append(model.encoder.dropout.register_forward_hook(hook_fn_dropout_normalized_encoder_output))
    
    ############################ Hook for Decoder ################################
    # Global variable to keep track of the token index
    token_idx = 0

    # Function to generate timestamp (token index)
    def get_timestamp():
        return str(token_idx)

    # Hook functions to capture input embedding
    def hook_fn_decoder_embedding(module, input, output):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp]['decoder_embedding'] = output.detach().clone()

    ## ------------ Hook function for Self-Attention Block --------------------
    def hook_fn_decoder_normalized_hidden_states(module, input, output, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_inputs[timestamp][f'decoder_layer_norm_{layer_index}_0'] = input
        decoder_outputs[timestamp][f'decoder_layer_norm_{layer_index}_0'] = output
        decoder_outputs[timestamp][f'input_to_layer_norm_{layer_index}'] = input[0]

    def hook_fn_decoder_self_attention_outputs(module, input, output, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_inputs[timestamp][f'decoder_self_attention_{layer_index}'] = input
        decoder_outputs[timestamp][f'decoder_self_attention_{layer_index}'] = output[0]

    def hook_fn_decoder_dropout_attention_output(module, input, output, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp][f'dropout_attention_output_{layer_index}'] = output

    ## ------------ Hook function for Cross-Attention Block --------------------
    def hook_fn_decoder_normalized_cross_attn_hidden_states(module, input, output, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_inputs[timestamp][f'decoder_layer_norm_{layer_index}_1'] = input
        decoder_outputs[timestamp][f'decoder_layer_norm_{layer_index}_1'] = output
        decoder_outputs[timestamp][f'input_to_cross_attn_layer_norm_{layer_index}'] = input[0]

    def hook_fn_decoder_cross_attention_outputs(module, input, output, layer_index):
        global token_idx
        timestamp = get_timestamp()
        key_value_states = encoder_outputs['dropout_normalized_encoder_output']
        query_state = input[0]

        inputs = {
            'query': input[0],
            'key': key_value_states,
            'value': key_value_states
        }

        decoder_inputs[timestamp][f'decoder_cross_attention_{layer_index}'] = inputs
        decoder_outputs[timestamp][f'decoder_cross_attention_{layer_index}'] = output[0]

    def hook_fn_decoder_dropout_cross_attn_output(module, input, output, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp][f'dropout_cross_attn_output_{layer_index}'] = output

    ## ------------ Hook function for Feed-Forward Block --------------------
    def hook_fn_decoder_normalized_forwarded_states(module, input, output, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_inputs[timestamp][f'decoder_layer_norm_{layer_index}_2'] = input
        decoder_outputs[timestamp][f'decoder_layer_norm_{layer_index}_2'] = output
        decoder_outputs[timestamp][f'input_to_ff_layer_norm_{layer_index}'] = input[0]

    def hook_fn_decoder_forwarded_states(module, input, output, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_inputs[timestamp][f'decoder_feed_forward_{layer_index}'] = input[0]
        decoder_outputs[timestamp][f'decoder_feed_forward_{layer_index}'] = output

    def hook_fn_decoder_dropout_forwarded_states(module, input, output, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp][f'dropout_forwarded_states_{layer_index}'] = output

    # Custom hooks to calculate residuals
    def hook_fn_decoder_residual_self_attention(layer_index):
        def hook(module, input, output):
            global token_idx
            timestamp = get_timestamp()
            input_to_layer_norm = decoder_outputs[timestamp][f'input_to_layer_norm_{layer_index}']
            decoder_outputs[timestamp][f'decoder_residual_self_attention_{layer_index}'] = input_to_layer_norm + output
        return hook

    def hook_fn_decoder_residual_cross_attention(layer_index):
        def hook(module, input, output):
            global token_idx
            timestamp = get_timestamp()
            input_to_layer_norm = decoder_outputs[timestamp][f'input_to_cross_attn_layer_norm_{layer_index}']
            decoder_outputs[timestamp][f'decoder_residual_cross_attention_{layer_index}'] = input_to_layer_norm + output
        return hook

    def hook_fn_decoder_residual_feed_forward(layer_index):
        def hook(module, input, output):
            global token_idx
            timestamp = get_timestamp()
            input_to_ff_layer_norm = decoder_outputs[timestamp][f'input_to_ff_layer_norm_{layer_index}']
            decoder_outputs[timestamp][f'decoder_residual_feed_forward_{layer_index}'] = input_to_ff_layer_norm + output
        return hook

    # Hook for Final Layer normalization and dropout for Decoder
    def hook_fn_normalized_decoder_output(module, input, output):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp]['decoder_layer_norm'] = output

    def hook_fn_dropout_normalized_decoder_output(module, input, output):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp]['dropout_normalized_decoder_output'] = output

    # Hook for the Decoder LM-Head
    def hook_fn_lm_head(module, input, output):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp]['decoder_lm_head'] = output
                
    
    # Register hook for embedding
    decoder_hooks.append(model.decoder.embed_tokens.register_forward_hook(lambda module, input, output: hook_fn_decoder_embedding(module, input, output)))

    # Register hooks to the decoder submodules
    for i, layer in enumerate(model.decoder.block):
        # Set layer_index attribute to all relevant submodules
        layer.layer[0].layer_norm.layer_index = i
        layer.layer[0].SelfAttention.layer_index = i
        layer.layer[0].dropout.layer_index = i
        layer.layer[1].layer_norm.layer_index = i
        layer.layer[1].EncDecAttention.layer_index = i
        layer.layer[1].dropout.layer_index = i
        layer.layer[2].layer_norm.layer_index = i
        layer.layer[2].DenseReluDense.layer_index = i
        layer.layer[2].dropout.layer_index = i

        decoder_hooks.append(layer.layer[0].layer_norm.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_normalized_hidden_states(module, input, output, layer_index=i)))
        decoder_hooks.append(layer.layer[0].SelfAttention.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_self_attention_outputs(module, input, output, layer_index=i)))
        decoder_hooks.append(layer.layer[0].dropout.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_dropout_attention_output(module, input, output, layer_index=i)))
        decoder_hooks.append(layer.layer[0].dropout.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_residual_self_attention(i)(module, input, output)))

        decoder_hooks.append(layer.layer[1].layer_norm.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_normalized_cross_attn_hidden_states(module, input, output, layer_index=i)))
        decoder_hooks.append(layer.layer[1].EncDecAttention.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_cross_attention_outputs(module, input, output, layer_index=i)))
        decoder_hooks.append(layer.layer[1].dropout.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_dropout_cross_attn_output(module, input, output, layer_index=i)))
        decoder_hooks.append(layer.layer[1].dropout.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_residual_cross_attention(i)(module, input, output)))

        decoder_hooks.append(layer.layer[2].layer_norm.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_normalized_forwarded_states(module, input, output, layer_index=i)))
        decoder_hooks.append(layer.layer[2].DenseReluDense.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_forwarded_states(module, input, output, layer_index=i)))
        decoder_hooks.append(layer.layer[2].dropout.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_dropout_forwarded_states(module, input, output, layer_index=i)))
        decoder_hooks.append(layer.layer[2].dropout.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_residual_feed_forward(i)(module, input, output)))

    # Register hook for Final Layer Normalization and dropout for Decoder
    decoder_hooks.append(model.decoder.final_layer_norm.register_forward_hook(lambda module, input, output: hook_fn_normalized_decoder_output(module, input, output)))
    decoder_hooks.append(model.decoder.dropout.register_forward_hook(lambda module, input, output: hook_fn_dropout_normalized_decoder_output(module, input, output)))

    # Register hook for the Decoder LM-Head
    decoder_hooks.append(model.lm_head.register_forward_hook(lambda module, input, output: hook_fn_lm_head(module, input, output)))
    
    
    # Function to increment token_idx
    def increment_token_idx():
        global token_idx
        token_idx += 1
        
    encoding = tokenizer(input_text, return_tensors='pt')
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    embedding_output = capture_encoder_embeddings(model, tokenizer, input_text)
    encoder_outputs['encoder_embedding'] = embedding_output
    
    # Initialize decoder_input_ids with the start token
    decoder_input_ids = torch.full(
        (input_ids.shape[0], 1), model.config.decoder_start_token_id, dtype=torch.long
    )
    
    # Reset token_idx before generating
    token_idx = 0
    max_length = model.config.max_position_embeddings  # Set the maximum length for generation
    generated_tokens = []
    
    for _ in range(max_length):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token_id.item())
        decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
        increment_token_idx()

        if next_token_id.item() == model.config.eos_token_id:
            break
        
    # Deregister hooks
    for handle in encoder_hooks:
        handle.remove()

    # Deregister hooks
    for handle in decoder_hooks:
        handle.remove()
        
    # Merge the encoder_outputs with timestep decoder_outputs to generate timestep wise outputs of the model
    outputs = {}

    for i in range(len(decoder_outputs)):
        outputs[f'{i}'] = {**encoder_outputs, **decoder_outputs[f'{i}']}
        
    return outputs, generated_tokens

