import torch
from collections import defaultdict


def build_jetmoe_tree(model, root='jet_moe'):
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

        # # Keep track of component type in a separate dictionary
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
    # add_component(ltree, root, model, parent=None)
    decoder_embeddings = add_component(ltree, 'decoder_embeddings', 'Embeddings', child=None)

    # Add jet_moe layers dynamically
    current_child = 'decoder_embeddings'
    for i, layer in enumerate(model.model.layers):
        decoder_layer_norm_0 = add_component(ltree, f'decoder_layer_norm_{i}_0', 'Layer_Norm', child=current_child)
        decoder_self_attention = add_component(ltree, f'decoder_self_attention_{i}', 'MoE_Self_Attention', child=f'decoder_layer_norm_{i}_0')
        decoder_residual_self_attention = add_component(ltree, f'decoder_residual_self_attention_{i}', 'Residual', child=[current_child, f'decoder_self_attention_{i}'])

        decoder_layer_norm_1 = add_component(ltree, f'decoder_layer_norm_{i}_1', 'Layer_Norm', child=f'decoder_self_attention_{i}')
        decoder_feed_forward = add_component(ltree, f'decoder_feed_forward_{i}', 'MoE_Feed_Forward', child=f'decoder_layer_norm_{i}_1')
        decoder_residual_feed_forward = add_component(ltree, f'decoder_residual_feed_forward_{i}', 'Residual', child=[f'decoder_residual_self_attention_{i}', f'decoder_feed_forward_{i}'])

        current_child = f'decoder_residual_feed_forward_{i}'

    if hasattr(model.model, 'norm'):
        decoder_final_layer_norm = add_component(ltree, 'decoder_layer_norm', 'Layer_Norm', child=current_child)
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

    # model_resource = (layer_tree, ltree, outputs, inputs)
    model_resource = {
        "layers": layer_tree,
        "graph": ltree,
        "outputs": outputs,
        "inputs": inputs
    }

    return model_resource, layer_stack


def extract_jetmoe_weights(model):
    # Initialize a dictionary to hold the weights
    weights_dict = {
        'decoder_embeddings': {},
        'decoder_layer_norm': {},
        'decoder_lm_head': {}
    }

    # Loop through the layers to initialize the weight dictionaries for each
    for i in range(model.config.num_hidden_layers):
        weights_dict[f'decoder_layer_norm_{i}_0'] = {}
        weights_dict[f'decoder_self_attention_{i}'] = {}
        weights_dict[f'decoder_layer_norm_{i}_1'] = {}
        weights_dict[f'decoder_feed_forward_{i}'] = {}

    # Extract the model's parameters and organize them into the dictionary
    for name, param in model.named_parameters():
        if isinstance(param, torch.Tensor):
            param_np = param.data.numpy()
        else:
            param_np = param

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

    if hasattr(model, 'lm_head'):
        lm_head_weights = model.lm_head.weight.data.numpy()
        weights_dict['decoder_lm_head']['lm_head.weight'] = lm_head_weights

    return weights_dict


def create_jetmoe_output(input_text, model, tokenizer, max_length, device):
    # Initialize variables
    token_idx = 0
    decoder_outputs = defaultdict(lambda: defaultdict(dict))
    decoder_inputs = defaultdict(lambda: defaultdict(dict))
    decoder_hooks = []

    # Function to generate timestamp (token index)
    def get_timestamp():
        return str(token_idx)

    # Hook functions to capture input embedding
    def hook_fn_decoder_embedding(module, input, output):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp]['decoder_embeddings'] = output.detach().clone()

    # Simplified hook function that stores only necessary outputs
    def hook_fn_decoder_normalized_hidden_states(module, input, output, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp][f'decoder_layer_norm_{layer_index}_0'] = output.detach().clone()

    def hook_fn_decoder_self_attention_outputs(module, input, output, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp][f'decoder_self_attention_{layer_index}'] = output[0].detach().clone()

    # Feed-Forward Hook Functions
    def hook_fn_decoder_normalized_forwarded_states(module, input, output, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp][f'decoder_layer_norm_{layer_index}_1'] = output.detach().clone()

    def hook_fn_decoder_forwarded_states(module, input, output, layer_index):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp][f'decoder_feed_forward_{layer_index}'] = output[0].detach().clone()


    # Custom hooks to calculate residuals
    def hook_fn_decoder_residual_self_attention(layer_index):
        def hook(module, input, output):
            global token_idx
            timestamp = get_timestamp()
            input_to_layer_norm = decoder_outputs[timestamp][f'decoder_layer_norm_{layer_index}_0']

            if isinstance(output, tuple):
                output = output[0]

            decoder_outputs[timestamp][f'decoder_residual_self_attention_{layer_index}'] = (input_to_layer_norm + output).detach().clone()
        return hook

    def hook_fn_decoder_residual_feed_forward(layer_index):
        def hook(module, input, output):
            global token_idx
            timestamp = get_timestamp()
            input_to_ff_layer_norm = decoder_outputs[timestamp][f'decoder_layer_norm_{layer_index}_1']

            if isinstance(output, tuple):
                output = output[0]

            decoder_outputs[timestamp][f'decoder_residual_feed_forward_{layer_index}'] = (input_to_ff_layer_norm + output).detach().clone()
        return hook

    # Hook for Final Layer normalization and dropout for Decoder
    def hook_fn_normalized_decoder_output(module, input, output):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp]['decoder_layer_norm'] = output.detach().clone()

    # Hook for the Decoder LM-Head
    def hook_fn_lm_head(module, input, output):
        global token_idx
        timestamp = get_timestamp()
        decoder_outputs[timestamp]['decoder_lm_head'] = output.detach().clone()

    # Register hook for embedding
    decoder_hooks.append(model.model.embed_tokens.register_forward_hook(hook_fn_decoder_embedding))

    # Register hooks to the decoder submodules
    for i, layer in enumerate(model.model.layers):
        decoder_hooks.append(layer.input_layernorm.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_normalized_hidden_states(module, input, output, layer_index=i)))
        decoder_hooks.append(layer.self_attention.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_self_attention_outputs(module, input, output, layer_index=i)))
        decoder_hooks.append(layer.self_attention.register_forward_hook(hook_fn_decoder_residual_self_attention(i)))

        # Feed-Forward Block Hooks
        decoder_hooks.append(layer.post_attention_layernorm.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_normalized_forwarded_states(module, input, output, layer_index=i)))
        decoder_hooks.append(layer.mlp.register_forward_hook(lambda module, input, output, i=i: hook_fn_decoder_forwarded_states(module, input, output, layer_index=i)))
        decoder_hooks.append(layer.mlp.register_forward_hook(hook_fn_decoder_residual_feed_forward(i)))

    decoder_hooks.append(model.model.norm.register_forward_hook(hook_fn_normalized_decoder_output))
    decoder_hooks.append(model.lm_head.register_forward_hook(hook_fn_lm_head))

    # Function to increment token_idx
    def increment_token_idx():
        global token_idx
        token_idx += 1

    encoding = tokenizer(input_text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    model = model.to(device)
    input_ids = input_ids.to(device)

    # Reset token_idx before generating
    token_idx = 0
    if max_length is None:
        max_length = model.config.max_position_embeddings
    
    generated_tokens = []
    
    for _ in range(max_length):
        outputs = model(input_ids=input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token_id.item())
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        increment_token_idx()

        if next_token_id.item() == model.config.eos_token_id:
            break
    
    # Deregister hooks
    for handle in decoder_hooks:
        handle.remove()
        
    return decoder_outputs, generated_tokens
