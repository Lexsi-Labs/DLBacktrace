import torch


def build_encoder_tree(model, root='bert'):
    # Initialize the tree structure
    ltree = {}
    layer_tree = {}
    inputs = []
    outputs = []
    intermediates = []

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
    embeddings = add_component(ltree, 'embeddings', 'Embeddings', child=None)

    # Add encoder layers dynamically
    current_child = 'embeddings'
    for i, layer in enumerate(model.bert.encoder.layer):
        attention = add_component(ltree, f'attention_{i}', 'Self_Attention', child=current_child)
        add_and_layer_norm_0 = add_component(ltree, f'add_and_layer_norm_{i}_0', 'Residual', child=[f'attention_{i}', current_child])
        feed_forward = add_component(ltree, f'feed_forward_{i}', 'Feed_Forward', child=f'add_and_layer_norm_{i}_0')
        add_and_layer_norm_1 = add_component(ltree, f'add_and_layer_norm_{i}_1', 'Residual', child=[f'feed_forward_{i}', f'add_and_layer_norm_{i}_0'])
        current_child = f'add_and_layer_norm_{i}_1'  # Update current_child to link this layer's output to the next layer's input

    # Optionally add pooler layer if present
    if hasattr(model.bert, 'pooler'):
        pooler = add_component(ltree, 'pooler', 'Pooler', child=current_child)
        current_child = 'pooler'

    if hasattr(model, 'classifier'):
        classifier = add_component(ltree, 'classifier', 'Classifier', child=current_child)
        current_child = 'classifier'

    # Classify components
    for name, component in ltree.items():
        if component['parent'] is None:
            outputs.append(component['name'])
        elif component['child'] is None:
            inputs.append(component['name'])
        else:
            intermediates.append(component['name'])

    model_resource = (layer_tree, ltree, outputs, inputs)
    return model_resource


def extract_encoder_weights(model):
    # Initialize a dictionary to hold the weights
    weights_dict = {
        'embeddings': {},
        'pooler': {},
        'dropout': {},
        'classifier': {}
    }

    for i in range(model.config.num_hidden_layers):
        weights_dict[f'attention_{i}'] = {}
        weights_dict[f'add_and_layer_norm_{i}_0'] = {}
        weights_dict[f'feed_forward_{i}'] = {}
        weights_dict[f'add_and_layer_norm_{i}_1'] = {}

    # Extract the model's parameters and organize them into the dictionary
    for name, param in model.bert.named_parameters():
        if 'embeddings' in name:
            weights_dict['embeddings'][name] = param.data.numpy()
        elif 'encoder.layer' in name:
            layer = name.split('.')[2]
            submodule = name.split('.')[3]
            if 'attention' in submodule and 'LayerNorm' not in name:
                weights_dict[f'attention_{layer}'][name] = param.data.numpy()
            elif 'attention.output.LayerNorm' in name:
                weights_dict[f'add_and_layer_norm_{layer}_0'][name] = param.data.numpy()
            elif 'intermediate' in submodule:
                weights_dict[f'feed_forward_{layer}'][name] = param.data.numpy()
            elif 'output' in submodule and 'LayerNorm' not in name:
                weights_dict[f'feed_forward_{layer}'][name] = param.data.numpy()
            elif 'output.LayerNorm' in name:
                weights_dict[f'add_and_layer_norm_{layer}_1'][name] = param.data.numpy()
        elif 'pooler' in name:
            weights_dict['pooler'][name] = param.data.numpy()

    for name, param in model.named_parameters():
        if 'dropout' in name:
            weights_dict['dropout'][name] = param.data.numpy()
        elif 'classifier' in name:
             weights_dict['classifier'][name] = param.data.numpy()

    return weights_dict


def create_encoder_output(model, input_ids=None, attention_mask=None, token_type_ids=None):
    all_layer_outputs = {}

    # Embeddings
    embedding_output = model.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
    all_layer_outputs['embeddings'] = embedding_output

    # iterate over each layer
    hidden_states = embedding_output

    for i, layer_module in enumerate(model.bert.encoder.layer):
        # code here
        with torch.no_grad():
            # Self-Attention and attention output
            attention_output = layer_module.attention.self(
                hidden_states,
                attention_mask=attention_mask,
            )[0]

            # Add + Layer Norm after attention
            attention_output = layer_module.attention.output.dense(attention_output)
            attention_output = layer_module.attention.output.dropout(attention_output)
            residual_attention_output = attention_output + hidden_states
            attention_output_norm = layer_module.attention.output.LayerNorm(residual_attention_output)

            # Feed Forward (Intermediate)
            intermediate_output = layer_module.intermediate(attention_output_norm)

            # Feed Forward Output
            feed_forward_output = layer_module.output.dense(intermediate_output)
            feed_forward_output = layer_module.output.dropout(feed_forward_output)
            residual_feed_forward_output = feed_forward_output + attention_output_norm
            feed_forward_output_norm = layer_module.output.LayerNorm(residual_feed_forward_output)


        # Save outputs add_and_layer_norm_0_0
        all_layer_outputs[f'attention_{i}'] = attention_output
        all_layer_outputs[f'add_and_layer_norm_{i}_0'] = attention_output_norm
        all_layer_outputs[f'feed_forward_{i}'] = feed_forward_output
        all_layer_outputs[f'add_and_layer_norm_{i}_1'] = feed_forward_output_norm

         # Update hidden states for the next layer
        hidden_states = feed_forward_output_norm

    # Pooler
    if hasattr(model.bert, 'pooler'):
        pooled_output = model.bert.pooler(hidden_states)
        all_layer_outputs['pooler'] = pooled_output

    if hasattr(model, 'dropout'):
        dropout_output = model.dropout(pooled_output)
        all_layer_outputs['dropout'] = dropout_output

    # Classifier
    if hasattr(model, 'classifier'):
        classifier = model.classifier(dropout_output)
        softmax_output = torch.nn.functional.softmax(classifier)
        all_layer_outputs['classifier'] = softmax_output

    return all_layer_outputs
