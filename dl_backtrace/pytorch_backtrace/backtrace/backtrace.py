import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dl_backtrace.pytorch_backtrace.backtrace.utils import contrast as UC
from dl_backtrace.pytorch_backtrace.backtrace.utils import prop as UP
from dl_backtrace.pytorch_backtrace.backtrace.config import activation_master
from dl_backtrace.pytorch_backtrace.backtrace.utils import helper as HP
from dl_backtrace.pytorch_backtrace.backtrace.utils import encoder as EN
from dl_backtrace.pytorch_backtrace.backtrace.utils import encoder_decoder as ED
from dl_backtrace.pytorch_backtrace.backtrace.utils import llama as LL

class Backtrace(object):
    """
    This is the constructor method for the Backtrace class. It initializes an instance of the class.
    It takes two optional parameters: model (a neural network model) and activation_dict (a dictionary that maps layer names to activation functions).
    """

    def __init__(self, model=None, activation_dict={}, model_type=None):
        if model_type == 'encoder':
            self.model = model
            self.model_type = model_type
            # create a tree-like structure for encoder model
            self.model_resource = EN.build_encoder_tree(model)
            # create a layer stack for encoder model
            self.create_layer_stack()
            # extract the encoder model weights
            self.model_weights = EN.extract_encoder_weights(model)
            # # calculate the output of each submodule of the encoder model
            # self.all_out_model = EN.create_encoder_output(model)
            self.activation_dict = None
            
        elif model_type == 'encoder_decoder':
            self.model = model
            self.model_type = model_type
            # create a tree-like structure and layer_stack for encoder-decoder model
            self.model_resource, self.layer_stack = ED.build_enc_dec_tree(model)
            # extract the encoder-decoder model weights
            self.model_weights = ED.extract_encoder_decoder_weights(model)  
            # # calculate the output of each submodule of the encoder-decoder model
            # self.all_out_model = ED.calculate_encoder_decoder_output(model)
            self.activation_dict = None
            
        elif model_type == 'llama':
            self.model = model
            self.model_type = model_type
            # create a tree-like structure and layer stack for llama model
            self.model_resource, self.layer_stack = LL.build_llama_tree(model)
            # extract the llama model weights
            self.model_weights = LL.extract_llama_weights(model)
            # # calculate the output of each submodule of the llama model
            # self.all_out_model = LL.create_llama_output(input_text, model, tokenizer, max_length, device)
            self.activation_dict = None 
            
        else:
            self.model_type = model_type
            # create a tree-like structure that represents the layers of the neural network model
            self.create_tree(model)
            # create a new model (an instance of tf.keras.Model) that produces the output of each layer in the neural network.
            self.create_model_output(model)
            # create a new model (an instance of tf.keras.Model) that produces the output of each layer in the neural network.
            self.create_every_model_output(model)
            # create a layer stack that defines the order in which layers should be processed during backpropagation.
            self.create_layer_stack()
            # checks if the model is sequential or not. If it's sequential, it adds the input layer to the layer stack.
            # identity

            inp_name = 'identity'
            self.layer_stack.append(inp_name)
            self.model_resource[1][inp_name] = {}
            self.model_resource[1][inp_name]["name"] = inp_name
            self.model_resource[1][inp_name]["type"] = "input"
            self.model_resource[1][inp_name]["parent"] = []
            self.model_resource[1][inp_name]["child"] = None
            self.model_resource[3].append(inp_name)
            self.sequential = True
            try:
                # calls the build_activation_dict method to build a dictionary that maps layer names to activation functions.
                # If that fails, it creates a temporary dictionary with default activation functions.
                if len(activation_dict) == 0:
                    self.build_activation_dict(model)
                else:
                    self.activation_dict = activation_dict

            except Exception as e:
                print(e)
                temp_dict = {}
                for l in model.layers:
                    temp_dict[l.name] = activation_master["None"]
                self.activation_dict = temp_dict 

    def build_activation_dict(self, model):
        model_resource = self.model_resource
        layer_list = list(model_resource[0].keys())
        activation_dict = {}
        activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax']  # You can add more activation functions
        for l in layer_list:
            activation_found = False
            try:  # could be activation for that layer
                for activation in activation_functions:
                    if activation in l.split('/')[1]:
                        activation_dict[l.split('/')[0]] = activation
                        activation_found = True
            except:
                activation_dict[l] = 'None'
        # activation_master :
        for key, value in activation_dict.items():
            activation_dict[key] = activation_master.get(value)
        self.activation_dict = activation_dict

    def create_tree(self, model):
        # create new layers same as tf version
        layers = list(model.named_children())
        activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax']
        layer_sequence = []
        i = 0
        while i < len(layers):
            current_layer, current_layer_obj = layers[i]
            # Check for layer/activation pair
            if i + 1 < len(layers):
                next_layer, next_layer_obj = layers[i+1]
                if any(af in next_layer.lower() for af in activation_functions):
                    layer_sequence.append((f"{current_layer}/{next_layer}", current_layer_obj))
                    i += 2 # Skip both
                    continue
            
            # Handle single layer that is not an activation function by itself
            if not any(af in current_layer.lower() for af in activation_functions):
                 layer_sequence.append((current_layer, current_layer_obj))
            i += 1
            
        # creating model_resource variable
        ltree = {}
        layer_tree = {}
        inputs = []
        outputs = []
        intermediates = []
        prev_layer_id = None
        num_layers = len(layer_sequence)
        for i, (layer_name, layer) in enumerate(layer_sequence):
            layer_id = layer_name
            ltree[layer_id] = {}
            layer_tree[layer_id] = layer
            layer_type = layer.__class__.__name__
            ltree[layer_id]["name"] = layer_id.split("/")[0]
            ltree[layer_id]["class"] = layer_type
            if i < num_layers - 1:
                ltree[layer_id]["type"] = "intermediate"
                intermediates.append(layer_id)
            else:
                ltree[layer_id]["type"] = "output"
                outputs.append(layer_id)
            if prev_layer_id is not None:
                ltree[layer_id]["child"] = [prev_layer_id]
                ltree[prev_layer_id]["parent"] = [layer_id]
            prev_layer_id = layer_id
        # Set child of the last layer as an empty list
        if prev_layer_id is not None:
            ltree[prev_layer_id]["parent"] = []
        layer_tree.pop('identity')
        ltree.pop('identity')
        self.model_resource = (layer_tree, ltree, outputs, inputs)

    def create_layer_stack(self):
        model_resource = self.model_resource
        start_layer = model_resource[2][0]
        layer_stack = [start_layer]
        temp_stack = [start_layer]
        while len(layer_stack) < len(model_resource[0]):
            start_layer = temp_stack.pop(0)
            if model_resource[1][start_layer]["child"]:
                child_nodes = model_resource[1][start_layer]["child"]
                for ch in child_nodes:
                    node_check = True
                    for pa in model_resource[1][ch]["parent"]:
                        if pa not in layer_stack:
                            node_check = False
                            break
                    if node_check:
                        if ch not in layer_stack:
                            layer_stack.append(ch)
                    temp_stack.append(ch)
        self.layer_stack = layer_stack

    def create_every_model_output(self, model):
        class ModelWithEveryOutputs(nn.Module):
            def __init__(self, base_model):
                super(ModelWithEveryOutputs, self).__init__()
                self.base_model = base_model
            def forward(self, x):
                outputs = []
                for layer_name, layer in self.base_model._modules.items():
                    if isinstance(x, tuple):
                        if isinstance(layer, nn.LSTM):
                            # Assuming you want to take the last LSTM output
                            x, _ = layer(x[0])  # Pass the first element of the tuple (assumes one LSTM layer)
                        else:
                            x = layer(x[0])  # Pass the first element of the tuple
                    else:
                        x = layer(x)
                    outputs.append((layer_name, x))
                return outputs
        self.every_out_model = ModelWithEveryOutputs(model)

    def create_model_output(self, model):
        class ModelWithOutputs(nn.Module):
            def __init__(self, base_model):
                super(ModelWithOutputs, self).__init__()
                self.base_model = base_model

            def forward(self, x):
                outputs = []
                for layer_name, layer in self.base_model._modules.items():
                    if isinstance(layer, nn.LSTM):
                        lstm_output, _ = layer(x)
                        if lstm_output.dim() == 3:
                            x = lstm_output[:, -1, :]  # Take the output of the last time step
                        else:
                            x = lstm_output
                    else:
                        x = layer(x)
                    outputs.append((layer_name, x))
                return outputs

        # all_out_model = ModelWithOutputs(model)
        self.all_out_model = ModelWithOutputs(model)
        model.eval()
        model_resource = self.model_resource
        self.layers = [[], []]
        for l in model_resource[0]:
            self.layers[0].append(l)
            self.layers[1].append(model_resource[0][l])

    def predict_every(self, inputs):
        every_out = self.every_out_model(inputs)
        activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax']
        every_temp_out = {}
        for i in range(len(every_out)):
            current_layer, current_layer_obj = every_out[i]
            try:
                next_layer, next_layer_obj = every_out[i + 1]
                current_layer_name = current_layer
                next_layer_name = next_layer
                next_layer_type = next_layer_name.lower()
                if any(af in next_layer_type for af in activation_functions):
                    if isinstance(next_layer_obj, tuple):
                        # Assuming you want the first tensor from the tuple
                        next_layer_tensor = next_layer_obj[0]
                    else:
                        next_layer_tensor = next_layer_obj
                    every_temp_out[
                        f"{current_layer_name}/{next_layer_name}"] = next_layer_tensor.detach().numpy().astype(
                        np.float32)
                    i += 1
                else:
                    if any(af in current_layer_name for af in activation_functions) is False:
                        if isinstance(current_layer_obj, tuple):
                            # Assuming you want the first tensor from the tuple
                            current_layer_tensor = current_layer_obj[0]
                        else:
                            current_layer_tensor = current_layer_obj
                        every_temp_out[current_layer_name] = current_layer_tensor.detach().numpy().astype(np.float32)
            except:
                if any(af in next_layer_type for af in activation_functions):
                    pass
                else:
                    if any(af in current_layer for af in activation_functions) is False:
                        if isinstance(current_layer_obj, tuple):
                            # Assuming you want the first tensor from the tuple
                            current_layer_tensor = current_layer_obj[0]
                        else:
                            current_layer_tensor = current_layer_obj
                        every_temp_out[current_layer] = current_layer_tensor.detach().cpu().numpy().astype(np.float32)
        return every_temp_out

    def predict(self, inputs):
        all_out = self.all_out_model(inputs)
        activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax']
        temp_out = {}
        for i in range(len(all_out)):
            current_layer, current_layer_obj = all_out[i]
            try:
                next_layer, next_layer_obj = all_out[i + 1]
                current_layer_name = current_layer
                next_layer_name = next_layer
                next_layer_type = next_layer_name.lower()
                if any(af in next_layer_type for af in activation_functions):
                    if isinstance(next_layer_obj, tuple):
                        # Assuming you want the first tensor from the tuple
                        next_layer_tensor = next_layer_obj[0]
                    else:
                        next_layer_tensor = next_layer_obj
                    temp_out[
                        f"{current_layer_name}/{next_layer_name}"] = next_layer_tensor.detach().cpu().numpy().astype(
                        np.float32)
                    i += 1
                else:
                    if any(af in current_layer_name for af in activation_functions) is False:
                        if isinstance(current_layer_obj, tuple):
                            # Assuming you want the first tensor from the tuple
                            current_layer_tensor = current_layer_obj[0]
                        else:
                            current_layer_tensor = current_layer_obj
                        temp_out[current_layer_name] = current_layer_tensor.detach().numpy().astype(np.float32)
            except:
                if any(af in next_layer_type for af in activation_functions):
                    pass
                else:
                    if any(af in current_layer for af in activation_functions) is False:
                        if isinstance(current_layer_obj, tuple):
                            # Assuming you want the first tensor from the tuple
                            current_layer_tensor = current_layer_obj[0]
                        else:
                            current_layer_tensor = current_layer_obj
                        temp_out[current_layer] = current_layer_tensor.detach().cpu().numpy().astype(np.float32)
        return temp_out

    def eval(
            self,
            all_out,
            mode="default",
            start_wt=[],
            multiplier=100.0,
            scaler=1,
            max_unit=0,
            predicted_token=None,
            thresholding=0.5,
            task="binary-classification",
    ):  
        # This method is used for evaluating layer-wise relevance based on different modes.
        if mode == "default":
            output = self.proportional_eval(
                all_out=all_out,
                start_wt=start_wt,
                multiplier=multiplier,
                scaler=scaler,
                max_unit=max_unit,
                predicted_token=predicted_token,
                thresholding=thresholding,
                task=task,
            )
            return output
        elif mode == "contrast":
            temp_output = self.contrast_eval(
                all_out=all_out, 
                multiplier=multiplier,
                scaler=scaler,
                thresholding=thresholding,
                task="binary-classification",
            )
            output = {}
            for k in temp_output[0].keys():
                output[k] = {}
                output[k]["Positive"] = temp_output[0][k]
                output[k]["Negative"] = temp_output[1][k]
            return output
        elif mode == "cuda_eval":
            output = self.cuda_proportional_eval(
                all_out=all_out,
                start_wt=start_wt,
                multiplier=multiplier,
                scaler=scaler,
                max_unit=max_unit,
                predicted_token=predicted_token,
                thresholding=thresholding,
                task=task,
            )
            return output

    def _safe_to_numpy(self, data):
        """Helper function to safely convert tensor or numpy array to numpy array."""
        if hasattr(data, 'detach'):
            return data.detach().numpy()
        elif hasattr(data, 'numpy'):
            return data.numpy()
        else:
            return data

    def proportional_eval(
            self, all_out, start_wt=[], multiplier=100.0, 
            scaler=1, max_unit=0, predicted_token=None,
            thresholding=0.5, task="binary-classification",
    ):
        model_resource = self.model_resource
        activation_dict = self.activation_dict
        inputcheck = False
        out_layer = model_resource[2][0]
        all_wt = {}
        if len(start_wt) == 0:
            if self.model_type == 'encoder':
                start_wt = UP.calculate_start_wt(self._safe_to_numpy(all_out[out_layer]), scaler=1)
                all_wt[out_layer] = start_wt * multiplier
                layer_stack = self.layer_stack
                all_wts = self.model_weights
            elif self.model_type == 'encoder_decoder' or self.model_type == 'llama':
                start_wt = UP.calculate_enc_dec_start_wt(self._safe_to_numpy(all_out[out_layer][0]), predicted_token)
                all_wt[out_layer] = start_wt * multiplier
                layer_stack = self.layer_stack
                all_wts = self.model_weights
            else:
                start_wt = UP.calculate_start_wt(all_out[out_layer], scaler, thresholding, task=task)
                start_wt = (start_wt/start_wt.max())
                all_wt[out_layer] = start_wt * multiplier
                layer_stack = self.layer_stack
                
        for start_layer in tqdm(layer_stack):
            if model_resource[1][start_layer]["child"]:
                child_nodes = model_resource[1][start_layer]["child"]
                for ch in child_nodes:
                    if ch not in all_wt:
                        if model_resource[1][start_layer]["class"] == 'LSTM':
                            all_wt[ch] = np.zeros_like(every_temp_out[ch][0])
                        else:
                            # Handle both tensor and numpy array cases
                            child_data = self._safe_to_numpy(all_out[ch][0])
                            all_wt[ch] = np.zeros_like(child_data)

                if model_resource[1][start_layer]["class"] == "Linear":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    temp_wt = UP.calculate_wt_fc(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource[1][start_layer]["class"] == "Conv2d":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    pad1 = l1.padding
                    strides1 = l1.stride
                    temp_wt = UP.calculate_wt_conv(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        pad1,
                        strides1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "ConvTranspose2d":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    pad1 = l1.padding
                    strides1 = l1.stride
                    temp_wt = UP.calculate_wt_conv2d_transpose(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        pad1,
                        strides1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "Conv1d":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    pad1 = l1.padding[0]
                    strides1 = l1.stride[0]
                    temp_wt = UP.calculate_wt_conv_1d(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        pad1, 
                        strides1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "ConvTranspose1d":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    pad1 = l1.padding[0]
                    strides1 = l1.stride[0]
                    dilation1= l1.dilation[0]
                    temp_wt = UP.calculate_wt_conv1d_transpose(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        pad1, 
                        strides1,
                        dilation1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "Reshape":
                    temp_wt = UP.calculate_wt_rshp(
                        all_wt[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource[1][start_layer]["class"] == "Flatten":
                    temp_wt = UP.calculate_wt_rshp(
                        all_wt[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif (
                        model_resource[1][start_layer]["class"] == "AdaptiveAvgPool2d"
                ):
                    temp_wt = UP.calculate_wt_gavgpool(
                        all_wt[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "MaxPool2d":
                    l1 = model_resource[0][start_layer]
                    pool_size = (l1.kernel_size, l1.kernel_size) if isinstance(l1.kernel_size, int) else l1.kernel_size
                    # Note: calculate_wt_maxpool internally converts these to tuples, so pass single values
                    padding = l1.padding if isinstance(l1.padding, int) else l1.padding[0]
                    strides = l1.stride if isinstance(l1.stride, int) else l1.stride[0]
                    temp_wt = UP.calculate_wt_maxpool(
                        all_wt[start_layer], all_out[child_nodes[0]][0], pool_size, padding, strides
                    )
                    all_wt[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "AvgPool2d":
                    l1 = model_resource[0][start_layer]
                    pool_size = (l1.kernel_size, l1.kernel_size) if isinstance(l1.kernel_size, int) else l1.kernel_size
                    # Note: calculate_wt_avgpool internally converts these to tuples, so pass single values
                    padding = l1.padding if isinstance(l1.padding, int) else l1.padding[0]
                    strides = l1.stride if isinstance(l1.stride, int) else l1.stride[0]
                    temp_wt = UP.calculate_wt_avgpool(
                        all_wt[start_layer], all_out[child_nodes[0]][0], pool_size, padding, strides
                    )
                    all_wt[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "MaxPool1d":
                    l1 = model_resource[0][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.stride
                    temp_wt = UP.calculate_wt_maxpool_1d(
                        all_wt[start_layer], all_out[child_nodes[0]][0], l1.kernel_size,pad1,strides1
                    )
                    all_wt[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "AvgPool1d":
                    l1 = model_resource[0][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.stride
                    temp_wt = UP.calculate_wt_avgpool_1d(
                        all_wt[start_layer], all_out[child_nodes[0]][0], l1.kernel_size,pad1,strides1
                    )
                    all_wt[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "Concatenate":
                    temp_wt = UP.calculate_wt_concat(
                        all_wt[start_layer],
                        [all_out[ch] for ch in child_nodes],
                        model_resource[0][start_layer].axis,
                    )
                    for ind, ch in enumerate(child_nodes):
                        all_wt[ch] += temp_wt[ind]
                elif model_resource[1][start_layer]["class"] == "Add":
                    temp_wt = UP.calculate_wt_add(
                        all_wt[start_layer], [all_out[ch] for ch in child_nodes]
                    )
                    for ind, ch in enumerate(child_nodes):
                        all_wt[ch] += temp_wt[ind]
                elif model_resource[1][start_layer]["class"] == "LSTM":
                    l1 = model_resource[0][start_layer]
                    return_sequence = l1.return_sequences
                    units = l1.units
                    num_of_cells = l1.input_shape[1]
                    lstm_obj_f = UP.LSTM_forward(
                        num_of_cells, units, l1.weights, return_sequence, False
                    )
                    lstm_obj_b = UP.LSTM_backtrace(
                        num_of_cells,
                        units,
                        [i.numpy() for i in l1.weights],
                        return_sequence,
                        False,
                    )
                    temp_out_f = lstm_obj_f.calculate_lstm_wt(
                        every_temp_out[child_nodes[0]][0]
                    )
                    temp_wt = lstm_obj_b.calculate_lstm_wt(
                        all_wt[start_layer], lstm_obj_f.compute_log
                    )
                    all_wt[child_nodes[0]] += temp_wt
                    
                elif model_resource[1][start_layer]["class"] == "Self_Attention":
                    weights = all_wts[start_layer]
                    self_attention_weights = HP.rename_self_attention_keys(weights)
                    config = self.model.config
                    temp_wt = UP.calculate_wt_self_attention_parallel(
                        all_wt[start_layer],
                        self._safe_to_numpy(all_out[child_nodes[0]][0]),
                        self_attention_weights,
                        config
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource[1][start_layer]["class"] == 'Residual':
                    temp_wt = UP.calculate_wt_residual(
                        all_wt[start_layer],
                        [self._safe_to_numpy(all_out[ch]) for ch in child_nodes],
                    )

                    for ind, ch in enumerate(child_nodes):
                        all_wt[ch] += temp_wt[ind]
                elif model_resource[1][start_layer]["class"] == 'Feed_Forward':
                    weights = all_wts[start_layer]
                    feed_forward_weights = HP.rename_feed_forward_keys(weights)
                    temp_wt = UP.calculate_wt_feed_forward_parallel(
                        all_wt[start_layer],
                        self._safe_to_numpy(all_out[child_nodes[0]][0]),
                        feed_forward_weights, 
                    )
                    all_wt[child_nodes[0]] += temp_wt
                    
                elif model_resource[1][start_layer]["class"] == 'LLAMA_Feed_Forward':
                    weights = all_wts[start_layer]
                    feed_forward_weights = HP.rename_llama_feed_forward_keys(weights)
                    temp_wt = UP.calculate_wt_llama_feed_forward_parallel(
                        all_wt[start_layer],
                        self._safe_to_numpy(all_out[child_nodes[0]][0]),
                        feed_forward_weights,
                    )
                    all_wt[child_nodes[0]] += temp_wt
                    
                elif model_resource[1][start_layer]["class"] == "Pooler":
                    weights = all_wts[start_layer]
                    pooler_weights = HP.rename_pooler_keys(weights)
                    temp_wt = UP.calculate_wt_pooler(
                        all_wt[start_layer],
                        self._safe_to_numpy(all_out[child_nodes[0]][0]),
                        pooler_weights
                    )
                    all_wt[child_nodes[0]] += temp_wt
                    
                elif model_resource[1][start_layer]["class"] == "Classifier":
                    weights = all_wts[start_layer]
                    classifier_weights = HP.rename_classifier_keys(weights)
                    temp_wt = UP.calculate_wt_classifier(
                        all_wt[start_layer],
                        self._safe_to_numpy(all_out[child_nodes[0]][0]),
                        classifier_weights
                    )
                    all_wt[child_nodes[0]] += temp_wt
                    
                elif model_resource[1][start_layer]["class"] == "LM_Head":
                    weights = all_wts[start_layer]
                    lm_head_weights = HP.rename_decoder_lm_head(weights)
                    temp_wt = UP.calculate_wt_lm_head_parallel(
                        all_wt[start_layer],
                        self._safe_to_numpy(all_out[child_nodes[0]][0]),
                        lm_head_weights
                    )
                    all_wt[child_nodes[0]] += temp_wt
                    
                elif model_resource[1][start_layer]["class"] == 'Layer_Norm':
                    temp_wt = all_wt[start_layer]
                    all_wt[child_nodes[0]] += temp_wt
                
                elif model_resource[1][start_layer]["class"] == 'Cross_Attention':
                    weights = all_wts[start_layer]
                    cross_attention_weights = HP.rename_cross_attention_keys(weights)
                    config = self.model.config
                    temp_wt = UP.calculate_wt_cross_attention_parallel(
                        all_wt[start_layer],
                        [self._safe_to_numpy(all_out[ch][0]) for ch in child_nodes],
                        cross_attention_weights,
                        config
                    )

                    for ind, ch in enumerate(child_nodes):
                        all_wt[ch] += temp_wt[ind]

                elif model_resource[1][start_layer]["class"] == "Embedding":
                    temp_wt = all_wt[start_layer]
                    temp_wt = np.mean(temp_wt,axis=1)
                    all_wt[child_nodes[0]] = all_wt[child_nodes[0]] + temp_wt
                else:
                    temp_wt = all_wt[start_layer]
                    # Handle shape mismatches for ResNet-like architectures
                    if temp_wt.shape != all_wt[child_nodes[0]].shape:
                        # Try to handle common ResNet skip connection patterns
                        target_shape = all_wt[child_nodes[0]].shape
                        if len(temp_wt.shape) == len(target_shape):
                                                         # Same number of dimensions, try reshaping or interpolation
                             if temp_wt.shape[0] != target_shape[0]:  # Different channels
                                 if temp_wt.shape[0] > target_shape[0]:
                                     # Downsample channels by taking every nth channel
                                     factor = temp_wt.shape[0] // target_shape[0]
                                     temp_wt = temp_wt[::factor][:target_shape[0]]
                                 elif target_shape[0] % temp_wt.shape[0] == 0:
                                     # Upsample channels by repeating if evenly divisible
                                     factor = target_shape[0] // temp_wt.shape[0]
                                     temp_wt = np.repeat(temp_wt, factor, axis=0)
                                 else:
                                     # If not evenly divisible, just take first n channels or pad with zeros
                                     if temp_wt.shape[0] < target_shape[0]:
                                         # Pad with zeros
                                         padding = [(0, target_shape[0] - temp_wt.shape[0])] + [(0, 0)] * (len(temp_wt.shape) - 1)
                                         temp_wt = np.pad(temp_wt, padding, mode='constant')
                                     else:
                                         # Truncate
                                         temp_wt = temp_wt[:target_shape[0]]
                            
                                                         # Handle spatial dimension mismatches
                             if len(temp_wt.shape) >= 3 and temp_wt.shape[1:3] != target_shape[1:3]:
                                 # Simple approach for spatial dimension changes
                                 if temp_wt.shape[1] > target_shape[1] and temp_wt.shape[2] > target_shape[2]:  # Spatial downsampling
                                     # Simple subsampling for downsampling
                                     factor_h = max(1, temp_wt.shape[1] // target_shape[1])
                                     factor_w = max(1, temp_wt.shape[2] // target_shape[2])
                                     temp_wt = temp_wt[:, ::factor_h, ::factor_w][:, :target_shape[1], :target_shape[2]]
                                 elif temp_wt.shape[1] < target_shape[1] and temp_wt.shape[2] < target_shape[2]:  # Spatial upsampling
                                     # Simple padding for upsampling
                                     pad_h = target_shape[1] - temp_wt.shape[1]
                                     pad_w = target_shape[2] - temp_wt.shape[2]
                                     padding = [(0, 0), (0, pad_h), (0, pad_w)] + [(0, 0)] * (len(temp_wt.shape) - 3)
                                     temp_wt = np.pad(temp_wt, padding, mode='constant')
                                 else:
                                     # Mixed case or exact match on one dimension, just crop/pad to fit
                                     if temp_wt.shape[1] != target_shape[1]:
                                         if temp_wt.shape[1] > target_shape[1]:
                                             temp_wt = temp_wt[:, :target_shape[1]]
                                         else:
                                             pad_h = target_shape[1] - temp_wt.shape[1]
                                             padding = [(0, 0), (0, pad_h)] + [(0, 0)] * (len(temp_wt.shape) - 2)
                                             temp_wt = np.pad(temp_wt, padding, mode='constant')
                                     if temp_wt.shape[2] != target_shape[2]:
                                         if temp_wt.shape[2] > target_shape[2]:
                                             temp_wt = temp_wt[:, :, :target_shape[2]]
                                         else:
                                             pad_w = target_shape[2] - temp_wt.shape[2]
                                             padding = [(0, 0), (0, 0), (0, pad_w)] + [(0, 0)] * (len(temp_wt.shape) - 3)
                                             temp_wt = np.pad(temp_wt, padding, mode='constant')
                        
                        # Final shape check and fallback
                        if temp_wt.shape != target_shape:
                            print(f"Warning: Shape mismatch {temp_wt.shape} != {target_shape}, using zero tensor")
                            temp_wt = np.zeros_like(all_wt[child_nodes[0]])
                    
                    all_wt[child_nodes[0]] += temp_wt
        if max_unit > 0 and scaler == 0:
            temp_dict = {}
            for k in all_wt.keys():
                temp_dict[k] = UC.weight_normalize(all_wt[k], max_val=max_unit)
            all_wt = temp_dict
        elif scaler > 0:
            temp_dict = {}
            for k in all_wt.keys():
                temp_dict[k] = UC.weight_scaler(all_wt[k], scaler=scaler)
            all_wt = temp_dict

        return all_wt

    def cuda_proportional_eval(
            self, all_out, start_wt=[], multiplier=100.0, 
            scaler=1, max_unit=0, predicted_token=None,
            thresholding=0.5, task="binary-classification",
    ):
        """
        CUDA-accelerated proportional evaluation method.
        Uses CUDA implementations for supported layer types (Linear, Conv2d, MaxPool2d).
        Falls back to original implementations for unsupported layers.
        """
        from dl_backtrace.pytorch_backtrace.backtrace.utils import prop as UP
        from dl_backtrace.pytorch_backtrace.backtrace.utils import contrast as UC
        
        model_resource = self.model_resource
        activation_dict = self.activation_dict
        out_layer = model_resource[2][0]
        all_wt = {}
        
        # Initialize starting weights (same as original)
        if len(start_wt) == 0:
            if self.model_type == 'encoder':
                start_wt = UP.calculate_start_wt(self._safe_to_numpy(all_out[out_layer]), scaler=1)
                all_wt[out_layer] = start_wt * multiplier
                layer_stack = self.layer_stack
                all_wts = self.model_weights
            elif self.model_type == 'encoder_decoder' or self.model_type == 'llama':
                start_wt = UP.calculate_enc_dec_start_wt(self._safe_to_numpy(all_out[out_layer][0]), predicted_token)
                all_wt[out_layer] = start_wt * multiplier
                layer_stack = self.layer_stack
                all_wts = self.model_weights
            else:
                start_wt = UP.calculate_start_wt(all_out[out_layer], scaler, thresholding, task=task)
                start_wt = (start_wt/start_wt.max())
                all_wt[out_layer] = start_wt * multiplier
                layer_stack = self.layer_stack
        
        # Process each layer in the stack
        for start_layer in tqdm(layer_stack):
            if model_resource[1][start_layer]["child"]:
                child_nodes = model_resource[1][start_layer]["child"]
                
                # Initialize child weights if not present
                for ch in child_nodes:
                    if ch not in all_wt:
                        child_data = self._get_layer_data(all_out, ch)
                        all_wt[ch] = np.zeros_like(child_data)
                            
                layer_class = model_resource[1][start_layer]["class"]
                
                # Process different layer types with CUDA acceleration where available
                if layer_class == "Linear":
                    self._cuda_process_linear_layer(start_layer, child_nodes, all_wt, all_out, model_resource, activation_dict)
                elif layer_class == "Conv2d":
                    self._cuda_process_conv2d_layer(start_layer, child_nodes, all_wt, all_out, model_resource, activation_dict)
                elif layer_class == "MaxPool2d":
                    self._cuda_process_maxpool2d_layer(start_layer, child_nodes, all_wt, all_out, model_resource)
                elif layer_class == "AdaptiveAvgPool2d":
                    self._cuda_process_adaptavgpool2d_layer(start_layer, child_nodes, all_wt, all_out, model_resource)
                elif layer_class == "Dropout":
                    # Pass-through layer, no special processing needed, just add relevance.
                    all_wt[child_nodes[0]] += all_wt[start_layer]
                elif layer_class == "Flatten":
                    temp_wt = UP.calculate_wt_rshp(
                        all_wt[start_layer], self._get_layer_data(all_out, child_nodes[0])
                    )
                    all_wt[child_nodes[0]] += temp_wt
                else:
                    # Generic pass-through for other unsupported layers
                    print(f"CUDA eval not supported for layer type: {layer_class}. Passing through.")
                    temp_wt = all_wt[start_layer]
                    # Apply same shape compatibility logic as in default mode
                    if temp_wt.shape != all_wt[child_nodes[0]].shape:
                        target_shape = all_wt[child_nodes[0]].shape
                        if len(temp_wt.shape) == len(target_shape):
                            if temp_wt.shape[0] != target_shape[0]:  # Different channels
                                if temp_wt.shape[0] > target_shape[0]:
                                    factor = temp_wt.shape[0] // target_shape[0]
                                    temp_wt = temp_wt[::factor][:target_shape[0]]
                                elif target_shape[0] % temp_wt.shape[0] == 0:
                                    factor = target_shape[0] // temp_wt.shape[0]
                                    temp_wt = np.repeat(temp_wt, factor, axis=0)
                                else:
                                    if temp_wt.shape[0] < target_shape[0]:
                                        padding = [(0, target_shape[0] - temp_wt.shape[0])] + [(0, 0)] * (len(temp_wt.shape) - 1)
                                        temp_wt = np.pad(temp_wt, padding, mode='constant')
                                    else:
                                        temp_wt = temp_wt[:target_shape[0]]
                            
                            if len(temp_wt.shape) >= 3 and temp_wt.shape[1:3] != target_shape[1:3]:
                                if temp_wt.shape[1] > target_shape[1] and temp_wt.shape[2] > target_shape[2]:
                                    factor_h = max(1, temp_wt.shape[1] // target_shape[1])
                                    factor_w = max(1, temp_wt.shape[2] // target_shape[2])
                                    temp_wt = temp_wt[:, ::factor_h, ::factor_w][:, :target_shape[1], :target_shape[2]]
                                elif temp_wt.shape[1] < target_shape[1] and temp_wt.shape[2] < target_shape[2]:
                                    pad_h = target_shape[1] - temp_wt.shape[1]
                                    pad_w = target_shape[2] - temp_wt.shape[2]
                                    padding = [(0, 0), (0, pad_h), (0, pad_w)] + [(0, 0)] * (len(temp_wt.shape) - 3)
                                    temp_wt = np.pad(temp_wt, padding, mode='constant')
                                else:
                                    if temp_wt.shape[1] != target_shape[1]:
                                        if temp_wt.shape[1] > target_shape[1]:
                                            temp_wt = temp_wt[:, :target_shape[1]]
                                        else:
                                            pad_h = target_shape[1] - temp_wt.shape[1]
                                            padding = [(0, 0), (0, pad_h)] + [(0, 0)] * (len(temp_wt.shape) - 2)
                                            temp_wt = np.pad(temp_wt, padding, mode='constant')
                                    if temp_wt.shape[2] != target_shape[2]:
                                        if temp_wt.shape[2] > target_shape[2]:
                                            temp_wt = temp_wt[:, :, :target_shape[2]]
                                        else:
                                            pad_w = target_shape[2] - temp_wt.shape[2]
                                            padding = [(0, 0), (0, 0), (0, pad_w)] + [(0, 0)] * (len(temp_wt.shape) - 3)
                                            temp_wt = np.pad(temp_wt, padding, mode='constant')
                        
                        if temp_wt.shape != target_shape:
                            print(f"Warning: CUDA pass-through shape mismatch {temp_wt.shape} != {target_shape}, using zero tensor")
                            temp_wt = np.zeros_like(all_wt[child_nodes[0]])
                    
                    all_wt[child_nodes[0]] += temp_wt
        
        # Apply normalization/scaling (same as original)
        if max_unit > 0 and scaler == 0:
            temp_dict = {}
            for k in all_wt.keys():
                temp_dict[k] = UC.weight_normalize(all_wt[k], max_val=max_unit)
            all_wt = temp_dict
        elif scaler > 0:
            temp_dict = {}
            for k in all_wt.keys():
                temp_dict[k] = UC.weight_scaler(all_wt[k], scaler=scaler)
            all_wt = temp_dict

        return all_wt
    
    def _get_layer_data(self, all_out, layer_name):
        """Helper function to extract layer data from all_out, handling different formats."""
        if isinstance(all_out[layer_name], dict) and 0 in all_out[layer_name]:
            data = all_out[layer_name][0]
        else:
            data = all_out[layer_name]
        
        data = self._safe_to_numpy(data)
        
        return data
    
    def _cuda_process_linear_layer(self, start_layer, child_nodes, all_wt, all_out, model_resource, activation_dict):
        """Process Linear layer with CUDA implementation if available."""
        l1 = model_resource[0][start_layer]
        w1 = l1.state_dict()['weight']
        b1 = l1.state_dict()['bias']
        
        input_data = self._get_layer_data(all_out, child_nodes[0])
        
        # weight_sample = all_wt[start_layer][0] if len(all_wt[start_layer].shape) > 1 else all_wt[start_layer]
        # input_sample = all_out[child_nodes[0]][0]
        
        # # Handle batch dimension for Linear layers
        if len(input_data.shape) > 1 and input_data.shape[0] > 1:
            input_sample = input_data[0]
            weight_sample = all_wt[start_layer][0] if len(all_wt[start_layer].shape) > 1 else all_wt[start_layer]
        else:
            input_sample = input_data.flatten() if len(input_data.shape) > 1 else input_data
            weight_sample = all_wt[start_layer].flatten() if len(all_wt[start_layer].shape) > 1 else all_wt[start_layer]
        
        # Try CUDA implementation first
        try:
            if torch.cuda.is_available():
                from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.Linear.cuda_version.wt_fc_ops import calculate_wt_fc_interface as linear_cuda
                
                # Convert parameters for CUDA function
                row_specific_weights_tensor = torch.tensor(weight_sample, dtype=torch.float32).contiguous().cuda()
                input_activations_tensor = torch.tensor(input_sample, dtype=torch.float32).contiguous().cuda()
                weights_matrix_tensor = w1.contiguous().cuda()
                bias_vector_tensor = b1.contiguous().cuda()
                
                # Get activation parameters
                activation_params = activation_dict[model_resource[1][start_layer]["name"]]
                
                has_lower_bound = True if activation_params["range"]["l"] is not None else False
                lower_threshold = activation_params["range"]["l"]
                has_upper_bound = True if activation_params["range"]["u"] is not None else False
                upper_threshold = activation_params["range"]["u"]
                is_non_mono = bool(activation_params["type"] == "non_mono")
                
                # Map activation function to enum
                if activation_params.get("func"):
                    if "relu" in str(activation_params["func"]).lower():
                        activation_func = 1
                    elif "sigmoid" in str(activation_params["func"]).lower():
                        activation_func = 2
                    else:
                        activation_func = 0
                else:
                    activation_func = 0
                
                temp_wt = linear_cuda(
                    row_specific_weights=row_specific_weights_tensor,
                    input_activations=input_activations_tensor,
                    weights_matrix=weights_matrix_tensor,
                    bias_vector=bias_vector_tensor,
                    has_lower_bound=has_lower_bound,
                    lower_threshold=lower_threshold,
                    has_upper_bound=has_upper_bound,
                    upper_threshold=upper_threshold,
                    is_non_mono=is_non_mono,
                    activation_func=activation_func
                ).cpu().numpy()
                
                print(f"[CUDA] Successfully processed Linear layer: {start_layer}")
            else:
                raise RuntimeError("CUDA not available")
                
        except Exception as e:
            print(f"[FALLBACK] Linear layer {start_layer}: {e}")
            # Fallback to original implementation
            from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.Linear.original_version import calculate_wt_fc as linear_original
            temp_wt = linear_original(
                weight_sample, input_sample, w1.detach().numpy(), b1.detach().numpy(),
                activation_dict[model_resource[1][start_layer]["name"]]
            )
        
        # Reshape result to match expected output shape
        if len(all_wt[child_nodes[0]].shape) > 1:
            temp_wt = temp_wt.reshape(all_wt[child_nodes[0]].shape)
            
        all_wt[child_nodes[0]] += temp_wt
    
    def _cuda_process_conv2d_layer(self, start_layer, child_nodes, all_wt, all_out, model_resource, activation_dict):
        """Process Conv2d layer with CUDA implementation if available."""
        l1 = model_resource[0][start_layer]
        w1 = l1.state_dict()['weight']
        b1 = l1.state_dict()['bias']
        pad1 = l1.padding
        strides1 = l1.stride
        
        input_data = self._get_layer_data(all_out, child_nodes[0])
        
        # # Handle batch dimension for Conv2D
        if len(input_data.shape) == 4:
            input_sample = input_data[0]
            weight_sample = all_wt[start_layer][0] if len(all_wt[start_layer].shape) == 4 else all_wt[start_layer]
        else:
            input_sample = input_data
            weight_sample = all_wt[start_layer]
        
        # weight_sample = all_wt[start_layer][0] if len(all_wt[start_layer].shape) == 4 else all_wt[start_layer]
        # input_sample = all_out[child_nodes[0]][0]
        
        # Try CUDA implementation first
        try:
            if torch.cuda.is_available():
                from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.Conv2D.cuda_version import calculate_wt_conv_cuda_optimized
                
                activation_params = activation_dict[model_resource[1][start_layer]["name"]].copy()
                
                # Convert tensors and call CUDA function
                grad_output_scales = torch.tensor(weight_sample, dtype=torch.float32).contiguous().cuda()
                input_activations = torch.tensor(input_sample, dtype=torch.float32).contiguous().cuda()
                kernel_weights_orig_shape = w1.contiguous().cuda()
                bias = b1.contiguous().cuda()
                
                temp_wt = calculate_wt_conv_cuda_optimized(
                    grad_output_scales, input_activations, kernel_weights_orig_shape,
                    bias, pad1, strides1, activation_params
                ).cpu().numpy()
                
                print(f"[CUDA] Successfully processed Conv2d layer: {start_layer}")
            else:
                raise RuntimeError("CUDA not available")
                
        except Exception as e:
            print(f"[FALLBACK] Conv2d layer {start_layer}: {e}")
            # Fallback to original implementation
            from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.Conv2D.original_version import calculate_wt_conv as conv2d_original
            temp_wt = conv2d_original(
                weight_sample, input_sample, w1.detach().numpy(), b1.detach().numpy(),
                pad1, strides1, activation_dict[model_resource[1][start_layer]["name"]]
            )
        
        # Handle the result shape to match target
        target_shape = all_wt[child_nodes[0]].shape
        temp_wt = temp_wt.T
        
        if len(target_shape) == 4 and len(temp_wt.shape) == 3:
            temp_wt = temp_wt[np.newaxis, ...]
        
        if temp_wt.shape != target_shape:
            if temp_wt.size == np.prod(target_shape):
                temp_wt = temp_wt.reshape(target_shape)
            else:
                temp_wt = np.zeros(target_shape)
        
        all_wt[child_nodes[0]] += temp_wt
    
    def _cuda_process_maxpool2d_layer(self, start_layer, child_nodes, all_wt, all_out, model_resource):
        """Process MaxPool2d layer with CUDA implementation if available."""
        l1 = model_resource[0][start_layer]
        pool_size = torch.tensor((l1.kernel_size, l1.kernel_size) if isinstance(l1.kernel_size, int) else l1.kernel_size, dtype=torch.int32).contiguous().cuda()
        padding = torch.tensor(l1.padding if isinstance(l1.padding, int) else l1.padding[0], dtype=torch.int32).contiguous().cuda()
        strides = torch.tensor(l1.stride if isinstance(l1.stride, int) else l1.stride[0], dtype=torch.int32).contiguous().cuda()
        
        input_data = self._get_layer_data(all_out, child_nodes[0])
        
        # Handle batch dimension for MaxPool
        if len(input_data.shape) == 4:
            input_sample = input_data[0]
            weight_sample = all_wt[start_layer][0] if len(all_wt[start_layer].shape) == 4 else all_wt[start_layer]
        else:
            input_sample = input_data
            weight_sample = all_wt[start_layer]
        
        # weight_sample = all_wt[start_layer][0] if len(all_wt[start_layer].shape) == 4 else all_wt[start_layer]
        # input_sample = all_out[child_nodes[0]][0]
        
        # Try CUDA implementation first
        try:
            if torch.cuda.is_available():
                from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.MaxPool2D.cuda_version import calculate_wt_maxpool_cuda
                
                temp_wt = calculate_wt_maxpool_cuda(
                    torch.tensor(weight_sample).contiguous().cuda(),
                    torch.tensor(input_sample).contiguous().cuda(),
                    pool_size, padding, strides
                ).cpu().numpy()
                
                print(f"[CUDA] Successfully processed MaxPool2d layer: {start_layer}")
            else:
                raise RuntimeError("CUDA not available")
                
        except Exception as e:
            print(f"[FALLBACK] MaxPool2d layer {start_layer}: {e}")
            # Fallback to original implementation
            from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.MaxPool2D.original_version import calculate_wt_maxpool as maxpool_original
            temp_wt = maxpool_original(weight_sample, input_sample, pool_size, padding, strides)
        
        # Handle the result shape to match target
        target_shape = all_wt[child_nodes[0]].shape
        temp_wt = temp_wt.T
        
        if len(target_shape) == 4 and len(temp_wt.shape) == 3:
            temp_wt = temp_wt[np.newaxis, ...]
        
        if temp_wt.shape != target_shape:
            if temp_wt.size == np.prod(target_shape):
                temp_wt = temp_wt.reshape(target_shape)
            else:
                temp_wt = np.zeros(target_shape)
        
        all_wt[child_nodes[0]] += temp_wt

    def _cuda_process_adaptavgpool2d_layer(self, start_layer, child_nodes, all_wt, all_out, model_resource):
        """Process AdaptiveAvgPool2d layer with CUDA implementation if available."""
        input_data = self._get_layer_data(all_out, child_nodes[0])
        
        # Handle batch dimension
        if len(input_data.shape) == 4:
            input_sample = input_data[0]
            weight_sample = all_wt[start_layer][0] if len(all_wt[start_layer].shape) == 4 else all_wt[start_layer]
        else:
            input_sample = input_data
            weight_sample = all_wt[start_layer]
        
        try:
            if torch.cuda.is_available():
                from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.AdaptiveAvgPool2D.cuda_version.wt_gavgpool_ops import fused_weighted_gavgpool as calculate_wt_gavgpool_cuda
                row_specific_weights_tensor = torch.tensor(weight_sample, dtype=torch.float32).cuda()
                input_activations_tensor = torch.tensor(input_sample, dtype=torch.float32).cuda()
                
                temp_wt = calculate_wt_gavgpool_cuda(
                    row_specific_weights_tensor,
                    input_activations_tensor
                ).cpu().numpy()
                
                print(f"[CUDA] Successfully processed AdaptiveAvgPool2d layer: {start_layer}")
            else:
                raise RuntimeError("CUDA not available")
        except Exception as e:
            print(f"[FALLBACK] AdaptiveAvgPool2d layer {start_layer}: {e}")
            # Fallback to original implementation
            from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.AdaptiveAvgPool2D.original_version import calculate_wt_gavgpool as gavgpool_original
            temp_wt = gavgpool_original(weight_sample, input_sample)
        
        if len(all_wt[child_nodes[0]].shape) > 1:
            temp_wt = temp_wt.reshape(all_wt[child_nodes[0]].shape)
        
        temp_wt_transposed = temp_wt.T
        
        # Handle shape mismatches for CUDA AdaptiveAvgPool2d
        if temp_wt_transposed.shape != all_wt[child_nodes[0]].shape:
            target_shape = all_wt[child_nodes[0]].shape
            print(f"Shape mismatch in CUDA AdaptiveAvgPool2d: {temp_wt_transposed.shape} != {target_shape}")
            
            # If transpose didn't fix it, try without transpose
            if temp_wt.shape == target_shape:
                temp_wt_transposed = temp_wt
            else:
                # Apply the same shape compatibility logic as in the default mode
                if len(temp_wt_transposed.shape) == len(target_shape):
                    if temp_wt_transposed.shape[0] != target_shape[0]:  # Different channels
                        if temp_wt_transposed.shape[0] > target_shape[0]:
                            factor = temp_wt_transposed.shape[0] // target_shape[0]
                            temp_wt_transposed = temp_wt_transposed[::factor][:target_shape[0]]
                        elif target_shape[0] % temp_wt_transposed.shape[0] == 0:
                            factor = target_shape[0] // temp_wt_transposed.shape[0]
                            temp_wt_transposed = np.repeat(temp_wt_transposed, factor, axis=0)
                        else:
                            if temp_wt_transposed.shape[0] < target_shape[0]:
                                padding = [(0, target_shape[0] - temp_wt_transposed.shape[0])] + [(0, 0)] * (len(temp_wt_transposed.shape) - 1)
                                temp_wt_transposed = np.pad(temp_wt_transposed, padding, mode='constant')
                            else:
                                temp_wt_transposed = temp_wt_transposed[:target_shape[0]]
                
                # Final check and fallback
                if temp_wt_transposed.shape != target_shape:
                    print(f"Warning: CUDA AdaptiveAvgPool2d shape mismatch {temp_wt_transposed.shape} != {target_shape}, using zero tensor")
                    temp_wt_transposed = np.zeros_like(all_wt[child_nodes[0]])
        
        all_wt[child_nodes[0]] += temp_wt_transposed            
            
    def contrast_eval(self, all_out, multiplier=100.0,
                            scaler=None,thresholding=0.5,
                            task="binary-classification"):
        model_resource = self.model_resource
        activation_dict = self.activation_dict
        inputcheck = False
        out_layer = model_resource[2][0]
        all_wt_pos = {}
        all_wt_neg = {}
        start_wt_pos, start_wt_neg = UC.calculate_start_wt(all_out[out_layer],scaler,thresholding,task)
        all_wt_pos[out_layer] = start_wt_pos * multiplier
        all_wt_neg[out_layer] = start_wt_neg * multiplier
        layer_stack = [out_layer]

        while len(layer_stack) > 0:
            start_layer = layer_stack.pop(0)
            if model_resource[1][start_layer]["child"]:
                child_nodes = model_resource[1][start_layer]["child"]
                for ch in child_nodes:
                    if ch not in all_wt_pos:
                        all_wt_pos[ch] = np.zeros_like(all_out[ch][0])
                        all_wt_neg[ch] = np.zeros_like(all_out[ch][0])
                if model_resource[1][start_layer]["class"] == "Linear":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_fc(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource[1][start_layer]["class"] == "Conv2d":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    pad1 = l1.padding
                    strides1 = l1.stride
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_conv(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        pad1,
                        strides1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "ConvTranspose2d":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    pad1 = l1.padding
                    strides1 = l1.stride
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_conv2d_transpose(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        pad1, 
                        strides1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == 'Conv1d':
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    pad1 = l1.padding[0]
                    strides1 = l1.stride[0]
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_conv_1d(all_wt_pos[start_layer],
                                                                all_wt_neg[start_layer],
                                                                all_out[child_nodes[0]][0],
                                                                w1,b1, pad1, strides1,
                                                                activation_dict[model_resource[1][start_layer]['name']])
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "ConvTranspose1d":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    pad1 = l1.padding[0]
                    strides1 = l1.stride[0]
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_conv1d_transpose(all_wt_pos[start_layer],
                                                                            all_wt_neg[start_layer],
                                                                            all_out[child_nodes[0]][0],
                                                                            w1,b1, pad1, strides1,
                                                                            activation_dict[model_resource[1][start_layer]['name']])
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "Reshape":
                    temp_wt_pos = UC.calculate_wt_rshp(
                        all_wt_pos[start_layer], all_out[child_nodes[0]][0]
                    )
                    temp_wt_neg = UC.calculate_wt_rshp(
                        all_wt_neg[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif (
                        model_resource[1][start_layer]["class"] == "AdaptiveAvgPool2d"
                ):
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_gavgpool(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "Flatten":
                    temp_wt = UC.calculate_wt_rshp(
                        all_wt_pos[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt
                    temp_wt = UC.calculate_wt_rshp(
                        all_wt_neg[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt_neg[child_nodes[0]] += temp_wt
                elif (
                        model_resource[1][start_layer]["class"] == "AdaptiveAvgPool2d"
                ):
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_gavgpool(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "MaxPool2d":
                    l1 = model_resource[0][start_layer]
                    temp_wt = UC.calculate_wt_maxpool(
                        all_wt_pos[start_layer],
                        all_out[child_nodes[0]][0],
                        (l1.kernel_size, l1.kernel_size),
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt.T
                    temp_wt = UC.calculate_wt_maxpool(
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        (l1.kernel_size, l1.kernel_size),
                    )
                    all_wt_neg[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "MaxPool1d":
                    l1 = model_resource[0][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.stride
                    temp_wt = UC.calculate_wt_maxpool_1d(
                        all_wt_pos[start_layer],
                        all_out[child_nodes[0]][0],
                        l1.kernel_size, pad1, strides1
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt.T
                    temp_wt = UC.calculate_wt_maxpool_1d(
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        l1.kernel_size, pad1, strides1
                    )
                    all_wt_neg[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "AvgPool2d":
                    l1 = model_resource[0][start_layer]
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_avgpool(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        (l1.kernel_size, l1.kernel_size),
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "AvgPool1d":
                    l1 = model_resource[0][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.stride
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_avgpool_1d(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        l1.kernel_size, pad1, strides1
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "Concatenate":
                    temp_wt = UC.calculate_wt_concat(
                        all_wt_pos[start_layer],
                        [all_out[ch] for ch in child_nodes],
                        model_resource[0][start_layer].axis,
                    )
                    for ind, ch in enumerate(child_nodes):
                        all_wt_pos[ch] += temp_wt[ind]
                    temp_wt = UC.calculate_wt_concat(
                        all_wt_neg[start_layer],
                        [all_out[ch] for ch in child_nodes],
                        model_resource[0][start_layer].axis,
                    )
                    for ind, ch in enumerate(child_nodes):
                        all_wt_neg[ch] += temp_wt[ind]
                elif model_resource[1][start_layer]["class"] == "Add":
                    temp_wt = UC.calculate_wt_add(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        [all_out[ch] for ch in child_nodes],
                    )
                    for ind, ch in enumerate(child_nodes):
                        all_wt_pos[ch] += temp_wt[ind][0]
                        all_wt_neg[ch] += temp_wt[ind][1]
                elif model_resource[1][start_layer]["class"] == "LSTM":
                    l1 = model_resource[0][start_layer]
                    return_sequence = l1.return_sequences
                    units = l1.units
                    num_of_cells = l1.input_shape[1]
                    lstm_obj_f = UC.LSTM_forward(
                        num_of_cells, units, l1.weights, return_sequence, False
                    )
                    lstm_obj_b = UC.LSTM_backtrace(
                        num_of_cells,
                        units,
                        [i.numpy() for i in l1.weights],
                        return_sequence,
                        False,
                    )
                    temp_out_f = lstm_obj_f.calculate_lstm_wt(
                        all_out[child_nodes[0]][0]
                    )
                    temp_wt_pos, temp_wt_neg = lstm_obj_b.calculate_lstm_wt(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        lstm_obj_f.compute_log,
                    )
                    all_wt_pos[child_nodes[0]] = temp_wt_pos
                    all_wt_neg[child_nodes[0]] = temp_wt_neg
                elif model_resource[1][start_layer]["class"] == "Embedding":
                    temp_wt_pos = all_wt_pos[start_layer]
                    temp_wt_neg = all_wt_neg[start_layer]

                    temp_wt_pos = np.mean(temp_wt_pos,axis=1)
                    temp_wt_neg = np.mean(temp_wt_neg,axis=1)

                    all_wt_pos[child_nodes[0]] = all_wt_pos[child_nodes[0]] + temp_wt_pos
                    all_wt_neg[child_nodes[0]] = all_wt_neg[child_nodes[0]] + temp_wt_neg
                else:
                    temp_wt_pos = all_wt_pos[start_layer]
                    temp_wt_neg = all_wt_neg[start_layer]
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                for ch in child_nodes:
                    if not (ch in layer_stack):
                        layer_stack.append(ch)
        return all_wt_pos, all_wt_neg
