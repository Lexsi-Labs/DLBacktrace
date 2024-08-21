import numpy as np
import torch
import torch.nn as nn
from dl_backtrace.pytorch_backtrace.backtrace.utils import contrast as UC
from dl_backtrace.pytorch_backtrace.backtrace.utils import prop as UP
from dl_backtrace.pytorch_backtrace.backtrace.config import activation_master

class Backtrace(object):
    """
    This is the constructor method for the Backtrace class. It initializes an instance of the class.
    It takes two optional parameters: model (a neural network model) and activation_dict (a dictionary that maps layer names to activation functions).
    """

    def __init__(self, model=None, activation_dict={}):

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

        for i in range(len(layers) - 1):
            current_layer, current_layer_obj = layers[i]
            next_layer, next_layer_obj = layers[i + 1]
            current_layer_name = current_layer
            next_layer_name = next_layer

            next_layer_type = next_layer_name.lower()
            if any(af in next_layer_type for af in activation_functions):
                layer_sequence.append((f"{current_layer_name}/{next_layer_name}", current_layer_obj))
                i += 1
            else:
                if any(af in current_layer_name for af in activation_functions) is False:
                    layer_sequence.append((current_layer_name, current_layer_obj))

        # creating model_resource variable
        layer_sequence
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
            mode,
            start_wt=[],
            multiplier=100.0,
            scaler=0,
            max_unit=0,
    ):
        # This method is used for evaluating layer-wise relevance based on different modes.
        if mode == "default":
            output = self.proportional_eval(
                all_out=all_out,
                start_wt=start_wt,
                multiplier=multiplier,
                scaler=0,
                max_unit=0,
            )
            return output
        elif mode == "contrast":
            temp_output = self.contrast_eval(all_out=all_out, multiplier=multiplier)
            output = {}
            for k in temp_output[0].keys():
                output[k] = {}
                output[k]["Positive"] = temp_output[0][k]
                output[k]["Negative"] = temp_output[1][k]
            return output

    def proportional_eval(
            self, all_out, start_wt=[], multiplier=100.0, scaler=0, max_unit=0
    ):
        model_resource = self.model_resource
        activation_dict = self.activation_dict
        inputcheck = False
        out_layer = model_resource[2][0]
        all_wt = {}
        if len(start_wt) == 0:
            start_wt = UP.calculate_start_wt(all_out[out_layer])
        all_wt[out_layer] = start_wt * multiplier
        layer_stack = self.layer_stack

        for start_layer in layer_stack:
            if model_resource[1][start_layer]["child"]:
                child_nodes = model_resource[1][start_layer]["child"]
                for ch in child_nodes:
                    if ch not in all_wt:
                        if model_resource[1][start_layer]["class"] == 'LSTM':
                            all_wt[ch] = np.zeros_like(every_temp_out[ch][0])
                        else:
                            all_wt[ch] = np.zeros_like(all_out[ch][0])

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
                    temp_wt = UP.calculate_wt_conv(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
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
                    temp_wt = UP.calculate_wt_maxpool(
                        all_wt[start_layer], all_out[child_nodes[0]][0], (l1.kernel_size, l1.kernel_size)
                    )
                    all_wt[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "AvgPool2d":
                    l1 = model_resource[0][start_layer]
                    temp_wt = UP.calculate_wt_avgpool(
                        all_wt[start_layer], all_out[child_nodes[0]][0], (l1.kernel_size, l1.kernel_size)
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
                else:
                    temp_wt = all_wt[start_layer]
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

    def contrast_eval(self, all_out, multiplier=100.0):
        model_resource = self.model_resource
        activation_dict = self.activation_dict
        inputcheck = False
        out_layer = model_resource[2][0]
        all_wt_pos = {}
        all_wt_neg = {}
        start_wt_pos, start_wt_neg = UC.calculate_start_wt(all_out[out_layer])
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
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_conv(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
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
                else:
                    temp_wt_pos = all_wt_pos[start_layer]
                    temp_wt_neg = all_wt_neg[start_layer]
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                for ch in child_nodes:
                    if not (ch in layer_stack):
                        layer_stack.append(ch)
        return all_wt_pos, all_wt_neg
