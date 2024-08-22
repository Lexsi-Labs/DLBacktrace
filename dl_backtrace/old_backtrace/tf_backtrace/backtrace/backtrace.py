import numpy as np
from tensorflow.keras import Model

from .config import activation_master
from .utils import contrast as UC
from .utils import prop as UP


class Backtrace(object):
    """
    This is the constructor method for the Backtrace class. It initializes an instance of the class.
    It takes two optional parameters: model (a neural network model) and activation_dict (a dictionary that maps layer names to activation functions).
    """
    def __init__(self, model=None, activation_dict={}):

        #create a tree-like structure that represents the layers of the neural network model
        self.create_tree(model.layers)

        #create a new model (an instance of tf.keras.Model) that produces the output of each layer in the neural network.
        self.create_model_output(model)

        #create a layer stack that defines the order in which layers should be processed during backpropagation.
        self.create_layer_stack()

        #checks if the model is sequential or not. If it's sequential, it adds the input layer to the layer stack.
        if (
            len(self.model_resource[3]) == 0
            or model.__module__.split(".")[-1] == "sequential"
        ):
            inp_name = model.input.name
            self.layer_stack.append(inp_name)
            self.model_resource[1][inp_name] = {}
            self.model_resource[1][inp_name]["name"] = inp_name
            self.model_resource[1][inp_name]["type"] = "input"
            self.model_resource[1][inp_name]["parent"] = []
            self.model_resource[1][inp_name]["child"] = None
            self.model_resource[3].append(inp_name)
            self.sequential = True
        else:
            self.sequential = False
        try:

            #calls the build_activation_dict method to build a dictionary that maps layer names to activation functions.
            #If that fails, it creates a temporary dictionary with default activation functions.
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
        # Builds an activation dictionary by inspecting the activation functions of the layers in the model.
        activation_dict = {}
        for l in model.layers:
            if not hasattr(l, "activation"):
                activation_dict[l.name] = activation_master["None"]
                continue
            a1 = l.activation
            func_name = str(a1).split(" ")
            if func_name[0] == "<function":
                activation_dict[l.name] = activation_master.get(
                    func_name[1], activation_master["None"]
                )
            else:
                a2 = a1.activation
                func_name = str(a2).split(" ")
                if func_name[0] == "<function":
                    activation_dict[l.name] = activation_master.get(
                        func_name[1], activation_master["None"]
                    )
        self.activation_dict = activation_dict

    def create_tree(self, layers):
        #Creates a tree structure representing the layers of the model.
        #categorizes layers as input, output, or intermediate layers and establishes parent-child relationships between layers.
        ltree = {}
        layer_tree = {}
        inputs = []
        outputs = []
        intermediates = []
        for l in layers:
            ltree[l.output.name] = {}
            layer_tree[l.output.name] = l
            ltree[l.output.name]["name"] = l.name.split("/")[0]
            ltree[l.output.name]["class"] = type(l).__name__
            if not isinstance(l.input, list):
                if l.input.name == l.output.name:
                    ltree[l.output.name]["type"] = "input"
                    ltree[l.output.name]["parent"] = []
                    ltree[l.output.name]["child"] = None
                    inputs.append(l.output.name)
                else:
                    ltree[l.output.name]["type"] = "output"
                    ltree[l.output.name]["parent"] = []
                    ltree[l.output.name]["child"] = [l.input.name]
                    outputs.append(l.output.name)
                if l.input.name in ltree:
                    if ltree[l.input.name]["type"] != "input":
                        ltree[l.input.name]["type"] = "intermediate"
                        intermediates.append(l.input.name)
                    if l.input.name != l.output.name:
                        ltree[l.input.name]["parent"].append(l.output.name)
            else:
                ltree[l.output.name]["type"] = "output"
                ltree[l.output.name]["child"] = [i.name for i in l.input]
                ltree[l.output.name]["parent"] = []
                outputs.append(l.output.name)
                for i in l.input:
                    if i.name in ltree:
                        if i.name != l.output.name:
                            ltree[i.name]["parent"].append(l.output.name)
                        if ltree[i.name]["type"] != "input":
                            ltree[i.name]["type"] = "intermediate"
                            intermediates.append(i.name)

        outputs = list(set(outputs) - set(intermediates))
        self.model_resource = (layer_tree, ltree, outputs, inputs)

    def create_layer_stack(self):
        #Creates a layer stack that defines the order in which layers should be processed during backpropagation.
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

    def create_model_output(self, model):
        #Creates a new model that produces the output of each layer in the neural network.
        self.layers = [[], []]
        for l in self.model_resource[0]:
            #     if l not in model_resource[3]:
            self.layers[0].append(l)
            self.layers[1].append(self.model_resource[0][l])

        self.all_out_model = Model(
            inputs=model.input, outputs=[layer.output for layer in self.layers[1]]
        )

    def predict(self, inputs):
        #takes input data inputs and performs a forward pass through the neural network model to compute the output of each layer.
        #returns a dictionary that maps layer names to their corresponding outputs.
        all_out = self.all_out_model(inputs, training=False)
        temp_out = {}
        for i, v in enumerate(self.layers[0]):
            temp_out[v] = all_out[i].numpy()
        if self.sequential:
            temp_out[self.layer_stack[-1]] = inputs
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
        #This method is used for evaluating layer-wise relevance based on different modes.
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
        #This method computes layer-wise relevance in the "default" mode.
        #iteratively calculates relevance for each layer based on the layer's type (e.g., Dense, Conv2D, LSTM) and activation function.
        #returns a dictionary mapping layer names to their relevance scores.
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
            #             print("===========================================")
            #         print(layer_stack)
            if model_resource[1][start_layer]["child"]:
                child_nodes = model_resource[1][start_layer]["child"]
                for ch in child_nodes:
                    if ch not in all_wt:
                        all_wt[ch] = np.zeros_like(all_out[ch][0])
                #                 print(start_layer, child_nodes,all_out[start_layer].shape)
                if model_resource[1][start_layer]["class"] == "Dense":
                    print('dense')
                    l1 = model_resource[0][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
                    temp_wt = UP.calculate_wt_fc(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt[child_nodes[0]] += temp_wt

                elif model_resource[1][start_layer]["class"] == "Conv2D":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
                    temp_wt = UP.calculate_wt_conv(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt[child_nodes[0]] += temp_wt
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
                    model_resource[1][start_layer]["class"] == "GlobalAveragePooling2D"
                ):
                    temp_wt = UP.calculate_wt_gavgpool(
                        all_wt[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource[1][start_layer]["class"] == "MaxPooling2D":
                    l1 = model_resource[0][start_layer]
                    temp_wt = UP.calculate_wt_maxpool(
                        all_wt[start_layer], all_out[child_nodes[0]][0], l1.pool_size
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource[1][start_layer]["class"] == "AveragePooling2D":
                    l1 = model_resource[0][start_layer]
                    temp_wt = UP.calculate_wt_avgpool(
                        all_wt[start_layer], all_out[child_nodes[0]][0], l1.pool_size
                    )
                    all_wt[child_nodes[0]] += temp_wt
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
                    print('lstm')
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
                        all_out[child_nodes[0]][0]
                    )
                    temp_wt = lstm_obj_b.calculate_lstm_wt(
                        all_wt[start_layer], lstm_obj_f.compute_log
                    )
                    all_wt[child_nodes[0]] += temp_wt

                elif model_resource[1][start_layer]["class"] == "Embedding":
                    print('embedding')
                    temp_wt = all_wt[start_layer]
                    temp_wt = np.mean(temp_wt,axis=1)

                    all_wt[child_nodes[0]] = all_wt[child_nodes[0]] + temp_wt

                else:
                    print('else')
                    temp_wt = all_wt[start_layer]
                    all_wt[child_nodes[0]] += temp_wt.astype('float64')

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
        #This method computes layer-wise relevance in the "contrast" mode.
        #calculates positive and negative relevance scores for each layer
        #returns a dictionary that maps layer names to dictionaries containing positive and negative relevance values.
        model_resource = self.model_resource
        print(model_resource)
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
                if model_resource[1][start_layer]["class"] == "Dense":
                    print('dense')
                    l1 = model_resource[0][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
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
                elif model_resource[1][start_layer]["class"] == "Conv2D":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_conv(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
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
                    model_resource[1][start_layer]["class"] == "GlobalAveragePooling2D"
                ):
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_gavgpool(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
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
                    model_resource[1][start_layer]["class"] == "GlobalAveragePooling2D"
                ):
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_gavgpool(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource[1][start_layer]["class"] == "MaxPooling2D":
                    l1 = model_resource[0][start_layer]
                    temp_wt = UC.calculate_wt_maxpool(
                        all_wt_pos[start_layer],
                        all_out[child_nodes[0]][0],
                        l1.pool_size,
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt
                    temp_wt = UC.calculate_wt_maxpool(
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        l1.pool_size,
                    )
                    all_wt_neg[child_nodes[0]] += temp_wt
                elif model_resource[1][start_layer]["class"] == "AveragePooling2D":
                    l1 = model_resource[0][start_layer]
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_avgpool(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        l1.pool_size,
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
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
                    print('lstm')
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
                    print('embedding layer')
                    temp_wt_pos = all_wt_pos[start_layer]
                    temp_wt_neg = all_wt_neg[start_layer]

                    temp_wt_pos = np.mean(temp_wt_pos,axis=1)
                    temp_wt_neg = np.mean(temp_wt_neg,axis=1)

                    all_wt_pos[child_nodes[0]] = all_wt_pos[child_nodes[0]] + temp_wt_pos
                    all_wt_neg[child_nodes[0]] = all_wt_neg[child_nodes[0]] + temp_wt_neg


                else:
                    print('else')
                    temp_wt_pos = all_wt_pos[start_layer]
                    temp_wt_neg = all_wt_neg[start_layer]
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                for ch in child_nodes:
                    if not (ch in layer_stack):
                        layer_stack.append(ch)
        return all_wt_pos, all_wt_neg
