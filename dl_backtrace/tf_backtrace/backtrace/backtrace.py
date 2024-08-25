import gc
from datetime import datetime
from tqdm import tqdm
import tensorflow.keras as keras  
import numpy as np  
import tensorflow as tf
from dl_backtrace.tf_backtrace.backtrace.utils import utils_prop as UP
from dl_backtrace.tf_backtrace.backtrace.utils import utils_contrast as UC
from dl_backtrace.tf_backtrace.backtrace.activation_info import activation_master
from dl_backtrace.tf_backtrace.backtrace.utils import encoder as EN
from dl_backtrace.tf_backtrace.backtrace.utils import encoder_decoder as ED
from dl_backtrace.tf_backtrace.backtrace.utils import helper as HP

from tensorflow.keras import backend as K  
from tensorflow.keras import losses  
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.layers import (  
    LSTM,
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    MaxPooling2D,
    Reshape,
)
from tensorflow.keras.models import Model, Sequential  
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop  
from tensorflow.keras.regularizers import l2  
from tensorflow.keras.utils import get_custom_objects  
from numpy.lib.stride_tricks import as_strided  


class Backtrace(object):
    def __init__(self, model=None, activation_dict={}, model_type=None):
        self.model = model
        self.model_type = model_type
        if model_type == 'encoder':
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
            # create a tree-like structure and layer_stack for encoder-decoder model
            self.model_resource, self.layer_stack = ED.build_enc_dec_tree(model)
            # extract the encoder-decoder model weights
            self.model_weights = ED.extract_encoder_decoder_weights(model)
            # # calculate the output of each submodule of the encoder-decoder model
            # self.all_out_model = ED.calculate_encoder_decoder_output(model)
            self.activation_dict = None 
        
        else:
            self.create_tree(model.layers)
            self.create_model_output(model)
            self.create_layer_stack()

            if len(self.model_resource["inputs"])==0 or model.__module__.split(".")[-1]=='sequential':
                inp_name = model.input.name
                self.layer_stack.append(inp_name)
                self.model_resource["graph"][inp_name] = {}
                self.model_resource["graph"][inp_name]["name"] = inp_name
                self.model_resource["graph"][inp_name]["type"] = "input"
                self.model_resource["graph"][inp_name]["parent"] = []
                self.model_resource["graph"][inp_name]["child"] = None
                self.model_resource["inputs"].append(inp_name)
                self.sequential = True
            else:
                self.sequential = False
            try:
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

    def build_activation_dict(self,model):
        activation_dict = {}
        for l in model.layers:
            if not hasattr(l,"activation"):
                activation_dict[l.name] = activation_master["None"]
                continue
            a1 = l.activation
            func_name = str(a1).split(" ")
            if func_name[0] == "<function":
                activation_dict[l.name] = activation_master.get(func_name[1],activation_master["None"])
            else:
                a2 = a1.activation
                func_name = str(a2).split(" ")
                if func_name[0] == "<function":
                    activation_dict[l.name] = activation_master.get(func_name[1],activation_master["None"])
        self.activation_dict = activation_dict

    def create_tree(self, layers):
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
        self.model_resource = {"layers":layer_tree, "graph":ltree, "outputs":outputs, "inputs":inputs}

    def create_layer_stack(self):
        model_resource = self.model_resource
        start_layer = model_resource["outputs"][0]
        layer_stack = [start_layer]
        temp_stack = [start_layer]
        while len(layer_stack) < len(model_resource["layers"]):
            start_layer = temp_stack.pop(0)
            if model_resource["graph"][start_layer]["child"]:
                child_nodes = model_resource["graph"][start_layer]["child"]
                for ch in child_nodes:
                    node_check = True
                    for pa in model_resource["graph"][ch]["parent"]:
                        if pa not in layer_stack:
                            node_check = False
                            break
                    if node_check:
                        if ch not in layer_stack:
                            layer_stack.append(ch)
                    temp_stack.append(ch)
        self.layer_stack = layer_stack

    def create_model_output(self, model):
        self.layers = [[], []]
        for l in self.model_resource["layers"]:
            self.layers[0].append(l)
            self.layers[1].append(self.model_resource["layers"][l])

        self.all_out_model = Model(
            inputs=model.input, outputs=[layer.output for layer in self.layers[1]]
        )

    def predict(self, inputs):
        all_out = self.all_out_model(inputs, training=False)
        temp_out = {}
        for i, v in enumerate(self.layers[0]):
            temp_out[v] = all_out[i].numpy()
        if self.sequential:
            temp_out[self.layer_stack[-1]] = inputs
        return temp_out
    
    def eval(self, all_out, start_wt=[], mode="default",multiplier=100.0, 
                scaler=None, max_unit=0,thresholding=0.5,
                task="binary-classification",predicted_token=None):

        if mode=="default":
            output = self.proportional_eval(all_out=all_out,
                                 start_wt=start_wt ,
                                 multiplier=multiplier, 
                                 scaler=scaler, 
                                 max_unit=max_unit,
                                thresholding=thresholding,
                                task=task,
                                predicted_token=predicted_token)
            return output
        elif mode=="contrast":
            temp_output = self.contrast_eval(all_out=all_out,
                                    multiplier=multiplier, 
                                    scaler=scaler,
                                    thresholding=thresholding,
                                    task=task)
            output = {}
            for k in temp_output[0].keys():
                output[k] = {}
                output[k]["Positive"] = temp_output[0][k]
                output[k]["Negative"] = temp_output[1][k]
            return output

    def proportional_eval(self, all_out, start_wt=[] ,
                        multiplier=100.0, scaler=None, max_unit=0, 
                        predicted_token=None, thresholding=0.5,
                        task="binary-classification"):
        model_resource = self.model_resource
        activation_dict = self.activation_dict
        inputcheck = False
        out_layer = model_resource["outputs"][0]
        all_wt = {}
        if len(start_wt) == 0:
            if self.model_type == 'encoder':
                start_wt = UP.calculate_start_wt(all_out[out_layer])
                all_wt[out_layer] = start_wt * multiplier
                layer_stack = self.layer_stack
                all_wts = self.model_weights

            elif self.model_type == 'encoder_decoder':
                start_wt = UP.calculate_enc_dec_start_wt(all_out[out_layer][0], predicted_token)
                all_wt[out_layer] = start_wt * multiplier
                layer_stack = self.layer_stack
                all_wts = self.model_weights

            else:
                start_wt = UP.calculate_start_wt(all_out[out_layer],scaler,thresholding,task=task)
                all_wt[out_layer] = start_wt * multiplier
                layer_stack = self.layer_stack

        for start_layer in tqdm(layer_stack):
            if model_resource["graph"][start_layer]["child"]:
                child_nodes = model_resource["graph"][start_layer]["child"]
                for ch in child_nodes:
                    if ch not in all_wt:
                        all_wt[ch] = np.zeros_like(all_out[ch][0])
                if model_resource["graph"][start_layer]["class"] == "Dense":
                    l1 = model_resource["layers"][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
                    temp_wt = UP.calculate_wt_fc(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        activation_dict[model_resource["graph"][start_layer]["name"]],
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == "Conv2D":
                    l1 = model_resource["layers"][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
                    pad1 = l1.padding
                    strides1 = l1.strides
                    temp_wt = UP.calculate_wt_conv(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        pad1, 
                        strides1,
                        activation_dict[model_resource["graph"][start_layer]["name"]],
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == "Conv2DTranspose":
                    l1 = model_resource["layers"][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
                    pad1 = l1.padding
                    strides1 = l1.strides
                    temp_wt = UP.calculate_wt_conv2d_transpose(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        pad1, 
                        strides1,
                        activation_dict[model_resource["graph"][start_layer]["name"]],
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == "Conv1D":
                    l1 = model_resource["layers"][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
                    pad1 = l1.padding
                    strides1 = l1.strides[0]
                    temp_wt = UP.calculate_wt_conv_1d(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        pad1, 
                        strides1,
                        activation_dict[model_resource["graph"][start_layer]["name"]],
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == "Conv1DTranspose":
                    l1 = model_resource["layers"][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
                    pad1 = l1.padding
                    strides1 = l1.strides[0]
                    temp_wt = UP.calculate_wt_conv1d_transpose(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        pad1, 
                        strides1,
                        activation_dict[model_resource["graph"][start_layer]["name"]],
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == "Reshape":
                    temp_wt = UP.calculate_wt_rshp(
                        all_wt[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == "Flatten":
                    temp_wt = UP.calculate_wt_rshp(
                        all_wt[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == "GlobalAveragePooling2D":
                    temp_wt = UP.calculate_wt_gavgpool(
                        all_wt[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == "GlobalAveragePooling1D":
                    temp_wt = UP.calculate_wt_gavgpool_1d(
                        all_wt[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == "GlobalMaxPooling2D":
                    temp_wt = UP.calculate_wt_gmaxpool_2d(
                        all_wt[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == "GlobalMaxPooling1D":
                    temp_wt = UP.calculate_wt_gmaxpool_1d(
                        all_wt[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == 'ZeroPadding2D':
                    l1 = model_resource["layers"][start_layer]
                    pad1 = l1.padding
                    temp_wt = UP.calculate_wt_zero_pad(all_wt[start_layer],
                                              all_out[child_nodes[0]][0],
                                              pad1)
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == 'MaxPooling2D':
                    l1 = model_resource["layers"][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.strides
                    temp_wt = UP.calculate_wt_maxpool(all_wt[start_layer],
                                              all_out[child_nodes[0]][0],
                                              l1.pool_size, pad1, strides1)
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == 'MaxPooling1D':
                    l1 = model_resource["layers"][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.strides
                    temp_wt = UP.calculate_wt_maxpool_1d(all_wt[start_layer],
                                              all_out[child_nodes[0]][0],
                                              l1.pool_size, pad1, strides1)
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == 'AveragePooling2D':
                    l1 = model_resource["layers"][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.strides
                    temp_wt = UP.calculate_wt_avgpool(all_wt[start_layer],
                                              all_out[child_nodes[0]][0],
                                              l1.pool_size, pad1, strides1)
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == 'AveragePooling1D':
                    l1 = model_resource["layers"][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.strides
                    temp_wt = UP.calculate_wt_avgpool_1d(all_wt[start_layer],
                                              all_out[child_nodes[0]][0],
                                              l1.pool_size, pad1, strides1)
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == "Concatenate":
                    temp_wt = UP.calculate_wt_concat(
                        all_wt[start_layer],
                        [all_out[ch] for ch in child_nodes],
                        model_resource["layers"][start_layer].axis,
                    )
                    for ind, ch in enumerate(child_nodes):
                        all_wt[ch] += temp_wt[ind]
                elif model_resource["graph"][start_layer]["class"] == "Add":
                    temp_wt = UP.calculate_wt_add(
                        all_wt[start_layer], [all_out[ch] for ch in child_nodes]
                    )
                    for ind, ch in enumerate(child_nodes):
                        all_wt[ch] += temp_wt[ind]
                elif model_resource["graph"][start_layer]["class"] == "LSTM":
                    l1 = model_resource["layers"][start_layer]
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
                elif model_resource["graph"][start_layer]["class"] == "Embedding":
                    temp_wt = all_wt[start_layer]
                    temp_wt = np.mean(temp_wt,axis=1)
                    all_wt[child_nodes[0]] = all_wt[child_nodes[0]] + temp_wt
                elif model_resource["graph"][start_layer]["class"] == "TextVectorization":
                    temp_wt = all_wt[start_layer]
                    all_wt[child_nodes[0]] = all_wt[child_nodes[0]] + temp_wt.sum()
                elif model_resource["graph"][start_layer]["class"] == "Self_Attention":
                    weights = all_wts[start_layer]
                    self_attention_weights = HP.rename_self_attention_keys(weights)
                    temp_wt = UP.calculate_wt_self_attention(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        self_attention_weights,
                    )
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == 'Residual':
                    temp_wt = UP.calculate_wt_residual(
                        all_wt[start_layer],
                        [all_out[ch] for ch in child_nodes],
                    )
                    for ind, ch in enumerate(child_nodes):
                        all_wt[ch] += temp_wt[ind]
                elif model_resource["graph"][start_layer]["class"] == 'Feed_Forward':
                    weights = all_wts[start_layer]
                    feed_forward_weights = HP.rename_feed_forward_keys(weights)
                    temp_wt = UP.calculate_wt_feed_forward(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        feed_forward_weights
                    )
                    all_wt[child_nodes[0]] += temp_wt             
                elif model_resource["graph"][start_layer]["class"] == "Pooler":
                    weights = all_wts[start_layer]
                    pooler_weights = HP.rename_pooler_keys(weights)
                    temp_wt = UP.calculate_wt_pooler(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        pooler_weights
                    )
                    all_wt[child_nodes[0]] += temp_wt               
                elif model_resource["graph"][start_layer]["class"] == "Classifier":
                    weights = all_wts[start_layer]
                    classifier_weights = HP.rename_classifier_keys(weights)
                    temp_wt = UP.calculate_wt_classifier(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        classifier_weights
                    )
                    all_wt[child_nodes[0]] += temp_wt                   
                elif model_resource["graph"][start_layer]["class"] == "LM_Head":
                    weights = all_wts[start_layer]
                    lm_head_weights = HP.rename_decoder_lm_head(weights)

                    temp_wt = UP.calculate_wt_lm_head(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],    #.detach().numpy(),
                        lm_head_weights
                    )
                    all_wt[child_nodes[0]] += temp_wt     
                elif model_resource["graph"][start_layer]["class"] == 'Layer_Norm':
                    temp_wt = all_wt[start_layer]
                    child_shape = all_wt[child_nodes[0]].shape
                    # Ensure temp_wt has the same shape as all_wt[child_nodes[0]]
                    if all_wt[child_nodes[0]].shape != temp_wt.shape:
                        temp_wt = np.squeeze(temp_wt, axis=0) if len(temp_wt.shape) == 3 else temp_wt
                    all_wt[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == 'Cross_Attention':
                    weights = all_wts[start_layer]
                    cross_attention_weights = HP.rename_cross_attention_keys(weights)
                    temp_wt = UP.calculate_wt_cross_attention(
                        all_wt[start_layer],
                        [all_out[ch][0] for ch in child_nodes],
                        cross_attention_weights,
                    )
                    for ind, ch in enumerate(child_nodes):
                        all_wt[ch] += temp_wt[ind]
                else:
                    temp_wt = all_wt[start_layer]
                    all_wt[child_nodes[0]] += temp_wt
        if max_unit>0:
            temp_dict = {}
            for k in all_wt.keys():
                temp_dict[k] = UC.weight_normalize(all_wt[k],max_val=max_unit)
            all_wt = temp_dict
        return all_wt

    def contrast_eval(self, all_out ,
                        multiplier=100.0, scaler=None,
                        thresholding=0.5,task="binary-classification"):
        model_resource = self.model_resource
        activation_dict = self.activation_dict
        inputcheck = False
        out_layer = model_resource["outputs"][0]
        all_wt_pos = {}
        all_wt_neg = {}
        start_wt_pos,start_wt_neg = UC.calculate_start_wt(all_out[out_layer], scaler,thresholding,task)
        all_wt_pos[out_layer] = start_wt_pos*multiplier
        all_wt_neg[out_layer] = start_wt_neg*multiplier
        layer_stack = self.layer_stack
        for start_layer in tqdm(layer_stack):
            if model_resource["graph"][start_layer]["child"]:
                child_nodes = model_resource["graph"][start_layer]["child"]
                for ch in child_nodes:
                    if ch not in all_wt_pos:
                        all_wt_pos[ch] = np.zeros_like(all_out[ch][0])
                        all_wt_neg[ch] = np.zeros_like(all_out[ch][0])
                if model_resource["graph"][start_layer]["class"] == 'Dense':
                    l1 = model_resource["layers"][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_fc(all_wt_pos[start_layer],
                                                              all_wt_neg[start_layer],
                                                              all_out[child_nodes[0]][0],
                                                              w1,b1,
                                                              activation_dict[model_resource["graph"][start_layer]['name']])
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource["graph"][start_layer]["class"] == 'Conv2D':
                    l1 = model_resource["layers"][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
                    pad1 = l1.padding
                    strides1 = l1.strides
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_conv(all_wt_pos[start_layer],
                                                                all_wt_neg[start_layer],
                                                                all_out[child_nodes[0]][0],
                                                                w1,b1, pad1, strides1,
                                                                activation_dict[model_resource["graph"][start_layer]['name']])
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource["graph"][start_layer]["class"] == "Conv2DTranspose":
                    l1 = model_resource["layers"][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
                    pad1 = l1.padding
                    strides1 = l1.strides
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_conv2d_transpose(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        pad1, 
                        strides1,
                        activation_dict[model_resource["graph"][start_layer]["name"]],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource["graph"][start_layer]["class"] == 'Conv1D':
                    l1 = model_resource["layers"][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
                    pad1 = l1.padding
                    strides1 = l1.strides[0]
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_conv_1d(all_wt_pos[start_layer],
                                                                all_wt_neg[start_layer],
                                                                all_out[child_nodes[0]][0],
                                                                w1,b1, pad1, strides1,
                                                                activation_dict[model_resource["graph"][start_layer]['name']])
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource["graph"][start_layer]["class"] == "Conv1DTranspose":
                    l1 = model_resource["layers"][start_layer]
                    w1 = l1.weights[0]
                    b1 = l1.weights[1]
                    pad1 = l1.padding
                    strides1 = l1.strides[0]
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_conv1d_transpose(all_wt_pos[start_layer],
                                                                            all_wt_neg[start_layer],
                                                                            all_out[child_nodes[0]][0],
                                                                            w1,b1, pad1, strides1,
                                                                            activation_dict[model_resource["graph"][start_layer]['name']])
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource["graph"][start_layer]["class"] == 'Reshape':
                    temp_wt_pos = UC.calculate_wt_rshp(all_wt_pos[start_layer],
                                                    all_out[child_nodes[0]][0])
                    temp_wt_neg = UC.calculate_wt_rshp(all_wt_neg[start_layer],
                                                    all_out[child_nodes[0]][0])
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource["graph"][start_layer]["class"] == 'GlobalAveragePooling2D':
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_gavgpool(all_wt_pos[start_layer],
                                                    all_wt_neg[start_layer],
                                                    all_out[child_nodes[0]][0])
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource["graph"][start_layer]["class"] == 'GlobalAveragePooling1D':
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_gavgpool_1d(all_wt_pos[start_layer],
                                                    all_wt_neg[start_layer],
                                                    all_out[child_nodes[0]][0])
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource["graph"][start_layer]["class"] == 'Flatten':
                    temp_wt = UC.calculate_wt_rshp(all_wt_pos[start_layer],
                                              all_out[child_nodes[0]][0])
                    all_wt_pos[child_nodes[0]] += temp_wt
                    temp_wt = UC.calculate_wt_rshp(all_wt_neg[start_layer],
                                              all_out[child_nodes[0]][0])
                    all_wt_neg[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == 'ZeroPadding2D':
                    l1 = model_resource["layers"][start_layer]
                    pad1 = l1.padding
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_zero_pad(all_wt_pos[start_layer],
                                                    all_wt_neg[start_layer],
                                                      all_out[child_nodes[0]][0],
                                                      pad1)
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource["graph"][start_layer]["class"] == 'MaxPooling2D':
                    l1 = model_resource["layers"][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.strides
                    temp_wt = UC.calculate_wt_maxpool(all_wt_pos[start_layer],
                                              all_out[child_nodes[0]][0],
                                              l1.pool_size, pad1, strides1)
                    all_wt_pos[child_nodes[0]] += temp_wt
                    temp_wt = UC.calculate_wt_maxpool(all_wt_neg[start_layer],
                                              all_out[child_nodes[0]][0],
                                              l1.pool_size, pad1, strides1)
                    all_wt_neg[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == 'MaxPooling1D':
                    l1 = model_resource["layers"][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.strides
                    temp_wt = UC.calculate_wt_maxpool_1d(all_wt_pos[start_layer],
                                              all_out[child_nodes[0]][0],
                                              l1.pool_size, pad1, strides1)
                    all_wt_pos[child_nodes[0]] += temp_wt
                    temp_wt = UC.calculate_wt_maxpool_1d(all_wt_neg[start_layer],
                                              all_out[child_nodes[0]][0],
                                              l1.pool_size, pad1, strides1)
                    all_wt_neg[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == 'GlobalMaxPooling2D':
                    temp_wt = UC.calculate_wt_gmaxpool_2d(all_wt_pos[start_layer],
                                              all_out[child_nodes[0]][0])
                    all_wt_pos[child_nodes[0]] += temp_wt
                    temp_wt = UC.calculate_wt_gmaxpool_2d(all_wt_neg[start_layer],
                                              all_out[child_nodes[0]][0])
                    all_wt_neg[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == 'GlobalMaxPooling1D':
                    temp_wt = UC.calculate_wt_gmaxpool_1d(all_wt_pos[start_layer],
                                              all_out[child_nodes[0]][0])
                    all_wt_pos[child_nodes[0]] += temp_wt
                    temp_wt = UC.calculate_wt_gmaxpool_1d(all_wt_neg[start_layer],
                                              all_out[child_nodes[0]][0])
                    all_wt_neg[child_nodes[0]] += temp_wt
                elif model_resource["graph"][start_layer]["class"] == 'AveragePooling2D':
                    l1 = model_resource["layers"][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.strides
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_avgpool(all_wt_pos[start_layer],
                                                                   all_wt_neg[start_layer],
                                                                   all_out[child_nodes[0]][0],
                                                                   l1.pool_size, pad1, strides1)
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource["graph"][start_layer]["class"] == 'AveragePooling1D':
                    l1 = model_resource["layers"][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.strides
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_avgpool_1d(all_wt_pos[start_layer],
                                                                   all_wt_neg[start_layer],
                                                                   all_out[child_nodes[0]][0],
                                                                   l1.pool_size, pad1, strides1)
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource["graph"][start_layer]["class"] == "Concatenate":
                    temp_wt = UC.calculate_wt_concat(all_wt_pos[start_layer],
                                                 [all_out[ch] for ch in child_nodes],
                                                 model_resource["layers"][start_layer].axis)
                    for ind, ch in enumerate(child_nodes):
                        all_wt_pos[ch]+=temp_wt[ind]
                    temp_wt = UC.calculate_wt_concat(all_wt_neg[start_layer],
                                                 [all_out[ch] for ch in child_nodes],
                                                 model_resource["layers"][start_layer].axis)
                    for ind, ch in enumerate(child_nodes):
                        all_wt_neg[ch]+=temp_wt[ind]

                elif model_resource["graph"][start_layer]["class"] == "Add":
                    temp_wt = UC.calculate_wt_add(all_wt_pos[start_layer],
                                               all_wt_neg[start_layer],
                                               [all_out[ch] for ch in child_nodes])
                    for ind, ch in enumerate(child_nodes):
                        all_wt_pos[ch]+=temp_wt[ind][0]
                        all_wt_neg[ch]+=temp_wt[ind][1]
                elif model_resource["graph"][start_layer]["class"] == "LSTM":
                    l1 = model_resource["layers"][start_layer]
                    return_sequence = l1.return_sequences
                    units = l1.units
                    num_of_cells = l1.input_shape[1]
                    lstm_obj_f = UC.LSTM_forward(num_of_cells, units, l1.weights, return_sequence, False)
                    lstm_obj_b = UC.LSTM_backtrace(num_of_cells, units, [i.numpy() for i in l1.weights], return_sequence, False)
                    temp_out_f = lstm_obj_f.calculate_lstm_wt(all_out[child_nodes[0]][0])
                    temp_wt_pos,temp_wt_neg = lstm_obj_b.calculate_lstm_wt(all_wt_pos[start_layer], 
                                                                           all_wt_neg[start_layer],
                                                                           lstm_obj_f.compute_log)
                    all_wt_pos[child_nodes[0]] = temp_wt_pos
                    all_wt_neg[child_nodes[0]] = temp_wt_neg
                elif model_resource["graph"][start_layer]["class"] == "Embedding":
                    temp_wt_pos = all_wt_pos[start_layer]
                    temp_wt_neg = all_wt_neg[start_layer]

                    temp_wt_pos = np.mean(temp_wt_pos,axis=1)
                    temp_wt_neg = np.mean(temp_wt_neg,axis=1)

                    all_wt_pos[child_nodes[0]] = all_wt_pos[child_nodes[0]] + temp_wt_pos
                    all_wt_neg[child_nodes[0]] = all_wt_neg[child_nodes[0]] + temp_wt_neg
                elif model_resource["graph"][start_layer]["class"] == "TextVectorization":
                    temp_wt_pos = all_wt_pos[start_layer]
                    temp_wt_neg = all_wt_neg[start_layer]

                    all_wt_pos[child_nodes[0]] = all_wt_pos[child_nodes[0]] + temp_wt_pos.sum()
                    all_wt_neg[child_nodes[0]] = all_wt_neg[child_nodes[0]] + temp_wt_neg.sum()
                else:
                    temp_wt_pos = all_wt_pos[start_layer]
                    temp_wt_neg = all_wt_neg[start_layer]
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                
        return all_wt_pos,all_wt_neg
