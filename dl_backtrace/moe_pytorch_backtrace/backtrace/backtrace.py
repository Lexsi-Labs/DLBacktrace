import numpy as np
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from dl_backtrace.moe_pytorch_backtrace.backtrace.utils import contrast as UC
from dl_backtrace.moe_pytorch_backtrace.backtrace.utils import prop as UP
from dl_backtrace.moe_pytorch_backtrace.backtrace.config import activation_master
from dl_backtrace.moe_pytorch_backtrace.backtrace.core import jetmoe as jetmoe, olmoe as olmoe, qwen3_moe as qwen3_moe, gpt_oss as gpt_oss, helper as helper
from dl_backtrace.moe_pytorch_backtrace.backtrace.utils import default_v2 as UD2


def t2np32(t):
        # Cast any torch tensor to float32 on CPU before numpy() to avoid bf16 errors
        if isinstance(t, torch.Tensor):
            return t.detach().to(torch.float32).cpu().numpy()
        return t

def _layer_idx(name: str):
    m = re.search(r'_(\d+)$', name)
    return int(m.group(1)) if m else None

def build_attention_plan(layer_stack, config):
    plan = {}
    for name in layer_stack:
        if not name.startswith("decoder_self_attention_"):
            continue
        idx = _layer_idx(name)
        if idx is None:
            continue
        is_sliding = (config.layer_types[idx] == "sliding_attention")
        plan[name] = {
            "attn_type": "sliding" if is_sliding else "full",
            "window": config.sliding_window if is_sliding else None,
        }
    return plan

def get_model_config(model):
    """
    Extracts the configuration object from a model, handling wrapped models.

    Args:
        model: A model instance or a wrapper around a model.

    Returns:
        The model's configuration object.

    Raises:
        AttributeError: If no configuration could be found.
    """
    # Check if the object has an underlying model
    if hasattr(model, 'model'):
        # It's a wrapper around another model
        inner_model = model.model
        if hasattr(inner_model, 'config'):
            return inner_model.config
    elif hasattr(model, 'config'):
        # It's the base model itself
        return model.config

    raise AttributeError("The provided object does not have a 'config' attribute.")


def get_tensor_or_raise(all_in, all_out, key, where="all_out"):
    x = all_out.get(key, None)
    if x is None:
        x = all_in.get(key, None)
        where = "all_in" if x is not None else where
    if x is None:
        raise KeyError(
            f"Missing tensor for node '{key}'. "
            f"Have keys(all_out)={list(all_out.keys())} "
            f"keys(all_in)={list(all_in.keys())}"
        )
    # unwrap tuples/lists from hooks
    if isinstance(x, (tuple, list)):
        x = x[0]
    if not torch.is_tensor(x):
        raise TypeError(f"Node '{key}' is not a tensor (got {type(x)}) from {where}.")
    return x

def tensor_to_numpy(x):
    """Convert Tensor, scalar, list/tuple, or ndarray to a NumPy array with memory-efficient float32."""
    if isinstance(x, np.ndarray):
        # Downcast float64 to float32 to save memory
        return x.astype(np.float32) if x.dtype == np.float64 else x

    if isinstance(x, torch.Tensor):
        # IMPORTANT: cast to float32 *before* .numpy() to avoid bfloat16 error
        return x.detach().to(torch.float32).cpu().numpy()

    if isinstance(x, (int, float)):
        return np.array(x, dtype=np.float32)

    if isinstance(x, (list, tuple)):
        arrs = []
        for xi in x:
            converted = tensor_to_numpy(xi) if not isinstance(xi, np.ndarray) else xi
            if hasattr(converted, 'dtype') and converted.dtype == np.float64:
                converted = converted.astype(np.float32)
            arrs.append(converted)
        try:
            return np.stack(arrs)
        except Exception:
            return np.array(arrs, dtype=object)

    raise TypeError(f"Cannot convert type {type(x)} to numpy")


class Backtrace(object):
    """
    This is the constructor method for the Backtrace class. It initializes an instance of the class.
    It takes two optional parameters: model (a neural network model) and activation_dict (a dictionary that maps layer names to activation functions).
    """

    def __init__(self, model=None, activation_dict={}, model_type=None):
        # if model_type == 'encoder':
        #     self.model = model
        #     self.model_type = model_type
        #     # create a tree-like structure for encoder model
        #     self.model_resource = EN.build_encoder_tree(model)
        #     # create a layer stack for encoder model
        #     self.create_layer_stack()
        #     # extract the encoder model weights
        #     self.model_weights = EN.extract_encoder_weights(model)
        #     # # calculate the output of each submodule of the encoder model
        #     # self.all_out_model = EN.create_encoder_output(model)
        #     self.activation_dict = None
            
        # elif model_type == 'encoder_decoder':
        #     self.model = model
        #     self.model_type = model_type
        #     # create a tree-like structure and layer_stack for encoder-decoder model
        #     self.model_resource, self.layer_stack = ED.build_enc_dec_tree(model)
        #     # extract the encoder-decoder model weights
        #     self.model_weights = ED.extract_encoder_decoder_weights(model)  
        #     # # calculate the output of each submodule of the encoder-decoder model
        #     # self.all_out_model = ED.calculate_encoder_decoder_output(model)
        #     self.activation_dict = None
            
        # elif model_type == 'llama':
        #     self.model = model
        #     self.model_type = model_type
        #     # create a tree-like structure and layer stack for llama model
        #     self.model_resource, self.layer_stack = LL.build_llama_tree(model)
        #     # extract the llama model weights
        #     self.model_weights = LL.extract_llama_weights(model)
        #     # # calculate the output of each submodule of the llama model
        #     # self.all_out_model = LL.create_llama_output(input_text, model, tokenizer, max_length, device)
        #     self.activation_dict = None

        if model_type == 'gpt_oss':
            self.model = model
            self.model_type = model_type
            # create a tree-like structure and layer stack for gpt_oss model
            self.model_resource, self.layer_stack = gpt_oss.build_gpt_oss_tree(model)
            # extract the gpt_oss model weights
            self.model_weights = gpt_oss.extract_gpt_oss_weights(model)
            # # calculate the output of each submodule of the gpt_oss model
            self.all_out_model = gpt_oss.create_gpt_oss_output(input_text, model, tokenizer, max_length, device)
            self.activation_dict = None
            
            self.all_layer_expert_relevance = {}    # Storing the relevance of experts

        elif model_type == 'qwen3_moe':
            self.model = model
            self.model_type = model_type
            # create a tree-like structure and layer stack for qwen3_moe model
            self.model_resource, self.layer_stack = qwen3_moe.build_qwen3_moe_tree(model)
            # extract the qwen3_moe model weights
            self.model_weights = qwen3_moe.extract_qwen3_moe_weights(model)
            # calculate the output of each submodule of the qwen3_moe model
            self.all_out_model = qwen3_moe.create_qwen3_moe_output(input_text, model, tokenizer, max_length, device)
            self.activation_dict = None
                
            self.all_layer_expert_relevance = {}    # Storing the relevance of experts
            
        elif model_type == 'jetmoe':
            self.model = model
            self.model_type = model_type
            # create a tree-like structure and layer stack for jetmoe model
            self.model_resource, self.layer_stack = jetmoe.build_jetmoe_tree(model)
            # extract the jetmoe model weights
            self.model_weights = jetmoe.extract_jetmoe_weights(model)
            # calculate the output of each submodule of the jetmoe model
            self.all_out_model = jetmoe.create_jetmoe_output(input_text, model, tokenizer, max_length, device)
            self.activation_dict = None
            
            self.all_layer_expert_relevance = {}    # Storing the relevance of experts
            
        elif model_type == 'olmoe':
            self.model = model
            self.model_type = model_type 
            # create a tree-like structure and layer stack for jetmoe model
            self.model_resource, self.layer_stack = olmoe.build_olmoe_tree(model)
            # extract the jetmoe model weights
            self.model_weights = olmoe.extract_olmoe_weights(model)
            # calculate the output of each submodule of the jetmoe model
            self.all_out_model = olmoe.create_olmoe_output(input_text, model, tokenizer, max_length, device)
            self.activation_dict = None
            
            self.all_layer_expert_relevance = {}    # Storing the relevance of experts
            
    #     else:
    #         self.model_type = model_type
    #         # create a tree-like structure that represents the layers of the neural network model
    #         self.create_tree(model)
    #         # create a new model (an instance of tf.keras.Model) that produces the output of each layer in the neural network.
    #         self.create_model_output(model)
    #         # create a new model (an instance of tf.keras.Model) that produces the output of each layer in the neural network.
    #         self.create_every_model_output(model)
    #         # create a layer stack that defines the order in which layers should be processed during backpropagation.
    #         self.create_layer_stack()
    #         # checks if the model is sequential or not. If it's sequential, it adds the input layer to the layer stack.
    #         # identity

    #         inp_name = 'identity'
    #         self.layer_stack.append(inp_name)
    #         self.model_resource[1][inp_name] = {}
    #         self.model_resource[1][inp_name]["name"] = inp_name
    #         self.model_resource[1][inp_name]["type"] = "input"
    #         self.model_resource[1][inp_name]["parent"] = []
    #         self.model_resource[1][inp_name]["child"] = None
    #         self.model_resource[3].append(inp_name)
    #         self.sequential = True
    #         try:
    #             # calls the build_activation_dict method to build a dictionary that maps layer names to activation functions.
    #             # If that fails, it creates a temporary dictionary with default activation functions.
    #             if len(activation_dict) == 0:
    #                 self.build_activation_dict(model)
    #             else:
    #                 self.activation_dict = activation_dict

    #         except Exception as e:
    #             print(e)
    #             temp_dict = {}
    #             for l in model.layers:
    #                 temp_dict[l.name] = activation_master["None"]
    #             self.activation_dict = temp_dict 

    # def build_activation_dict(self, model):
    #     model_resource = self.model_resource
    #     layer_list = list(model_resource[0].keys())
    #     activation_dict = {}
    #     activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax']  # You can add more activation functions
    #     for l in layer_list:
    #         activation_found = False
    #         try:  # could be activation for that layer
    #             for activation in activation_functions:
    #                 if activation in l.split('/')[1]:
    #                     activation_dict[l.split('/')[0]] = activation
    #                     activation_found = True
    #         except:
    #             activation_dict[l] = 'None'
    #     # activation_master :
    #     for key, value in activation_dict.items():
    #         activation_dict[key] = activation_master.get(value)
    #     self.activation_dict = activation_dict

    # def create_tree(self, model):
    #     # create new layers same as tf version
    #     layers = list(model.named_children())
    #     activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax']
    #     layer_sequence = []
    #     for i in range(len(layers) - 1):
    #         current_layer, current_layer_obj = layers[i]
    #         next_layer, next_layer_obj = layers[i + 1]
    #         current_layer_name = current_layer
    #         next_layer_name = next_layer

    #         next_layer_type = next_layer_name.lower()
    #         if any(af in next_layer_type for af in activation_functions):
    #             layer_sequence.append((f"{current_layer_name}/{next_layer_name}", current_layer_obj))
    #             i += 1
    #         else:
    #             if any(af in current_layer_name for af in activation_functions) is False:
    #                 layer_sequence.append((current_layer_name, current_layer_obj))
    #     # creating model_resource variable
    #     layer_sequence
    #     ltree = {}
    #     layer_tree = {}
    #     inputs = []
    #     outputs = []
    #     intermediates = []
    #     prev_layer_id = None
    #     num_layers = len(layer_sequence)
    #     for i, (layer_name, layer) in enumerate(layer_sequence):
    #         layer_id = layer_name
    #         ltree[layer_id] = {}
    #         layer_tree[layer_id] = layer
    #         layer_type = layer.__class__.__name__
    #         ltree[layer_id]["name"] = layer_id.split("/")[0]
    #         ltree[layer_id]["class"] = layer_type
    #         if i < num_layers - 1:
    #             ltree[layer_id]["type"] = "intermediate"
    #             intermediates.append(layer_id)
    #         else:
    #             ltree[layer_id]["type"] = "output"
    #             outputs.append(layer_id)
    #         if prev_layer_id is not None:
    #             ltree[layer_id]["child"] = [prev_layer_id]
    #             ltree[prev_layer_id]["parent"] = [layer_id]
    #         prev_layer_id = layer_id
    #     # Set child of the last layer as an empty list
    #     if prev_layer_id is not None:
    #         ltree[prev_layer_id]["parent"] = []
    #     layer_tree.pop('identity')
    #     ltree.pop('identity')
    #     self.model_resource = (layer_tree, ltree, outputs, inputs)

    # def create_layer_stack(self):
    #     model_resource = self.model_resource
    #     start_layer = model_resource[2][0]
    #     layer_stack = [start_layer]
    #     temp_stack = [start_layer]
    #     while len(layer_stack) < len(model_resource[0]):
    #         start_layer = temp_stack.pop(0)
    #         if model_resource[1][start_layer]["child"]:
    #             child_nodes = model_resource[1][start_layer]["child"]
    #             for ch in child_nodes:
    #                 node_check = True
    #                 for pa in model_resource[1][ch]["parent"]:
    #                     if pa not in layer_stack:
    #                         node_check = False
    #                         break
    #                 if node_check:
    #                     if ch not in layer_stack:
    #                         layer_stack.append(ch)
    #                 temp_stack.append(ch)
    #     self.layer_stack = layer_stack

    # def create_every_model_output(self, model):
    #     class ModelWithEveryOutputs(nn.Module):
    #         def __init__(self, base_model):
    #             super(ModelWithEveryOutputs, self).__init__()
    #             self.base_model = base_model
    #         def forward(self, x):
    #             outputs = []
    #             for layer_name, layer in self.base_model._modules.items():
    #                 if isinstance(x, tuple):
    #                     if isinstance(layer, nn.LSTM):
    #                         # Assuming you want to take the last LSTM output
    #                         x, _ = layer(x[0])  # Pass the first element of the tuple (assumes one LSTM layer)
    #                     else:
    #                         x = layer(x[0])  # Pass the first element of the tuple
    #                 else:
    #                     x = layer(x)
    #                 outputs.append((layer_name, x))
    #             return outputs
    #     self.every_out_model = ModelWithEveryOutputs(model)

    # def create_model_output(self, model):
    #     class ModelWithOutputs(nn.Module):
    #         def __init__(self, base_model):
    #             super(ModelWithOutputs, self).__init__()
    #             self.base_model = base_model

    #         def forward(self, x):
    #             outputs = []
    #             for layer_name, layer in self.base_model._modules.items():
    #                 if isinstance(layer, nn.LSTM):
    #                     lstm_output, _ = layer(x)
    #                     if lstm_output.dim() == 3:
    #                         x = lstm_output[:, -1, :]  # Take the output of the last time step
    #                     else:
    #                         x = lstm_output
    #                 else:
    #                     x = layer(x)
    #                 outputs.append((layer_name, x))
    #             return outputs

    #     # all_out_model = ModelWithOutputs(model)
    #     self.all_out_model = ModelWithOutputs(model)
    #     model.eval()
    #     model_resource = self.model_resource
    #     self.layers = [[], []]
    #     for l in model_resource[0]:
    #         self.layers[0].append(l)
    #         self.layers[1].append(model_resource[0][l])

    # def predict_every(self, inputs):
    #     every_out = self.every_out_model(inputs)
    #     activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax']
    #     every_temp_out = {}
    #     for i in range(len(every_out)):
    #         current_layer, current_layer_obj = every_out[i]
    #         try:
    #             next_layer, next_layer_obj = every_out[i + 1]
    #             current_layer_name = current_layer
    #             next_layer_name = next_layer
    #             next_layer_type = next_layer_name.lower()
    #             if any(af in next_layer_type for af in activation_functions):
    #                 if isinstance(next_layer_obj, tuple):
    #                     # Assuming you want the first tensor from the tuple
    #                     next_layer_tensor = next_layer_obj[0]
    #                 else:
    #                     next_layer_tensor = next_layer_obj
    #                 every_temp_out[
    #                     f"{current_layer_name}/{next_layer_name}"] = next_layer_tensor.detach().numpy().astype(
    #                     np.float32)
    #                 i += 1
    #             else:
    #                 if any(af in current_layer_name for af in activation_functions) is False:
    #                     if isinstance(current_layer_obj, tuple):
    #                         # Assuming you want the first tensor from the tuple
    #                         current_layer_tensor = current_layer_obj[0]
    #                     else:
    #                         current_layer_tensor = current_layer_obj
    #                     every_temp_out[current_layer_name] = current_layer_tensor.detach().numpy().astype(np.float32)
    #         except:
    #             if any(af in next_layer_type for af in activation_functions):
    #                 pass
    #             else:
    #                 if any(af in current_layer for af in activation_functions) is False:
    #                     if isinstance(current_layer_obj, tuple):
    #                         # Assuming you want the first tensor from the tuple
    #                         current_layer_tensor = current_layer_obj[0]
    #                     else:
    #                         current_layer_tensor = current_layer_obj
    #                     every_temp_out[current_layer] = current_layer_tensor.detach().cpu().numpy().astype(np.float32)
    #     return every_temp_out

    # def predict(self, inputs):
    #     all_out = self.all_out_model(inputs)
    #     activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax']
    #     temp_out = {}
    #     for i in range(len(all_out)):
    #         current_layer, current_layer_obj = all_out[i]
    #         try:
    #             next_layer, next_layer_obj = all_out[i + 1]
    #             current_layer_name = current_layer
    #             next_layer_name = next_layer
    #             next_layer_type = next_layer_name.lower()
    #             if any(af in next_layer_type for af in activation_functions):
    #                 if isinstance(next_layer_obj, tuple):
    #                     # Assuming you want the first tensor from the tuple
    #                     next_layer_tensor = next_layer_obj[0]
    #                 else:
    #                     next_layer_tensor = next_layer_obj
    #                 temp_out[
    #                     f"{current_layer_name}/{next_layer_name}"] = next_layer_tensor.detach().cpu().numpy().astype(
    #                     np.float32)
    #                 i += 1
    #             else:
    #                 if any(af in current_layer_name for af in activation_functions) is False:
    #                     if isinstance(current_layer_obj, tuple):
    #                         # Assuming you want the first tensor from the tuple
    #                         current_layer_tensor = current_layer_obj[0]
    #                     else:
    #                         current_layer_tensor = current_layer_obj
    #                     temp_out[current_layer_name] = current_layer_tensor.detach().numpy().astype(np.float32)
    #         except:
    #             if any(af in next_layer_type for af in activation_functions):
    #                 pass
    #             else:
    #                 if any(af in current_layer for af in activation_functions) is False:
    #                     if isinstance(current_layer_obj, tuple):
    #                         # Assuming you want the first tensor from the tuple
    #                         current_layer_tensor = current_layer_obj[0]
    #                     else:
    #                         current_layer_tensor = current_layer_obj
    #                     temp_out[current_layer] = current_layer_tensor.detach().cpu().numpy().astype(np.float32)
    #     return temp_out

    def eval(
            self,
            all_in,
            all_out,
            mode="default",
            start_wt=[],
            multiplier=100.0,
            scaler=0,
            max_unit=0,
            predicted_token=None,
            thresholding=0.5,
            task="binary-classification",
    ):
        # This method is used for evaluating layer-wise relevance based on different modes.
        if mode == "default":
            output = self.proportional_eval(
                all_in=all_in,
                all_out=all_out,
                start_wt=start_wt,
                multiplier=multiplier,
                scaler=0,
                max_unit=0,
                predicted_token=predicted_token,
                thresholding=0.5,
                task="binary-classification",
            )
            return output
        # elif mode == "contrast":
        #     temp_output = self.contrast_eval(
        #         all_out=all_out, 
        #         multiplier=multiplier,
        #         scaler=0,
        #         thresholding=0.5,
        #         task="binary-classification",
        #     )
        #     output = {}
        #     for k in temp_output[0].keys():
        #         output[k] = {}
        #         output[k]["Positive"] = temp_output[0][k]
        #         output[k]["Negative"] = temp_output[1][k]
        #     return output

    def proportional_eval(
            self, all_in, all_out, start_wt=[], multiplier=100.0, 
            scaler=0, max_unit=0, predicted_token=None,
            thresholding=0.5, task="binary-classification", get_layer_implementation=None
    ):
        if get_layer_implementation is None:
            get_layer_implementation = lambda x: "original"

        model_resource = self.model_resource
        activation_dict = self.activation_dict
        layer_stack = self.layer_stack
        all_wts = self.model_weights
        inputcheck = False
        all_wt = {}

        out_layer = model_resource['outputs'][0]
        raw_out = get_tensor_or_raise(all_in, all_out, out_layer, where="out_layer")
        out_np = tensor_to_numpy(raw_out)

        if len(start_wt) == 0:
            start_wt = UD2.calculate_start_wt(out_np, scaler=scaler, task='generation')
        
        all_wt[out_layer] = start_wt * multiplier     
                
        for start_layer in tqdm(layer_stack):
            if model_resource['graph'][start_layer]["child"]:
                child_nodes = model_resource['graph'][start_layer]["child"]
                for ch in child_nodes:
                    if ch not in all_wt:
                        x = get_tensor_or_raise(all_in, all_out, ch, where=f"child of {start_layer}")
                        all_wt[ch] = np.zeros(x.shape, dtype=np.float64)

                if model_resource['graph'][start_layer]["class"] == "LM_Head":
                    weights = all_wts[start_layer]
                    lm_head_weights = helper.rename_decoder_lm_head(weights)
                    impl = get_layer_implementation("LM_Head")
                    temp_wt = UD2.launch_lm_head(  # .calculate_wt_lm_head_parallel(
                        impl,
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0].detach().numpy(),
                        lm_head_weights,
                        b=None,
                        act=None
                    )
                    all_wt[child_nodes[0]] += temp_wt

                elif model_resource['graph'][start_layer]["class"] == 'Layer_Norm':
                    temp_wt = all_wt[start_layer]
                    all_wt[child_nodes[0]] += temp_wt

                elif model_resource['graph'][start_layer]["class"] == 'Residual':
                    temp_wt = UP.calculate_wt_residual(
                        all_wt[start_layer],
                        [all_out[ch].detach().numpy() for ch in child_nodes],
                    )

                    for ind, ch in enumerate(child_nodes):
                        all_wt[ch] += temp_wt[ind]

                # elif model_resource['graph'][start_layer]["class"] == "Embedding":
                #     temp_wt = all_wt[start_layer]
                #     temp_wt = np.mean(temp_wt,axis=1)
                #     all_wt[child_nodes[0]] = all_wt[child_nodes[0]] + temp_wt
                
                # -------------------- For JetMoE ---------------------
                elif model_resource['graph'][start_layer]["class"] == 'JetMoE_Feed_Forward':
                    weights = all_wts[start_layer]
                    feed_forward_weights = helper.rename_jetmoe_feed_forward_keys(weights)
                    
                    temp_wt, ff_expert = UD2.launch_jetmoe_feed_forward(  #) UP.calculate_wt_jetmoe_feed_forward(
                        impl,
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0].detach().numpy(),
                        feed_forward_weights,
                        self.model
                    )
                    
                    all_wt[child_nodes[0]] += temp_wt
                    layer = f"{start_layer}_ff_expert"
                    self.all_layer_expert_relevance[layer] = ff_expert
                
                elif model_resource['graph'][start_layer]["class"] == 'JetMoE_Self_Attention':
                    weights = all_wts[start_layer]
                    self_attention_weights = helper.rename_jetmoe_self_attention_keys(weights)
                    
                    temp_wt, attention_expert =UD2.launch_jetmoe_self_attention(   # UP.calculate_wt_jetmoe_self_attention_parallel(
                        impl,
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0],
                        self_attention_weights, 
                        self.model
                    )
                    
                    all_wt[child_nodes[0]] += temp_wt
                    layer = f"{start_layer}_attention_expert"
                    self.all_layer_expert_relevance[layer] = attention_expert

                # -------------------- For OLMoE ---------------------
                elif model_resource['graph'][start_layer]["class"] == 'OLMoE_Feed_Forward':
                    weights = all_wts[start_layer]
                    feed_forward_weights = helper.rename_olmoe_feed_forward_keys(weights)
                    
                    temp_wt, ff_expert = UD2.launch_olmoe_feed_forward(    # UP.calculate_wt_olmoe_feed_forward_parallel(
                        impl,
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0].detach().numpy(),
                        feed_forward_weights,
                        self.model 
                    )
                    
                    all_wt[child_nodes[0]] += temp_wt
                    layer = f"{start_layer}_ff_expert"
                    self.all_layer_expert_relevance[layer] = ff_expert
                
                elif model_resource['graph'][start_layer]["class"] == "Self_Attention":
                    weights = all_wts[start_layer]
                    self_attention_weights = helper.rename_self_attention_keys(weights)
                    config = get_model_config(self.model)
                    temp_wt = UP.calculate_wt_self_attention_parallel(
                        all_wt[start_layer],
                        all_out[child_nodes[0]][0].detach().numpy(),
                        self_attention_weights,
                        config
                    )
                    all_wt[child_nodes[0]] += temp_wt

                # -------------------- For Qwen3-MoE ---------------------
                elif model_resource["graph"][start_layer]["class"] == 'Qwen_Feed_Forward':
                    weights = all_wts[start_layer]
                    feed_forward_weights = helper.rename_qwenmoe_feed_forward_keys(weights)
                    config = get_model_config(self.model)

                    temp_wt, ff_expert = UD2.launch_qwen3_moe_feed_forward(
                        impl,
                        all_wt[start_layer],
                        all_in[child_nodes[0]].detach().numpy(),  # all_out[child_nodes[0]].detach().numpy(),
                        feed_forward_weights,
                        config
                    )

                    all_wt[child_nodes[0]] += temp_wt

                    layer = f"{start_layer}_ff_expert"
                    self.all_layer_expert_relevance[layer] = ff_expert

                elif model_resource["graph"][start_layer]["class"] == 'Grouped_Query_Attention':
                    weights = all_wts[start_layer]
                    # print(f"weights: {weights.keys()}")
                    self_attention_weights = helper.rename_self_attention_keys(weights)
                    config = get_model_config(self.model)

                    temp_wt = UD2.launch_qwen3_moe_self_attention(    # calculate_wt_self_attention_parallel(
                        impl,
                        all_wt[start_layer],
                        all_in[child_nodes[0]].detach().numpy(),  # all_out[child_nodes[0]].detach().numpy(),
                        self_attention_weights,
                        config
                    )

                    all_wt[child_nodes[0]] += temp_wt

                # -------------------- For GPT-OSS MoE ---------------------
                elif model_resource["graph"][start_layer]["class"] == 'GPT_OSS_Feed_Forward':
                    weights = all_wts[start_layer]
                    feed_forward_weights = helper.rename_gptoss_feed_forward_keys(weights) #rename_feed_forward_keys(weights)
                    config = get_model_config(self.model)

                    temp_wt, ff_expert = UD2.launch_gpt_oss_feed_forward(   # calculate_wt_gpt_oss_feed_forward_parallel(
                        impl,
                        all_wt[start_layer],
                        t2np32(all_in[child_nodes[0]]),  # all_out[child_nodes[0]].detach().numpy(),
                        feed_forward_weights,
                        config
                    )

                    all_wt[child_nodes[0]] += temp_wt

                    layer = f"{start_layer}_ff_expert"
                    self.all_layer_expert_relevance[layer] = ff_expert

                elif model_resource["graph"][start_layer]["class"] in 'GPT_OSS_Self_Attention':
                    weights = all_wts[start_layer]
                    # print(f"weights: {weights.keys()}")
                    self_attention_weights = helper.rename_self_attention_keys(weights)
                    config = get_model_config(self.model)

                    ATTN_PLAN = build_attention_plan(layer_stack, config)
                    attn_info = ATTN_PLAN.get(start_layer, {"attn_type": "full", "window": None})

                    temp_wt = UD2.launch_gpt_oss_self_attention(    # calculate_wt_self_attention_parallel(
                        impl,
                        all_wt[start_layer],
                        t2np32(all_in[child_nodes[0]]),  # all_out[child_nodes[0]].detach().numpy(),
                        self_attention_weights,
                        config,
                        attn_type=attn_info["attn_type"],
                        sliding_window=attn_info["window"],
                    )

                    all_wt[child_nodes[0]] += temp_wt
                    print(f"temp_wt: {np.sum(temp_wt):.2f}")
                
                # Default calling 
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
