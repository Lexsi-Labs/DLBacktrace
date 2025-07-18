# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/graph_builder.py
import torch
import networkx as nx

from .config import ATEN_HYPERPARAMS_TABLE, ATEN_DEFAULTS_TABLE

def normalize_func_name(target):
    """
    Normalize FX/ATen targets to a base function name (e.g., 'unsqueeze').

    Parameters:
        target: The target attribute from an FX node, which can be a function, method, or string.

    Returns:
        A string representing the normalized function name.
    """
    name = str(target)
    if name.startswith('aten.'):
        return name.split('.')[1]  # Extracts 'unsqueeze' from 'aten.unsqueeze.default'
    elif '.' in name:
        return name.split('.')[0]  # Handles cases like 'torch.nn.functional.relu'
    return name  # Fallback for other formats


def extract_layer_hyperparams(node, extracted_weights):
    layer_hyperparams = {}
    func_name = node.target.__name__ if callable(node.target) else str(node.target)
    func_name = str(func_name).split('.')[0]
    default_flag = False

    if func_name in ATEN_HYPERPARAMS_TABLE:
        expected_params = ATEN_HYPERPARAMS_TABLE[func_name]
        args_node = iter(node.args)

        for param in expected_params:
            value, arg_flag, flag_skip = None, False, False
            try:
                while True:
                    x_itr = next(args_node)
                    if isinstance(x_itr, torch.fx.node.Node):
                        if param in x_itr.name:
                            value = extracted_weights.get(x_itr.name)
                            arg_flag = True
                            layer_hyperparams[param] = value
                            break
                    elif isinstance(x_itr, (tuple, list)) and any(isinstance(item, torch.fx.node.Node) for item in x_itr):
                        x_itr_list = []
                        for item in x_itr:
                            value = extracted_weights.get(item.name, None) if isinstance(item, torch.fx.node.Node) else item
                            x_itr_list.append(value)
                            flag_skip = True
                        layer_hyperparams[param] = x_itr_list
                        arg_flag = True
                        break
                    elif x_itr is None and param == "bias":
                        value = None
                        arg_flag = True
                        layer_hyperparams[param] = value
                        break
                    elif isinstance(x_itr, (bool, int, float, list, tuple, torch.memory_format)):
                        value = x_itr
                        break
            except StopIteration:
                value = ATEN_DEFAULTS_TABLE.get(param, None)
                default_flag = True

            if value is not None and not arg_flag:
                if func_name.startswith(("conv", "pool")) and param in {"stride", "padding", "dilation", "output_padding"}:
                    dim = func_name[-2:]
                    num_dims = 1 if dim == "1d" else 2 if dim == "2d" else 3 if dim == "3d" else 1
                    value = tuple(value if isinstance(value, (list, tuple)) else [value] * num_dims)

                if param in {"stride", "padding", "dilation", "output_padding", "kernel_size", "normalized_shape"}:
                    layer_hyperparams[param] = tuple(value) if isinstance(value, (list, tuple)) else (value,)
                elif param in {"momentum", "eps"}:
                    layer_hyperparams[param] = float(value)
                elif param in {"groups", "num_groups"}:
                    layer_hyperparams[param] = int(value)
                elif param == "bias" and default_flag:
                    layer_hyperparams[param] = bool(value)
                elif param in {"ceil_mode", "count_include_pad", "affine", "track_running_stats", "elementwise_affine"}:
                    layer_hyperparams[param] = bool(value)
                elif param == "dim" or func_name == "softmax":
                    layer_hyperparams[param] = value if isinstance(value, int) else 1
                elif param in {"shape", "dims", "dim0", "dim1", "sizes"}:
                    layer_hyperparams[param] = tuple(value) if isinstance(value, (list, tuple)) else (value,)
                else:
                    layer_hyperparams[param] = value
            elif flag_skip:
                continue
            else:
                layer_hyperparams[param] = value

    if "bias" not in layer_hyperparams:
        layer_hyperparams["bias"] = None

    return layer_hyperparams


def build_graph(tracer, extracted_weights, verbose=False):
    graph = nx.DiGraph()

    for node in tracer.graph.nodes:
        name = node.name
        graph.add_node(name, 
                       parents=[], 
                       children=[], 
                       method_args=[],
                       node_type=node.op,
                       layer=None,
                       layer_type=None,
                       layer_name=None,
                       func_name=None,
                       func_module=None,
                       layer_hyperparams=None)

    for node in tracer.graph.nodes:
        name = node.name
        node_type = node.op
        func_name = None
        func_module = None
        layer_type = "CustomOp"
        layer_name = "Unknown_Operation"
        layer_hyperparams = None

        if node_type == 'call_function':
            func_name = node.target.__name__ if callable(node.target) else str(node.target)
            func_module = node.target.__module__ if callable(node.target) else "Unknown"

            if func_module == "torch._ops.aten":
                func_name = normalize_func_name(func_name)

                # Layer classification
                ATEN_LAYER_MAP = {
                    "DL_Layer": {'conv2d', 'conv_transpose2d', 'max_pool2d', 'avg_pool2d', 'adaptive_avg_pool2d', 'lstm'},
                    "Activation": {'relu', 'relu_', 'sigmoid', 'softmax', 'gelu', 'tanh', 'silu', '_softmax'},
                    "MLP_Layer": {'linear', 'addmm'},
                    "Dynamic_Size": {'sym_size'},
                    "Mathematical_Operation": {'matmul', 'add', 'add_', 'zeros', 'rsub', 'pow', 'rsqrt', 'neg',
                                               'triu', 'gt', 'eq', 'mul_', 'mul', 'cos', 'sin', 'mm', 'bmm',
                                               'logical_not', 'where', 'sub', 'ge'},
                    "Normalization": {'batch_norm', 'layer_norm', 'dropout'},
                    "Vector_Operation": {'flatten', 'transpose', 'permute', 'reshape', 'cat', 'stack', 'split',
                                         'chunk', 'unsqueeze', 'squeeze', 'slice', 'select', 'view', 'unflatten',
                                         'contiguous', 'mean', 'expand', 'to', 'clone', 'copy_', 'arange', 'full',
                                         '_to_copy', '_unsafe_view', 'slice_scatter', 'copy', 'full_like', 'any', 'scalar_tensor'},
                    "masked_fill": {'masked_fill'},
                    "NLP_Embedding": {'embedding', 'embedding_bag'},
                    "Attention": {'scaled_dot_product_attention'}
                }

                for category, ops in ATEN_LAYER_MAP.items():
                    if func_name in ops:
                        layer_name = category
                        break
                else:
                    layer_name = "Unknown_Operation"
                    if verbose:
                        print(f"[Unknown ATen] {func_name} ← {node.target}")

                layer_type = "ATen_Operation"
                layer_hyperparams = extract_layer_hyperparams(node, extracted_weights)

            elif func_module == "builtins":
                layer_name = "Python_Built-in_Function"
                layer_type = "Operation"

            elif func_name == "getitem":
                layer_name = "Indexing_Operation"
                layer_type = "Operation"
            else:
                layer_type = "Operation"

        elif node_type == "placeholder":
            layer_name = "Placeholder"
            layer_type = "Placeholder"

        elif node_type == "output":
            layer_name = "Output"
            layer_type = "Output"

        # Save metadata
        graph.nodes[name].update({
            "layer": node.target,
            "layer_name": layer_name,
            "layer_type": layer_type,
            "func_name": func_name,
            "func_module": func_module,
            "layer_hyperparams": layer_hyperparams,
            "method_args": [arg for arg in node.args if not isinstance(arg, torch.fx.Node)],
            "node_type": node_type
        })

        for input_node in node.all_input_nodes:
            if isinstance(input_node, torch.fx.Node):
                parent = input_node.name
                graph.nodes[name]['parents'].append(parent)
                graph.nodes[parent]['children'].append(name)
                graph.add_edge(parent, name)

        if verbose:
            print(f"🔹 {name} → {layer_type} [{layer_name}] | inputs: {graph.nodes[name]['parents']}")

    layer_stack = list(nx.topological_sort(graph))
    return graph, layer_stack
