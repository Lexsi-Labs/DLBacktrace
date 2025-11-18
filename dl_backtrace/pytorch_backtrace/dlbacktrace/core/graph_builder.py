# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/graph_builder.py

import logging
import torch
import networkx as nx
from typing import Any, Dict, List, Optional, Union, Tuple

from .config import ATEN_HYPERPARAMS_TABLE, ATEN_DEFAULTS_TABLE

# Set up logger for this module
logger = logging.getLogger(__name__)

def normalize_func_name(target: Any) -> str:
    """
    Normalize FX/ATen targets to a base function name (e.g., 'unsqueeze').

    Args:
        target: The target attribute from an FX node, which can be a function, method, or string.

    Returns:
        A string representing the normalized function name.
        
    Raises:
        TypeError: If target cannot be converted to string
    """
    if target is None:
        logger.warning("Target is None, returning 'unknown'")
        return "unknown"
    
    try:
        name = str(target)
        if name.startswith('aten.'):
            result = name.split('.')[1]  # Extracts 'unsqueeze' from 'aten.unsqueeze.default'
            logger.debug(f"Normalized ATen function: {name} -> {result}")
            return result
        elif '.' in name:
            result = name.split('.')[0]  # Handles cases like 'torch.nn.functional.relu'
            logger.debug(f"Normalized module function: {name} -> {result}")
            return result
        else:
            logger.debug(f"Function name unchanged: {name}")
            return name  # Fallback for other formats
            
    except Exception as e:
        logger.error(f"Failed to normalize function name from {target}: {e}")
        return "unknown"


def extract_layer_hyperparams(node: Any, extracted_weights: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract layer hyperparameters from an FX node.
    
    Args:
        node: FX node containing layer information
        extracted_weights: Dictionary of extracted weight tensors
        
    Returns:
        Dictionary of layer hyperparameters
        
    Raises:
        ValueError: If node is None or invalid
        TypeError: If extracted_weights is not a dictionary
    """
    if node is None:
        raise ValueError("Node cannot be None")
    
    if not isinstance(extracted_weights, dict):
        raise TypeError("extracted_weights must be a dictionary")
    
    layer_hyperparams = {}
    
    try:
        func_name = node.target.__name__ if callable(node.target) else str(node.target)
        func_name = str(func_name).split('.')[0]
        default_flag = False
        
        logger.debug(f"Extracting hyperparameters for function: {func_name}")

        if func_name in ATEN_HYPERPARAMS_TABLE:
            expected_params = ATEN_HYPERPARAMS_TABLE[func_name]
            args_node = iter(node.args)

            # 🔧 SPECIAL HANDLING: Fix arange parameter extraction
            if func_name == "arange":
                logger.debug(f"🔧 Special handling for arange parameter extraction")
                args_list = list(node.args)
                logger.debug(f"🔧 arange args: {args_list}")
                
                # Handle different arange signatures
                if len(args_list) == 1:
                    # torch.arange(5) -> start=0, end=5
                    end_value = args_list[0]
                    layer_hyperparams["start"] = 0
                    layer_hyperparams["end"] = end_value
                    layer_hyperparams["step"] = 1
                    logger.debug(f"🔧 arange: Single arg {end_value} -> start=0, end={end_value}, step=1")
                elif len(args_list) == 2:
                    # torch.arange(1, 4) -> start=1, end=4
                    start_value, end_value = args_list[0], args_list[1]
                    layer_hyperparams["start"] = start_value
                    layer_hyperparams["end"] = end_value
                    layer_hyperparams["step"] = 1
                    logger.debug(f"🔧 arange: Two args {start_value}, {end_value} -> start={start_value}, end={end_value}, step=1")
                elif len(args_list) == 3:
                    # torch.arange(1, 4, 2) -> start=1, end=4, step=2
                    start_value, end_value, step_value = args_list[0], args_list[1], args_list[2]
                    layer_hyperparams["start"] = start_value
                    layer_hyperparams["end"] = end_value
                    layer_hyperparams["step"] = step_value
                    logger.debug(f"🔧 arange: Three args {start_value}, {end_value}, {step_value} -> start={start_value}, end={end_value}, step={step_value}")
                else:
                    # Fallback to default extraction
                    logger.warning(f"🔧 arange: Unexpected number of args {len(args_list)}, using default extraction")
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
                                        logger.debug(f"Found {param} from node {x_itr.name}")
                                        break
                                elif isinstance(x_itr, (tuple, list)) and any(isinstance(item, torch.fx.node.Node) for item in x_itr):
                                    x_itr_list = []
                                    for item in x_itr:
                                        value = extracted_weights.get(item.name, None) if isinstance(item, torch.fx.node.Node) else item
                                        x_itr_list.append(value)
                                        flag_skip = True
                                    layer_hyperparams[param] = x_itr_list
                                    arg_flag = True
                                    logger.debug(f"Found {param} from list/tuple with {len(x_itr_list)} items")
                                    break
                                elif x_itr is None and param == "bias":
                                    value = None
                                    arg_flag = True
                                    layer_hyperparams[param] = value
                                    logger.debug(f"Found {param} as None (bias)")
                                    break
                                elif isinstance(x_itr, (bool, int, float, list, tuple, torch.memory_format)):
                                    value = x_itr
                                    logger.debug(f"Found {param} as literal value: {value}")
                                    break
                        except StopIteration:
                            value = ATEN_DEFAULTS_TABLE.get(param, None)
                            default_flag = True
                            logger.debug(f"Using default value for {param}: {value}")
                
                # Handle remaining parameters (dtype, layout, device, pin_memory)
                for param in ["dtype", "layout", "device", "pin_memory"]:
                    if param not in layer_hyperparams:
                        value = ATEN_DEFAULTS_TABLE.get(param, None)
                        layer_hyperparams[param] = value
                        logger.debug(f"Using default value for {param}: {value}")
            else:
                # Default parameter extraction for other operations
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
                                    logger.debug(f"Found {param} from node {x_itr.name}")
                                    break
                            elif isinstance(x_itr, (tuple, list)) and any(isinstance(item, torch.fx.node.Node) for item in x_itr):
                                x_itr_list = []
                                for item in x_itr:
                                    value = extracted_weights.get(item.name, None) if isinstance(item, torch.fx.node.Node) else item
                                    x_itr_list.append(value)
                                    flag_skip = True
                                layer_hyperparams[param] = x_itr_list
                                arg_flag = True
                                logger.debug(f"Found {param} from list/tuple with {len(x_itr_list)} items")
                                break
                            elif x_itr is None and param == "bias":
                                value = None
                                arg_flag = True
                                layer_hyperparams[param] = value
                                logger.debug(f"Found {param} as None (bias)")
                                break
                            elif isinstance(x_itr, (bool, int, float, list, tuple, torch.memory_format)):
                                value = x_itr
                                logger.debug(f"Found {param} as literal value: {value}")
                                break
                    except StopIteration:
                        value = ATEN_DEFAULTS_TABLE.get(param, None)
                        default_flag = True
                        logger.debug(f"Using default value for {param}: {value}")

                    if value is not None and not arg_flag:
                        # Handle dimension-specific parameters
                        if func_name.startswith(("conv", "pool")) and param in {"stride", "padding", "dilation", "output_padding"}:
                            dim = func_name[-2:]
                            num_dims = 1 if dim == "1d" else 2 if dim == "2d" else 3 if dim == "3d" else 1
                            value = tuple(value if isinstance(value, (list, tuple)) else [value] * num_dims)
                            logger.debug(f"Expanded {param} to {num_dims}D: {value}")

                        # Type-specific parameter handling
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

        # Ensure bias parameter is always present
        if "bias" not in layer_hyperparams:
            layer_hyperparams["bias"] = None
            logger.debug("Added default bias=None")

        logger.debug(f"Extracted {len(layer_hyperparams)} hyperparameters for {func_name}")
        return layer_hyperparams
        
    except Exception as e:
        logger.error(f"Failed to extract hyperparameters from node: {e}")
        # Return minimal hyperparameters to prevent crashes
        return {"bias": None}


def build_graph(tracer: Any, extracted_weights: Dict[str, Any], verbose: bool = False) -> Tuple[nx.DiGraph, List[str]]:
    """
    Build a NetworkX graph from the traced model with comprehensive metadata.
    
    Args:
        tracer: Traced model graph module
        extracted_weights: Dictionary of extracted weight tensors
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (graph, layer_stack) where:
        - graph: NetworkX DiGraph with node metadata
        - layer_stack: Topologically sorted list of node names
        
    Raises:
        ValueError: If tracer is None or invalid
        TypeError: If extracted_weights is not a dictionary
    """
    if tracer is None:
        raise ValueError("Tracer cannot be None")
    
    if not isinstance(extracted_weights, dict):
        raise TypeError("extracted_weights must be a dictionary")
    
    # Validate tracer has required attributes
    if not hasattr(tracer, 'graph') or not hasattr(tracer.graph, 'nodes'):
        raise ValueError("Tracer must have a graph attribute with nodes")
    
    logger.info(f"Building graph from tracer with {len(tracer.graph.nodes)} nodes")
    
    try:
        graph = nx.DiGraph()

        # Initialize all nodes with default metadata
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

        # Process each node and extract metadata
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

                    # Layer classification mapping
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

                    # Find layer category
                    for category, ops in ATEN_LAYER_MAP.items():
                        if func_name in ops:
                            layer_name = category
                            break
                    else:
                        layer_name = "Unknown_Operation"
                        if verbose:
                            logger.warning(f"Unknown ATen operation: {func_name} ← {node.target}")

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

            # Save metadata to graph node
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

            # Build parent-child relationships
            for input_node in node.all_input_nodes:
                if isinstance(input_node, torch.fx.Node):
                    parent = input_node.name
                    graph.nodes[name]['parents'].append(parent)
                    graph.nodes[parent]['children'].append(name)
                    graph.add_edge(parent, name)

            if verbose:
                logger.debug(f"🔹 {name} → {layer_type} [{layer_name}] | inputs: {graph.nodes[name]['parents']}")

        # Create topological sort for execution order
        layer_stack = list(nx.topological_sort(graph))
        
        logger.info(f"Graph built successfully: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        logger.debug(f"Layer stack length: {len(layer_stack)}")
        
        return graph, layer_stack
        
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        raise RuntimeError(f"Graph building failed: {e}") from e