# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/execution_engine.py
import os
import uuid
import shutil
import io
import torch
import numpy as np
import pickle
import joblib
import inspect
import zstandard as zstd
from tempfile import TemporaryFile
import lz4.frame  # only keep if used later

class DiskCacheManager:
    def __init__(self, cache_dir="node_io_cache", compression_level=2):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.zstd_level = compression_level

    def save_tensor(self, tensor):
        """
        Compresses and saves a tensor using joblib and zstandard (optional).
        """
        filename = f"{uuid.uuid4().hex}.pt.zstd"
        path = os.path.join(self.cache_dir, filename)

        tensor = tensor.cpu()  # Always move to CPU before saving

        # Use temporary file + compression
        with open(path, "wb") as f:
            cctx = zstd.ZstdCompressor(level=self.zstd_level)
            with cctx.stream_writer(f) as compressor:
                joblib.dump(tensor, compressor)

        return path

    def load_tensor(self, path):
        """
        Loads a tensor saved with joblib + zstd decompression.
        Uses BytesIO to allow seekability for joblib.load.
        """
        try:
            with open(path, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    data = reader.read()  # Can use reader.readall() if read hangs
                    return joblib.load(io.BytesIO(data))
        except Exception as e:
            print(f"[load_tensor] ❌ Failed to load {path}: {e}")
            raise


    def clear_cache(self):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def shutdown(self):
        self.clear_cache()

    def get_all_cached_files(self):
        return os.listdir(self.cache_dir)

    def get_cache_size_mb(self):
        return sum(
            os.path.getsize(os.path.join(self.cache_dir, f))
            for f in os.listdir(self.cache_dir)
        ) / 1e6


def _load_if_path(val, cache_manager):
    if isinstance(val, str) and val.endswith(".pt.zstd") and os.path.isfile(val):
        return cache_manager.load_tensor(val)
    return val

def _sanitize_input_tensor(t):
    return t.detach() if isinstance(t, torch.nn.Parameter) else t


def _process_output_tuple(output):
    if isinstance(output, tuple):
        return tuple(_process_output_tuple(o) for o in output)
    elif isinstance(output, list):
        return [_process_output_tuple(o) for o in output]
    elif isinstance(output, (torch.Tensor, int)):
        return output
    try:
        return torch.tensor(output)
    except Exception:
        return output


def _process_layer_input(layer_in):
    list_inp = False
    # ✅ Only unwrap if it's a singleton *list of tensors*
    while isinstance(layer_in, (list, tuple)) and len(layer_in) == 1:
        if isinstance(layer_in[0], (torch.Tensor, int, float, np.ndarray)):
            layer_in = layer_in[0]
        else:
            break
    return layer_in, isinstance(layer_in, list)


def flatten_paths(val):
    if isinstance(val, list):
        flat = []
        for v in val:
            flat.extend(flatten_paths(v))
        return flat
    elif isinstance(val, str):
        return [val]
    else:
        return [val]


def execute_aten_operation(func_name, aten_op, layer_in, layer_hyperparams, method_args, parents, node_io, node_name):
    try:
        if func_name == "linear":
            return aten_op(layer_in, layer_hyperparams["weight"], layer_hyperparams["bias"])

        elif func_name == "conv2d":
            return aten_op(layer_in, layer_hyperparams["weight"], layer_hyperparams["bias"],
                           layer_hyperparams["stride"], layer_hyperparams["padding"],
                           layer_hyperparams["dilation"], layer_hyperparams["groups"])

        elif func_name == "max_pool2d":
            return aten_op(layer_in,
                           layer_hyperparams["kernel_size"],
                           layer_hyperparams["stride"],
                           layer_hyperparams["padding"],
                           layer_hyperparams["dilation"],
                           layer_hyperparams["ceil_mode"])

        elif func_name == "adaptive_avg_pool2d":
            return aten_op(layer_in, layer_hyperparams["output_size"])

        elif func_name == "dropout":
            return aten_op(layer_in, layer_hyperparams["p"], layer_hyperparams["train"])

        elif func_name in ("relu", "relu_", "gelu", "tanh"):
            return aten_op(layer_in)

        elif "unsqueeze" in func_name :
            return aten_op(layer_in, layer_hyperparams["dim"])
            
        elif "squeeze" in func_name :
            return aten_op(layer_in, layer_hyperparams["dim"])

        elif func_name == "layer_norm":
            return aten_op(layer_in,
                           layer_hyperparams["normalized_shape"],
                           layer_hyperparams["weight"],
                           layer_hyperparams["bias"],
                           layer_hyperparams["eps"],
                           False)

        elif func_name == "batch_norm":
            return aten_op(layer_in,
                           layer_hyperparams["weight"],
                           layer_hyperparams["bias"],
                           layer_hyperparams["running_mean"],
                           layer_hyperparams["running_var"],
                           False,
                           layer_hyperparams["momentum"],
                           layer_hyperparams["eps"],
                           False)

        elif func_name == "scaled_dot_product_attention":
            return aten_op(layer_in[0], layer_in[1], layer_in[2],
                           layer_hyperparams["attn_mask"],
                           layer_hyperparams["dropout_p"],
                           layer_hyperparams["is_causal"])

        elif func_name == "lstm":
            return aten_op(layer_in[0],
                           (layer_in[1], layer_in[2]),
                           layer_hyperparams["params"],
                           layer_hyperparams["has_biases"],
                           layer_hyperparams["num_layers"],
                           layer_hyperparams["dropout"],
                           False,
                           layer_hyperparams["bidirectional"],
                           layer_hyperparams["batch_first"])

        elif func_name == "reshape":
            if "shape" in layer_hyperparams:
                if isinstance(layer_hyperparams["shape"], (tuple, list)) and isinstance(layer_in,list):
                    if len(layer_in)>0:
                        for item in range(0,len(layer_hyperparams['shape'])):
                            if isinstance(layer_hyperparams['shape'][item], torch.fx.node.Node):
                                idx = parents.index(str(layer_hyperparams['shape'][item]))
                                value = layer_in[idx]
                                layer_hyperparams['shape'][item] = value
                    layer_in = layer_in[0]
                output = aten_op(layer_in,layer_hyperparams['shape'])
            return output

        elif func_name in ("masked_fill","masked_fill_"):
            mask = None
            if isinstance(layer_in, list) and len(layer_in) > 1:
                mask = layer_in[1]  # Second argument should be the mask tensor
                layer_in = layer_in[0]  # First argument is the input tensor
            value = layer_hyperparams["value"]
            if isinstance(value, torch.Tensor):
                value = value.item()  # Convert single-element tensor to Python scalar
            if value == float('-inf'):
                value = -torch.finfo(layer_in.dtype).max  # Use max negative finite value
            output = aten_op(layer_in, mask, value)
            return output

        elif func_name == "view":
            if "shape" in layer_hyperparams:
                if isinstance(layer_hyperparams["shape"], (tuple, list)) and isinstance(layer_in,list):
                    if len(layer_in)>0:
                        for item in range(0,len(layer_hyperparams['shape'])):
                            if isinstance(layer_hyperparams['shape'][item], torch.fx.node.Node):
                                idx = parents.index(str(layer_hyperparams['shape'][item]))
                                value = layer_in[idx]
                                layer_hyperparams['shape'][item] = value
                    layer_in = layer_in[0]
            output = aten_op(layer_in,layer_hyperparams['shape'])
            return output

        elif func_name == "select":
            return aten_op(layer_in,
                           layer_hyperparams["dim"],
                           layer_hyperparams["index"])

        elif func_name == "arange":
            start = layer_hyperparams.get("start", None)
            end = layer_hyperparams.get("end", None)
            step = layer_hyperparams.get("step", 1)
            dtype = layer_hyperparams.get("dtype", None)
            device = layer_hyperparams.get("device", None)

            # Function to resolve parameters to scalars
            def resolve_param(param):
                if isinstance(param, torch.Tensor):
                    return param.item()
                elif isinstance(param, torch.fx.Node):
                    return node_io[str(param)]['output_values'].item()
                elif isinstance(param, torch.SymInt):
                    return int(param.node._value) if hasattr(param.node, "_value") else 1
                return param

            # Resolve parameters
            start = resolve_param(start)
            end = resolve_param(end)
            step = resolve_param(step)

            def get_overload_name(op_overload_obj):
                try:
                    return str(op_overload_obj).split(".")[-1]
                except Exception:
                    return None

            arange_overload = get_overload_name(layer)

            try:
                if arange_overload == "start":
                    # aten::arange.start(start, end, *, dtype, device)
                    output = aten_op(start, end-1, dtype=dtype, device=device)
                elif arange_overload == "default":
                    # aten::arange.default(end, *, dtype, device)
                    output = aten_op(end-1, dtype=dtype, device=device)
                else:
                    # Other overloads like aten::arange.Scalar support step
                    output = aten_op(start, end, step, dtype=dtype, device=device)
            except Exception as e:
                raise RuntimeError(f"Failed to call aten::arange with resolved args. "
                                f"start={start}, end={end}, step={step}, dtype={dtype}, device={device}. "
                                f"Overload: {arange_overload}. Error: {str(e)}")
            return output

        elif func_name == "slice":
            if isinstance(layer_in, list):
                if isinstance(layer_in[0],torch.Tensor) and isinstance(layer_in[1],int):
                    layer_hyperparams['end'] = layer_in[1]
                    layer_in = layer_in[0]
            return aten_op(layer_in,
                           layer_hyperparams["dim"],
                           layer_hyperparams["start"],
                           layer_hyperparams["end"],
                           layer_hyperparams.get("step", 1))
            
        elif func_name == "sym_size":
            if isinstance(layer_in,(tuple,list)):
                layer_in = layer_in[0]
            output = aten_op(layer_in,layer_hyperparams['dim'])
            return output
        elif func_name == "to":
            output = aten_op(layer_in,*method_args)
            return output
        elif func_name in ("mean","softmax"):
            output = aten_op(layer_in,*method_args)
            return output
        elif func_name in ("unflatten","flatten"):
            output = aten_op(layer_in,*method_args)
            return output
        elif func_name == "embedding":
            indices = layer_in[0] if isinstance(layer_in, (list, tuple)) else layer_in
            if not torch.is_floating_point(indices) and indices.dtype in (torch.int32, torch.int64):
                pass
            else:
                indices = indices.long()
            return aten_op(layer_hyperparams["weight"],
                           indices,
                           layer_hyperparams["padding_idx"],
                           layer_hyperparams["scale_grad_by_freq"],
                           layer_hyperparams["sparse"])
        elif func_name in ("mul", "mul_"):            
            valid_inputs = [x for x in layer_in if isinstance(x, torch.Tensor)]
            
            # ⚠️ Safety check: restrict to exactly 2 inputs
            if len(valid_inputs) != 2:
                raise RuntimeError(f"{node_name} [DLBacktraceFX] `{func_name}` expects 2 tensors, got {len(valid_inputs)}: {valid_inputs}")
    
            a, b = valid_inputs
    
            # --- Safe broadcasting if shapes differ ---
            if a.dtype != b.dtype:
                common_dtype = torch.promote_types(a.dtype, b.dtype)
                a = a.to(dtype=common_dtype)
                b = b.to(dtype=common_dtype)
    
            try:
                output = aten_op(a, b, *method_args)
            except Exception as e:
                raise RuntimeError(f"{node_name} failed in {func_name} with shapes {a.shape}, {b.shape}: {e}")
            return output
        elif func_name in {"mm", "bmm"}:
            if isinstance(layer_in, list) and len(layer_in) == 2:
                output = aten_op(layer_in[0], layer_in[1])
            else:
                raise RuntimeError(f"{func_name} expects 2 tensor inputs. Got: {layer_in}")
            return output
        elif func_name in {"add", "add_", "mul", "sub", "div", "rsub", "pow", "gt", "ge", "lt", "eq"}:
            if isinstance(layer_in,list):
                if len(layer_in) == 2:
                    output = aten_op(layer_in[0],layer_in[1],*method_args)
                elif len(layer_in) == 3:
                    output = aten_op(layer_in[0],layer_in[1],layer_in[2],*method_args)
                else:
                    output = aten_op(layer_in,*method_args)
            else:
                output = aten_op(layer_in,*method_args)
            return output

        elif "scalar_tensor" in func_name:
            value = layer_hyperparams.get("value", 0)
            dtype = layer_hyperparams.get("dtype", torch.float32)
            device = layer_hyperparams.get("device", torch.device("cpu"))
            layout = layer_hyperparams.get("layout", torch.strided)
            pin_memory = layer_hyperparams.get("pin_memory", False)

            # Ensure value is a scalar
            if isinstance(value, (list, tuple)):
                value = value[0] if value else 0
            elif isinstance(value, torch.Tensor):
                value = value.item()

            # Create the scalar tensor
            output = aten_op(
                value,
                dtype=dtype,
                layout=layout,
                device=device,
                pin_memory=pin_memory
            )
            return output

            
        elif func_name == "matmul":
            if isinstance(layer_in, list) and len(layer_in) == 2:
                a, b = layer_in[0], layer_in[1]
                if a.dtype != b.dtype:
                    target_dtype = torch.promote_types(a.dtype, b.dtype)
                    a = a.to(dtype=target_dtype)
                    b = b.to(dtype=target_dtype)

                output = aten_op(a, b, *method_args)
            else:
                raise RuntimeError(f"[DLBacktraceFX] matmul expects 2 inputs but got: {layer_in}")
            return output

        elif func_name == "full":
            size = layer_hyperparams.get("size", None) or layer_hyperparams.get("sizes", None)
            fill_value = layer_hyperparams.get("fill_value", 0)
            dtype = layer_hyperparams.get("dtype", torch.float32)
            device = layer_hyperparams.get("device", torch.device("cpu"))

            # 🔧 Evaluate shape
            evaluated_size = []
            for s in size:
                if isinstance(s, torch.SymInt):
                    s = int(s.node._value) if hasattr(s.node, "_value") else 1
                elif isinstance(s, torch.fx.Node):
                    s_output = node_io[str(s)]['output_values']
                    if isinstance(s_output, torch.Tensor):
                        s = int(s_output.item())
                    else:
                        s = int(s_output)
                elif isinstance(s, torch.Tensor):
                    s = int(s.item())
                evaluated_size.append(int(s))

            # 🎯 Ensure fill_value is scalar
            if isinstance(fill_value, torch.Tensor):
                fill_value = fill_value.item()

            # ⚠️ Use list instead of tuple
            output = aten_op(evaluated_size, fill_value, dtype=dtype, device=device)
            return output

        
        elif func_name == "full_like":
            return aten_op(layer_in,
                           fill_value=layer_hyperparams["fill_value"],
                           dtype=layer_hyperparams["dtype"],
                           layout=layer_hyperparams["layout"],
                           device=layer_hyperparams["device"],
                           pin_memory=layer_hyperparams["pin_memory"])

        elif func_name == "expand":
            if isinstance(layer_in,list):
                if len(layer_in)>1:
                    layer_in = layer_in[0]
                if isinstance(layer_hyperparams['sizes'], (tuple, list)):
                    for item in range(len(layer_hyperparams['sizes'])):
                        if isinstance(layer_hyperparams['sizes'][item], torch.fx.node.Node):
                            layer_hyperparams['sizes'][item] = node_io[str(layer_hyperparams['sizes'][item])]['output_values']
                output = aten_op(layer_in,layer_hyperparams['sizes'],implicit=layer_hyperparams['implicit'])
            
            return output

        elif "permute" in func_name:
            if isinstance(layer_in,tuple):
                layer_in = layer_in[0]
            elif isinstance(layer_in, list):
                layer_in = layer_in[0]
            return aten_op(layer_in, layer_hyperparams["dims"])

        elif func_name == "transpose":
            return aten_op(layer_in, *method_args)
            
        elif func_name == "sigmoid":
            return aten_op(layer_in, *method_args)
        
        elif func_name == "zeros":
            return aten_op(layer_hyperparams["sizes"])

        elif func_name == "_softmax":
            return aten_op(layer_in,
                           layer_hyperparams["dim"],
                           layer_hyperparams["half_to_float"])

        elif func_name == "mean":
            return aten_op(layer_in, *method_args)

        elif func_name == "where":
            cond, x, y = layer_in
            return aten_op(cond, x, y)

        elif func_name == "contiguous":
            return aten_op(layer_in)

        elif func_name == "_to_copy":
            return aten_op(layer_in,
                           memory_format=layer_hyperparams.get("memory_format", torch.contiguous_format),
                           non_blocking=layer_hyperparams.get("non_blocking", False))

        elif func_name == "logical_not":
            return aten_op(layer_in)

        elif func_name == "any":
            return aten_op(layer_in,
                           layer_hyperparams["dim"],
                           layer_hyperparams["keepdim"])

        elif func_name in {"flatten", "unflatten"}:
            return aten_op(layer_in, *method_args)

        elif func_name == "rsqrt":
            if isinstance(layer_in, list) and len(layer_in) == 1:
                output = aten_op(layer_in[0], *method_args)
            elif isinstance(layer_in, torch.Tensor):
                output = aten_op(layer_in, *method_args)
            else:
                raise RuntimeError(f"[DLBacktraceFX] rsqrt expects 1 input, got {type(layer_in)}: {layer_in}")
            return output

        elif func_name == "triu":
            if isinstance(layer_in, list):
                if len(layer_in) != 1:
                    raise ValueError(f"Expected 1 tensor for `triu`, but got: {len(layer_in)}")
                layer_in = layer_in[0]
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"`triu` expects a tensor input but got: {type(layer_in)}")
            return aten_op(layer_in, layer_hyperparams.get("diagonal", 0))

        else:
            print("=================",func_name)
            if layer_hyperparams:
                print(layer_hyperparams)
            print(method_args)
            print(aten_op)
            output = aten_op(layer_in,*method_args)
            print("at-ma",layer_in.shape,aten_op, method_args,func_name)
            
    except Exception as e:
        print(f"[Execution Error] Node `{node_name}` failed in `{func_name}`: {e}")
        #return layer_in


def run_execution(graph, layer_stack, model, extracted_weights, inputs, tracer, cache_manager):
    tensor_map = {}
    node_io = {}


    model_signature = inspect.signature(model.forward)
    expected_input_names = list(model_signature.parameters.keys())

    if len(inputs) != len(expected_input_names):
        raise ValueError("Mismatch between model input count and provided inputs.")

    inp_map = dict(zip(expected_input_names, inputs))
    if len(extracted_weights) < len(layer_stack):
        tensor_map[layer_stack[len(extracted_weights)]] = inputs

    for node_name in layer_stack:
        node_data = graph.nodes[node_name]
        parents = node_data["parents"]
        layer_type = node_data["layer_type"]
        node_type = node_data["node_type"]
        method_args = node_data["method_args"]
        layer = node_data["layer"]
        layer_name = node_data["layer_name"]
        func_name = node_data["func_name"]
        func_module = node_data["func_module"]
        layer_hyperparams = node_data["layer_hyperparams"]
        children = node_data["children"]

        if any(p not in tensor_map for p in parents):
            continue

        layer_in = [_sanitize_input_tensor(tensor_map[p]) for p in parents]
        output = layer_in

        try:
            if layer_type == "Placeholder":
                output = extracted_weights.get(node_name, inp_map.get(node_name, layer_in))
                special_weight_keywords = [
                    '_layernorm_weight',
                    '_proj_weight',
                    '_head_weight'
                ]
                
                if any(key in node_name for key in special_weight_keywords):
                    layer_type = "future_use"
                elif "weight" in node_name:
                    layer_type = "Weight"
                elif "bias" in node_name:
                    layer_type = "Bias"
                elif "running_mean" in node_name:
                    layer_type = "bn_running_mean"
                elif "running_var" in node_name:
                    layer_type = "bn_running_var"
                elif "num_batches_tracked" in node_name:
                    layer_type = "bn_num_batches_tracked"
                elif "embeddings_token_type_ids" in node_name:
                    layer_type = "embeddings_token_type_ids"
                elif "embeddings_position_ids" in node_name:
                    layer_type = "embeddings_position_ids"


            elif layer_type == "Model_Layer":
                resolved_layer = model
                for attr in layer.split("."):
                    resolved_layer = getattr(resolved_layer, attr)
                output = resolved_layer(*layer_in)

            elif layer_type == "Output":
                output = layer_in[0]

            elif layer_type == "CustomOp":
                output = layer_in

            elif layer_type in ("ATen_Operation", "Operation"):
                layer_in = []
                for node_x in parents:
                    if node_io[node_x]['layer_type'] in (
                        "Weight", "Bias", "bn_running_mean", "bn_running_var",
                        "bn_num_batches_tracked", "embeddings"
                    ):
                        continue

                    val = node_io[node_x]['output_values']

                    try:
                        if isinstance(val, (list, tuple)):
                            loaded_vals = []
                            for v in val:
                                loaded = _load_if_path(v, cache_manager)
                                loaded_vals.append(loaded)
                            layer_in.extend(loaded_vals)
                        else:
                            loaded = _load_if_path(val, cache_manager)
                            layer_in.append(loaded)
                    except Exception as ex:
                        print(f"[{node_name}] ❌ Error loading `{val}` from parent `{node_x}`: {ex}")
                        raise


                aten_op = layer
                layer_in, _ = _process_layer_input(layer_in)
                output = execute_aten_operation(func_name, aten_op, layer_in, layer_hyperparams, method_args, parents, node_io, node_name)

        except Exception as e:
            print(f"[Execution Error] Node `{node_name}` failed in `{func_name}`: {e}")
            output = layer_in

        processed_output = _process_output_tuple(output)
        try:
            # Save output to disk and get the filepath
            if isinstance(processed_output, torch.Tensor):
                output_path = cache_manager.save_tensor(processed_output)
            else:
                output_path = processed_output  # Don't cache scalars or ints
            input_paths = []
            for t in layer_in:
                if isinstance(t, torch.Tensor):
                    path = cache_manager.save_tensor(t)
                    input_paths.append(path)
                else:
                    input_paths.append(t)  # Leave as-is for int, tuple, etc.


            input_paths = flatten_paths(input_paths)
            output_path = flatten_paths(output_path)
            node_io[node_name] = {
                "input_sources": parents,
        
                # Optionally offload this too if inputs are large (currently in-memory)
                "input_values": input_paths,
        
                "output_values": output_path,  # 💾 Disk-stored output
                "layer_type": layer_type,
                "node_type": node_type,
                "method_args": method_args,
                "layer": layer,
                "layer_name": layer_name,
                "func_name": func_name,
                "func_module": func_module,
                "output_children": children,
                "layer_hyperparams": layer_hyperparams,
            }
        
            # Optional cleanup
            del processed_output
            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"[Node IO Error] Failed to serialize `{node_name}`: {e}")
            node_io[node_name] = {
                "input_sources": parents,
                "input_values": str(layer_in),
                "output_values": str(processed_output),
                "layer_type": layer_type,
                "node_type": node_type,
                "method_args": str(method_args),
                "layer": str(layer),
                "layer_name": str(layer_name),
                "func_name": str(func_name),
                "func_module": str(func_module),
                "output_children": str(children),
                "layer_hyperparams": str(layer_hyperparams),
            }
        
        tensor_map[node_name] = output_path  # store path instead of tensor


    return node_io


class ExecutionEngine:
    def __init__(self, model, extracted_weights, fx_graph, layer_stack, tracer, exported_program,cache_manager):
        self.model = model
        self.extracted_weights = extracted_weights
        self.graph = fx_graph
        self.layer_stack = layer_stack
        self.tracer = tracer
        self.exported_program = exported_program
        self.cache_manager = cache_manager  # ✅ MISSING assignment


    def run(self, inputs):
        return run_execution(
            graph=self.graph,
            layer_stack=self.layer_stack,
            model=self.model,
            extracted_weights=self.extracted_weights,
            inputs=inputs,
            tracer=self.tracer,
            cache_manager=self.cache_manager
        )
