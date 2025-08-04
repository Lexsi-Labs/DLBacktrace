# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/execution_engine_noncache.py

import inspect
import torch
import numpy as np

def _load_if_path(val, cache_manager):
    if isinstance(val, str) and val.endswith(".pt.zstd") and os.path.isfile(val):
        return cache_manager.load_tensor(val)
    return val

def sanitize(x):
        return x.to(torch.float32) if isinstance(x, torch.Tensor) and x.dtype == torch.bool else x 

def _sanitize_input_tensor(t):
    return t.detach() if isinstance(t, torch.nn.Parameter) else t


def _process_output_tuple(output):
    if isinstance(output, (torch.Tensor, int)):
        return output  # ✅ Return directly, not wrapped in a list

    result = []

    def flatten(x):
        if isinstance(x, torch.Tensor):
            result.append(x)
        elif isinstance(x, (list, tuple)):
            for item in x:
                flatten(item)
        elif x is not None:
            try:
                result.append(torch.tensor(x))
            except Exception:
                pass

    flatten(output)

    if len(result) == 1:
        return result[0]  # ✅ return just the tensor
    elif len(result) > 1:
        return tuple(result)
    else:
        return None


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


def execute_aten_operation(func_name, aten_op, layer_in, layer_hyperparams, method_args, parents, node_io, node_name,tensor_map, children=None):
    try:
        # --- Special-case native batch_norm to preserve full tuple ---
        if func_name in ("_native_batch_norm_legit_no_training", "batch_norm"):
            # Determine behavior based on the type of layer_in
            if isinstance(layer_in, (list, tuple)) and len(layer_in) >= 5:
                # Inputs are passed as a tuple/list → strict mode
                inp, weight, bias, running_mean, running_var = layer_in[:5]
                momentum, eps = method_args
                output = aten_op(inp, weight, bias, running_mean, running_var, momentum, eps)
                return output
            else:
                # Inputs passed individually or as a non-strict struct → fallback to legacy
                output = aten_op(
                    layer_in,
                    layer_hyperparams.get("weight"),
                    layer_hyperparams.get("bias"),
                    layer_hyperparams.get("running_mean"),
                    layer_hyperparams.get("running_var"),
                    layer_hyperparams.get("momentum"),
                    layer_hyperparams.get("eps"),
                )
                return output

        # --- Handle FX getitem nodes: tuple-indexing instead of tensor slicing ---
        if func_name == "getitem":
            parent = parents[0]
            tup = node_io[parent]["output_values"]
            idx = method_args[0]

            # --- Error checking ---
            if tup is None:
                raise RuntimeError(f"[ERROR] Cannot apply getitem on None from parent node '{parent}'")

            if not isinstance(tup, (tuple, list)):
                raise TypeError(f"[ERROR] Invalid type for getitem: {type(tup)} from node '{parent}'")

            return tup[idx]

        if func_name == "linear":
            if isinstance(layer_in, list):
                if len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    ip = []
                    for p in parents:
                        if node_io[p]['layer_type'] == "future_use":
                            continue
                        elif node_io[p]['layer_type'] == "Weight":
                            continue
                        else:
                            ip.append(tensor_map[p])
                        if len(ip)==1:
                            layer_in = ip[0]
                print("nodename layerin",node_name,len(layer_in),parents)

            if isinstance(layer_hyperparams["bias"],bool):
                output = aten_op(layer_in, layer_hyperparams["weight"])
                return output
            else:
                output = aten_op(layer_in, layer_hyperparams["weight"], layer_hyperparams["bias"])
                return output

        elif func_name == "conv2d": 
            # unwrap single-element list inputs
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 

            bias = layer_hyperparams["bias"]
            if isinstance(bias, bool):
                bias = None  # ✅ Fix: convert True/False to None

            return aten_op(layer_in, layer_hyperparams["weight"], bias,
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

        elif func_name in ("relu", "relu_", "gelu", "tanh","silu"):
            return aten_op(layer_in)

        elif "unsqueeze" in func_name :           
            return aten_op(layer_in, layer_hyperparams["dim"])
            
        elif "squeeze" in func_name :
            return aten_op(layer_in, layer_hyperparams["dim"])

        elif func_name == "layer_norm":
            if isinstance(layer_in, list):
                # Extract the first tensor, assuming it's the actual input
                tensor_candidates = [x for x in layer_in if isinstance(x, torch.Tensor)]
                if not tensor_candidates:
                    raise RuntimeError(f"[{node_name}] ❌ No tensor found in `layer_in` list: {layer_in}")
                layer_in = tensor_candidates[0]
                pass

            if not isinstance(layer_in, torch.Tensor):
                raise RuntimeError(f"[{node_name}] ❌ `layer_norm` expected Tensor input, got {type(layer_in)}")
        
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

        elif func_name in ("masked_fill", "masked_fill_"):
            #print(f"\n[{node_name}] ➕ Executing `{func_name}`")
            #print(f"[{node_name}] Parents: {parents}")
            #print(f"[{node_name}] method_args: {method_args}")
            #print(f"[{node_name}] layer_hyperparams: {layer_hyperparams}")

            if isinstance(layer_in, list) and len(layer_in) > 1:
                mask = layer_in[1]
                layer_in = layer_in[0]

                # ✅ Ensure mask is boolean
                if mask.dtype != torch.bool:
                    mask = mask != 0
            else:
                raise RuntimeError(f"[{node_name}] ❌ Expected list with input and mask for `{func_name}`, got: {layer_in}")

            value = layer_hyperparams.get("value", 0.0)
            if isinstance(value, torch.Tensor):
                value = value.item()  # Convert single-element tensor to Python scalar

            if value == float('-inf'):
                value = -torch.finfo(layer_in.dtype).max  # Use max negative finite value

            #print(f"[{node_name}] input shape: {getattr(layer_in, 'shape', None)}, dtype: {getattr(layer_in, 'dtype', None)}")
            #print(f"[{node_name}] mask shape: {getattr(mask, 'shape', None)}, dtype: {getattr(mask, 'dtype', None)}")
            #print(f"[{node_name}] fill value: {value}")

            try:
                output = aten_op(layer_in, mask, value)
                #print(f"[{node_name}] ✅ `masked_fill` success → output shape: {output.shape}")
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `masked_fill` failed with shapes: input={layer_in.shape}, mask={mask.shape}, value={value}. Error: {e}")

            return output

        elif func_name == "view":
            shape = layer_hyperparams.get("shape", [])
            resolved_shape = []

            # 🛑 Fallback: use method_args[0] if shape is None or contains None
            if not shape or any(s is None for s in shape):
                if method_args and isinstance(method_args[0], (list, tuple)):
                    shape = method_args[0]
                    print(f"[view] ⛑️ Fallback to shape from method_args: {shape}")

            def resolve_param_view(p):
                try:
                    # Use output_values from node_io if symbolic
                    if isinstance(p, (torch.fx.Node, str)):
                        p_key = str(p)
                        if p_key in node_io:
                            val = node_io[p_key]["output_values"]
                            return val.item() if isinstance(val, torch.Tensor) else int(val)
                        else:
                            raise KeyError(f"[view] ❌ `{p_key}` not found in `node_io`")
                    elif isinstance(p, torch.Tensor):
                        return int(p.item())
                    elif isinstance(p, (int, torch.SymInt)):
                        return int(p)
                    elif p is None:
                        raise ValueError("[view] ❌ Cannot resolve None in shape")
                    return int(p)
                except Exception as e:
                    raise ValueError(f"[view] ❌ Failed to resolve dim: {p} ({type(p)}) → {e}")

            for idx, dim in enumerate(shape):
                try:
                    resolved = resolve_param_view(dim)
                    resolved_shape.append(resolved)
                except Exception as e:
                    print(f"[view] ❌ Failed to resolve dim at index {idx}: {e}")
                    raise RuntimeError(f"[view] ❌ Cannot resolve shape for `view`: {shape}")

            if isinstance(layer_in, list) and len(layer_in) > 0:
                layer_in = layer_in[0]

            #print(f"[view] ✅ Resolved shape: {resolved_shape}")
            try:
                output = aten_op(layer_in, resolved_shape)
            except Exception as e:
                raise RuntimeError(f"[view] ❌ Execution failed: input shape={getattr(layer_in, 'shape', None)}, resolved_shape={resolved_shape}, error={e}")
            
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
            if len(parents) == 1 and end == 10:
                end = tensor_map[parents[0]]
            

            # Function to resolve parameters to scalars
            def resolve_param(param):
                if isinstance(param, torch.Tensor):
                    return param.item()
                elif isinstance(param, torch.fx.Node):
                    return tensor_map[str(param)].item()
                    #return node_io[str(param)]['output_values'].item()
                elif isinstance(param, torch.SymInt):
                    return tensor_map[str(param)].item()
                    #return int(param.node._value) if hasattr(param.node, "_value") else 1
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

            arange_overload = get_overload_name(aten_op)

            try:
                if arange_overload in ("start_step", "Scalar", "Scalar_"):
                    # aten::arange.start_step(start, end, step, *, dtype, device)
                    output = aten_op(start, end, step, dtype=dtype, device=device)
                elif "start" in str(arange_overload) :
                    # aten::arange.start(start, end, *, dtype, device)
                    output = aten_op(start, end, dtype=dtype, device=device)
                elif "default" in str(arange_overload) :
                    # aten::arange.default(end, *, dtype, device)
                    output = aten_op(end, dtype=dtype, device=device)
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
                if isinstance(layer_in[0], torch.Tensor) and isinstance(layer_in[1], int):
                    # code here
                    layer_hyperparams['end'] = layer_in[1]
                    layer_in = layer_in[0]
                    print(f"[{node_name}] layer_in shape: {layer_in.shape}")

                if not isinstance(layer_in, torch.Tensor):
                    raise RuntimeError(f"[{node_name}] ❌ `slice` expected Tensor input, got {type(layer_in)}: {layer_in}")
            
             # ✅ Print unconditionally here if it's a tensor
            elif isinstance(layer_in, torch.Tensor):
                print(f"[{node_name}] layer_in shape: {layer_in.shape}")

            output = aten_op(
                layer_in,
                layer_hyperparams["dim"],
                layer_hyperparams["start"],
                layer_hyperparams["end"],
                layer_hyperparams.get("step", 1)
            )
            print(f"[{node_name}] output shape: {output.shape}") 
            return output
        
        elif func_name == "sym_size":
            if isinstance(layer_in,(tuple,list)):
                layer_in = layer_in[0]
            output = aten_op(layer_in,layer_hyperparams['dim'])
            return output

        elif func_name == "_assert_tensor_metadata":
            # Skip or pass-through this op, as it's only a debug consistency check
            print(f"[{node_name}] ℹ️ Skipping `_assert_tensor_metadata` (no-op).")
            if isinstance(layer_in, list) and len(layer_in) > 0:
                return layer_in[0]
            return layer_in

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
            layer_in = []
            for node_x in parents: 
                if "weight" in node_x:
                    continue
                else:
                    layer_in.append(node_io[node_x]['output_values'])

            indices = layer_in[0] if isinstance(layer_in, (list, tuple)) else layer_in
            # 🔄 Ensure indices are LongTensor
            if not torch.is_floating_point(indices) and indices.dtype in (torch.int32, torch.int64):
                pass  # Already correct
            else:
                indices = indices.long()
        
            return aten_op(layer_hyperparams["weight"],
                    indices,
                    layer_hyperparams["padding_idx"],
                    layer_hyperparams["scale_grad_by_freq"],
                    layer_hyperparams["sparse"]) 

        elif func_name in ("mul", "mul_"):

            # Extract operands a and b robustly
            a = b = None
            if isinstance(layer_in, list):
                if len(layer_in) == 2:
                    a, b = layer_in
                elif len(layer_in) == 1 and len(method_args) == 1:
                    a = layer_in[0]
                    b = method_args[0]
                else:
                    raise RuntimeError(
                        f"[{node_name}] [DLBacktraceFX] `{func_name}` expects 2 inputs, got list of length {len(layer_in)} and method_args of length {len(method_args)}"
                    )
            elif isinstance(layer_in, (int, float, torch.Tensor)) and len(method_args) == 1:
                a = layer_in
                b = method_args[0]
            else:
                raise RuntimeError(
                    f"[{node_name}] [DLBacktraceFX] `{func_name}` expects 2 inputs, got: {type(layer_in)} + {method_args}"
                )

            # Resolve SymInt or symbolic Node values
            if hasattr(a, 'node') and hasattr(a.node, '_value'):
                a = a.node._value
            if hasattr(b, 'node') and hasattr(b.node, '_value'):
                b = b.node._value

            # Utility: unwrap 0-dim tensors carefully
            def unwrap_scalar(x):
                if isinstance(x, torch.nn.Parameter):
                    x = x.detach()
                if isinstance(x, torch.Tensor) and x.ndim == 0:
                    val = x.item()
                    if isinstance(val, (int, float)) and abs(val) > 1e10:
                        return x.to(dtype=torch.float32)  # keep as tensor
                    return val
                if isinstance(x, (int, float)):
                    if abs(x) > 1e10:
                        return torch.tensor(x, dtype=torch.float32)
                    return x
                return x

            a = unwrap_scalar(a)
            b = unwrap_scalar(b)

            # Final promotion to safe tensor if needed
            def to_safe_tensor(val, ref):
                if isinstance(val, (int, float)):
                    if abs(val) > 1e10:
                        return torch.tensor(val, dtype=torch.float32, device=ref.device if isinstance(ref, torch.Tensor) else 'cpu')
                    return torch.tensor(val, dtype=ref.dtype if isinstance(ref, torch.Tensor) else torch.float32,
                                        device=ref.device if isinstance(ref, torch.Tensor) else 'cpu')
                return val

            a = to_safe_tensor(a, b)
            b = to_safe_tensor(b, a)

            # Execute op
            try:
                output = aten_op(a, b)
            except Exception as e:
                raise RuntimeError(
                    f"[{node_name}] ❌ failed in `{func_name}` with a={type(a)}, b={type(b)}; "
                    f"shapes: {getattr(a, 'shape', None)}, {getattr(b, 'shape', None)}. Error: {e}"
                )

            print(f"node_name: {node_name}, mul shape: {output.shape}")
            return output

        elif func_name in {"add", "add_", "sub", "div", "rsub", "pow", "gt", "ge", "lt", "eq"}:
            output = None
            layer_in = layer_in if isinstance(layer_in, list) else [layer_in]
            print(f"[{node_name}] 🔍 `{func_name}` with {[x.shape if isinstance(x, torch.Tensor) else type(x) for x in layer_in]}")

            # Ensure exactly 2 inputs
            if len(layer_in) != 2:
                recovered_inputs = [tensor_map[p] for p in parents if p in tensor_map]
                for inp in recovered_inputs:
                    if len(layer_in) < 2:
                        layer_in.append(inp)
                if len(layer_in) < 2 and func_name in {"rsub", "pow"} and method_args:
                    layer_in.append(method_args[0])
                if len(layer_in) != 2:
                    raise RuntimeError(f"[{node_name}] ❌ `{func_name}` requires exactly 2 inputs, got {len(layer_in)}")

            # ✅ Safe unpacking before branching
            arg0, arg1 = layer_in[0], layer_in[1] if len(layer_in) > 1 else None
            arg0 = sanitize(arg0)
            arg1 = sanitize(arg1)

            try:
                # Handle special ops manually
                if func_name == "add_":
                    alpha = method_args[0] if method_args else 1
                    output = arg0.add_(arg1 * alpha if alpha != 1 else arg1)

                elif func_name == "add":
                    alpha = method_args[0] if method_args else 1
                    output = torch.add(arg0, arg1, alpha=alpha)

                elif func_name == "sub":
                    alpha = method_args[0] if method_args else 1
                    output = torch.sub(arg0, arg1, alpha=alpha)

                elif func_name == "div":
                    eps = 1e-6
                    if isinstance(arg1, torch.Tensor):
                        arg1 = torch.where(arg1 == 0, torch.full_like(arg1, eps), arg1)
                    elif isinstance(arg1, (int, float)) and arg1 == 0:
                        arg1 = eps
                    output = arg0 / arg1

                elif func_name == "rsub":
                    if arg1 is None and method_args:
                        minuend = method_args[0]
                        minuend = minuend.item() if isinstance(minuend, torch.Tensor) else minuend
                        output = torch.sub(minuend, arg0)
                    else:
                        output = torch.sub(arg1, arg0)

                elif func_name == "pow":
                    base, exponent = arg0, arg1
                    if base is None and method_args:
                        base = method_args[0]
                        base = base.item() if isinstance(base, torch.Tensor) else base
                    # Clamp base to avoid NaNs from negative fractional exponents
                    if isinstance(base, torch.Tensor):
                        base = torch.clamp(base, min=1e-6)
                    output = torch.pow(base, exponent)

                # === Scalar comparison ops (gt, ge, lt, eq) ===
                elif func_name in {"gt", "ge", "lt", "le", "eq"}:
                    # Direct PyTorch comparison ops (avoid aten_op)
                    if func_name == "gt":
                        output = arg0 > arg1
                    elif func_name == "ge":
                        output = arg0 >= arg1
                    elif func_name == "lt":
                        output = arg0 < arg1
                    elif func_name == "eq":
                        output = arg0 == arg1

                # === Fallback for unknown cases ===
                else:
                    # Fallback for rare ops
                    output = aten_op(arg0, arg1, *method_args) if arg1 is not None else aten_op(arg0, *method_args)

            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `{func_name}` failed in aten_op: {e}")

            if output is None:
                raise RuntimeError(f"[{node_name}] ❌ `{func_name}` did not compute output")

            # NaN check
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any().item()
                print(f"[{node_name}] ✅ {func_name} output shape: {output.shape}, NaN: {has_nan}")
                if has_nan:
                    print(f"[ERROR:NaN] {node_name} produced NaNs in `{func_name}`")

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
        
        elif func_name == "bmm":
            if isinstance(layer_in, list):
                #for i, t in enumerate(layer_in):
                #print(f"    ↪ Input {i}: shape={getattr(t, 'shape', t)}, type={type(t)}")
                if len(layer_in) != 2:
                    raise RuntimeError(f"[{node_name}] ❌ `bmm` expects exactly 2 tensor inputs, got {len(layer_in)}")
                a, b = layer_in
            else:
                raise RuntimeError(f"[{node_name}] ❌ `bmm` expects list of 2 tensors, got {type(layer_in)}")

            # Ensure both are tensors
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `bmm` inputs must be tensors: got {type(a)} and {type(b)}")

            # Optional dtype promotion
            if a.dtype != b.dtype:
                promoted_dtype = torch.promote_types(a.dtype, b.dtype)
                a = a.to(promoted_dtype)
                b = b.to(promoted_dtype)

            try:
                output = aten_op(a, b)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `bmm` failed with shapes {a.shape}, {b.shape}: {e}")

            return output

        elif func_name == "full":
            # 1. Try to extract `size`
            size = layer_hyperparams.get("size") or layer_hyperparams.get("sizes")
            fall_back_flag = False
            # 2. If `size` is missing or invalid, try method_args fallback
            if not size or all(s is None for s in size):
                if method_args and isinstance(method_args[0], (list, tuple)):
                    size = method_args[0]
                    fall_back_flag = True
                else:
                    print(f"[{node_name}] ⚠️ size is still None → Trying layer_in fallback...")

                    # 3. Last resort: try to resolve from layer_in if it's symbolic
                if fall_back_flag:
                    size_op = []
                    for p in size:
                        parent_val = tensor_map[str(p)]
                        if isinstance(parent_val, (int, float)):
                            size_op.append(int(parent_val))
                        else:
                            print(f"[{node_name}] ❌ Cannot resolve size from parent `{p}`: {type(parent_val)} → {parent_val}")
                    size =  size_op
            # 4. Logging final size

            # 5. Remaining parameters
            fill_value = layer_hyperparams.get("fill_value", 0)
            dtype = layer_hyperparams.get("dtype", torch.float32)
            device = layer_hyperparams.get("device", torch.device("cpu"))

            if isinstance(fill_value, torch.Tensor):
                fill_value = fill_value.item()

            try:
                output = aten_op(size, fill_value, dtype=dtype, device=device)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ Failed to execute `full` with size={size}, fill_value={fill_value}: {e}")
            
            return output

        
        elif func_name == "full_like":
            return aten_op(layer_in,
                           fill_value=layer_hyperparams["fill_value"],
                           dtype=layer_hyperparams["dtype"],
                           layout=layer_hyperparams["layout"],
                           device=layer_hyperparams["device"],
                           pin_memory=layer_hyperparams["pin_memory"])

        elif func_name == "expand":
            # --- 1) Map any torch.fx.Node in the declared sizes to its actual tensor/output value ---
            raw_sizes = layer_hyperparams.get("sizes", None)
            if isinstance(raw_sizes, (list, tuple)):
                sizes = []
                for s in raw_sizes:
                    if isinstance(s, torch.fx.Node):
                        # retrieve the recorded output value for that node
                        sizes.append(self.node_io[str(s)]["output_values"])
                    else:
                        sizes.append(s)
            else:
                sizes = raw_sizes

            # --- 2) Helper to resolve ints, tensors, SymInt, etc. ---
            def resolve_param(param):
                if isinstance(param, torch.Tensor):
                    return int(param.item())
                elif isinstance(param, torch.fx.Node):
                    value = node_io.get(str(param), {}).get("output_values", None)
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            return int(value.item())
                        else:
                            raise ValueError(f"[resolve_param] Tensor has multiple elements: {value.shape}")
                    elif isinstance(value, (int, float)):
                        return int(value)
                    elif value is None:
                        raise ValueError(f"[resolve_param] No output value found for node: {param}")
                    else:
                        raise TypeError(f"[resolve_param] Unexpected output value type: {type(value)}") 
                elif isinstance(param, torch.SymInt):
                    # fallback to stored _value if available
                    return int(param.node._value) if hasattr(param.node, "_value") else 1
                else:
                    return int(param)

            # --- 3) Unwrap single-element lists for the input tensor ---
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0]
                print(f"layer_in: {layer_in.shape}")

            print(f"initial sizes: {sizes}")

            # --- 4) If we still don't have all sizes, fall back to method args ---
            if not sizes or any(s is None for s in sizes):
                if method_args and isinstance(method_args[0], (list, tuple)):
                    print(f"[expand] ⛑️ Falling back to method_args: {method_args[0]}")
                    sizes = [resolve_param(x) for x in method_args[0]]
                else:
                    raise RuntimeError(f"[expand] ❌ Cannot resolve sizes for node `{node_name}`")

            # --- 5) Final resolution of each size element ---
            resolved_sizes = []
            for idx, s in enumerate(sizes):
                print(f"idx: {idx}, raw size: {s}")
                try:
                    resolved_sizes.append(resolve_param(s))
                except Exception as e:
                    raise RuntimeError(f"[expand] ❌ Error resolving size at index {idx}: {e}")

            print(f"resolved_sizes: {resolved_sizes}, implicit: {layer_hyperparams.get('implicit', False)}")

            # --- 6) Call through to the ATen expand operation ---
            return aten_op(
                layer_in,
                resolved_sizes,
                implicit=layer_hyperparams.get("implicit", False)
            )
        

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
            try:
                cond_, x_, y_ = torch.broadcast_tensors(cond, x, y)
                return aten_op(cond_, x_, y_)
            except Exception as e:
                # Safe fallback: expand cond to match if it's missing a dimension
                if cond.ndim == x.ndim - 1 and cond.shape[0] == x.shape[0] and cond.shape[-2:] == x.shape[-2:]:
                    cond_expanded = cond.unsqueeze(1).expand_as(x)
                    print(f"[{node_name}] ⚠️ Expanded cond to shape {cond_expanded.shape}")
                    return aten_op(cond_expanded, x, y)
                raise RuntimeError(
                    f"[{node_name}] ❌ Shape mismatch in `where`: "
                    f"cond: {cond.shape}, x: {x.shape}, y: {y.shape}. Error: {e}"
                )


        elif func_name == "contiguous":
            return aten_op(layer_in)

        elif func_name == "_to_copy":
            if isinstance(layer_in, list) and len(layer_in) == 1:
                layer_in = layer_in[0]
            
            if layer_in is None:
                raise RuntimeError(f"[DLBacktraceFX] ❌ _to_copy received `None` as input at node `{node_name}` → likely due to skipped or failed parent node.")
            
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[DLBacktraceFX] ❌ _to_copy expected a Tensor but got {type(layer_in)} at node `{node_name}` → value: {layer_in}")
                       
            output = aten_op(layer_in,
                           memory_format=layer_hyperparams.get("memory_format", torch.contiguous_format),
                           non_blocking=layer_hyperparams.get("non_blocking", False))
            
            return output

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
                print(node_name)
                print(layer_in.shape,"rsqrt shape input")
                output = aten_op(layer_in, *method_args)
                print(output.shape,"rsqrt shape")
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

        elif func_name == "mm":
            # Ensure `layer_in` is a list of exactly two tensors
            if isinstance(layer_in, list):
                print(f"  ↪ Number of inputs: {len(layer_in)}")
                for i, inp in enumerate(layer_in):
                    print(f"    ↪ Input {i}: shape={getattr(inp, 'shape', 'N/A')}, type={type(inp)}")
                
                if len(layer_in) != 2:
                    raise RuntimeError(f"[{node_name}] ❌ `mm` expects 2 tensor inputs, got {len(layer_in)}")

                a, b = layer_in
            else:
                raise RuntimeError(f"[{node_name}] ❌ `mm` expects list of tensors, got: {type(layer_in)}")

            # Validate inputs are tensors
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `mm` inputs must be tensors: got {type(a)} and {type(b)}")

            # Optional: dtype promotion
            if a.dtype != b.dtype:
                target_dtype = torch.promote_types(a.dtype, b.dtype)
                a = a.to(dtype=target_dtype)
                b = b.to(dtype=target_dtype)

            try:
                output = aten_op(a, b)
            except Exception as e:
                raise RuntimeError(
                    f"[{node_name}] ❌ `mm` failed with shapes {a.shape}, {b.shape}: {e}"
                )

            return output

        elif func_name == "cat":
            
            # Step 1: Get tensor list from method_args (preferred)
            tensors = []
            if method_args and isinstance(method_args[0], (list, tuple)):
                for t in method_args[0]:
                    key = str(t)
                    if key in node_io:
                        val = node_io[key]["output_values"]
                        if isinstance(val, torch.Tensor):
                            tensors.append(val)
                        else:
                            raise RuntimeError(f"[{node_name}] ⚠️ Non-tensor in method_args[0]: {type(val)}")
                    else:
                        raise KeyError(f"[{node_name}] ❌ Parent node `{key}` not found in node_io")
            else:
                raise RuntimeError(f"[{node_name}] ❌ `cat` method_args[0] is not a list: {method_args}")

            # Step 2: Get and validate the dimension
            dim = method_args[1] if len(method_args) > 1 else layer_hyperparams.get("dim", 0)
            if not isinstance(dim, int):
                try:
                    dim = int(dim)
                except Exception as e:
                    raise RuntimeError(f"[{node_name}] ❌ Invalid dim value `{dim}`: {e}")

            for i, t in enumerate(tensors):
                print(f"  ↪ Tensor {i}: shape={t.shape}, dtype={t.dtype}")

            try:
                output = aten_op(tensors, dim)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `cat` failed: {e}")

            return output

        elif func_name == "clone":
            if isinstance(layer_in, list):
                if len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    raise RuntimeError(f"[{node_name}] ❌ `clone` expects 1 input tensor, got: {layer_in}")

            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `clone` expected Tensor input, got {type(layer_in)}")

            try:
                output = aten_op(
                    layer_in,
                    memory_format=layer_hyperparams.get("memory_format", torch.contiguous_format)
                )
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `clone` failed: {e}")

            return output

        elif func_name == "cos":
            if isinstance(layer_in, list):
                if len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    raise RuntimeError(f"[{node_name}] ❌ `cos` expects a single input tensor, got: {layer_in}")
            
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `cos` expects a Tensor, got {type(layer_in)}")

            try:
                output = aten_op(layer_in)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `cos` failed: {e}")
            
            return output

        elif func_name == "sin":
            # Unwrap if it's a singleton list
            if isinstance(layer_in, list):
                if len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    raise RuntimeError(f"[{node_name}] ❌ `sin` expects a single input tensor but got a list of {len(layer_in)} elements.")

            # Type check
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `sin` expected a Tensor but got {type(layer_in)} → {layer_in}")

            try:
                output = aten_op(layer_in)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `sin` failed: {e}")

            return output

        elif func_name == "neg":
            # Unwrap if it's a singleton list
            if isinstance(layer_in, list):
                if len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    raise RuntimeError(f"[{node_name}] ❌ `neg` expects a single input tensor but got list of {len(layer_in)} elements.")

            # Type check
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `neg` expected a Tensor but got {type(layer_in)} → {layer_in}")

            try:
                output = aten_op(layer_in)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `neg` failed: {e}")

            return output
        
        elif func_name == "copy":
            
            # Unpack inputs
            if isinstance(layer_in, list) and len(layer_in) == 2:
                self_tensor, src_tensor = layer_in
            else:
                raise RuntimeError(
                    f"[{node_name}] ❌ `copy` expects a list of 2 tensors, but got: {layer_in}"
                )

            # Validate input types
            if not isinstance(self_tensor, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `copy` expected `self` to be Tensor but got {type(self_tensor)}")

            if not isinstance(src_tensor, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `copy` expected `src` to be Tensor but got {type(src_tensor)}")

            non_blocking = layer_hyperparams.get("non_blocking", False)

            try:
                output = aten_op(self_tensor, src_tensor, non_blocking=non_blocking)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `copy` failed: {e}")

            return output
        
        elif func_name == "slice_scatter":
            # Step 1: Unpack inputs
            if isinstance(layer_in, list) and len(layer_in) == 2:
                self_tensor, src_tensor = layer_in
            else:
                raise RuntimeError(f"[{node_name}] ❌ `slice_scatter` expects 2 inputs (self, src), got: {layer_in}")

            
            # Step 2: Resolve hyperparams with defaults and checks
            INT64_MAX = 9223372036854775807
            dim   = layer_hyperparams.get("dim", 0)
            start = layer_hyperparams.get("start", 0)
            end   = layer_hyperparams.get("end", None)
            step  = layer_hyperparams.get("step", 1)

            # Fix symbolic placeholder for start/end
            if isinstance(start, int) and start >= INT64_MAX:
                print(f"[{node_name}] ⚠️ Detected symbolic placeholder for `start` ({start}) → resetting to 0")
                start = 0

            if isinstance(end, int) and end >= INT64_MAX:
                end = self_tensor.shape[dim]
                print(f"[{node_name}] ⚠️ Detected symbolic placeholder for `end` ({INT64_MAX}) → using {end}")

            if end is None:
                end = self_tensor.shape[dim]
                print(f"[{node_name}] ℹ️ `end` not specified → using {end}")

            # Step 3: Validate dimensions
            try:
                slice_size = end - start
                src_size = src_tensor.shape[dim]
                self_size = self_tensor.shape[dim]

                if slice_size != src_size:
                    print(f"[{node_name}] ⚠️ Shape mismatch: src[{dim}] = {src_size} != target slice length = {slice_size}")
            except Exception as e:
                print(f"[{node_name}] ⚠️ Failed to inspect shapes: {e}")

            # Optional: Show small slice for verification
            try:
                preview = self_tensor.narrow(dim, start, min(end - start, self_tensor.shape[dim] - start))
                print(f"[{node_name}] 🔍 self_tensor slice preview (dim={dim}, start={start}, end={end}): shape={preview.shape}")
            except Exception as e:
                print(f"[{node_name}] ⚠️ Could not preview slice: {e}")

            # Step 4: Execute
            try:
                output = aten_op(self_tensor, src_tensor, dim, start, end, step)
                print(f"[{node_name}] ✅ `slice_scatter` success → output shape: {output.shape}")
            except Exception as e:
                raise RuntimeError(
                    f"[{node_name}] ❌ `slice_scatter` failed: self={type(self_tensor)}, src={type(src_tensor)}, "
                    f"dim={dim}, start={start}, end={end}, step={step}. Error: {e}"
                )

            return output
        
        elif func_name == "_unsafe_view":
            # 🔍 Step 1: Extract tensor from layer_in
            if isinstance(layer_in, list):
                input_tensor = None
                for item in layer_in:
                    if isinstance(item, torch.Tensor):
                        input_tensor = item
                        break
                if input_tensor is None:
                    raise RuntimeError(f"[{node_name}] ❌ No valid tensor found in inputs: {layer_in}")
            else:
                input_tensor = layer_in

            # 🔍 Step 2: Resolve shape from method_args
            def resolve_param_view(p):
                try:
                    if isinstance(p, (torch.fx.Node, str)):
                        p_key = str(p)
                        if p_key in node_io:
                            val = node_io[p_key]["output_values"]
                            return val.item() if isinstance(val, torch.Tensor) else int(val)
                        else:
                            raise KeyError(f"[{node_name}] ❌ `{p_key}` not found in `node_io`")
                    elif isinstance(p, torch.Tensor):
                        return int(p.item())
                    elif isinstance(p, (int, torch.SymInt)):
                        return int(p)
                    elif p is None:
                        raise ValueError(f"[{node_name}] ❌ Cannot resolve `None` in shape.")
                    return int(p)
                except Exception as e:
                    raise ValueError(f"[{node_name}] ❌ Failed to resolve shape param: {p} ({type(p)}) → {e}")

            raw_shape = method_args[0] if method_args and isinstance(method_args[0], (list, tuple)) else []
            resolved_shape = [resolve_param_view(p) for p in raw_shape]

            # ✅ Step 3: Execute
            try:
                output = aten_op(input_tensor, resolved_shape)
                print(f"[{node_name}] ✅ `_unsafe_view` success → output shape: {output.shape}")
            except Exception as e:
                raise RuntimeError(
                    f"[{node_name}] ❌ `_unsafe_view` failed: input={getattr(input_tensor, 'shape', None)}, "
                    f"shape={resolved_shape}, error={e}"
                )

            return output

        elif func_name == "einsum":
            if not method_args or not isinstance(method_args[0], str):
                raise RuntimeError(f"[{node_name}] ❌ `einsum` requires equation string as the first method_arg")

            equation = method_args[0]

            # Ensure input is a list of tensors
            if not isinstance(layer_in, (list, tuple)):
                layer_in = [layer_in]

            print(f"[{node_name}] 🧪 einsum equation: {equation}")
            for i, t in enumerate(layer_in):
                if isinstance(t, torch.Tensor):
                    print(f"[{node_name}] ↪ Input {i} shape: {t.shape}")
                else:
                    print(f"[{node_name}] ⚠️ Input {i} is not a tensor: {type(t)}")

            try:
                # Don't unpack the tensor list — pass it as a list
                output = aten_op(equation, layer_in)
                print(f"[{node_name}] ✅ einsum output shape: {output.shape}")
            except Exception as e:
                raise RuntimeError(
                    f"[{node_name}] ❌ `einsum` failed with equation '{equation}' and inputs: {layer_in}. Error: {e}"
                )

            return output

        elif func_name == "index_select":
            if not isinstance(layer_in, list) or len(layer_in) != 2:
                raise RuntimeError(f"[{node_name}] ❌ `index_select` expected 2 inputs from parents, got: {layer_in}")

            input_tensor, index_tensor = layer_in

            if not method_args or len(method_args) < 1:
                raise RuntimeError(f"[{node_name}] ❌ `index_select` missing `dim` in method_args: {method_args}")

            dim = method_args[0]
            if isinstance(dim, torch.Tensor):
                dim = int(dim.item())
            elif not isinstance(dim, int):
                raise TypeError(f"[{node_name}] ❌ `dim` must be int, got {type(dim)}: {dim}")

            if index_tensor.dtype not in (torch.int32, torch.int64):
                index_tensor = index_tensor.to(dtype=torch.int64)

            # ✅ Match length to peer tensor (e.g. einsum_5)
            peer_shape = node_io.get('einsum_5', {}).get('output_values', None)
            if isinstance(peer_shape, torch.Tensor):
                expected_dim_size = peer_shape.shape[dim]
                if index_tensor.shape[0] > expected_dim_size:
                    print(f"[{node_name}] ⚠️ Trimming index_tensor from {index_tensor.shape[0]} to {expected_dim_size} to match einsum_5 shape")
                    index_tensor = index_tensor[:expected_dim_size]

            print(f"[{node_name}] ✅ index_select(dim={dim}) on input shape {input_tensor.shape} with index shape {index_tensor.shape}")
            return aten_op(input_tensor, dim, index_tensor)

        elif func_name == "addmm":
            if isinstance(layer_in, (list, tuple)):
                for idx, item in enumerate(layer_in):
                    print(f"idx: {idx}, item: {item.shape}") 
            else:
                print(f"layer_in shape: {layer_in.shape}")

            if isinstance(layer_in, list) and len(layer_in) == 3:
                bias, mat1, mat2 = layer_in
                print(f"bias: {bias.shape}, mat1: {mat1.shape}, mat2: {mat2.shape}") 

                if not all(isinstance(x, torch.Tensor) for x in (bias, mat1, mat2)):
                    raise TypeError(f"[{node_name}] ❌ Expected Tensors for `addmm`, got {[type(x) for x in layer_in]}")

                try:
                    return torch.addmm(bias, mat1, mat2)
                except Exception as e:
                    raise RuntimeError(
                        f"[{node_name}] ❌ torch.addmm failed with shapes: "
                        f"bias={bias.shape}, mat1={mat1.shape}, mat2={mat2.shape}. Error: {e}"
                    )
            else:
                raise RuntimeError(f"[{node_name}] ❌ `addmm` expects 3 input tensors (bias, mat1, mat2), got: {layer_in}")
        
        else:
            inputs = layer_in if isinstance(layer_in, (list, tuple)) else [layer_in]
            output = aten_op(*inputs, *method_args)
            return output 
            
    except Exception as e:
        print(f"[Execution Error] Node `{node_name}` failed in `{func_name}`: {e}")
        #return layer_in

def run_execution_nocache(graph, layer_stack, model, extracted_weights, inputs, tracer):
    print(f"Executing `run_execution_nocache` ...!")
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
        method_args = node_data["method_args"]
        layer = node_data["layer"]
        func_name = node_data["func_name"]
        layer_hyperparams = node_data["layer_hyperparams"]
        children = node_data["children"]

        if any(p not in tensor_map for p in parents):
            continue
        
        layer_in = [_sanitize_input_tensor(tensor_map[p]) for p in parents]
        output = layer_in
        # Handle embedding function
        if func_name == "embedding":
            pass
                    
        try:
            if layer_type == "Placeholder":
                output = extracted_weights.get(node_name, inp_map.get(node_name, layer_in)) 

                special_weight_keywords = [
                    '_layernorm_weight',
                    '_proj_weight',
                    '_head_weight',
                    '_norm_weight'
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
                if func_name == "linear":
                    for node_x in parents:
                        if node_io[node_x]['layer_type'] in ("future_use","Weight","Bias","bn_running_mean","bn_running_var","bn_num_batches_tracked","embeddings"):
                            continue
                        else:
                            layer_in.append(node_io[node_x]['output_values'])
                elif func_name == "addmm":
                    # ✅ For addmm, include all parent outputs, including weight/bias
                    for node_x in parents:
                        layer_in.append(node_io[node_x]['output_values'])

                else:
                    for node_x in parents:
                        if func_name == "_native_batch_norm_legit_no_training":
                            # ✅ Include *all* parents for batch norm op
                            layer_in.append(node_io[node_x]['output_values'])
                        elif node_io[node_x]['layer_type'] in ("Weight", "Bias", "bn_running_mean", "bn_running_var", "bn_num_batches_tracked", "embeddings"):
                            continue
                        else:
                            layer_in.append(node_io[node_x]['output_values']) 

                if func_name == "addmm":
                    pass  # ⛔ Don't process or unwrap inputs
                else:
                    layer_in, _ = _process_layer_input(layer_in)

                output = execute_aten_operation(func_name, layer, layer_in, layer_hyperparams, method_args, parents, node_io, node_name,tensor_map, children=children)
                
        except Exception as e:
            print(f"[Execution Error - NoCache] Node `{node_name}` failed in `{func_name}`: {e}")
            output = layer_in

        # At the end of node execution
        processed_output = _process_output_tuple(output)

        if isinstance(processed_output, torch.Tensor) and torch.isnan(processed_output).any():
            print(f"[ERROR:NaN] Node `{node_name}` produced NaNs → shape: {processed_output.shape}")

        # Accept both single-tensor and tensor-tuples
        if isinstance(processed_output, (torch.Tensor, int)):
            tensor_map[node_name] = processed_output
        elif isinstance(processed_output, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in processed_output):
            tensor_map[node_name] = processed_output
        else:
            raise RuntimeError(f"[{node_name}] ❌ No valid tensor found in processed_output.") 

        # normalize input_values to never be a bare list
        iv = layer_in
        if isinstance(iv, list):
            if len(iv) == 1:
                iv = iv[0]
            else:
                iv = tuple(iv)

        # record everything in node_io
        node_io[node_name] = {
            "input_sources":    parents,
            "input_values":     iv,
            "output_values":    processed_output,
            "layer_type":       layer_type,
            "node_type":        node_data["node_type"],
            "method_args":      method_args,
            "layer":            layer,
            "layer_name":       node_data["layer_name"],
            "func_name":        func_name,
            "func_module":      node_data["func_module"],
            "output_children":  children,
            "layer_hyperparams":layer_hyperparams,
        }

    return node_io

class ExecutionEngineNoCache:
    def __init__(self, model, extracted_weights, fx_graph, layer_stack, tracer, exported_program):
        self.model = model
        self.extracted_weights = extracted_weights
        self.graph = fx_graph
        self.layer_stack = layer_stack
        self.tracer = tracer
        self.exported_program = exported_program

    def run(self, inputs):
        return run_execution_nocache(
            graph=self.graph,
            layer_stack=self.layer_stack,
            model=self.model,
            extracted_weights=self.extracted_weights,
            inputs=inputs,
            tracer=self.tracer
        )
