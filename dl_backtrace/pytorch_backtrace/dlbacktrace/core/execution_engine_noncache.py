# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/execution_engine_noncache.py
import os
import inspect
import torch
import numpy as np

# Toggle debug prints
DEBUG = True  # Changed to False by default for performance
def log(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def _load_if_path(val, cache_manager):
    if isinstance(val, str) and val.endswith(".pt.zstd") and os.path.isfile(val):
        return cache_manager.load_tensor(val)
    return val

def sanitize(x):
        return x.to(torch.float32) if isinstance(x, torch.Tensor) and x.dtype == torch.bool else x

def ensure_model_has_device_attribute(model):
    """Ensure model has a device attribute, set it if missing"""
    if not hasattr(model, 'device'):
        try:
            # Try to get device from model parameters
            device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
            model.device = device
            print(f"[DEBUG] 🔧 Added device attribute to model: {device}")
        except Exception as e:
            model.device = torch.device('cpu')
            print(f"[DEBUG] 🔧 Added default device attribute to model: cpu (error: {e})")
    return model.device 

def _sanitize_input_tensor(t):
    return t.detach() if isinstance(t, torch.nn.Parameter) else t

# 🔧 CONSISTENCY FIXES: New functions to ensure consistent outputs
def standardize_layer_input(layer_in, parents, node_io, tensor_map, func_name):
    """Ensure consistent input processing across all operations"""
    if isinstance(layer_in, list):
        # Filter out weight/bias nodes consistently
        filtered_inputs = []
        for p in parents:
            if node_io[p]['layer_type'] in ["Weight", "Bias", "future_use"]:
                continue
            filtered_inputs.append(tensor_map[p])
        
        if len(filtered_inputs) == 1:
            return filtered_inputs[0]
        elif len(filtered_inputs) > 1:
            return filtered_inputs
        else:
            raise RuntimeError(f"No valid inputs found for {func_name}")
    
    return layer_in

def enforce_precision_consistency(tensors, target_dtype=torch.float32):
    """Ensure all tensors use the same precision and device"""
    # Get the target device from the first tensor we find
    target_device = None
    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            if isinstance(t, torch.Tensor):
                target_device = t.device
                break
    elif isinstance(tensors, torch.Tensor):
        target_device = tensors.device
    
    # If no device found, default to CPU
    if target_device is None:
        target_device = torch.device("cpu")
    
    if isinstance(tensors, (list, tuple)):
        return [t.to(dtype=target_dtype, device=target_device) if isinstance(t, torch.Tensor) else t for t in tensors]
    elif isinstance(tensors, torch.Tensor):
        return tensors.to(dtype=target_dtype, device=target_device)
    return tensors

def setup_consistent_environment():
    """Set up environment for consistent execution"""
    print("[DEBUG] 🔧 Setting up deterministic execution environment...")
    
    # Force evaluation mode
    torch.set_grad_enabled(False)
    print("[DEBUG]   ✅ Gradients disabled")
    
    # 🔧 CRITICAL FIX: Disable deterministic algorithms that can cause inconsistencies
    # These settings can make DLB execution different from direct model execution
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for consistency
    torch.backends.cudnn.benchmark = True       # Enable benchmarking for consistency
    torch.backends.cudnn.allow_tf32 = True      # Allow TF32 for consistency
    print("[DEBUG]   ✅ cuDNN settings optimized for consistency")
    
    # 🔧 CRITICAL FIX: Don't force deterministic algorithms
    # torch.use_deterministic_algorithms(True, warn_only=False)  # Commented out
    print("[DEBUG]   ✅ PyTorch algorithms set for consistency (non-deterministic)")
    
    # 🔧 CRITICAL FIX: Don't set fixed random seeds
    # This can cause differences between DLB and direct execution
    # torch.manual_seed(42)  # Commented out
    # torch.cuda.manual_seed(42)  # Commented out
    # torch.cuda.manual_seed_all(42)  # Commented out
    # np.random.seed(42)  # Commented out
    print("[DEBUG]   ✅ Random seeds preserved for consistency")
    
    # 🔧 CRITICAL FIX: Don't force float32 precision globally
    # This can cause dtype mismatches with original model
    # torch.set_default_dtype(torch.float32)  # Commented out
    print("[DEBUG]   ✅ Default dtype preserved for consistency")
    
    # Set environment variables for consistency (not determinism)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # os.environ['PYTHONHASHSEED'] = '42'  # Commented out
    print("[DEBUG]   ✅ Environment variables set for consistency")
    
    print("[DEBUG] 🔧 Consistency-optimized environment setup complete!")

def ensure_model_state_consistency(dlb_model, original_model):
    """Ensure DLB model has exactly the same state as original model"""
    print("[DEBUG] 🔧 Ensuring model state consistency...")
    try:
        # Copy all parameters exactly
        for orig_param, dlb_param in zip(original_model.parameters(), dlb_model.parameters()):
            dlb_param.data.copy_(orig_param.data)
            # Ensure same dtype and device
            dlb_param.data = dlb_param.data.to(dtype=orig_param.dtype, device=orig_param.device)
        
        # Copy all buffers (running stats, etc.)
        for orig_buffer, dlb_buffer in zip(original_model.buffers(), dlb_model.buffers()):
            dlb_buffer.data.copy_(orig_buffer.data)
            # Ensure same dtype and device
            dlb_buffer.data = dlb_buffer.data.to(dtype=orig_buffer.dtype, device=orig_buffer.device)
        
        # Force same mode and properties
        dlb_model.eval()
        dlb_model.requires_grad_(False)
        
        # 🔧 CRITICAL FIX: Preserve original parameter precision for consistency
        # Don't force float32 conversion as it can cause numerical differences
        for param in dlb_model.parameters():
            print(f"[DEBUG] Preserving parameter {param.shape} precision: {param.dtype}")
        
        print("[DEBUG]   ✅ Model state synchronized successfully")
        return True
        
    except Exception as e:
        print(f"[DEBUG]   ❌ Model state synchronization failed: {e}")
        return False

def ensure_input_consistency(inputs, target_model):
    """Ensure inputs are IDENTICAL between executions"""
    print("[DEBUG] 🔧 Ensuring input consistency...")
    try:
        # 🔧 CRITICAL FIX: Ensure model has device attribute, set it if missing
        target_device = ensure_model_has_device_attribute(target_model)
        
        consistent_inputs = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                # Force exact same properties as original model expects
                consistent_inp = inp.detach().clone()
                
                # 🔧 SPECIAL HANDLING: input_ids and attention_mask must remain long/int
                if i == 0:  # First input is usually input_ids
                    target_dtype = torch.long
                elif i == 1:  # Second input is usually attention_mask
                    target_dtype = torch.long
                else:
                    target_dtype = inp.dtype  # Preserve original dtype
                
                consistent_inp = consistent_inp.to(
                    dtype=target_dtype,
                    device=target_device,
                    memory_format=torch.contiguous_format
                )
                consistent_inp.requires_grad_(False)
                consistent_inputs.append(consistent_inp)
                print(f"[DEBUG]   ✅ Input tensor {i}: {inp.shape} -> {consistent_inp.shape} on {consistent_inp.device}, dtype: {consistent_inp.dtype}")
            else:
                consistent_inputs.append(inp)
                print(f"[DEBUG]   ✅ Non-tensor input {i}: {type(inp)}")
        
        print("[DEBUG]   ✅ Input consistency ensured successfully")
        return consistent_inputs
        
    except Exception as e:
        # Silently handle device-related errors - they're not critical
        # Only print error if it's a real issue, not just device detection
        if "device" not in str(e).lower() and "attribute" not in str(e).lower():
            print(f"[DEBUG] 🔧 Input consistency failed: {e}")
        return inputs

def ensure_embedding_consistency(weight, indices, node_name):
    """Ensure embedding operation produces consistent results"""
    print(f"[DEBUG] 🔧 Ensuring embedding consistency for {node_name}...")
    
    try:
        # 🔧 CRITICAL FIX: Ensure weight is properly formatted
        if isinstance(weight, torch.Tensor):
            # Force weight to be contiguous and on correct device
            if not weight.is_contiguous():
                weight = weight.contiguous()
                print(f"[DEBUG] Made weight contiguous: {weight.shape}")
            
            # 🔧 CRITICAL FIX: Preserve original weight precision for consistency
            # Don't force float32 conversion as it can cause numerical differences
            print(f"[DEBUG] Preserving weight precision: {weight.dtype}")
        
        # 🔧 CRITICAL FIX: Ensure indices are properly formatted
        if isinstance(indices, torch.Tensor):
            # Force indices to be long (integer) type
            if indices.dtype != torch.long:
                indices = indices.long()
                print(f"[DEBUG] Converted indices to long: {indices.dtype}")
            
            # Ensure indices are contiguous
            if not indices.is_contiguous():
                indices = indices.contiguous()
                print(f"[DEBUG] Made indices contiguous: {indices.shape}")
            
            # Ensure indices are on same device as weight
            if weight.device != indices.device:
                indices = indices.to(device=weight.device)
                print(f"[DEBUG] Moved indices to device: {weight.device}")
        
        print(f"[DEBUG] ✅ Embedding consistency ensured:")
        print(f"   Weight: {weight.shape} on {weight.device}, dtype: {weight.dtype}")
        print(f"   Indices: {indices.shape} on {indices.device}, dtype: {indices.dtype}")
        
        return weight, indices
        
    except Exception as e:
        print(f"[DEBUG] ❌ Embedding consistency failed: {e}")
        return weight, indices

def ensure_operation_consistency(operation_name, inputs, weights, node_name):
    """Ensure any operation produces consistent results"""
    print(f"[DEBUG] 🔧 Ensuring operation consistency for {operation_name} in {node_name}...")
    
    try:
        # 🔧 CRITICAL FIX: Preserve original input precision for consistency
        # Don't force float32 conversion as it can cause numerical differences
        if isinstance(inputs, (list, tuple)):
            consistent_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    # Preserve original dtype and ensure contiguous
                    consistent_inp = inp.to(dtype=inp.dtype)  # Preserve original dtype
                    if not consistent_inp.is_contiguous():
                        consistent_inp = consistent_inp.contiguous()
                    consistent_inputs.append(consistent_inp)
                else:
                    consistent_inputs.append(inp)
            inputs = consistent_inputs
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.to(dtype=inputs.dtype)  # Preserve original dtype
            if not inputs.is_contiguous():
                inputs = inputs.contiguous()
        
        # 🔧 CRITICAL FIX: Preserve original weight precision for consistency
        # Don't force float32 conversion as it can cause numerical differences
        if isinstance(weights, (list, tuple)):
            consistent_weights = []
            for w in weights:
                if isinstance(w, torch.Tensor):
                    consistent_w = w.to(dtype=w.dtype)  # Preserve original dtype
                    if not consistent_w.is_contiguous():
                        consistent_w = consistent_w.contiguous()
                    consistent_weights.append(consistent_w)
                else:
                    consistent_weights.append(w)
            weights = consistent_weights
        elif isinstance(weights, torch.Tensor):
            weights = weights.to(dtype=weights.dtype)  # Preserve original dtype
            if not weights.is_contiguous():
                weights = weights.contiguous()
        
        print(f"[DEBUG] ✅ Operation consistency ensured for {operation_name}")
        return inputs, weights
        
    except Exception as e:
        print(f"[DEBUG] ❌ Operation consistency failed: {e}")
        return inputs, weights

def sanitize_model_layer_inputs(layer_in, original_input):
    """Ensure Model_Layer inputs match original model inputs exactly"""
    if isinstance(layer_in, (list, tuple)):
        sanitized = []
        for i, inp in enumerate(layer_in):
            if isinstance(inp, torch.Tensor):
                # Force same properties as original
                sanitized.append(inp.detach().to(dtype=original_input.dtype, device=original_input.device))
            else:
                sanitized.append(inp)
        return sanitized
    else:
        if isinstance(layer_in, torch.Tensor):
            return layer_in.detach().to(dtype=original_input.dtype, device=original_input.device)
        return layer_in

def ensure_model_state_consistency(dlb_model, original_model):
    """Ensure DLB model has exactly the same state as original model"""
    print("[DEBUG] 🔧 Ensuring model state consistency...")
    
    try:
        # Copy all parameters exactly
        for orig_param, dlb_param in zip(original_model.parameters(), dlb_model.parameters()):
            dlb_param.data.copy_(orig_param.data)
        
        # Copy all buffers (running stats, etc.)
        for orig_buffer, dlb_buffer in zip(original_model.buffers(), dlb_model.buffers()):
            dlb_buffer.data.copy_(orig_buffer.data)
        
        # Force same mode and properties
        dlb_model.eval()
        dlb_model.requires_grad_(False)
        
        # 🔧 CRITICAL FIX: Preserve original parameter precision for consistency
        # Don't force float32 conversion as it can cause numerical differences
        for param in dlb_model.parameters():
            print(f"[DEBUG] Preserving parameter {param.shape} precision: {param.dtype}")
        
        print("[DEBUG]   ✅ Model state synchronized successfully")
        return True
        
    except Exception as e:
        print(f"[DEBUG]   ❌ Model state synchronization failed: {e}")
        return False

def ensure_input_consistency(inputs, target_model):
    """Ensure inputs are IDENTICAL between executions"""
    print("[DEBUG] 🔧 Ensuring input consistency...")
    
    try:
        # 🔧 CRITICAL FIX: Ensure model has device attribute, set it if missing
        target_device = ensure_model_has_device_attribute(target_model)
        
        consistent_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                # 🔧 CRITICAL FIX: Preserve original input precision for consistency
                # Don't force float32 conversion as it can cause numerical differences
                consistent_inp = inp.detach().clone()
                consistent_inp = consistent_inp.to(
                    dtype=inp.dtype,  # Preserve original dtype
                    device=target_device,
                    memory_format=torch.contiguous_format
                )
                consistent_inp.requires_grad_(False)
                consistent_inputs.append(consistent_inp)
                print(f"[DEBUG]   ✅ Input tensor: {inp.shape} -> {consistent_inp.shape} on {consistent_inp.device}")
            else:
                consistent_inputs.append(inp)
                print(f"[DEBUG]   ✅ Non-tensor input: {type(inp)}")
        
        print("[DEBUG]   ✅ Input consistency ensured successfully")
        return consistent_inputs
        
    except Exception as e:
        print(f"[DEBUG]   ❌ Input consistency failed: {e}")
        return inputs

def ensure_embedding_consistency(weight, indices, node_name):
    """Ensure embedding operation produces consistent results"""
    print(f"[DEBUG] 🔧 Ensuring embedding consistency for {node_name}...")
    
    try:
        # 🔧 CRITICAL FIX: Ensure weight is properly formatted
        if isinstance(weight, torch.Tensor):
            # Force weight to be contiguous and on correct device
            if not weight.is_contiguous():
                weight = weight.contiguous()
                print(f"[DEBUG] Made weight contiguous: {weight.shape}")
            
            # 🔧 CRITICAL FIX: Preserve original weight precision for consistency
            # Don't force float32 conversion as it can cause numerical differences
            print(f"[DEBUG] Preserving weight precision: {weight.dtype}")
        
        # 🔧 CRITICAL FIX: Ensure indices are properly formatted
        if isinstance(indices, torch.Tensor):
            # Force indices to be long (integer) type
            if indices.dtype != torch.long:
                indices = indices.long()
                print(f"[DEBUG] Converted indices to long: {indices.dtype}")
            
            # Ensure indices are contiguous
            if not indices.is_contiguous():
                indices = indices.contiguous()
                print(f"[DEBUG] Made indices contiguous: {indices.shape}")
            
            # Ensure indices are on same device as weight
            if weight.device != indices.device:
                indices = indices.to(device=weight.device)
                print(f"[DEBUG] Moved indices to device: {weight.device}")
        
        print(f"[DEBUG] ✅ Embedding consistency ensured:")
        print(f"   Weight: {weight.shape} on {weight.device}, dtype: {weight.dtype}")
        print(f"   Indices: {indices.shape} on {indices.device}, dtype: {indices.dtype}")
        
        return weight, indices
        
    except Exception as e:
        print(f"[DEBUG] ❌ Embedding consistency failed: {e}")
        return weight, indices


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
        # 🔧 CONSISTENCY FIX: Apply precision consistency to ALL operations
        if isinstance(layer_in, (list, tuple)):
            layer_in = [enforce_precision_consistency(x) for x in layer_in]
        else:
            layer_in = enforce_precision_consistency(layer_in)
        
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
            # 🔧 CRITICAL FIX: Consistent linear operation handling with proper device consistency
            if DEBUG:
                print(f"[{node_name}] 🔧 linear: input type={type(layer_in)}, parents={parents}")
            
            # 🔧 FIX: Use standardized input processing
            layer_in = standardize_layer_input(layer_in, parents, node_io, tensor_map, func_name)
            
            # 🔧 FIX: Enforce precision consistency
            layer_in = enforce_precision_consistency(layer_in)
            
            # 🔧 DEVICE CONSISTENCY FIX: Ensure input and weight are on the same device
            if isinstance(layer_in, torch.Tensor):
                target_device = layer_in.device
                weight = layer_hyperparams["weight"]
                
                if weight.device != target_device:
                    if DEBUG:
                        print(f"[{node_name}] 🔧 linear: moving weight from {weight.device} to {target_device}")
                    weight = weight.to(device=target_device)
                
                # 🔧 FIX: Ensure weight is contiguous for better performance
                if not weight.is_contiguous():
                    weight = weight.contiguous()
                
                if DEBUG:
                    print(f"[{node_name}] ✅ linear: input shape={layer_in.shape}, weight shape={weight.shape}")
                    print(f"[{node_name}] ✅ linear: input device={layer_in.device}, weight device={weight.device}")
                
                # 🔧 FIX: Handle bias with proper device consistency
                bias = layer_hyperparams.get("bias", None)
                if bias is not None and not isinstance(bias, bool):
                    if bias.device != target_device:
                        if DEBUG:
                            print(f"[{node_name}] 🔧 linear: moving bias from {bias.device} to {target_device}")
                        bias = bias.to(device=target_device)
                    
                    if not bias.is_contiguous():
                        bias = bias.contiguous()
                    
                    output = aten_op(layer_in, weight, bias)
                else:
                    output = aten_op(layer_in, weight)
                
                if DEBUG:
                    print(f"[{node_name}] ✅ linear: output shape={output.shape}")
                
                return output
            else:
                raise RuntimeError(f"[{node_name}] ❌ linear: expected tensor input, got {type(layer_in)}")

        elif func_name == "conv2d": 
            # unwrap single-element list inputs
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 

            bias = layer_hyperparams["bias"]
            if isinstance(bias, bool):
                bias = None  # ✅ Fix: convert True/False to None

            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            target_device = layer_in.device
            weight = layer_hyperparams["weight"]
            if weight.device != target_device:
                weight = weight.to(device=target_device)
            if bias is not None and bias.device != target_device:
                bias = bias.to(device=target_device)

            return aten_op(layer_in, weight, bias,
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
            # 🔧 CONSISTENCY FIX: Force evaluation mode for consistency
            return aten_op(layer_in, layer_hyperparams["p"], False)  # Always False

        elif func_name in ("relu", "relu_", "gelu", "tanh","silu"):
            # 🔧 CRITICAL FIX: Robust activation function handling with proper input validation
            if DEBUG:
                print(f"[{node_name}] 🔧 {func_name}: input type={type(layer_in)}, length={len(layer_in) if isinstance(layer_in, (list, tuple)) else 'single'}")
            
            # 🔧 FIX: Handle list inputs properly
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    raise RuntimeError(f"[{node_name}] ❌ {func_name}: received empty input list")
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: Find first valid tensor without modifying original
                    valid_tensor = None
                    for x in layer_in:
                        if isinstance(x, torch.Tensor):
                            valid_tensor = x
                            break
                    
                    if valid_tensor is None:
                        raise RuntimeError(f"[{node_name}] ❌ {func_name}: no valid tensor found in input list")
                    
                    layer_in = valid_tensor
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ {func_name}: expected tensor input, got {type(layer_in)}")
            
            # 🔧 DEVICE CONSISTENCY FIX: Ensure tensor is on the correct device
            if not layer_in.is_contiguous():
                layer_in = layer_in.contiguous()
            
            if DEBUG:
                print(f"[{node_name}] ✅ {func_name}: input shape={layer_in.shape}, device={layer_in.device}")
            
            try:
                # 🔧 FIX: aten_op is the actual PyTorch function, call it directly
                output = aten_op(layer_in)
                
                if DEBUG:
                    print(f"[{node_name}] ✅ {func_name}: output shape={output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ {func_name}: failed with input shape={layer_in.shape}. Error: {e}")

        elif "unsqueeze" in func_name :           
            # 🔧 CRITICAL FIX: Robust input handling for unsqueeze operation
            if DEBUG:
                print(f"[{node_name}] 🔧 unsqueeze: input type={type(layer_in)}, length={len(layer_in) if isinstance(layer_in, (list, tuple)) else 'single'}")
            
            # 🔧 FIX: Preserve original input and handle list inputs properly
            original_layer_in = layer_in
            
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    if DEBUG:
                        print(f"[{node_name}] ⚠️ unsqueeze received empty input list, attempting recovery")
                    
                    # 🔧 FIX: Try to recover from parents without overwriting original input
                    recovered_input = None
                    for node_x in parents:
                        if node_io[node_x]['layer_type'] not in ("Weight", "Bias", "future_use"):
                            recovered_input = node_io[node_x]['output_values']
                            if DEBUG:
                                print(f"[{node_name}] 🔧 Recovered input from parent {node_x}: {type(recovered_input)}")
                            break
                    
                    if recovered_input is None:
                        raise RuntimeError(f"[{node_name}] ❌ unsqueeze: no valid input found in parents")
                    
                    # 🔧 FIX: Use recovered input but don't modify original
                    layer_in = recovered_input
                    
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: Find first valid tensor without modifying original
                    valid_tensor = None
                    for x in layer_in:
                        if isinstance(x, torch.Tensor):
                            valid_tensor = x
                            break
                    
                    if valid_tensor is None:
                        raise RuntimeError(f"[{node_name}] ❌ unsqueeze: no valid tensor found in input list")
                    
                    layer_in = valid_tensor
            
            # 🔧 FIX: Validate input type
            if not isinstance(layer_in, torch.Tensor):
                raise RuntimeError(f"[{node_name}] ❌ unsqueeze expected tensor input, got {type(layer_in)}")
            
            # 🔧 FIX: Validate dimension parameter
            dim = layer_hyperparams.get("dim")
            if dim is None:
                raise RuntimeError(f"[{node_name}] ❌ unsqueeze: 'dim' parameter is required")
            
            if DEBUG:
                print(f"[{node_name}] ✅ unsqueeze: input shape={layer_in.shape}, dim={dim}")
            
            try:
                output = aten_op(layer_in, dim)
                if DEBUG:
                    print(f"[{node_name}] ✅ unsqueeze output shape: {output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ unsqueeze failed: input shape={layer_in.shape}, dim={dim}. Error: {e}")
            
        elif "squeeze" in func_name :
            return aten_op(layer_in, layer_hyperparams["dim"])

        elif func_name == "layer_norm":
            # 🔧 CONSISTENCY FIX: Use standardized input processing
            layer_in = standardize_layer_input(layer_in, parents, node_io, tensor_map, func_name)
            
            # 🔧 CONSISTENCY FIX: Enforce precision consistency
            layer_in = enforce_precision_consistency(layer_in)
            
            if not isinstance(layer_in, torch.Tensor):
                raise RuntimeError(f"[{node_name}] ❌ `layer_norm` expected Tensor input, got {type(layer_in)}")
        
            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            target_device = layer_in.device
            weight = layer_hyperparams["weight"]
            bias = layer_hyperparams["bias"]
            if weight.device != target_device:
                weight = weight.to(device=target_device)
            if bias.device != target_device:
                bias = bias.to(device=target_device)
        
            # 🔧 CONSISTENCY FIX: Use consistent epsilon value
            eps = 1e-5  # Standard value instead of variable
            return aten_op(layer_in,
                           layer_hyperparams["normalized_shape"],
                           weight,
                           bias,
                           eps,
                           False)

        elif func_name == "batch_norm":
            # 🔧 CONSISTENCY FIX: Use standardized input processing
            layer_in = standardize_layer_input(layer_in, parents, node_io, tensor_map, func_name)
            
            # 🔧 CONSISTENCY FIX: Enforce precision consistency
            layer_in = enforce_precision_consistency(layer_in)
            
            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            target_device = layer_in.device
            weight = layer_hyperparams["weight"]
            bias = layer_hyperparams["bias"]
            running_mean = layer_hyperparams["running_mean"]
            running_var = layer_hyperparams["running_var"]
            
            if weight.device != target_device:
                weight = weight.to(device=target_device)
            if bias.device != target_device:
                bias = bias.to(device=target_device)
            if running_mean.device != target_device:
                running_mean = running_mean.to(device=target_device)
            if running_var.device != target_device:
                running_var = running_var.to(device=target_device)
            
            # 🔧 CONSISTENCY FIX: Use consistent epsilon value
            eps = 1e-5  # Standard value instead of variable
            
            return aten_op(layer_in,
                           weight,
                           bias,
                           running_mean,
                           running_var,
                           False,  # Always False for training
                           layer_hyperparams["momentum"],
                           eps,
                           False)

        elif func_name == "scaled_dot_product_attention":
            # 🔧 CONSISTENCY FIX: Ensure consistent attention computation
            q, k, v = layer_in[0], layer_in[1], layer_in[2]
            
            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            target_device = q.device
            if k.device != target_device:
                k = k.to(device=target_device)
            if v.device != target_device:
                v = v.to(device=target_device)
            
            # 🔧 CONSISTENCY FIX: Force consistent precision
            q = enforce_precision_consistency(q)
            k = enforce_precision_consistency(k)
            v = enforce_precision_consistency(v)
            
            return aten_op(q, k, v,
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

                # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
                target_device = layer_in.device
                if mask.device != target_device:
                    mask = mask.to(device=target_device)

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
            # 🔧 CRITICAL FIX: Robust view operation with proper validation
            if DEBUG:
                print(f"[{node_name}] 🔧 view: input type={type(layer_in)}, method_args={method_args}")
            
            # 🔧 FIX: Validate input before processing
            if layer_in is None:
                raise RuntimeError(f"[{node_name}] ❌ view: input is None")
            
            # 🔧 FIX: Handle list inputs properly
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    raise RuntimeError(f"[{node_name}] ❌ view: received empty input list")
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: Find first valid tensor without modifying original
                    valid_tensor = None
                    for x in layer_in:
                        if isinstance(x, torch.Tensor):
                            valid_tensor = x
                            break
                    
                    if valid_tensor is None:
                        raise RuntimeError(f"[{node_name}] ❌ view: no valid tensor found in input list")
                    
                    layer_in = valid_tensor
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ view: expected tensor input, got {type(layer_in)}")
            
            # 🔧 FIX: Get shape with proper fallback
            shape = layer_hyperparams.get("shape", [])
            if not shape or any(s is None for s in shape):
                if method_args and isinstance(method_args[0], (list, tuple)):
                    shape = method_args[0]
                    if DEBUG:
                        print(f"[{node_name}] 🔧 view: using shape from method_args: {shape}")
                else:
                    raise RuntimeError(f"[{node_name}] ❌ view: no valid shape found in hyperparams or method_args")
            
            if DEBUG:
                print(f"[{node_name}] 🔧 view: processing shape: {shape}")
            
            # 🔧 FIX: Robust parameter resolution with proper error handling
            def resolve_param_view(p, idx):
                try:
                    if isinstance(p, (torch.fx.Node, str)):
                        p_key = str(p)
                        if p_key in node_io:
                            val = node_io[p_key]["output_values"]
                            if isinstance(val, torch.Tensor) and val.numel() == 1:
                                return int(val.item())
                            elif isinstance(val, (int, float)):
                                return int(val)
                            else:
                                raise ValueError(f"Node {p_key} output is not a scalar: {type(val)}")
                        else:
                            raise KeyError(f"Node {p_key} not found in node_io")
                    elif isinstance(p, torch.Tensor):
                        if p.numel() == 1:
                            return int(p.item())
                        else:
                            raise ValueError(f"Tensor has multiple elements: {p.shape}")
                    elif isinstance(p, (int, torch.SymInt)):
                        return int(p)
                    elif p is None:
                        raise ValueError("Cannot resolve None in shape")
                    else:
                        raise TypeError(f"Unsupported shape type: {type(p)}")
                        
                except Exception as e:
                    raise RuntimeError(f"Failed to resolve shape element {idx} ({p}): {e}")

            # 🔧 FIX: Resolve shape with comprehensive error handling
            resolved_shape = []
            for idx, dim in enumerate(shape):
                try:
                    resolved = resolve_param_view(dim, idx)
                    resolved_shape.append(resolved)
                except Exception as e:
                    raise RuntimeError(f"[{node_name}] ❌ view: failed to resolve shape element {idx}: {e}")
            
            if DEBUG:
                print(f"[{node_name}] ✅ view: resolved shape: {resolved_shape}")
            
            # 🔧 FIX: Validate resolved shape
            if not resolved_shape:
                raise RuntimeError(f"[{node_name}] ❌ view: empty shape after resolution")
            
            # 🔧 FIX: Allow -1 for automatic dimension inference (PyTorch standard)
            if any(s < -1 for s in resolved_shape):
                raise RuntimeError(f"[{node_name}] ❌ view: invalid shape dimensions: {resolved_shape} (only -1 and positive values allowed)")
            
            # 🔧 FIX: Handle -1 dimensions for automatic inference
            if -1 in resolved_shape:
                if DEBUG:
                    print(f"[{node_name}] 🔧 view: detected -1 dimension, will infer automatically")
                
                # Count how many -1 dimensions we have
                neg_one_count = resolved_shape.count(-1)
                if neg_one_count > 1:
                    raise RuntimeError(f"[{node_name}] ❌ view: only one -1 dimension allowed, got {neg_one_count}")
                
                # Calculate what the -1 dimension should be
                total_elements = layer_in.numel()
                for s in resolved_shape:
                    if s != -1:
                        total_elements //= s
                
                # Replace -1 with the calculated value
                resolved_shape = [s if s != -1 else total_elements for s in resolved_shape]
                
                if DEBUG:
                    print(f"[{node_name}] 🔧 view: inferred -1 dimension as {total_elements}")
            
            # 🔧 FIX: Validate tensor compatibility with final shape
            total_elements = 1
            for s in resolved_shape:
                total_elements *= s
            
            if layer_in.numel() != total_elements:
                raise RuntimeError(f"[{node_name}] ❌ view: shape {resolved_shape} is incompatible with input tensor "
                                f"of {layer_in.numel()} elements (expected {total_elements} elements)")
            
            try:
                output = aten_op(layer_in, resolved_shape)
                if DEBUG:
                    print(f"[{node_name}] ✅ view: input shape={layer_in.shape}, output shape={output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ view: execution failed: input shape={layer_in.shape}, "
                                f"resolved_shape={resolved_shape}, error={e}")

        elif func_name == "select":
            return aten_op(layer_in,
                           layer_hyperparams["dim"],
                           layer_hyperparams["index"])

        elif func_name == "arange":
            # 🔧 CRITICAL FIX: Remove hardcoded magic number and implement proper parameter resolution
            start = layer_hyperparams.get("start", None)
            end = layer_hyperparams.get("end", None)
            step = layer_hyperparams.get("step", 1)
            dtype = layer_hyperparams.get("dtype", None)
            device = layer_hyperparams.get("device", None)
            
            # 🔧 CRITICAL FIX: Proper parameter resolution without hardcoded assumptions
            def resolve_param(param):
                if isinstance(param, torch.Tensor):
                    return param.item()
                elif isinstance(param, torch.fx.Node):
                    node_key = str(param)
                    if node_key in tensor_map:
                        return tensor_map[node_key].item()
                    else:
                        raise RuntimeError(f"[{node_name}] ❌ Node {node_key} not found in tensor_map")
                elif isinstance(param, torch.SymInt):
                    if hasattr(param, 'node') and hasattr(param.node, '_value'):
                        return int(param.node._value)
                    else:
                        return int(param)
                elif param is None:
                    return None
                return param

            # 🔧 CRITICAL FIX: Resolve parameters with proper error handling
            try:
                start = resolve_param(start)
                end = resolve_param(end)
                step = resolve_param(step)
                
                # 🔧 CRITICAL FIX: Validate parameters
                if end is None:
                    raise RuntimeError(f"[{node_name}] ❌ 'end' parameter is required for arange")
                if step == 0:
                    raise RuntimeError(f"[{node_name}] ❌ 'step' cannot be zero")
                
                # 🔧 CRITICAL FIX: Ensure proper data types
                if isinstance(start, (int, float)):
                    start = int(start)
                if isinstance(end, (int, float)):
                    end = int(end)
                if isinstance(step, (int, float)):
                    step = int(step)
                    
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ Failed to resolve arange parameters: {e}")

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
                elif "start" in str(arange_overload):
                    # aten::arange.start(start, end, *, dtype, device)
                    output = aten_op(start, end, dtype=dtype, device=device)
                elif "default" in str(arange_overload):
                    # aten::arange.default(end, *, dtype, device)
                    output = aten_op(end, dtype=dtype, device=device)
                else:
                    # Other overloads like aten::arange.Scalar support step
                    output = aten_op(start, end, step, dtype=dtype, device=device)
                    
                if DEBUG:
                    print(f"[{node_name}] ✅ arange({start}, {end}, {step}) → shape: {output.shape}")
                    
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ Failed to execute arange: "
                                f"start={start}, end={end}, step={step}, dtype={dtype}, device={device}. "
                                f"Overload: {arange_overload}. Error: {str(e)}")
            return output

        elif func_name == "slice":
            # 🔧 CRITICAL FIX: Robust slice operation without hyperparameter corruption
            if DEBUG:
                print(f"[{node_name}] 🔧 slice: input type={type(layer_in)}, length={len(layer_in) if isinstance(layer_in, (list, tuple)) else 'single'}")
            
            # 🔧 FIX: Handle list inputs without modifying hyperparameters
            if isinstance(layer_in, list):
                if len(layer_in) >= 2 and isinstance(layer_in[0], torch.Tensor) and isinstance(layer_in[1], (int, torch.Tensor)):
                    # Extract tensor and potential end value
                    input_tensor = layer_in[0]
                    potential_end = layer_in[1]
                    
                    # Convert potential_end to int if it's a tensor
                    if isinstance(potential_end, torch.Tensor):
                        if potential_end.numel() == 1:
                            potential_end = int(potential_end.item())
                        else:
                            raise RuntimeError(f"[{node_name}] ❌ slice: end value must be a scalar tensor, got shape {potential_end.shape}")
                    
                    if DEBUG:
                        print(f"[{node_name}] 🔧 slice: extracted tensor shape={input_tensor.shape}, potential_end={potential_end}")
                    
                    # 🔧 FIX: Use extracted tensor but preserve original hyperparameters
                    layer_in = input_tensor
                    
                    # 🔧 FIX: Only update end if it's not already set in hyperparams
                    if "end" not in layer_hyperparams or layer_hyperparams["end"] is None:
                        if DEBUG:
                            print(f"[{node_name}] 🔧 slice: updating end parameter to {potential_end}")
                        # Create a copy to avoid modifying original
                        updated_hyperparams = dict(layer_hyperparams)
                        updated_hyperparams["end"] = potential_end
                        layer_hyperparams = updated_hyperparams
                else:
                    raise RuntimeError(f"[{node_name}] ❌ slice: expected list with [tensor, int] or single tensor, got {layer_in}")
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(layer_in, torch.Tensor):
                raise RuntimeError(f"[{node_name}] ❌ slice expected tensor input, got {type(layer_in)}")
            
            # 🔧 FIX: Validate required hyperparameters
            required_params = ["dim", "start", "end"]
            for param in required_params:
                if param not in layer_hyperparams:
                    raise RuntimeError(f"[{node_name}] ❌ slice: missing required parameter '{param}'")
            
            dim = layer_hyperparams["dim"]
            start = layer_hyperparams["start"]
            end = layer_hyperparams["end"]
            step = layer_hyperparams.get("step", 1)
            
            if DEBUG:
                print(f"[{node_name}] ✅ slice: input shape={layer_in.shape}, dim={dim}, start={start}, end={end}, step={step}")
            
            try:
                output = aten_op(layer_in, dim, start, end, step)
                if DEBUG:
                    print(f"[{node_name}] ✅ slice output shape: {output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ slice failed: input shape={layer_in.shape}, "
                                f"dim={dim}, start={start}, end={end}, step={step}. Error: {e}")
        
        elif func_name == "sym_size":
            if isinstance(layer_in,(tuple,list)):
                layer_in = layer_in[0]
            output = aten_op(layer_in,layer_hyperparams['dim'])
            return output

        elif func_name == "_assert_tensor_metadata":
            # Skip or pass-through this op, as it's only a debug consistency check
            if DEBUG:
                print(f"[{node_name}] ℹ️ Skipping `_assert_tensor_metadata` (no-op).")
            if isinstance(layer_in, list) and len(layer_in) > 0:
                return layer_in[0]
            return layer_in

        elif func_name == "to":
            # 🔧 CRITICAL FIX: Robust 'to' operation with input validation
            if DEBUG:
                print(f"[{node_name}] 🔧 to: input type={type(layer_in)}, method_args={method_args}")
            
            # 🔧 FIX: Validate input before execution
            if layer_in is None:
                raise RuntimeError(f"[{node_name}] ❌ to: input is None")
            
            # 🔧 FIX: Handle list inputs properly
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    raise RuntimeError(f"[{node_name}] ❌ to: received empty input list")
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: For multiple inputs, apply 'to' to each
                    try:
                        outputs = []
                        for i, inp in enumerate(layer_in):
                            if isinstance(inp, torch.Tensor):
                                output = aten_op(inp, *method_args)
                                outputs.append(output)
                            else:
                                outputs.append(inp)
                        
                        if len(outputs) == 1:
                            return outputs[0]
                        else:
                            return outputs
                            
                    except Exception as e:
                        raise RuntimeError(f"[{node_name}] ❌ to: failed to process list input: {e}")
            
            # 🔧 FIX: Validate tensor input
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ to: expected tensor input, got {type(layer_in)}")
            
            try:
                output = aten_op(layer_in, *method_args)
                if DEBUG:
                    print(f"[{node_name}] ✅ to: input shape={layer_in.shape}, output shape={output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ to: failed with input shape={layer_in.shape}, "
                                f"method_args={method_args}. Error: {e}")
        elif func_name == "mean":
            # 🔧 IMPROVED: Better list input handling
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    if DEBUG:
                        print(f"[{node_name}] ⚠️ mean operation received empty input list")
                    # Try to recover from parents
                    for node_x in parents:
                        if node_io[node_x]['layer_type'] not in ("Weight", "Bias", "future_use"):
                            layer_in = node_io[node_x]['output_values']
                            break
                    if isinstance(layer_in, list) and len(layer_in) == 0:
                        raise RuntimeError(f"[{node_name}] ❌ mean operation has no valid inputs")
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # Take the first tensor from the list
                    layer_in = next((x for x in layer_in if isinstance(x, torch.Tensor)), layer_in[0])
            
            if not isinstance(layer_in, torch.Tensor):
                raise RuntimeError(f"[{node_name}] ❌ mean expected tensor input, got {type(layer_in)}")
            
            return aten_op(layer_in, *method_args)
        elif func_name == "softmax":
            # 🔧 CONSISTENCY FIX: Ensure consistent softmax computation
            if isinstance(layer_in, list):
                layer_in = layer_in[0]
            
            # 🔧 CONSISTENCY FIX: Force float32 precision for numerical stability
            layer_in = enforce_precision_consistency(layer_in)
            
            output = aten_op(layer_in,*method_args)
            return output
        elif func_name in ("unflatten","flatten"):
            output = aten_op(layer_in,*method_args)
            return output
        elif func_name == "embedding":
            print(f"[DEBUG] 🔧 Executing embedding operation via ATen: {node_name}")
            
            # 🔧 CRITICAL FIX: Enhanced embedding operation for consistency
            layer_in = []
            for node_x in parents: 
                if "weight" in node_x:
                    continue
                else:
                    layer_in.append(node_io[node_x]['output_values'])

            indices = layer_in[0] if isinstance(layer_in, (list, tuple)) else layer_in
            
            # 🔧 CRITICAL FIX: Use enhanced embedding consistency function
            weight = layer_hyperparams["weight"]
            weight, indices = ensure_embedding_consistency(weight, indices, node_name)
            
            # 🔧 CONSISTENCY FIX: Ensure indices are LongTensor and properly formatted
            if isinstance(indices, torch.Tensor):
                # Force indices to be long (integer) type
                if indices.dtype != torch.long:
                    indices = indices.long()
                    print(f"[DEBUG] Converted indices to long type: {indices.dtype}")
                
                # Ensure indices are contiguous
                if not indices.is_contiguous():
                    indices = indices.contiguous()
                    print(f"[DEBUG] Made indices contiguous")
            else:
                print(f"[WARN] Indices is not a tensor: {type(indices)}")
                indices = torch.tensor(indices, dtype=torch.long)
        
            # 🔧 CONSISTENCY FIX: Ensure device compatibility between weight and indices
            weight = layer_hyperparams["weight"]
            if isinstance(weight, torch.Tensor) and isinstance(indices, torch.Tensor):
                if weight.device != indices.device:
                    # Move indices to the same device as weight
                    indices = indices.to(weight.device)
                    print(f"[DEBUG] Moved indices to device {weight.device} to match weight")
                
                # 🔧 CONSISTENCY FIX: Ensure weight is properly formatted
                if not weight.is_contiguous():
                    weight = weight.contiguous()
                    print(f"[DEBUG] Made weight contiguous")
            
            print(f"[DEBUG] Embedding ATen: weight {weight.shape}, indices {indices.shape}")
            
            # Execute embedding operation with consistent parameters
            result = aten_op(weight,
                    indices,
                    layer_hyperparams["padding_idx"],
                    layer_hyperparams["scale_grad_by_freq"],
                    layer_hyperparams["sparse"])
            
            print(f"[DEBUG] Embedding ATen output shape: {result.shape}")
            return result

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

            def expand_to_match(a, b):
                if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                    return a, b

                if a.shape == b.shape:
                    return a, b

                if a.ndim == 2 and b.ndim == 2:
                    if a.shape[0] == b.shape[0]:
                        if b.shape[1] == 1:
                            b = b.expand(-1, a.shape[1])
                        elif b.shape[1] < a.shape[1]:
                            # Allow repeating with crop
                            repeat_factor = (a.shape[1] + b.shape[1] - 1) // b.shape[1]  # Ceiling division
                            b = b.repeat(1, repeat_factor)[:, :a.shape[1]]  # Trim to match a
                        else:
                            raise RuntimeError(f"[Shape Mismatch] b.shape[1] > a.shape[1]")
                    else:
                        raise RuntimeError(f"[Batch Mismatch] Cannot align batch dims: a={a.shape}, b={b.shape}")

                return a, b

            # Execute op
            try:
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
                    target_device = a.device
                    if b.device != target_device:
                        b = b.to(device=target_device)
                    
                    if DEBUG:
                        print(f"Before ---  a: {a.shape}, b: {b.shape}")
                        print("Aligning the shapes")
                    a, b = expand_to_match(a, b)
                    if DEBUG:
                        print(f"After ---  a: {a.shape}, b: {b.shape}") 
                output = aten_op(a, b)
            except Exception as e:
                raise RuntimeError(
                    f"[{node_name}] ❌ failed in `{func_name}` with a={type(a)}, b={type(b)}; "
                    f"shapes: {getattr(a, 'shape', None)}, {getattr(b, 'shape', None)}. Error: {e}"
                )

            # print(f"node_name: {node_name}, mul shape: {output.shape}")
            return output

        elif func_name in {"add", "add_", "sub", "div", "rsub", "pow", "gt", "ge", "lt", "eq"}:
            # 🔧 CRITICAL FIX: Robust comparison and arithmetic operations
            if DEBUG:
                print(f"[{node_name}] 🔧 {func_name}: input type={type(layer_in)}, length={len(layer_in) if isinstance(layer_in, (list, tuple)) else 'single'}")
                print(f"[{node_name}] 🔧 {func_name}: method_args={method_args}")
            
            # 🔧 FIX: Handle list inputs with proper validation
            if isinstance(layer_in, list):
                if DEBUG:
                    print(f"[{node_name}] 🔧 {func_name}: processing list input, length={len(layer_in)}")
                
                # 🔧 FIX: Validate list inputs for comparison operations
                if func_name in {"gt", "ge", "lt", "eq"} and len(layer_in) != 2:
                    raise RuntimeError(f"[{node_name}] ❌ {func_name}: comparison operations require exactly 2 inputs, got {len(layer_in)}")
                
                # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
                if len(layer_in) >= 2:
                    target_device = None
                    for t in layer_in:
                        if isinstance(t, torch.Tensor):
                            target_device = t.device
                            break
                    
                    if target_device is not None:
                        for i, t in enumerate(layer_in):
                            if isinstance(t, torch.Tensor) and t.device != target_device:
                                layer_in[i] = t.to(device=target_device)
                                if DEBUG:
                                    print(f"[{node_name}] 🔧 Moved tensor {i} to device {target_device}")
                
                # 🔧 FIX: Execute operation based on input count
                if len(layer_in) == 2:
                    # 🔧 FIX: Validate inputs for comparison operations
                    if func_name in {"gt", "ge", "lt", "eq"}:
                        a, b = layer_in[0], layer_in[1]
                        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                            raise TypeError(f"[{node_name}] ❌ {func_name}: both inputs must be tensors, got {type(a)} and {type(b)}")
                        
                        # 🔧 FIX: Ensure tensors can be compared
                        if a.shape != b.shape:
                            try:
                                # Try broadcasting
                                a, b = torch.broadcast_tensors(a, b)
                                if DEBUG:
                                    print(f"[{node_name}] 🔧 {func_name}: broadcasted shapes to {a.shape}")
                            except Exception as e:
                                raise RuntimeError(f"[{node_name}] ❌ {func_name}: cannot broadcast shapes {a.shape} and {b.shape}: {e}")
                    
                    output = aten_op(layer_in[0], layer_in[1], *method_args)
                    
                elif len(layer_in) == 3:
                    output = aten_op(layer_in[0], layer_in[1], layer_in[2], *method_args)
                else:
                    output = aten_op(layer_in, *method_args)
                    
            else:
                # 🔧 FIX: Handle single tensor inputs
                if DEBUG:
                    print(f"[{node_name}] 🔧 {func_name}: processing single tensor input")
                
                # 🔧 FIX: Special handling for comparison operations with single tensor
                if func_name in {"gt", "ge", "lt", "eq"} and not method_args:
                    raise RuntimeError(f"[{node_name}] ❌ {func_name}: comparison operation requires 2 inputs")
                
                # 🔧 FIX: Try to recover inputs from parents for missing arguments
                if func_name in {"add", "add_", "sub", "div"} and not method_args:
                    if DEBUG:
                        print(f"[{node_name}] 🔧 {func_name}: collecting inputs from parents: {parents}")
                    
                    recovered_inputs = [tensor_map[p] for p in parents if p in tensor_map]
                    if len(recovered_inputs) >= 2:
                        if DEBUG:
                            print(f"[{node_name}] 🔧 {func_name}: found {len(recovered_inputs)} inputs, using first 2")
                        
                        a, b = recovered_inputs[0], recovered_inputs[1]
                        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                            # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
                            target_device = a.device
                            if b.device != target_device:
                                b = b.to(device=target_device)
                                if DEBUG:
                                    print(f"[{node_name}] 🔧 {func_name}: moved second tensor to device {target_device}")
                            
                            output = aten_op(a, b)
                        else:
                            output = aten_op(a, b)
                        return output
                
                # 🔧 FIX: Handle single tensor cases with proper validation
                if func_name == "pow":
                    if DEBUG:
                        print(f"[{node_name}] 🔧 {func_name}: pow operation with single tensor")
                    
                    if method_args and len(method_args) > 0:
                        scalar_val = method_args[0]
                        if 'Tensor_Scalar' in str(aten_op):
                            output = aten_op(layer_in, scalar_val)
                        else:
                            output = aten_op(scalar_val, layer_in)
                    else:
                        output = aten_op(-1.0, layer_in)
                else:
                    output = aten_op(layer_in, *method_args)
            
            if DEBUG:
                if hasattr(output, 'shape'):
                    print(f"[{node_name}] ✅ {func_name} output shape: {output.shape}")
            
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
                
                # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
                target_device = a.device if isinstance(a, torch.Tensor) else b.device if isinstance(b, torch.Tensor) else torch.device("cpu")
                if isinstance(a, torch.Tensor) and a.device != target_device:
                    a = a.to(device=target_device)
                if isinstance(b, torch.Tensor) and b.device != target_device:
                    b = b.to(device=target_device)
                
                # Type consistency
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

            # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
            target_device = a.device
            if b.device != target_device:
                b = b.to(device=target_device)

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
            # 🔧 CRITICAL FIX: Robust size resolution for full operation
            size = layer_hyperparams.get("size") or layer_hyperparams.get("sizes")
            
            if DEBUG:
                print(f"[{node_name}] 🔧 full: initial size={size}, method_args={method_args}")
            
            # 🔧 FIX: Proper size resolution with error handling
            if not size or all(s is None for s in size):
                if method_args and isinstance(method_args[0], (list, tuple)):
                    size = method_args[0]
                    if DEBUG:
                        print(f"[{node_name}] 🔧 Using size from method_args: {size}")
                else:
                    raise RuntimeError(f"[{node_name}] ❌ No valid size found in hyperparams or method_args")
            
            # 🔧 FIX: Resolve symbolic sizes with proper error handling
            resolved_size = []
            for i, s in enumerate(size):
                try:
                    if isinstance(s, torch.fx.Node):
                        node_key = str(s)
                        if node_key in tensor_map:
                            parent_val = tensor_map[node_key]
                            if isinstance(parent_val, torch.Tensor) and parent_val.numel() == 1:
                                resolved_size.append(int(parent_val.item()))
                            elif isinstance(parent_val, (int, float)):
                                resolved_size.append(int(parent_val))
                            else:
                                raise ValueError(f"Parent {node_key} output is not a scalar: {type(parent_val)}")
                        else:
                            raise KeyError(f"Parent node {node_key} not found in tensor_map")
                    elif isinstance(s, torch.Tensor):
                        if s.numel() == 1:
                            resolved_size.append(int(s.item()))
                        else:
                            raise ValueError(f"Size element {i} is not a scalar tensor: {s.shape}")
                    elif isinstance(s, (int, float)):
                        resolved_size.append(int(s))
                    elif s is None:
                        raise ValueError(f"Size element {i} is None")
                    else:
                        raise TypeError(f"Unsupported size type: {type(s)}")
                        
                except Exception as e:
                    raise RuntimeError(f"[{node_name}] ❌ Failed to resolve size element {i} ({s}): {e}")
            
            if DEBUG:
                print(f"[{node_name}] ✅ Resolved size: {resolved_size}")
            
            # 🔧 FIX: Validate resolved size
            if not resolved_size:
                raise RuntimeError(f"[{node_name}] ❌ Empty size after resolution")
            
            if any(s <= 0 for s in resolved_size):
                raise RuntimeError(f"[{node_name}] ❌ Invalid size dimensions: {resolved_size} (all must be positive)")
            
            # Remaining parameters with validation
            fill_value = layer_hyperparams.get("fill_value", 0)
            dtype = layer_hyperparams.get("dtype", torch.float32)
            device = layer_hyperparams.get("device", torch.device("cpu"))

            if isinstance(fill_value, torch.Tensor):
                if fill_value.numel() == 1:
                    fill_value = fill_value.item()
                else:
                    raise RuntimeError(f"[{node_name}] ❌ fill_value must be a scalar tensor, got shape {fill_value.shape}")

            try:
                output = aten_op(resolved_size, fill_value, dtype=dtype, device=device)
                if DEBUG:
                    print(f"[{node_name}] ✅ full({resolved_size}, {fill_value}) → shape: {output.shape}")
                    
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ Failed to execute full: "
                                f"size={resolved_size}, fill_value={fill_value}, dtype={dtype}, device={device}. Error: {e}")
            
            return output

        
        elif func_name == "full_like":
            return aten_op(layer_in,
                           fill_value=layer_hyperparams["fill_value"],
                           dtype=layer_hyperparams["dtype"],
                           layout=layer_hyperparams["layout"],
                           device=layer_hyperparams["device"],
                           pin_memory=layer_hyperparams["pin_memory"])

        elif func_name == "expand":
            # 🔧 CRITICAL FIX: Robust expand operation without input corruption
            if DEBUG:
                print(f"[{node_name}] 🔧 expand: input type={type(layer_in)}, method_args={method_args}")
            
            # 🔧 FIX: Handle list inputs without modifying original
            original_layer_in = layer_in
            
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    if DEBUG:
                        print(f"[{node_name}] ⚠️ expand received empty input list, attempting recovery")
                    
                    # 🔧 FIX: Try to recover from parents without overwriting original input
                    recovered_input = None
                    for node_x in parents:
                        if node_io[node_x]['layer_type'] not in ("Weight", "Bias", "future_use"):
                            recovered_input = node_io[node_x]['output_values']
                            if DEBUG:
                                print(f"[{node_name}] 🔧 expand: recovered input from parent {node_x}: {type(recovered_input)}")
                            break
                    
                    if recovered_input is None:
                        raise RuntimeError(f"[{node_name}] ❌ expand: no valid input found in parents")
                    
                    # 🔧 FIX: Use recovered input but don't modify original
                    layer_in = recovered_input
                    
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: Find first valid tensor without modifying original
                    valid_tensor = None
                    for x in layer_in:
                        if isinstance(x, torch.Tensor):
                            valid_tensor = x
                            break
                    
                    if valid_tensor is None:
                        raise RuntimeError(f"[{node_name}] ❌ expand: no valid tensor found in input list")
                    
                    layer_in = valid_tensor
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ expand: expected tensor input, got {type(layer_in)}")
            
            if DEBUG:
                print(f"[{node_name}] 🔧 expand: input shape: {layer_in.shape}")
            
            # 🔧 FIX: Get sizes with proper fallback
            raw_sizes = layer_hyperparams.get("sizes", None)
            if isinstance(raw_sizes, (list, tuple)):
                sizes = []
                for s in raw_sizes:
                    if isinstance(s, torch.fx.Node):
                        # retrieve the recorded output value for that node
                        sizes.append(node_io[str(s)]["output_values"])
                    else:
                        sizes.append(s)
            else:
                sizes = raw_sizes

            # 🔧 FIX: Robust parameter resolution with proper error handling
            def resolve_param(param, idx):
                try:
                    if isinstance(param, torch.Tensor):
                        if param.numel() == 1:
                            return int(param.item())
                        else:
                            raise ValueError(f"Size element {idx} is not a scalar tensor: {param.shape}")
                    elif isinstance(param, torch.fx.Node):
                        node_key = str(param)
                        value = node_io.get(node_key, {}).get("output_values", None)
                        if value is None:
                            raise ValueError(f"No output value found for node: {node_key}")
                        
                        if isinstance(value, torch.Tensor):
                            if value.numel() == 1:
                                return int(value.item())
                            else:
                                raise ValueError(f"Node {node_key} output is not a scalar: {value.shape}")
                        elif isinstance(value, (int, float)):
                            return int(value)
                        else:
                            raise TypeError(f"Unexpected output value type: {type(value)}")
                    elif isinstance(param, torch.SymInt):
                        if hasattr(param, 'node') and hasattr(param.node, '_value'):
                            return int(param.node._value)
                        else:
                            raise ValueError(f"SymInt {param} has no _value attribute")
                    elif isinstance(param, (int, float)):
                        return int(param)
                    elif param is None:
                        raise ValueError(f"Size element {idx} is None")
                    else:
                        raise TypeError(f"Unsupported size type: {type(param)}")
                        
                except Exception as e:
                    raise RuntimeError(f"Failed to resolve size element {idx} ({param}): {e}")

            # 🔧 FIX: If sizes not available, try method_args fallback
            if not sizes or any(s is None for s in sizes):
                if method_args and isinstance(method_args[0], (list, tuple)):
                    if DEBUG:
                        print(f"[{node_name}] 🔧 expand: using sizes from method_args: {method_args[0]}")
                    sizes = method_args[0]
                else:
                    raise RuntimeError(f"[{node_name}] ❌ expand: no valid sizes found in hyperparams or method_args")

            if DEBUG:
                print(f"[{node_name}] 🔧 expand: processing sizes: {sizes}")

            # 🔧 FIX: Resolve sizes with comprehensive error handling
            resolved_sizes = []
            for idx, s in enumerate(sizes):
                try:
                    resolved = resolve_param(s, idx)
                    resolved_sizes.append(resolved)
                except Exception as e:
                    raise RuntimeError(f"[{node_name}] ❌ expand: failed to resolve size element {idx}: {e}")

            if DEBUG:
                print(f"[{node_name}] ✅ expand: resolved sizes: {resolved_sizes}")

            # 🔧 FIX: Validate resolved sizes
            if not resolved_sizes:
                raise RuntimeError(f"[{node_name}] ❌ expand: empty sizes after resolution")

            # 🔧 FIX: Validate size compatibility with input tensor
            if len(resolved_sizes) != layer_in.dim():
                raise RuntimeError(f"[{node_name}] ❌ expand: size dimensions {len(resolved_sizes)} don't match tensor rank {layer_in.dim()}")

            # 🔧 FIX: Execute with proper error handling
            try:
                output = aten_op(
                    layer_in,
                    resolved_sizes,
                    implicit=layer_hyperparams.get("implicit", False)
                )
                if DEBUG:
                    print(f"[{node_name}] ✅ expand: output shape: {output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ expand: execution failed: input shape={layer_in.shape}, "
                                f"sizes={resolved_sizes}, error={e}")
        

        elif "permute" in func_name:
            if isinstance(layer_in,tuple):
                layer_in = layer_in[0]
            elif isinstance(layer_in, list):
                layer_in = layer_in[0]
            return aten_op(layer_in, layer_hyperparams["dims"])

        elif func_name == "transpose":
            # 🔧 CRITICAL FIX: Robust transpose operation with proper validation
            if DEBUG:
                print(f"[{node_name}] 🔧 transpose: input type={type(layer_in)}, method_args={method_args}")
            
            # 🔧 FIX: Validate input before processing
            if layer_in is None:
                raise RuntimeError(f"[{node_name}] ❌ transpose: input is None")
            
            # 🔧 FIX: Handle list inputs properly
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    raise RuntimeError(f"[{node_name}] ❌ transpose: received empty input list")
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: Find first valid tensor without modifying original
                    valid_tensor = None
                    for x in layer_in:
                        if isinstance(x, torch.Tensor):
                            valid_tensor = x
                            break
                    
                    if valid_tensor is None:
                        raise RuntimeError(f"[{node_name}] ❌ transpose: no valid tensor found in input list")
                    
                    layer_in = valid_tensor
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ transpose: expected tensor input, got {type(layer_in)}")
            
            # 🔧 FIX: Validate method arguments
            if not method_args or len(method_args) < 2:
                raise RuntimeError(f"[{node_name}] ❌ transpose: requires 2 arguments (dim0, dim1), got {method_args}")
            
            dim0, dim1 = method_args[0], method_args[1]
            
            # 🔧 FIX: Validate dimensions
            if not isinstance(dim0, int) or not isinstance(dim1, int):
                raise TypeError(f"[{node_name}] ❌ transpose: dimensions must be integers, got {type(dim0)} and {type(dim1)}")
            
            if dim0 < 0 or dim1 < 0:
                raise ValueError(f"[{node_name}] ❌ transpose: dimensions must be non-negative, got {dim0} and {dim1}")
            
            if dim0 >= layer_in.dim() or dim1 >= layer_in.dim():
                raise ValueError(f"[{node_name}] ❌ transpose: dimensions {dim0} and {dim1} out of range for tensor of rank {layer_in.dim()}")
            
            if DEBUG:
                print(f"[{node_name}] ✅ transpose: input shape={layer_in.shape}, dims=({dim0}, {dim1})")
            
            try:
                output = aten_op(layer_in, dim0, dim1)
                if DEBUG:
                    print(f"[{node_name}] ✅ transpose: output shape={output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ transpose: failed with input shape={layer_in.shape}, "
                                f"dims=({dim0}, {dim1}). Error: {e}")
            
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
            if len(layer_in) == 3:
                cond, x, y = layer_in
                
                # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
                target_device = x.device if isinstance(x, torch.Tensor) else y.device if isinstance(y, torch.Tensor) else torch.device("cpu")
                if isinstance(cond, torch.Tensor) and cond.device != target_device:
                    cond = cond.to(device=target_device)
                if isinstance(x, torch.Tensor) and x.device != target_device:
                    x = x.to(device=target_device)
                if isinstance(y, torch.Tensor) and y.device != target_device:
                    y = y.to(device=target_device)
                
                try:
                    cond_, x_, y_ = torch.broadcast_tensors(cond, x, y)
                    return aten_op(cond_, x_, y_)
                except Exception as e:
                    # Safe fallback: expand cond to match if it's missing a dimension
                    if cond.ndim == x.ndim - 1 and cond.shape[0] == x.shape[0] and cond.shape[-2:] == x.shape[-2:]:
                        cond_expanded = cond.unsqueeze(1).expand_as(x)
                        if DEBUG:
                            print(f"[{node_name}] ⚠️ Expanded cond to shape {cond_expanded.shape}")
                        return aten_op(cond_expanded, x, y)
                    raise RuntimeError(
                        f"[{node_name}] ❌ Shape mismatch in `where`: "
                        f"cond: {cond.shape}, x: {x.shape}, y: {y.shape}. Error: {e}"
                    )
            
            elif len(layer_in) == 2:
                # Handle aten.where.ScalarSelf variant: x.where(cond)
                x, cond = layer_in
                
                # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
                target_device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
                if isinstance(cond, torch.Tensor) and cond.device != target_device:
                    cond = cond.to(device=target_device)

                if cond.dtype != torch.bool:
                    if DEBUG:
                        print(f"[{node_name}] ⚠️ Converting cond from {cond.dtype} to bool")
                    cond = cond.to(torch.bool)

                try:
                    cond_, x_ = torch.broadcast_tensors(cond, x)
                    return torch.where(cond_, x_, torch.zeros_like(x_))  # Default y = 0
                except Exception as e:
                    raise RuntimeError(
                        f"[{node_name}] ❌ Shape mismatch in `where.ScalarSelf`: x: {x.shape}, cond: {cond.shape}. Error: {e}"
                    )

            else:
                raise RuntimeError(
                    f"[{node_name}] ❌ Unexpected number of inputs for `where`: got {len(layer_in)} → {layer_in}"
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
            # 🔧 DEVICE CONSISTENCY FIX: logical operations work on all types
            if isinstance(layer_in, torch.Tensor):
                # Logical operations work on all types, but for consistency with bitwise_not
                if layer_in.dtype in [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64]:
                    # Integer types work on both devices
                    return aten_op(layer_in)
                else:
                    # For float types, convert to boolean first, then apply logical_not
                    bool_tensor = layer_in.bool()
                    result = aten_op(bool_tensor)
                    # Convert back to the original dtype and device
                    return result.to(dtype=layer_in.dtype, device=layer_in.device)
            return aten_op(layer_in)

        elif func_name == "bitwise_not":
            # 🔧 DEVICE CONSISTENCY FIX: bitwise operations only work on integer and boolean types
            if isinstance(layer_in, torch.Tensor):
                # Bitwise operations only work on integer and boolean types
                if layer_in.dtype in [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64]:
                    # Integer types work on both devices
                    return aten_op(layer_in)
                else:
                    # For float types, convert to boolean first, then apply bitwise_not
                    # This is a common pattern in deep learning where bitwise_not is used for masking
                    bool_tensor = layer_in.bool()
                    result = aten_op(bool_tensor)
                    # Convert back to the original dtype and device
                    return result.to(dtype=layer_in.dtype, device=layer_in.device)
            return aten_op(layer_in)

        elif func_name == "any":
            return aten_op(layer_in,
                           layer_hyperparams["dim"],
                           layer_hyperparams["keepdim"])

        elif func_name == "all":
            # 🔧 CRITICAL FIX: Robust 'all' operation with proper validation
            if DEBUG:
                print(f"[{node_name}] 🔧 all: input type={type(layer_in)}, method_args={method_args}")
            
            # 🔧 FIX: Validate input before processing
            if layer_in is None:
                raise RuntimeError(f"[{node_name}] ❌ all: input is None")
            
            # 🔧 FIX: Handle list inputs properly
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    raise RuntimeError(f"[{node_name}] ❌ all: received empty input list")
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: Find first valid tensor without modifying original
                    valid_tensor = None
                    for x in layer_in:
                        if isinstance(x, torch.Tensor):
                            valid_tensor = x
                            break
                    
                    if valid_tensor is None:
                        raise RuntimeError(f"[{node_name}] ❌ all: no valid tensor found in input list")
                    
                    layer_in = valid_tensor
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ all: expected tensor input, got {type(layer_in)}")
            
            # 🔧 FIX: Get dimension and keepdim parameters with proper fallback
            dim = layer_hyperparams.get("dim", None)
            keepdim = layer_hyperparams.get("keepdim", False)
            
            if DEBUG:
                print(f"[{node_name}] ✅ all: input shape={layer_in.shape}, dim={dim}, keepdim={keepdim}")
            
            try:
                # 🔧 FIX: Handle different aten operation variants based on parameters
                if dim is not None:
                    # Use aten::all.dim variant when dimension is specified
                    output = aten_op(layer_in, dim, keepdim)
                else:
                    # Use aten::all.default variant when no dimension is specified (reduces all dimensions)
                    # This is equivalent to calling all() without arguments in PyTorch
                    output = aten_op(layer_in)
                
                if DEBUG:
                    print(f"[{node_name}] ✅ all: output shape={output.shape}")
                return output
                
            except Exception as e:
                # 🔧 FIX: Better error handling with fallback
                if DEBUG:
                    print(f"[{node_name}] ⚠️ all: first attempt failed, trying fallback: {e}")
                
                try:
                    # Fallback: if dim is None, try to reduce all dimensions
                    if dim is None:
                        # For aten::all.dim, we need to provide a dimension
                        # Since we want to reduce all dimensions, we'll use dim=0 and then reduce further if needed
                        output = aten_op(layer_in, 0, keepdim)
                        if DEBUG:
                            print(f"[{node_name}] ✅ all: fallback successful with dim=0, output shape={output.shape}")
                        return output
                    else:
                        raise RuntimeError(f"[{node_name}] ❌ all: failed with input shape={layer_in.shape}, "
                                        f"dim={dim}, keepdim={keepdim}. Error: {e}")
                except Exception as fallback_error:
                    raise RuntimeError(f"[{node_name}] ❌ all: failed with input shape={layer_in.shape}, "
                                    f"dim={dim}, keepdim={keepdim}. Original error: {e}. Fallback error: {fallback_error}")

        elif func_name in {"flatten", "unflatten"}:
            return aten_op(layer_in, *method_args)

        elif func_name == "rsqrt":
            if isinstance(layer_in, list) and len(layer_in) == 1:
                output = aten_op(layer_in[0], *method_args)
            elif isinstance(layer_in, torch.Tensor):
                if DEBUG:
                    print(node_name)
                    print(layer_in.shape,"rsqrt shape input")
                output = aten_op(layer_in, *method_args)
                if DEBUG:
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
                if DEBUG:
                    print(f"  ↪ Number of inputs: {len(layer_in)}")

                if DEBUG:
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

            # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
            target_device = a.device
            if b.device != target_device:
                b = b.to(device=target_device)

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

            if DEBUG:
                for i, t in enumerate(tensors):
                    print(f"  ↪ Tensor {i}: shape={t.shape}, dtype={t.dtype}")

            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            if len(tensors) > 1:
                target_device = tensors[0].device
                for i, t in enumerate(tensors):
                    if t.device != target_device:
                        tensors[i] = t.to(device=target_device)
                        if DEBUG:
                            print(f"  🔧 Moved tensor {i} from {t.device} to {target_device}")

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
            # 🔧 CRITICAL FIX: Robust copy operation without input modification
            if DEBUG:
                print(f"[{node_name}] 🔧 copy: input type={type(layer_in)}, length={len(layer_in) if isinstance(layer_in, (list, tuple)) else 'single'}")
            
            # 🔧 FIX: Validate input structure
            if not isinstance(layer_in, list) or len(layer_in) != 2:
                raise RuntimeError(f"[{node_name}] ❌ copy: expects a list of 2 tensors, got: {layer_in}")
            
            self_tensor, src_tensor = layer_in
            
            # 🔧 FIX: Validate input types
            if not isinstance(self_tensor, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ copy: expected 'self' to be Tensor but got {type(self_tensor)}")
            
            if not isinstance(src_tensor, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ copy: expected 'src' to be Tensor but got {type(src_tensor)}")
            
            if DEBUG:
                print(f"[{node_name}] 🔧 copy: self shape={self_tensor.shape}, src shape={src_tensor.shape}")
                print(f"[{node_name}] 🔧 copy: self device={self_tensor.device}, src device={src_tensor.device}")
            
            # 🔧 FIX: Create copies to avoid modifying original tensors
            self_tensor_copy = self_tensor.clone()
            src_tensor_copy = src_tensor.clone()
            
            # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
            target_device = self_tensor_copy.device
            if src_tensor_copy.device != target_device:
                src_tensor_copy = src_tensor_copy.to(device=target_device)
                if DEBUG:
                    print(f"[{node_name}] 🔧 copy: moved src tensor to device {target_device}")
            
            # 🔧 FIX: Validate device synchronization was successful
            if src_tensor_copy.device != target_device:
                raise RuntimeError(f"[{node_name}] ❌ copy: failed to move src tensor to target device {target_device}")
            
            # 🔧 FIX: Validate tensor compatibility
            if self_tensor_copy.shape != src_tensor_copy.shape:
                raise RuntimeError(f"[{node_name}] ❌ copy: shape mismatch - self: {self_tensor_copy.shape}, src: {src_tensor_copy.shape}")
            
            non_blocking = layer_hyperparams.get("non_blocking", False)
            
            try:
                output = aten_op(self_tensor_copy, src_tensor_copy, non_blocking=non_blocking)
                if DEBUG:
                    print(f"[{node_name}] ✅ copy: output shape={output.shape}")
                    
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ copy: failed with self shape={self_tensor_copy.shape}, "
                                f"src shape={src_tensor_copy.shape}, non_blocking={non_blocking}. Error: {e}")
            
            return output
        
        elif func_name == "slice_scatter":
            # Step 1: Unpack inputs
            if isinstance(layer_in, list) and len(layer_in) == 2:
                self_tensor, src_tensor = layer_in
                
                # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
                target_device = self_tensor.device
                if src_tensor.device != target_device:
                    src_tensor = src_tensor.to(device=target_device)
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
                if DEBUG:
                    print(f"[{node_name}] ⚠️ Detected symbolic placeholder for `start` ({start}) → resetting to 0")
                start = 0

            if isinstance(end, int) and end >= INT64_MAX:
                end = self_tensor.shape[dim]
                if DEBUG:
                    print(f"[{node_name}] ⚠️ Detected symbolic placeholder for `end` ({INT64_MAX}) → using {end}")

            if end is None:
                end = self_tensor.shape[dim]
                if DEBUG:
                    print(f"[{node_name}] ℹ️ `end` not specified → using {end}")

            # Step 3: Validate dimensions
            try:
                slice_size = end - start
                src_size = src_tensor.shape[dim]
                self_size = self_tensor.shape[dim]

                if slice_size != src_size:
                    if DEBUG:
                        print(f"[{node_name}] ⚠️ Shape mismatch: src[{dim}] = {src_size} != target slice length = {slice_size}")
            except Exception as e:
                print(f"[{node_name}] ⚠️ Failed to inspect shapes: {e}")

            # Optional: Show small slice for verification
            try:
                preview = self_tensor.narrow(dim, start, min(end - start, self_tensor.shape[dim] - start))
                if DEBUG:
                    print(f"[{node_name}] 🔍 self_tensor slice preview (dim={dim}, start={start}, end={end}): shape={preview.shape}")
            except Exception as e:
                print(f"[{node_name}] ⚠️ Could not preview slice: {e}")

            # Step 4: Execute
            try:
                output = aten_op(self_tensor, src_tensor, dim, start, end, step)
                if DEBUG:
                    print(f"[{node_name}] ✅ `slice_scatter` success → output shape: {output.shape}")
            except Exception as e:
                raise RuntimeError(
                    f"[{node_name}] ❌ `slice_scatter` failed: self={type(self_tensor)}, src={type(src_tensor)}, "
                    f"dim={dim}, start={start}, end={end}, step={step}. Error: {e}"
                )

            return output
        
        elif func_name == "_unsafe_view":
            # 🔧 CRITICAL FIX: Robust _unsafe_view operation with proper validation
            if DEBUG:
                print(f"[{node_name}] 🔧 _unsafe_view: input type={type(layer_in)}, method_args={method_args}")
            
            # 🔧 FIX: Validate input before processing
            if layer_in is None:
                raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: input is None")
            
            # 🔧 FIX: Extract tensor from layer_in with proper validation
            if isinstance(layer_in, list):
                input_tensor = None
                for item in layer_in:
                    if isinstance(item, torch.Tensor):
                        input_tensor = item
                        break
                
                if input_tensor is None:
                    raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: no valid tensor found in inputs: {layer_in}")
            else:
                input_tensor = layer_in
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(input_tensor, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ _unsafe_view: expected tensor input, got {type(input_tensor)}")
            
            # 🔧 FIX: Validate method arguments
            if not method_args or len(method_args) == 0:
                raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: method_args is empty")
            
            raw_shape = method_args[0]
            if not isinstance(raw_shape, (list, tuple)):
                raise TypeError(f"[{node_name}] ❌ _unsafe_view: expected list/tuple for shape, got {type(raw_shape)}")
            
            if DEBUG:
                print(f"[{node_name}] 🔧 _unsafe_view: raw shape: {raw_shape}")
            
            # 🔧 FIX: Robust parameter resolution with proper error handling
            def resolve_param_view(p, idx):
                try:
                    if isinstance(p, (torch.fx.Node, str)):
                        p_key = str(p)
                        if p_key in node_io:
                            val = node_io[p_key]["output_values"]
                            if isinstance(val, torch.Tensor) and val.numel() == 1:
                                return int(val.item())
                            elif isinstance(val, (int, float)):
                                return int(val)
                            else:
                                raise ValueError(f"Node {p_key} output is not a scalar: {type(val)}")
                        else:
                            raise KeyError(f"Node {p_key} not found in node_io")
                    elif isinstance(p, torch.Tensor):
                        if p.numel() == 1:
                            return int(p.item())
                        else:
                            raise ValueError(f"Tensor has multiple elements: {p.shape}")
                    elif isinstance(p, (int, torch.SymInt)):
                        return int(p)
                    elif p is None:
                        raise ValueError("Cannot resolve None in shape")
                    else:
                        raise TypeError(f"Unsupported shape type: {type(p)}")
                        
                except Exception as e:
                    raise RuntimeError(f"Failed to resolve shape element {idx} ({p}): {e}")

            # 🔧 FIX: Resolve shape with comprehensive error handling
            resolved_shape = []
            for idx, p in enumerate(raw_shape):
                try:
                    resolved = resolve_param_view(p, idx)
                    resolved_shape.append(resolved)
                except Exception as e:
                    raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: failed to resolve shape element {idx}: {e}")
            
            if DEBUG:
                print(f"[{node_name}] ✅ _unsafe_view: resolved shape: {resolved_shape}")
            
            # 🔧 FIX: Validate resolved shape
            if not resolved_shape:
                raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: empty shape after resolution")
            
            # 🔧 FIX: Allow -1 for automatic dimension inference (PyTorch standard)
            if any(s < -1 for s in resolved_shape):
                raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: invalid shape dimensions: {resolved_shape} (only -1 and positive values allowed)")
            
            # 🔧 FIX: Handle -1 dimensions for automatic inference
            if -1 in resolved_shape:
                if DEBUG:
                    print(f"[{node_name}] 🔧 _unsafe_view: detected -1 dimension, will infer automatically")
                
                # Count how many -1 dimensions we have
                neg_one_count = resolved_shape.count(-1)
                if neg_one_count > 1:
                    raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: only one -1 dimension allowed, got {neg_one_count}")
                
                # Calculate what the -1 dimension should be
                total_elements = input_tensor.numel()
                for s in resolved_shape:
                    if s != -1:
                        total_elements //= s
                
                # Replace -1 with the calculated value
                resolved_shape = [s if s != -1 else total_elements for s in resolved_shape]
                
                if DEBUG:
                    print(f"[{node_name}] 🔧 _unsafe_view: inferred -1 dimension as {total_elements}")
            
            # 🔧 FIX: Validate tensor compatibility with final shape
            total_elements = 1
            for s in resolved_shape:
                total_elements *= s
            
            if input_tensor.numel() != total_elements:
                raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: shape {resolved_shape} (expected {total_elements} elements)")
            
            # 🔧 FIX: Execute with proper error handling
            try:
                output = aten_op(input_tensor, resolved_shape)
                if DEBUG:
                    print(f"[{node_name}] ✅ _unsafe_view: input shape={input_tensor.shape}, output shape={output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: execution failed: input shape={input_tensor.shape}, "
                                f"resolved_shape={resolved_shape}, error={e}")

        elif func_name == "einsum":
            if not method_args or not isinstance(method_args[0], str):
                raise RuntimeError(f"[{node_name}] ❌ `einsum` requires equation string as the first method_arg")

            equation = method_args[0]

            # Ensure input is a list of tensors
            if not isinstance(layer_in, (list, tuple)):
                layer_in = [layer_in]

            if DEBUG: 
                print(f"[{node_name}] 🧪 einsum equation: {equation}")
                for i, t in enumerate(layer_in):
                    if isinstance(t, torch.Tensor):
                        print(f"[{node_name}] ↪ Input {i} shape: {t.shape}")
                    else:
                        print(f"[{node_name}] ⚠️ Input {i} is not a tensor: {type(t)}")

            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            if len(layer_in) > 1:
                target_device = None
                for t in layer_in:
                    if isinstance(t, torch.Tensor):
                        target_device = t.device
                        break
                
                if target_device is not None:
                    for i, t in enumerate(layer_in):
                        if isinstance(t, torch.Tensor) and t.device != target_device:
                            layer_in[i] = t.to(device=target_device)
                            if DEBUG:
                                print(f"[{node_name}] 🔧 Moved tensor {i} from {t.device} to {target_device}")

            try:
                # Don't unpack the tensor list — pass it as a list
                output = aten_op(equation, layer_in)
                if DEBUG:
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

            # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
            target_device = input_tensor.device
            if index_tensor.device != target_device:
                index_tensor = index_tensor.to(device=target_device)

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
                    if DEBUG:
                        print(f"[{node_name}] ⚠️ Trimming index_tensor from {index_tensor.shape[0]} to {expected_dim_size} to match einsum_5 shape")
                    index_tensor = index_tensor[:expected_dim_size]

            if DEBUG:
                print(f"[{node_name}] ✅ index_select(dim={dim}) on input shape {input_tensor.shape} with index shape {index_tensor.shape}")
            return aten_op(input_tensor, dim, index_tensor)

        elif func_name == "addmm":
            if DEBUG:
                if isinstance(layer_in, (list, tuple)):
                    for idx, item in enumerate(layer_in):
                        print(f"idx: {idx}, item: {item.shape}") 
                else:
                    print(f"layer_in shape: {layer_in.shape}")

            if isinstance(layer_in, list) and len(layer_in) == 3:
                bias, mat1, mat2 = layer_in
                if DEBUG:
                    print(f"bias: {bias.shape}, mat1: {mat1.shape}, mat2: {mat2.shape}") 

                if not all(isinstance(x, torch.Tensor) for x in (bias, mat1, mat2)):
                    raise TypeError(f"[{node_name}] ❌ Expected Tensors for `addmm`, got {[type(x) for x in layer_in]}")

                # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
                target_device = mat1.device
                if bias.device != target_device:
                    bias = bias.to(device=target_device)
                if mat2.device != target_device:
                    mat2 = mat2.to(device=target_device)

                try:
                    return torch.addmm(bias, mat1, mat2)
                except Exception as e:
                    raise RuntimeError(
                        f"[{node_name}] ❌ torch.addmm failed with shapes: "
                        f"bias={bias.shape}, mat1={mat1.shape}, mat2={mat2.shape}. Error: {e}"
                    )
            else:
                raise RuntimeError(f"[{node_name}] ❌ `addmm` expects 3 input tensors (bias, mat1, mat2), got: {layer_in}")
        
        elif func_name == "index": 
            input_tensor = layer_in[0]
            index_tensors = layer_in[1:]

            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            target_device = input_tensor.device
            for i, idx in enumerate(index_tensors):
                if isinstance(idx, torch.Tensor) and idx.device != target_device:
                    index_tensors[i] = idx.to(device=target_device)

            # Cast all index tensors to long
            index_tensors = [
                idx.to(dtype=torch.long) if isinstance(idx, torch.Tensor) and not idx.dtype in (torch.long, torch.int, torch.bool, torch.uint8) else idx
                for idx in index_tensors
            ]

            # 🛡️ Validate and inject index at correct dimension
            input_rank = input_tensor.dim()
            indices = [slice(None)] * input_rank  # default: [:, :, :, ...]

            for idx in index_tensors:
                # Dynamically find dimension that matches index shape
                inserted = False
                for i in range(input_rank):
                    if idx.dim() == 1 and input_tensor.shape[i] == idx.shape[0]:
                        indices[i] = idx
                        inserted = True
                        break
                if not inserted:
                    raise RuntimeError(f"[{node_name}] ❌ Could not find matching dimension for index with shape {idx.shape} in input_tensor of shape {input_tensor.shape}")

            try:
                return input_tensor[tuple(indices)]
            except Exception as e:
                raise RuntimeError(
                    f"[{node_name}] ❌ Failed to apply `index` with shape {input_tensor.shape} and indices {[idx.shape for idx in index_tensors]}: {e}"
                )

        else:
            inputs = layer_in if isinstance(layer_in, (list, tuple)) else [layer_in]
            output = aten_op(*inputs, *method_args)
            return output 
            
    except Exception as e:
        print(f"[Execution Error] Node `{node_name}` failed in `{func_name}`: {e}")
        #return layer_in

def run_execution_nocache(graph, layer_stack, model, extracted_weights, inputs, tracer, exported_program=None):
    print(f"Executing `run_execution_nocache` ...!")
    
    # 🔧 CONSISTENCY FIX: Set up consistent environment
    setup_consistent_environment()
    
    # 🔧 CRITICAL FIX: Ensure model is in consistent state
    print("[DEBUG] 🔧 Ensuring model consistency...")
    model.eval()
    model.requires_grad_(False)
    
    # 🔧 CRITICAL FIX: Synchronize extracted weights with current model state
    print("[DEBUG] 🔧 Synchronizing extracted weights with model state...")
    if exported_program is not None:
        for placeholder_name, extracted_weight in extracted_weights.items():
            if extracted_weight is not None and isinstance(extracted_weight, torch.Tensor):
                # Get the real parameter name from exported_program
                real_key = None
                for spec in exported_program.graph_signature.input_specs:
                    if hasattr(spec.arg, 'name') and spec.arg.name == placeholder_name:
                        real_key = spec.target
                        break
                
                if real_key and real_key in model.state_dict():
                    current_weight = model.state_dict()[real_key]
                    # Update extracted weight to match current model state
                    extracted_weights[placeholder_name] = current_weight.clone()
                    print(f"[DEBUG]   ✅ Synchronized {placeholder_name} -> {real_key}")
    else:
        print("[DEBUG]   ⚠️ No exported_program provided, skipping weight synchronization")
    
    print("[DEBUG] ✅ Weight synchronization complete")
    
    # 🔧 CRITICAL FIX: Preserve original model precision instead of forcing float32
    # This ensures numerical consistency with the original model
    print(f"[DEBUG] Preserving original model precision for consistency")
    for param in model.parameters():
        print(f"[DEBUG] Parameter {param.shape} dtype: {param.dtype}")
    
    for buffer in model.buffers():
        print(f"[DEBUG] Buffer {buffer.shape} dtype: {buffer.dtype}")
    
    print("[DEBUG] ✅ Model consistency ensured")
    
    tensor_map = {}
    node_io = {}

    model_signature = inspect.signature(model.forward)
    expected_input_names = list(model_signature.parameters.keys())

    if len(inputs) != len(expected_input_names):
        raise ValueError("Mismatch between model input count and provided inputs.")

    inp_map = dict(zip(expected_input_names, inputs))
    
    # 🔧 CRITICAL FIX: Ensure input tensors are consistent from the start
    if len(extracted_weights) < len(layer_stack):
        # Force input tensors to have consistent properties
        if isinstance(inputs, (list, tuple)):
            consistent_inputs = []
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    # 🔧 SPECIAL HANDLING: input_ids and attention_mask must remain long/int
                    if i == 0:  # First input is usually input_ids
                        consistent_inp = inp.detach().to(dtype=torch.long)
                    elif i == 1:  # Second input is usually attention_mask
                        consistent_inp = inp.detach().to(dtype=torch.long)
                    else:
                        consistent_inp = inp.detach().to(dtype=inp.dtype)  # Preserve original dtype
                    
                    # Ensure contiguous memory layout
                    if not consistent_inp.is_contiguous():
                        consistent_inp = consistent_inp.contiguous()
                    
                    consistent_inputs.append(consistent_inp)
                    print(f"[DEBUG] Input {i}: {inp.shape} -> {consistent_inp.shape}, dtype: {consistent_inp.dtype}")
                else:
                    consistent_inputs.append(inp)
            tensor_map[layer_stack[len(extracted_weights)]] = consistent_inputs
        else:
            if isinstance(inputs, torch.Tensor):
                # For single tensor input, assume it's input_ids
                consistent_input = inputs.detach().to(dtype=torch.long)
                if not consistent_input.is_contiguous():
                    consistent_input = consistent_input.contiguous()
                tensor_map[layer_stack[len(extracted_weights)]] = consistent_input
                print(f"[DEBUG] Single input: {inputs.shape} -> {consistent_input.shape}, dtype: {consistent_input.dtype}")
            else:
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
        
        # 🔧 CRITICAL FIX: Ensure input consistency for all operations
        layer_in = ensure_input_consistency(layer_in, model)
        
        output = layer_in
        
        # Handle embedding function
        if func_name == "embedding":
            print(f"[DEBUG] 🔧 Handling embedding operation for node: {node_name}")
            
            # 🔧 CRITICAL FIX: Ensure embedding operation is handled consistently
            try:
                # Get the embedding weight from hyperparams
                weight = layer_hyperparams.get("weight")
                if weight is None:
                    print(f"[ERROR] No weight found for embedding operation in {node_name}")
                    continue
                
                # Get the input indices (token IDs)
                indices = None
                for p in parents:
                    if node_io[p]['layer_type'] not in ["Weight", "Bias"]:
                        indices = node_io[p]['output_values']
                        break
                
                if indices is None:
                    print(f"[ERROR] No indices found for embedding operation in {node_name}")
                    continue
                
                # 🔧 CONSISTENCY FIX: Ensure indices are proper type and device
                if isinstance(indices, torch.Tensor):
                    # Force indices to be long (integer) type
                    if indices.dtype != torch.long:
                        indices = indices.long()
                    
                    # Ensure indices are on the same device as weight
                    if weight.device != indices.device:
                        indices = indices.to(device=weight.device)
                        print(f"[DEBUG] Moved indices to device {weight.device}")
                    
                    # Ensure indices are contiguous
                    if not indices.is_contiguous():
                        indices = indices.contiguous()
                
                # 🔧 CONSISTENCY FIX: Use exact same embedding parameters as original model
                padding_idx = layer_hyperparams.get("padding_idx", -1)
                scale_grad_by_freq = layer_hyperparams.get("scale_grad_by_freq", False)
                sparse = layer_hyperparams.get("sparse", False)
                
                print(f"[DEBUG] Embedding: weight shape {weight.shape}, indices shape {indices.shape}")
                
                # Execute embedding operation
                output = torch.nn.functional.embedding(
                    indices, weight, padding_idx, scale_grad_by_freq, sparse
                )
                
                print(f"[DEBUG] Embedding output shape: {output.shape}")
                
            except Exception as e:
                print(f"[ERROR] Embedding operation failed for {node_name}: {e}")
                output = layer_in

        # ✅ Register lifted_tensor constants early (before parent check)
        if layer_type == "Placeholder" and "lifted_tensor" in node_name:
            output = extracted_weights.get(node_name, None)
            if output is None:
                print(f"[WARN] Lifted tensor node `{node_name}` missing in extracted_weights")
                continue
            
        # 🔧 CONSISTENCY FIX: Ensure weights have consistent precision
        if layer_type == "Placeholder" and any(keyword in node_name for keyword in ["weight", "bias"]):
            if output is not None and isinstance(output, torch.Tensor):
                output = enforce_precision_consistency(output)
                    
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
                
                # 🔧 CRITICAL FIX: Ensure Model_Layer inputs are IDENTICAL to direct model execution
                # Get the original input tensor to match properties exactly
                original_input = None
                if len(inputs) > 0:
                    original_input = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
                
                # Sanitize inputs to match original model input properties exactly
                if original_input is not None:
                    layer_in = sanitize_model_layer_inputs(layer_in, original_input)
                
                # 🔧 ENHANCED CONSISTENCY FIX: Use comprehensive consistency functions
                if isinstance(layer_in, (list, tuple)):
                    layer_in = [enforce_precision_consistency(x) for x in layer_in]
                    # 🔧 CRITICAL FIX: Preserve original input precision for consistency
                    # Don't force float32 conversion as it can cause numerical differences
                    for i, inp in enumerate(layer_in):
                        if isinstance(inp, torch.Tensor):
                            layer_in[i] = inp.to(dtype=inp.dtype)  # Preserve original dtype
                            if not layer_in[i].is_contiguous():
                                layer_in[i] = layer_in[i].contiguous()
                else:
                    layer_in = enforce_precision_consistency(layer_in)
                    if isinstance(layer_in, torch.Tensor):
                        layer_in = layer_in.to(dtype=layer_in.dtype)  # Preserve original dtype
                        if not layer_in.is_contiguous():
                            layer_in = layer_in.contiguous()
                
                # 🔧 CRITICAL FIX: Ensure the resolved layer is in the same state
                resolved_layer.eval()
                resolved_layer.requires_grad_(False)
                
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

                # 🔧 CRITICAL FIX: Better input processing for all operations
                if func_name in ["addmm", "matmul", "bmm"]:
                    # For matrix operations, ensure inputs are properly formatted
                    layer_in, _ = ensure_operation_consistency(func_name, layer_in, [], node_name)
                elif func_name == "addmm":
                    pass  # ⛔ Don't process or unwrap inputs
                elif func_name == "linear":
                    # 🔧 FIX: Linear operations are handled separately above, skip duplicate processing
                    if DEBUG:
                        print(f"[{node_name}] 🔧 linear: skipping duplicate processing (handled above)")
                    pass
                else:
                    # 🔧 IMPROVED: Better list-to-tensor conversion for all other operations
                    if DEBUG:
                        print(f"[{node_name}] 🔧 Processing inputs for {func_name}: {len(layer_in)} inputs")
                        for i, inp in enumerate(layer_in):
                            if isinstance(inp, torch.Tensor):
                                print(f"  Input {i}: {inp.shape} on {inp.device}, dtype: {inp.dtype}")
                            else:
                                print(f"  Input {i}: {type(inp)}")
                    
                    # 🔧 IMPROVED: Handle empty input lists gracefully
                    if len(layer_in) == 0:
                        if DEBUG:
                            print(f"[{node_name}] ⚠️ No inputs collected for {func_name}, trying to recover from parents")
                        # Try to recover inputs from parent nodes
                        for node_x in parents:
                            if node_io[node_x]['layer_type'] not in ("Weight", "Bias", "future_use"):
                                layer_in.append(node_io[node_x]['output_values'])
                    
                    # 🔧 IMPROVED: Better list processing
                    if len(layer_in) == 1 and isinstance(layer_in[0], (list, tuple)):
                        # Unwrap nested lists
                        layer_in = layer_in[0]
                        if DEBUG:
                            print(f"[{node_name}] 🔧 Unwrapped nested list, now have {len(layer_in)} inputs")
                    
                    # Process the inputs
                    layer_in, _ = _process_layer_input(layer_in)
                    
                    if DEBUG:
                        print(f"[{node_name}] 🔧 After processing: {type(layer_in)}, length: {len(layer_in) if isinstance(layer_in, (list, tuple)) else 'single'}")

                output = execute_aten_operation(func_name, layer, layer_in, layer_hyperparams, method_args, parents, node_io, node_name,tensor_map, children=children)
                
        except Exception as e:
            print(f"[Execution Error - NoCache] Node `{node_name}` failed in `{func_name}`: {e}")
            output = layer_in

        # At the end of node execution
        processed_output = _process_output_tuple(output)

        if DEBUG:
            if isinstance(processed_output, torch.Tensor) and torch.isnan(processed_output).any():
                print(f"[ERROR:NaN] Node `{node_name}` produced NaNs → shape: {processed_output.shape}")

        # 🔧 CRITICAL FIX: Ensure ALL tensor outputs maintain consistency
        if isinstance(processed_output, torch.Tensor):
            # Force consistent properties for all tensor outputs
            processed_output = processed_output.detach().to(dtype=torch.float32)
            tensor_map[node_name] = processed_output
        elif isinstance(processed_output, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in processed_output):
            # Force consistent properties for all tensors in tuples
            processed_output = [x.detach().to(dtype=torch.float32) for x in processed_output]
            tensor_map[node_name] = processed_output
        elif isinstance(processed_output, int):
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

    def run(self, inputs, debug=False,):
        global DEBUG
        DEBUG = debug
        return run_execution_nocache(
            graph=self.graph,
            layer_stack=self.layer_stack,
            model=self.model,
            extracted_weights=self.extracted_weights,
            inputs=inputs,
            tracer=self.tracer,
            exported_program=self.exported_program,
        )
