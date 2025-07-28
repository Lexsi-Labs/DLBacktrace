import ast  # Needed to safely parse stringified lists
import gc  # For memory cleanup
import numpy as np
import torch
from tqdm import tqdm
from dl_backtrace.pytorch_backtrace.dlbacktrace.utils import default as UD
from dl_backtrace.pytorch_backtrace.dlbacktrace.utils import default_v2 as UD2  # Import launch functions

# Toggle debug prints
DEBUG = False  # Changed to False by default for performance
def log(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

# Memory pool for reusing arrays
_memory_pool = {}
_pool_lock = False

def get_pooled_array(shape, dtype=np.float32):
    """Get a reusable array from memory pool to reduce allocations."""
    global _memory_pool, _pool_lock
    if _pool_lock:
        return np.empty(shape, dtype=dtype)
    
    key = (shape, dtype)
    if key in _memory_pool:
        arr = _memory_pool.pop(key)
        arr.fill(0)  # Clear previous data
        return arr
    return np.empty(shape, dtype=dtype)

def return_pooled_array(arr):
    """Return array to memory pool for reuse."""
    global _memory_pool, _pool_lock
    if _pool_lock or arr.size > 1e7:  # Don't pool very large arrays
        return
    
    key = (arr.shape, arr.dtype)
    if len(_memory_pool) < 50:  # Limit pool size
        _memory_pool[key] = arr

def clear_memory_pool():
    """Clear the memory pool and force garbage collection."""
    global _memory_pool
    _memory_pool.clear()
    gc.collect()

def tensor_to_numpy(x, copy=False):
    """Convert Tensor, scalar, list/tuple, or ndarray to a NumPy array with memory-efficient float32."""
    if isinstance(x, np.ndarray):
        if x.dtype == np.float64:
            # Convert to float32 in-place if possible, otherwise copy
            return x.astype(np.float32, copy=False) if not copy else x.astype(np.float32)
        return x.copy() if copy else x
    
    if isinstance(x, torch.Tensor):
        # Use float32 for memory efficiency and avoid unnecessary copies
        if x.is_cuda:
            x = x.cpu()
        
        # Try to avoid copy if tensor is already float32 and contiguous
        if x.dtype == torch.float32 and x.is_contiguous():
            result = x.detach().numpy()
            return result.copy() if copy else result
        else:
            return x.detach().to(torch.float32).numpy()
    
    if isinstance(x, (int, float)):
        return np.array(x, dtype=np.float32)
    
    if isinstance(x, (list, tuple)):
        # Process in batches for large lists
        if len(x) > 100:
            arrs = []
            for i in range(0, len(x), 100):
                batch = x[i:i+100]
                batch_arrs = [tensor_to_numpy(xi, copy=False) for xi in batch]
                arrs.extend(batch_arrs)
        else:
            arrs = [tensor_to_numpy(xi, copy=False) for xi in x]
        
        # Ensure consistent dtype
        arrs = [arr.astype(np.float32, copy=False) if arr.dtype == np.float64 else arr for arr in arrs]
        
        try:
            return np.stack(arrs)
        except Exception:
            return np.array(arrs, dtype=object)
    
    raise TypeError(f"Cannot convert type {type(x)} to numpy")

def process_input_for_eval(X):
    """Unwrap single-element or multi-element lists/tuples and convert to NumPy with memory optimization."""
    # Handle lists/tuples
    if isinstance(X, (list, tuple)):
        X = X[0] if len(X) == 1 else X[0]  # Assume first element is the actual input tensor

    # Convert torch.Tensor to NumPy with memory-efficient float32
    if isinstance(X, torch.Tensor):
        if X.is_cuda:
            X = X.cpu()
        
        # Avoid copy if already float32 and contiguous
        if X.dtype == torch.float32 and X.is_contiguous():
            return X.detach().numpy()
        else:
            return X.detach().to(torch.float32).numpy()

    # Already a NumPy array - ensure float32 for memory efficiency
    if isinstance(X, np.ndarray):
        return X.astype(np.float32, copy=False) if X.dtype == np.float64 else X

    raise TypeError(f"Cannot process input of type {type(X)}")

def get_relevance_from_child(child, parent, all_wt, node_io):
    """
    Return the slice of child's relevance that belongs to parent.
    If child's relevance is a list, pick the slot matching parent.
    """
    if child not in all_wt:
        if DEBUG:
            log(f"get_rel: no relevance for child '{child}'")
        return None
    
    r = all_wt[child]
    if isinstance(r, list):
        sources = node_io[child]["input_sources"]
        try:
            idx = sources.index(parent)
        except ValueError:
            if DEBUG:
                log(f"get_rel: parent '{parent}' not in sources of '{child}': {sources}")
            return None
        
        if idx >= len(r):
            if DEBUG:
                log(f"get_rel: index {idx} out of range for relevance list of '{child}'")
            return None
        
        if DEBUG:
            log(f"get_rel: child='{child}', parent='{parent}', idx={idx}, slice.shape={r[idx].shape}")
        return r[idx]
    
    if DEBUG:
        log(f"get_rel: child='{child}' single-output, shape={r.shape}")
    return r

def fast_align_relevance(r, target_shape):
    """
    Fast relevance alignment with minimal memory allocations.
    Handles most common cases efficiently.
    """
    # Early return if shapes already match
    if r.shape == target_shape:
        return r
    
    # Use float32 consistently
    if r.dtype != np.float32:
        r = r.astype(np.float32, copy=False)
    
    original_total = np.sum(r, dtype=np.float64)  # Use float64 for sum to avoid overflow
    
    # Handle simple cases first
    if r.size == np.prod(target_shape):
        result = r.reshape(target_shape)
        return result
    
    # Collapse extra leading dimensions efficiently
    while r.ndim > len(target_shape) and r.shape[0] == 1:
        r = r.squeeze(0)
    
    while r.ndim > len(target_shape):
        r = np.linalg.norm(r, axis=0, keepdims=False)
    
    # Handle dimension mismatches
    if r.ndim < len(target_shape):
        # Add missing dimensions
        missing_dims = len(target_shape) - r.ndim
        new_shape = r.shape + (1,) * missing_dims
        r = r.reshape(new_shape)
    
    # Handle size mismatches per dimension
    for ax, (dr, dt) in enumerate(zip(r.shape, target_shape)):
        if dr == dt:
            continue
        
        if dt == 1 and dr > 1:
            # Sum to singleton dimension
            r = r.sum(axis=ax, keepdims=True)
        elif dr > dt > 1 and dr % dt == 0:
            # Clean reduction
            factor = dr // dt
            new_shape = list(r.shape)
            new_shape[ax] = dt
            new_shape.insert(ax + 1, factor)
            r = r.reshape(new_shape).sum(axis=ax + 1)
        elif dt > dr > 0 and dt % dr == 0:
            # Clean expansion
            factor = dt // dr
            r = np.repeat(r, factor, axis=ax)
    
    # Final broadcast attempt
    if r.shape != target_shape:
        try:
            r = np.broadcast_to(r, target_shape)
            # Make a copy since broadcast_to returns a view
            result = get_pooled_array(target_shape, np.float32)
            result[:] = r
            r = result
        except ValueError:
            # Fallback to more aggressive alignment
            return align_relevance_fallback(r, target_shape, original_total)
    
    # Preserve total relevance
    final_total = np.sum(r, dtype=np.float64)
    if not np.isclose(final_total, original_total, rtol=1e-3) and final_total > 0:
        scale = original_total / (final_total + 1e-8)
        r *= np.float32(scale)
    
    return r

def align_relevance_fallback(r, target_shape, original_total):
    """
    Fallback alignment for complex cases.
    This is the original complex alignment logic, used only when fast alignment fails.
    """
    # Use the original complex logic as fallback
    return align_relevance(r, target_shape, original_total)

def align_relevance(r, target_shape, total_relevance=None):
    """
    Align array `r` to `target_shape` with intelligent fallbacks.
    This is the original implementation kept as fallback.
    """
    # Use float32 to reduce memory usage by ~50%
    r = np.array(r, dtype=np.float32)
    original_total = np.sum(r, dtype=np.float32) if total_relevance is None else total_relevance

    # 0) collapse extra leading dims
    while r.ndim > len(target_shape):
        if DEBUG:
            log(f"[INFO] collapsing extra dim → shape: {r.shape[1:]}")
        r = np.linalg.norm(r, axis=0)

    # 1) sum-out axes where target_dim==1 but r_dim>1
    for ax, (dr, dt) in enumerate(zip(r.shape, target_shape)):
        if dt == 1 and dr != 1:
            if DEBUG:
                log(f"[INFO] summing axis {ax} ({dr}→1)")
            r = r.sum(axis=ax, keepdims=True)

    # 2) reduce axes where dr % dt == 0
    for ax, (dr, dt) in enumerate(zip(r.shape, target_shape)):
        if dt > 1 and dr != dt and dr % dt == 0 and np.prod(r.shape) >= np.prod(target_shape):
            factor = dr // dt
            new_shape = list(r.shape)
            new_shape[ax] = dt
            new_shape.insert(ax + 1, factor)
            try:
                r = r.reshape(new_shape).sum(axis=ax + 1)
                if DEBUG:
                    log(f"[INFO] reduced axis {ax} by factor {factor} → {r.shape}")
            except:
                pass

    # 2.5) expand axes where target_dim>dr and dt % dr == 0 (upsample)
    for ax, (dr, dt) in enumerate(zip(r.shape, target_shape)):
        if dt > dr and dt % dr == 0:
            factor = dt // dr
            r = np.repeat(r, factor, axis=ax)
            if DEBUG:
                log(f"[INFO] upsampled axis {ax} by factor {factor} → {r.shape}")

    # Skip the complex fallback logic for performance - use fast broadcast
    if r.shape != target_shape:
        try:
            r = np.broadcast_to(r, target_shape).copy()
            if DEBUG:
                log(f"[INFO] final broadcast to {target_shape}")
        except Exception as e:
            raise ValueError(
                f"🚫 Cannot align relevance: incompatible shape {r.shape} → {target_shape}: {e}"
            )

    # Normalize relevance total
    final_total = np.sum(r)
    if not np.isclose(final_total, original_total, rtol=1e-3) and final_total > 0:
        scale = original_total / (final_total + 1e-8)
        r *= scale
        if DEBUG:
            log(f"[INFO] normalized relevance total from {final_total:.4f} to {original_total:.4f}")

    return r

def assign_embedding_relevance(node, info, all_wt, node_io):
    """
    Embedding layers: align relevance to token IDs shape; robust to shape mismatches.
    Optimized for memory efficiency.
    """
    if DEBUG:
        log(f"assign_embedding_relevance: node='{node}'")

    # 1) Aggregate relevance from all children efficiently
    R_out = None
    total_relevance = 0.0
    
    for c in info["output_children"]:
        rc = get_relevance_from_child(c, node, all_wt, node_io)
        if rc is None:
            continue

        # Convert to numpy efficiently
        if isinstance(rc, list):
            summed = None
            for x in rc:
                arr = tensor_to_numpy(x, copy=False)
                summed = arr if summed is None else (summed + arr)
            rc = summed
        elif isinstance(rc, torch.Tensor):
            rc = tensor_to_numpy(rc, copy=False)
        else:
            rc = np.asarray(rc, dtype=np.float32)

        if R_out is None:
            R_out = rc.copy()
        else:
            R_out += rc
        
        total_relevance += np.sum(rc, dtype=np.float64)

    if R_out is None:
        return

    if DEBUG:
        log(f"R_out ---  relevance value: {total_relevance:.4f}, shape: {R_out.shape}")

    # 2) Extract token_ids efficiently
    token_ids = None
    for val in info.get("input_values", []):
        if isinstance(val, torch.Tensor) and not torch.is_floating_point(val):
            token_ids = tensor_to_numpy(val, copy=False)
            break

    # 3) Fallback: try to get token_ids from node_io directly
    if token_ids is None:
        for key in ['input_ids', 0, 'ids']:
            val = node_io.get(key)
            if isinstance(val, torch.Tensor) and not torch.is_floating_point(val):
                token_ids = tensor_to_numpy(val, copy=False)
                if DEBUG:
                    log(f"[{node}] ℹ️ token_ids recovered from node_io['{key}']")
                break

    # Skip for non-token embedding nodes
    if token_ids is None:
        if DEBUG:
            log(f"[{node}] ⏭️ Skipping: could not extract token_ids (likely not token embedding)")
        return

    target_shape = token_ids.shape

    # 4) Handle previous relevance efficiently
    prev = all_wt.get(node)
    if isinstance(prev, list):
        summed = None
        for x in prev:
            arr = tensor_to_numpy(x, copy=False)
            if arr.shape == R_out.shape:
                summed = arr if summed is None else (summed + arr)
            elif DEBUG:
                log(f"[{node}] ⚠️ Skipping incompatible prev shape: {arr.shape}")
        prev = summed
    elif isinstance(prev, torch.Tensor):
        prev = tensor_to_numpy(prev, copy=False)

    # 5) Fast path for matching shapes
    if prev is not None and R_out.shape == prev.shape == target_shape:
        if DEBUG:
            log(f"[{node}] ℹ️ Relevance already aligned, fast accumulation.")
        all_wt[node] = prev + R_out
        return

    # 6) Align dimensions using fast alignment
    if R_out.shape != target_shape:
        try:
            R_out = fast_align_relevance(R_out, target_shape)
            if DEBUG:
                log(f"[{node}] ℹ️ Aligned relevance to {R_out.shape}")
        except Exception as e:
            raise ValueError(f"[{node}] ❌ Failed to align relevance: {e}")

    # 7) Accumulate into all_wt
    if prev is None:
        all_wt[node] = R_out
    else:
        if prev.shape != R_out.shape:
            try:
                prev = fast_align_relevance(prev, R_out.shape)
            except Exception as e:
                raise ValueError(f"[{node}] ❌ Shape mismatch in accumulation: {e}")
        all_wt[node] = prev + R_out

    if DEBUG:
        log(f"[{node}] ✅ Stored relevance with shape={all_wt[node].shape}, "
            f"value={np.sum(all_wt[node]):.4f}")

def run_evaluation(
    node_io,
    activation_master,
    mode="default",
    start_wt=None,
    multiplier=100.0,
    scaler=1.0,
    thresholding=0.5,
    task="binary-classification",
    get_layer_implementation=None,
):
    """
    Perform LRP-style backtrace through node_io with memory optimization.
    Returns: dict node_name -> relevance (np.ndarray or list of).
    """
    all_wt = {}
    activation_dict = {}
    
    # Set default layer implementation function if not provided
    if get_layer_implementation is None:
        get_layer_implementation = lambda x: "original"
    
    # Clear memory pool at start
    clear_memory_pool()
    
    # --- Step 1: seed the 'output' node ---
    raw_out = node_io["output"]["input_values"]
    if isinstance(raw_out, (list, tuple)):
        raw_out = raw_out[0]
    out_np = tensor_to_numpy(raw_out, copy=False)
    
    if DEBUG:
        log(f"out_np: {out_np.shape}")
    
    seed = UD.calculate_start_wt(out_np, scaler=scaler, thresholding=thresholding, task=task)
    
    if DEBUG:
        log(f"seed: {np.sum(seed):.8f},  shape: {seed.shape}")
    
    seed_np = tensor_to_numpy(seed, copy=False)
    if seed_np.size == 0 or np.all(seed_np == 0):
        if DEBUG:
            log("seed zero or empty → using ones") 
        seed_np = np.ones_like(out_np, dtype=np.float32)
    
    all_wt["output"] = seed_np * np.float32(multiplier)
    
    if DEBUG:
        log(f"seed output → {all_wt['output'].shape}, relevance of seed: {np.sum(all_wt['output']):.8f}")

    # --- Step 2: zero-initialize all other nodes efficiently ---
    node_names = list(reversed(list(node_io.keys())))
    
    for name in node_names:
        if DEBUG:
            log(f"{'*****' * 10}")
        if name == "output":
            continue
            
        info = node_io[name]
        layer = info["layer_name"]
        parents = info.get("input_sources", [])
        
        if DEBUG:
            log(f"init: node='{name}', layer='{layer}', parents={parents}")

        output_children = info['output_children']
        layer_name = info['layer_name']
        node_type = info["node_type"]
        func_name = info["func_name"]
        
        if DEBUG:
            log(f"output_children: {output_children}, layer_name: {layer_name}, node_type: {node_type}, func_name: {func_name}")

        if layer in ("DL_Layer", "MLP_Layer"):
            activation_dict[name] = "None"

        # Collect tensor inputs for this node efficiently
        inp_vals = info.get("input_values", [])
        if not isinstance(inp_vals, (list, tuple)):
            inp_vals = [inp_vals]
        
        tensor_inputs = []
        for p, v in zip(parents, inp_vals):
            lt = node_io[p].get("layer_type", "")
            if lt in ("Weight", "Bias", "bn_running_mean", "bn_running_var", "bn_num_batches_tracked"):
                continue
            if isinstance(v, (torch.Tensor, np.ndarray)):
                tensor_inputs.append(v)

        if not tensor_inputs:
            # Fallback to output shape
            out_val = tensor_to_numpy(info["output_values"], copy=False)
            zeros = [get_pooled_array(out_val.shape, np.float32)]
            if DEBUG:
                log(f"  zero-init fallback → {out_val.shape}")
        else:
            zeros = []
            for v in tensor_inputs:
                v_np = tensor_to_numpy(v, copy=False)
                zeros.append(get_pooled_array(v_np.shape, np.float32))
            
            if DEBUG:
                if len(zeros) > 1:
                    log(f"  zero-init multi-parent → {[z.shape for z in zeros]}")
                else:
                    log(f"  zero-init single-parent → {zeros[0].shape}")

        all_wt[name] = zeros if len(zeros) > 1 else zeros[0]
        
        if DEBUG:
            log(f"all_wt[{name}]: {len(zeros) if len(zeros) > 1 else zeros[0].shape}")

    # --- Step 3: backpropagate relevance with progress bar only if DEBUG ---
    iterator = tqdm(node_names, desc="Backtracing") if DEBUG else node_names
    
    for name in iterator:
        if name == "output":
            continue

        # Skip symbolic/meta nodes
        symbolic_keywords = ("sym_size", "sym_int", "symbolic", "shape", "size", "dim")
        if any(k in name for k in symbolic_keywords):
            if DEBUG:
                log(f"[SKIP] Skipping symbolic/meta node: {name}")
            continue

        info     = node_io[name]
        layer    = info["layer_name"]
        func     = info.get("func_name", "")
        parents  = info.get("input_sources", [])
        children = info.get("output_children", [])
        
        if DEBUG:
            log(f"\nbacktrace: node='{name}', layer='{layer}', func='{func}', parents='{parents}', children='{children}'")

        def add_rel(r, parent_idx=None):
            """Add relevance r into all_wt[name], aligning shapes as needed with memory optimization."""
            if r is None:
                return

            if name.startswith("p_model_") or "weight" in name.lower():
                if DEBUG:
                    log(f"[SKIP] skipping relevance alignment for placeholder/weight: '{name}'")
                return

            buf = all_wt[name]
            if DEBUG:
                log(f"  add_rel: '{name}' buf={'list' if isinstance(buf, list) else buf.shape}")

            if isinstance(buf, list):
                parts = r if isinstance(r, (list, tuple)) else [r]

                if parent_idx is not None:
                    if parent_idx < len(buf):
                        arr = np.asarray(r, dtype=np.float32)
                        # Ensure buffer is float32
                        if buf[parent_idx].dtype != np.float32:
                            buf[parent_idx] = buf[parent_idx].astype(np.float32, copy=False)
                        buf[parent_idx] += fast_align_relevance(arr, buf[parent_idx].shape)
                else:
                    for i in range(len(buf)):
                        if i < len(parts) and parts[i] is not None:
                            arr = np.asarray(parts[i], dtype=np.float32)
                            if buf[i].dtype != np.float32:
                                buf[i] = buf[i].astype(np.float32, copy=False)
                            buf[i] += fast_align_relevance(arr, buf[i].shape)

                all_wt[name] = buf
            else:
                arr = np.asarray(r, dtype=np.float32)
                if buf.dtype != np.float32:
                    buf = buf.astype(np.float32, copy=False)
                all_wt[name] = buf + fast_align_relevance(arr, buf.shape)

        # Process different layer types...
        # [Rest of the layer processing logic remains the same but with optimized tensor operations]
        
        # — Activation: sum children's relevance —
        if layer == "Activation":
            for c in children:
                add_rel(get_relevance_from_child(c, name, all_wt, node_io))
            continue

        # — MLP (linear) —
        if layer == "MLP_Layer":
            hp = info["layer_hyperparams"]
            W  = tensor_to_numpy(hp["weight"], copy=False)
            B  = None if isinstance(hp["bias"], bool) else tensor_to_numpy(hp["bias"], copy=False)
            X  = process_input_for_eval(info["input_values"])
            
            if DEBUG:
                log(f"  [MLP] X={X.shape}, W={W.shape}, B={'none' if B is None else B.shape}")
            
            for c in children:
                R = get_relevance_from_child(c, name, all_wt, node_io)
                if DEBUG:
                    log(f"relevance from child: {np.sum(R):.8f}, shape: {R.shape}")
                if R is not None:
                    impl = get_layer_implementation("MLP_Layer")
                    if DEBUG:
                        log(f"Using {impl} implementation for linear layer {name}")
                    δ = UD2.launch_linear(impl, R, X, W, B, activation_master[activation_dict[name]])
                    if DEBUG:
                        log(f"relevance at {name}: {np.sum(δ):.8f}, shape: {δ.shape}") 
                    add_rel(δ)
            continue

        # — Conv2d —
        if layer == "DL_Layer" and func == "conv2d":
            hp = info["layer_hyperparams"]
            W = tensor_to_numpy(hp["weight"], copy=False)
            B = None if isinstance(hp["bias"], bool) or hp["bias"] is None else tensor_to_numpy(hp["bias"], copy=False)
            stride, pad = hp["stride"], hp["padding"]
            X = process_input_for_eval(info["input_values"])
            
            if DEBUG:
                log(f"  [Conv2d] X={X.shape}, W={W.shape}, B={B.shape}, pad={pad}, stride={stride}, activation_master: {activation_master[activation_dict[name]]}")
            
            for c in children:
                R = get_relevance_from_child(c, name, all_wt, node_io)
                if R is not None:
                    impl = get_layer_implementation("DL_Layer")
                    if DEBUG:
                        log(f"Using {impl} implementation for conv2d layer {name}")
                    δ = UD2.launch_conv2d(impl, R, X, W, B, pad, stride, activation_master[activation_dict[name]])
                    add_rel(δ)
            continue

        # — Scaled dot-product attention —
        if layer == "Attention" and func == "scaled_dot_product_attention":
            vals = info.get("input_values", [])
            if isinstance(vals, (list, tuple)):
                if DEBUG:
                    log(f"number of inputs: {len(vals)}")
            
            if DEBUG:
                for idx, item in enumerate(vals):
                    log(f"idx: {idx}, item: {item.shape}")

            n = len(vals)
            if n == 4:
                Q, K, V, masked_fill = (tensor_to_numpy(v, copy=False) for v in vals)
                if DEBUG:
                    log(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}, masked_fill: {masked_fill.shape}")
            elif n == 3:
                Q, K, V = (tensor_to_numpy(v, copy=False) for v in vals)
                raw_mask = info.get("layer_hyperparams", {}).get("attn_mask", None)
                masked_fill = None if raw_mask is None else tensor_to_numpy(raw_mask, copy=False)
                if DEBUG:
                    log(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}, masked_fill: {getattr(masked_fill, 'shape', None)}")
            else:
                raise RuntimeError(f"[{name}] expected 3 or 4 inputs for attention, got {n}")

            for c in children:
                R = get_relevance_from_child(c, name, all_wt, node_io)
                if DEBUG:
                    log(f"R --- value: {np.sum(R):.8f},  shape: {R.shape}")
                if R is not None:
                    impl = get_layer_implementation("Attention")
                    if DEBUG:
                        log(f"Using {impl} implementation for attention layer {name}")
                    result = UD2.launch_self_attention(impl, R, Q, K, V, masked_fill)
                    RQ, RK, RV, R_masked_fill = result
                    if DEBUG:
                        log(f"RQ: {np.sum(RQ):.8f}, shape: {RQ.shape}")
                        log(f"RK: {np.sum(RK):.8f}, shape: {RK.shape}")
                        log(f"RV: {np.sum(RV):.8f}, shape: {RV.shape}")
                        log(f"R_masked_fill: {np.sum(R_masked_fill):.8f}, shape: {R_masked_fill.shape}")
                    add_rel([RQ, RK, RV, R_masked_fill])
            continue

        # — Embedding —
        if layer == "NLP_Embedding" and func == "embedding":
            assign_embedding_relevance(name, info, all_wt, node_io)
            continue
        
        # — Elementwise multiply —
        if layer == "Mathematical_Operation" and func in ("mul", "mul_"):
            vals = info.get("input_values", [])
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            if len(vals) < 2:
                if DEBUG:
                    log(f"vals has only {len(vals)} which is less than 2.")
                for c in children:
                    add_rel(get_relevance_from_child(c, name, all_wt, node_io))
            else:
                X, Y = tensor_to_numpy(vals[0], copy=False), tensor_to_numpy(vals[1], copy=False)
                if DEBUG:
                    log(f"X: {X.shape}, Y: {Y.shape}")
                
                for c in children:
                    R = get_relevance_from_child(c, name, all_wt, node_io)
                    if DEBUG:
                        log(f"child {c}: relevance: {np.sum(R):.8f}")
                    if R is None:
                        continue

                    # Parse parents efficiently
                    raw_parents = info.get("input_sources", [])
                    if DEBUG:
                        log(f"[DEBUG] Raw parents from info['input_sources']: {raw_parents} (type={type(raw_parents)})")

                    if isinstance(raw_parents, str):
                        try:
                            parent_names = ast.literal_eval(raw_parents)
                            if DEBUG:
                                log(f"[DEBUG] Parsed parents from string: {parent_names} (type={type(parent_names)})")
                        except Exception as e:
                            if DEBUG:
                                log(f"[DEBUG] Error parsing input_sources string with ast.literal_eval: {e}")
                            continue
                    else:
                        parent_names = raw_parents

                    if DEBUG:
                        log(f"[DEBUG] Number of parsed parents: {len(parent_names)}")

                    if len(parent_names) < 2:
                        if DEBUG:
                            log("Less than 2 parents found; cannot resolve mul operation properly.")
                        continue

                    # Detect weight nodes
                    def is_weight(name):
                        return "weight" in name.lower()

                    is_X_weight = is_weight(parent_names[0])
                    is_Y_weight = is_weight(parent_names[1])

                    if DEBUG:
                        log(f"[DEBUG] Parent 0: {parent_names[0]}, is_weight: {is_X_weight}")
                        log(f"[DEBUG] Parent 1: {parent_names[1]}, is_weight: {is_Y_weight}")

                    # Handle weight cases efficiently
                    if is_X_weight and not is_Y_weight:
                        if DEBUG:
                            log(f"X is a weight node '{parent_names[0]}', discarding its relevance.")
                        zeros_X = get_pooled_array(X.shape, np.float32)
                        add_rel([zeros_X, R])
                        return_pooled_array(zeros_X)
                    elif is_Y_weight and not is_X_weight:
                        if DEBUG:
                            log(f"Y is a weight node '{parent_names[1]}', discarding its relevance.")
                        zeros_Y = get_pooled_array(Y.shape, np.float32)
                        add_rel([R, zeros_Y])
                        return_pooled_array(zeros_Y)
                    else:
                        # Split relevance normally
                        Rx, Ry = UD.calculate_wt_mul(R, X, Y)
                        if DEBUG:
                            log(f"X--- relevance: {np.sum(Rx):.8f}, shape: {Rx.shape}") 
                            log(f"Y--- relevance: {np.sum(Ry):.8f}, shape: {Ry.shape}") 
                        add_rel([Rx, Ry])
            continue

        # For Residual Connections
        if layer == "Mathematical_Operation" and func == "add":
            vals = info.get("input_values", [])
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            if len(vals) < 2:
                if DEBUG:
                    log(f"vals has only {len(vals)} which is less than 2.")
                for c in children:
                    add_rel(get_relevance_from_child(c, name, all_wt, node_io))
            else:
                X, Y = tensor_to_numpy(vals[0], copy=False), tensor_to_numpy(vals[1], copy=False)
                if DEBUG:
                    log(f"X: {X.shape}, Y: {Y.shape}")

                inp = [X, Y]
                for c in children:
                    R = get_relevance_from_child(c, name, all_wt, node_io)
                    if DEBUG:
                        log(f"child {c}: relevance: {np.sum(R):.8f}")
                    if R is not None:
                        impl = get_layer_implementation("Mathematical_Operation_add")
                        if DEBUG:
                            log(f"Using {impl} implementation for add operation {name}")
                        Rx, Ry = UD2.launch_wt_add_equal(impl, R, inp)
                        if DEBUG:
                            log(f"X--- relevance: {np.sum(Rx):.8f}, shape: {Rx.shape}") 
                            log(f"Y--- relevance: {np.sum(Ry):.8f}, shape: {Ry.shape}") 
                        add_rel([Rx, Ry]) 
            continue

        # — Norm / Indexing / other simple ops — passthrough
        if layer in {"Normalization", "Indexing_Operation"} or (
           layer=="Mathematical_Operation" and func not in ("mul","mul_")):
            for c in children:
                add_rel(get_relevance_from_child(c, name, all_wt, node_io))
            continue

        # — Vector operations (optimized) —
        if layer == "Vector_Operation":
            vals = info.get("input_values", [])
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            
            tensor_inputs = [v for v in vals if isinstance(v, (torch.Tensor, np.ndarray))]
            
            if DEBUG:
                for idx, item in enumerate(tensor_inputs):
                    log(f"idx: {idx}, item: {item.shape}")

            if not tensor_inputs:
                for c in children:
                    add_rel(get_relevance_from_child(c, name, all_wt, node_io))
                continue

            base = tensor_inputs[0]
            inp = tensor_to_numpy(base, copy=False)
            shape = inp.shape
            hp = info.get("layer_hyperparams", {})
            
            if DEBUG:
                log(f"base: {base.shape}")
                log(f"inp: {inp.shape}")
                log(f"shape: {shape}")
                log(f"hp: {hp}") 

            for c in children:
                R = get_relevance_from_child(c, name, all_wt, node_io)
                if R is None:
                    continue 
                
                R = np.asarray(R, dtype=np.float32)
                if DEBUG:
                    log(f"R: {np.sum(R):.8f}, shape: {R.shape}")
                    log(f"    before VecOp '{func}', R={R.shape}")

                # Optimized vector operations
                if func == "mean":
                    dims = hp.get("dim", None) or hp.get("dims", None)
                    if isinstance(dims, int):
                        dims = (dims,)

                    if np.prod(R.shape) != np.prod(shape):
                        dims = tuple(i for i, (r, s) in enumerate(zip(R.shape, shape)) if r == 1 and s > 1)
                        if DEBUG:
                            log(f"[WARN] Overriding dims based on shape: inferred dims={dims}")

                    if DEBUG:
                        log(f"dims: {dims}")

                    tot = float(np.prod([shape[d] for d in dims]))
                    if DEBUG:
                        log(f"tot: {tot}")

                    if DEBUG:
                        log(f"R before broadcasting: {np.sum(R):.8f} and its shape: {R.shape}")
                    
                    R = np.broadcast_to(R, shape)
                    if DEBUG:
                        log(f"Broadcasted R: {np.sum(R):.8f} and its shape: {R.shape}")

                    R = R / tot
                    if DEBUG:
                        log(f"calculated relevance of node `mean`: {np.sum(R):.8f}, shape: {R.shape}")
                    
                elif func in {"view", "reshape", "flatten", "unflatten"}:
                    R = R.reshape(shape)
                    if DEBUG:
                        log(f"R ---  value: {np.sum(R):.8f},  shape: {R.shape}")

                elif func == "permute":
                    inv = np.argsort(hp.get("dims", []))
                    R = R.transpose(inv)
                    if DEBUG:
                        log(f"R ---  value: {np.sum(R):.8f},  shape: {R.shape}")

                elif func == "transpose":
                    d0, d1 = info.get("method_args", (None, None))
                    R = R.swapaxes(d0, d1)
                    if DEBUG:
                        log(f"R ---  value: {np.sum(R):.8f},  shape: {R.shape}")

                elif func == "squeeze":
                    d = hp.get("dim")
                    if d is not None and shape[d] == 1:
                        R = np.expand_dims(R, axis=d)
                        if DEBUG:
                            log(f"R ---  value: {np.sum(R):.8f},  shape: {R.shape}")

                elif func == "unsqueeze":
                    d = hp.get("dim")
                    if d is not None:
                        R = np.squeeze(R, axis=d)
                        if DEBUG:
                            log(f"R ---  value: {np.sum(R):.8f},  shape: {R.shape}")

                elif func == "slice":
                    dim = hp.get("dim", 0)
                    start = hp.get("start", 0)
                    end = hp.get("end", None)
                    step = hp.get("step", 1)

                    if all(x is not None for x in [dim, start, end]):
                        slicer = [slice(None)] * len(shape)
                        slicer[dim] = slice(start, end, step)

                        if DEBUG:
                            log(f"[DEBUG][slice] input shape={shape}, R shape={R.shape}")

                        tmp = get_pooled_array(shape, R.dtype)
                        reg = tmp[tuple(slicer)]

                        # Attempt to align R to reg shape efficiently
                        if R.shape != reg.shape:
                            if DEBUG:
                                log(f"[DEBUG][slice] reg.shape={reg.shape}, R.shape={R.shape}")

                            if R.ndim == reg.ndim + 1:
                                if DEBUG:
                                    log(f"[DEBUG][slice] Collapsing last dim via L2 norm: {R.shape} → ", end="")
                                R = np.linalg.norm(R, axis=-1)
                                if DEBUG:
                                    log(f"{R.shape}")

                            # Handle batch mismatch efficiently
                            if R.shape[0] != reg.shape[0] and reg.shape[0] == 1:
                                if DEBUG:
                                    log(f"[DEBUG][slice] Expanding batch dim: {R.shape} → ({reg.shape[0]}, ...)")
                                R = R[:1]  # Only use first sample
                            elif R.shape[0] != reg.shape[0] and R.shape[0] == 1 and reg.shape[0] > 1:
                                R = np.broadcast_to(R, reg.shape)

                            if R.shape != reg.shape:
                                R = fast_align_relevance(R, reg.shape)

                        tmp[tuple(slicer)] = R.reshape(reg.shape) if R.shape != reg.shape else R
                        R = tmp
                        if DEBUG:
                            log(f"[DEBUG][slice] Final R shape={R.shape}, sum={np.sum(R):.4f}")
                    elif DEBUG:
                        log(f"[WARN][slice] Incomplete slice params: dim={dim}, start={start}, end={end}")

                elif func == "select":
                    dim, idx = hp.get("dim", 0), hp.get("index", 0)
                    if isinstance(idx, torch.Tensor):
                        idx = int(idx)
                    sl = [slice(None)] * len(shape)
                    sl[dim] = idx
                    tmp = get_pooled_array(shape, R.dtype)
                    reg = tmp[tuple(sl)]
                    R2 = fast_align_relevance(R, reg.shape) if R.shape != reg.shape else R
                    tmp[tuple(sl)] = R2
                    return_pooled_array(tmp)
                    R = tmp
                    if DEBUG:
                        log(f"R ---  value: {np.sum(R):.8f},  shape: {R.shape}")

                elif func == "expand":
                    sizes = hp.get("sizes") or hp.get("size") or hp.get("shape")
                    if DEBUG:
                        log(f"sizes: {sizes}")
                    if sizes:
                        exp = tuple(sizes)
                        if DEBUG:
                            log(f"exp: {exp}")
                        
                        # Step 1: Resolve -1 or None using input shape
                        try:
                            resolved_exp = tuple(
                                s if isinstance(s, int) and s > 0 else R.shape[i]
                                for i, s in enumerate(exp)
                            )
                            if DEBUG:
                                log(f"resolved_exp: {resolved_exp}")

                            # Step 2: Reshape R to expanded shape if possible
                            if R.shape != resolved_exp:
                                R = R.reshape(resolved_exp)
                        except Exception:
                            if DEBUG:
                                log(f"[WARN] expand→reshape {exp} failed")

                        if DEBUG:
                            print(f"Input shape: {shape}, shape after `expand` operation: {exp}, resolved_exp: {resolved_exp}") 

                        # Step 3: Reduce relevance over broadcasted axes efficiently
                        try:
                            for ax, (o, e) in enumerate(zip(shape, resolved_exp)):
                                if isinstance(o, int) and isinstance(e, int):
                                    if o == 1 and e > 1:
                                        if DEBUG:
                                            log(f"Reducing axis {ax}: o={o}, e={e}") 
                                        R = R.sum(axis=ax, keepdims=True)
                                elif DEBUG:
                                    log(f"[WARN] Skipping axis {ax} due to unresolved symbolic sizes: o={o}, e={e}")
                            
                            # Final reshape back to input shape
                            R = R.reshape(shape)
                        except Exception as e:
                            # Fallback using fast alignment
                            if DEBUG:
                                log(f"[WARN] Standard expand reduction failed: {e}")
                            try:
                                R = fast_align_relevance(R, shape)
                            except Exception as e_fallback:
                                if DEBUG:
                                    log(f"[ERROR] Expand fallback failed: {e_fallback}")
                                raise ValueError(f"[{name}] ❌ Failed to handle expand relevance reduction properly.")

                        if DEBUG:
                            log(f"R--- shape: {R.shape}")
                            log(f"R ---  value: {np.sum(R):.8f},  shape: {R.shape}")      

                elif func == "cat":
                    dim = hp.get("dim", 0)
                    sizes = [tensor_to_numpy(t, copy=False).shape[dim] for t in tensor_inputs]
                    splits = np.cumsum(sizes)[:-1]
                    parts = np.split(R, splits, axis=dim)
                    if DEBUG:
                        log(f"    cat → {len(parts)} parts")

                    # Clamp negative values
                    parts = [np.maximum(p, 0) for p in parts]

                    # Normalize to preserve total sum
                    total = np.sum(R, dtype=np.float64)
                    current_sum = sum(np.sum(p, dtype=np.float64) for p in parts)
                    if current_sum > 0:
                        scale = total / current_sum
                        parts = [p * np.float32(scale) for p in parts]

                    for i, p in enumerate(parts):
                        add_rel(p, parent_idx=i)
                    continue

                elif func in {"contiguous", "to"}:
                    pass
                else:
                    # Fallback reshape if total elements match
                    if R.size == np.prod(shape):
                        R = R.reshape(shape)
                    elif DEBUG:
                        log(f"[WARN] VecOp fallback pass-through R={R.shape}")

                if DEBUG:
                    log(f"    after VecOp '{func}', R={R.shape}")
                    log(f"R ---  value: {np.sum(R):.8f},  shape: {R.shape}")
                add_rel(R)
            continue

        # — Fallback pass-through for everything else —
        for c in children:
            add_rel(get_relevance_from_child(c, name, all_wt, node_io))

        # Periodic garbage collection for large models
        if len(all_wt) % 100 == 0:
            gc.collect()

    # Final cleanup
    clear_memory_pool()
    return all_wt

class RelevancePropagator:
    """Encapsulates relevance propagation logic with memory optimization."""
    def __init__(self, graph, node_io, activation_master, get_layer_implementation=None):
        self.graph = graph
        self.node_io = node_io
        self.activation_master = activation_master
        self.get_layer_implementation = get_layer_implementation or (lambda x: "original")

    def propagate(
        self,
        start_wt=None,
        mode="default",
        multiplier=100.0,
        scaler=1.0,
        thresholding=0.5,
        task="binary-classification",
    ):
        return run_evaluation(
            self.node_io,
            self.activation_master,
            mode=mode,
            start_wt=start_wt,
            multiplier=multiplier,
            scaler=scaler,
            thresholding=thresholding,
            task=task,
            get_layer_implementation=self.get_layer_implementation,
        )
