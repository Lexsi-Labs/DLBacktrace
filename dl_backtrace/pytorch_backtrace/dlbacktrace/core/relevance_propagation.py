# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/relevance_propagation.py

import ast  # Needed to safely parse stringified lists
import gc  # For memory cleanup
import numpy as np
import torch
from tqdm import tqdm
from dl_backtrace.pytorch_backtrace.dlbacktrace.utils import default_v2 as UD2  # Import launch functions

# Toggle debug prints
DEBUG = False  # Changed to False by default for performance
def log(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


def tensor_to_numpy(x):
    """Convert Tensor, scalar, list/tuple, or ndarray to a NumPy array, preserving dtype when possible."""
    if isinstance(x, np.ndarray):
        # Only convert float64 to float32 to save memory
        return x.astype(np.float32) if x.dtype == np.float64 else x
    if isinstance(x, torch.Tensor):
        # Preserve original dtype - don't force float32
        return x.detach().cpu().numpy()
    if isinstance(x, (int, float)):
        return np.array(x, dtype=np.float32)
    if isinstance(x, (list, tuple)):
        arrs = []
        for xi in x:
            converted = tensor_to_numpy(xi) if not isinstance(xi, np.ndarray) else xi
            # Only convert float64 to float32
            if hasattr(converted, 'dtype') and converted.dtype == np.float64:
                converted = converted.astype(np.float32)
            arrs.append(converted)
        try:
            return np.stack(arrs)
        except Exception:
            return np.array(arrs, dtype=object)
    raise TypeError(f"Cannot convert type {type(x)} to numpy")


def process_input_for_eval(X):
    """Unwrap single-element or multi-element lists/tuples and convert to NumPy, preserving dtype when possible."""
    # Handle lists/tuples
    if isinstance(X, (list, tuple)):
        X = X[0] if len(X) == 1 else X[0]  # Assume first element is the actual input tensor

    # Convert torch.Tensor to NumPy - preserve original dtype
    if isinstance(X, torch.Tensor):
        if X.is_cuda: 
            X = X.cpu()
        
        # Preserve original dtype
        return X.detach().numpy()

    # Already a NumPy array - only convert float64 to float32
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


def align_relevance(r, target_shape, total_relevance=None):
    """
    Align array `r` to `target_shape` with intelligent fallbacks:
      0) Collapse extra leading dims,
      1) Sum-out axes where target_dim==1 but r_dim>1,
      2) Reduce axes where r_dim is multiple of target_dim,
      2.5) Expand axes where target_dim>r_dim and target_dim % r_dim == 0,
      3-4) Apply reduction, upsampling, or tiling logic,
      5) Final broadcast attempt,
      6-8) Handle 1D/2D mismatches.
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

    # 3–4) Approximate reduction or tiling for mismatches BEFORE trying broadcast
    if r.shape != target_shape:
        # 4a) approx reduce last dim for 2D
        if r.ndim == 2 and r.shape[1] > target_shape[-1] and r.shape[0] == target_shape[0]:
            try:
                factor = r.shape[1] // target_shape[-1]
                trimmed = r[:, :factor * target_shape[-1]]
                r = trimmed.reshape(r.shape[0], target_shape[-1], factor).mean(-1)
                log(f"[INFO] approx-reduced last dim → {r.shape}")
            except:
                pass

        # 4b) block-mean reduce last dim
        if r.ndim == len(target_shape) and r.shape[:-1] == target_shape[:-1]:
            from math import floor
            factor = floor(r.shape[-1] / target_shape[-1])
            if factor >= 1:
                try:
                    trimmed = r[..., :factor * target_shape[-1]]
                    r = trimmed.reshape(*r.shape[:-1], target_shape[-1], factor).mean(-1)
                    log(f"[INFO] approx block-mean reduced last dim (factor {factor}) → {r.shape}")
                except:
                    pass

        # 4c) repeat last dim if clean factor
        if r.ndim == len(target_shape) and r.shape[:-1] == target_shape[:-1]:
            dr, dt = r.shape[-1], target_shape[-1]
            if dt % dr == 0:
                factor = dt // dr
                try:
                    r = np.repeat(r, factor, axis=-1)
                    log(f"[INFO] repeated last dim {dr}→{dt} with factor {factor} → {r.shape}")
                except Exception as e:
                    log(f"[WARN] repeat fail on last dim: {e}")

        # 4d) approximate tiling of last dim
        if r.ndim == len(target_shape) and r.shape[:-1] == target_shape[:-1]:
            dr, dt = r.shape[-1], target_shape[-1]
            try:
                reps = -(-dt // dr)
                r = np.tile(r, reps)[..., :dt]
                log(f"[INFO] approx-tiled last dim from {dr}→{dt} with reps={reps} → {r.shape}")
            except Exception as e:
                log(f"[WARN] approx-tiling failed: {e}")

        # 4e) approx reduction for last two dims
        if r.ndim == len(target_shape) and np.prod(r.shape[-2:]) > np.prod(target_shape[-2:]):
            try:
                flat = r.reshape(*r.shape[:-2], -1)
                tgt_flat = np.prod(target_shape[-2:])
                factor = flat.shape[-1] // tgt_flat
                trimmed = flat[..., :factor * tgt_flat]
                r = trimmed.reshape(*flat.shape[:-1], tgt_flat, factor).mean(-1)
                r = r.reshape(*r.shape[:-1], *target_shape[-2:])
                log(f"[INFO] approx block-mean reduced last two dims → {r.shape}")
            except Exception as e:
                log(f"[WARN] reduction for last two dims failed: {e}")

        # 4f) tile last two dims if both smaller than target
        if (
            r.ndim == len(target_shape)
            and r.ndim >= 2
            and target_shape[-2] > 1
            and target_shape[-1] > 1
            and r.shape[-2] < target_shape[-2]
            and r.shape[-1] < target_shape[-1]
        ):
            try:
                reps_0 = -(-target_shape[-2] // r.shape[-2])
                reps_1 = -(-target_shape[-1] // r.shape[-1])
                r = np.tile(r, (*([1] * (r.ndim - 2)), reps_0, reps_1))
                r = r[..., :target_shape[-2], :target_shape[-1]]
                log(f"[INFO] approx-tiled last two dims from {r.shape[-2:]}, cut to {target_shape[-2:]}")
            except Exception as e:
                log(f"[WARN] tiling last two dims failed: {e}")

        # 4g) Final soft reduction of last dim
        if r.ndim == len(target_shape) and r.shape[:-1] == target_shape[:-1]:
            dr, dt = r.shape[-1], target_shape[-1]
            if dr != dt:
                try:
                    factor = dr / dt
                    new = np.linspace(0, dr, dt + 1, dtype=int)
                    chunks = [r[..., new[i]:new[i + 1]] for i in range(dt)]
                    r = np.stack([c.mean(-1) if c.shape[-1] > 0 else np.zeros_like(c[..., 0]) for c in chunks], axis=-1)
                    log(f"[INFO] final soft-reduced last dim {dr}→{dt} using linear split → {r.shape}")
                except Exception as e:
                    log(f"[WARN] final soft reduction failed: {e}")

    # 5) Final broadcast (after all intelligent resizing)
    if r.shape != target_shape:
        try:
            if r.ndim < len(target_shape):
                missing_dims = len(target_shape) - r.ndim
                r = r.reshape(r.shape + (1,) * missing_dims)
                log(f"[INFO] unsqueezed {missing_dims} trailing dims → shape: {r.shape}")
            r = np.broadcast_to(r, target_shape).copy()
            if DEBUG:
                log(f"[INFO] final broadcast to {target_shape}")
        except Exception as e:
            raise ValueError(
                f"🚫 Cannot align relevance: incompatible shape {r.shape} → {target_shape}: {e}"
            )

    # 6) handle 1D to 1D
    if r.ndim == 1 and len(target_shape) == 1:
        if r.shape[0] > target_shape[0]:
            r = r[:target_shape[0]]
            log(f"[INFO] truncated to {r.shape}")
        else:
            pad_width = target_shape[0] - r.shape[0]
            r = np.pad(r, (0, pad_width), mode='constant')
            log(f"[INFO] padded to {r.shape}")
        return r

    # 7) handle (B, S) style mismatch
    if r.ndim == 1 and len(target_shape) == 2:
        B, S = target_shape
        if r.shape[0] == S:
            r = np.tile(r[None, :], (B, 1))
            log(f"[INFO] expanded 1D to (B,S) → {r.shape}")
        elif r.shape[0] == B:
            r = np.tile(r[:, None], (1, S))
            log(f"[INFO] expanded 1D to (B,S) → {r.shape}")
        return r

    # 8) final fallback for 2D
    if r.ndim == 2 and len(target_shape) == 2:
        B, S = target_shape
        if r.shape[1] != S:
            r = r[:, :S] if r.shape[1] > S else np.pad(r, ((0, 0), (0, S - r.shape[1])), mode='constant')
            log(f"[INFO] truncated/padded seq dim → {r.shape}")
        if r.shape[0] != B:
            r = np.tile(r, (B, 1))[:B]
            log(f"[INFO] tiled/truncated batch dim → {r.shape}")
    elif r.size == np.prod(target_shape):
        try:
            r = r.reshape(target_shape)
            log(f"[INFO] reshaped to {target_shape}")
        except:
            pass

    if r.shape != target_shape:
        raise ValueError(f"Cannot align relevance from shape {r.shape} to {target_shape}")

    # Normalize relevance total if changed
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
    Skips non-token embeddings that don't receive input_ids.
    """
    if DEBUG:
        log(f"assign_embedding_relevance: node='{node}'")

    # 1) Aggregate relevance from all children
    R_out = None
    for c in info["output_children"]:
        rc = get_relevance_from_child(c, node, all_wt, node_io)
        if rc is None:
            continue

        if isinstance(rc, list):
            summed = None
            for x in rc:
                arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
                summed = arr if summed is None else (summed + arr)
            rc = summed
        elif isinstance(rc, torch.Tensor):
            rc = rc.detach().cpu().numpy()

        R_out = rc if R_out is None else (R_out + rc)

    if R_out is None:
        return

    if DEBUG:
        log(f"R_out ---  relevance value: {np.sum(R_out):.4f}, shape: {R_out.shape}")

    # 2) Try to extract token_ids from input_values
    token_ids = None
    for val in info.get("input_values", []):
        try:
            log(f"[{node}] 🔍 input type={type(val)}, dtype={val.dtype if isinstance(val, torch.Tensor) else 'N/A'}, shape={val.shape if hasattr(val, 'shape') else 'N/A'}")
        except Exception:
            pass
        if isinstance(val, torch.Tensor) and not torch.is_floating_point(val):
            token_ids = tensor_to_numpy(val)
            break

    # 3) Fallback: try to get token_ids from node_io directly
    if token_ids is None:
        for key in ['input_ids', 0, 'ids']:
            val = node_io.get(key)
            if isinstance(val, torch.Tensor) and not torch.is_floating_point(val):
                token_ids = tensor_to_numpy(val)
                if DEBUG:
                    log(f"[{node}] ℹ️ token_ids recovered from node_io['{key}']")
                break

    # Skip for non-token embedding nodes
    if token_ids is None:
        if DEBUG:
            log(f"[{node}] ⏭️ Skipping: could not extract token_ids (likely not token embedding)")
        return

    target_shape = token_ids.shape

    # 4) Retrieve any previous relevance
    prev = all_wt.get(node)
    if isinstance(prev, list):
        summed = None
        for x in prev:
            arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            if arr.shape == R_out.shape:
                summed = arr if summed is None else (summed + arr)
            elif DEBUG:
                log(f"[{node}] ⚠️ Skipping incompatible prev shape: {arr.shape}")
        prev = summed
    elif isinstance(prev, torch.Tensor):
        prev = prev.detach().cpu().numpy()

    # 5) Fast path for matching shapes
    if prev is not None and R_out.shape == prev.shape:
        if DEBUG:
            log(f"[{node}] ℹ️ Relevance already aligned, fast accumulation.")
        all_wt[node] = prev + R_out
        return

    # 6) Align dimensions
    if R_out.shape != target_shape:
        try:
            while R_out.ndim > len(target_shape):
                R_out = R_out.sum(axis=-1)  # preserve magnitude
                if DEBUG:
                    log(f"[{node}] ℹ️ collapsed to {R_out.shape}") 
            if R_out.shape != target_shape:
                R_out = align_relevance(R_out, target_shape)
                if DEBUG:
                    log(f"[{node}] ℹ️ Aligned relevance to {R_out.shape}")
        except Exception as e:
            raise ValueError(f"[{node}] ❌ Failed to align relevance: {e}")

    # 7) Accumulate into all_wt
    if prev is None:
        all_wt[node] = R_out.copy()
    else:
        if prev.shape != R_out.shape:
            if DEBUG:
                log(f"[{node}] ℹ️ Mismatch: prev={prev.shape}, current={R_out.shape}")
            try:
                expanded = R_out
                while expanded.ndim < prev.ndim:
                    expanded = expanded[..., None]
                R_out = np.broadcast_to(expanded, prev.shape)
            except Exception as e:
                raise ValueError(
                    f"[{node}] ❌ Shape mismatch in accumulation: "
                    f"prev={prev.shape}, current={R_out.shape}, error: {e}"
                )
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
    target_token_ids=None,
    get_layer_implementation=None,
):
    """
    Perform LRP-style backtrace through node_io.

    target_token_ids:
        - Only meaningful for task=="generation".
        - A token id (or list of token ids, batch-aligned) that we actually chose
          during decoding. For classification tasks this is ignored.

    Returns: dict node_name -> relevance (np.ndarray or list of).
    """
    all_wt = {}
    activation_dict = {}
    
    # Set default layer implementation function if not provided
    if get_layer_implementation is None:
        get_layer_implementation = lambda x: "original"
    
    # --- Step 1: seed the 'output' node ---
    raw_out = node_io["output"]["input_values"]
    if isinstance(raw_out, (list, tuple)):
        raw_out = raw_out[0]
    out_np = tensor_to_numpy(raw_out)
    
    if DEBUG:
        log(f"out_np: {out_np.shape}")
    
    seed = UD2.calculate_start_wt(
        out_np, 
        scaler=scaler, 
        task=task,
        target_indices=target_token_ids,  # NEW: only matters when task=="generation"
        thresholding=thresholding,
    )
    
    if DEBUG:
        log(f"seed: {np.sum(seed):.8f},  shape: {seed.shape}")
    
    seed_np = tensor_to_numpy(seed)
    if seed_np.size == 0 or np.all(seed_np == 0):
        if DEBUG:
            log("seed zero or empty → using ones") 
        seed_np = np.ones_like(out_np, dtype=np.float32)
    
    all_wt["output"] = seed_np * np.float32(multiplier)
    
    if DEBUG:
        log(f"seed output → {all_wt['output'].shape}, relevance of seed: {np.sum(all_wt['output']):.8f}")

    # --- Step 2: zero-initialize all other nodes ---
    for name in reversed(list(node_io.keys())):
        log(f"{'*****' * 10}")
        if name == "output":
            continue
            
        info = node_io[name]
        layer = info["layer_name"]
        parents = info.get("input_sources", [])
        
        if DEBUG:
            log(f"init: node='{name}', layer='{layer}', parents={parents}")

        ############################################
        output_children = info['output_children']
        layer_name = info['layer_name']
        node_type = info["node_type"]
        func_name = info["func_name"]
        
        if DEBUG:
            log(f"output_children: {output_children}, layer_name: {layer_name}, node_type: {node_type}, func_name: {func_name}")

        if layer in ("DL_Layer", "MLP_Layer"):
            activation_dict[name] = "None"

        # collect tensor inputs for this node
        inp_vals = info.get("input_values", [])
        if not isinstance(inp_vals, (list, tuple)):
            inp_vals = [inp_vals]
        
        tensor_inputs = []
        for p, v in zip(parents, inp_vals):
            lt = node_io[p].get("layer_type", "")
            if lt in ("Weight", "Bias", "bn_running_mean", "bn_running_var", "bn_num_batches_tracked"):
                continue
            if func_name == "addmm" and lt == 'future_use':  # Special case: skip 'future_use' type for addmm  
                continue
            if isinstance(v, (torch.Tensor, np.ndarray)):
                tensor_inputs.append(v) 

        if not tensor_inputs:
            # fallback to output shape
            out_val = tensor_to_numpy(info["output_values"])
            zeros = [np.zeros_like(out_val, dtype=np.float32)]
            if DEBUG:
                log(f"  zero-init fallback → {out_val.shape}")
        else:
            zeros = [np.zeros_like(tensor_to_numpy(v), dtype=np.float32) for v in tensor_inputs]
            if DEBUG:
                if len(zeros) > 1:
                    log(f"  zero-init multi-parent → {[z.shape for z in zeros]}")
                else:
                    log(f"  zero-init single-parent → {zeros[0].shape}")

        all_wt[name] = zeros if len(zeros) > 1 else zeros[0]
        
        if DEBUG:
            log(f"all_wt[{name}]: {len(zeros) if len(zeros) > 1 else zeros[0].shape}")

    # --- Step 3: backpropagate relevance ---
    for name in tqdm(list(node_io.keys())[::-1], desc="Backtracing"):
        if name == "output":
            continue

        # 🚫 Skip symbolic/meta nodes like shape, size, etc.
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
            """Add relevance r into all_wt[name], aligning shapes as needed."""
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
                            if DEBUG:
                                log(f"[FIX] casting buf[{parent_idx}] from {buf[parent_idx].dtype} to float32")
                            buf[parent_idx] = buf[parent_idx].astype(np.float32)

                        buf[parent_idx] += align_relevance(arr, buf[parent_idx].shape)

                else:
                    for i in range(len(buf)):
                        if i < len(parts) and parts[i] is not None:
                            arr = np.asarray(parts[i], dtype=np.float32)
                            if buf[i].dtype != np.float32:
                                if DEBUG:
                                    log(f"[FIX] casting buf[{i}] from {buf[i].dtype} to float32")
                                buf[i] = buf[i].astype(np.float32)

                            buf[i] += align_relevance(arr, buf[i].shape)

                all_wt[name] = buf
            else:
                arr = np.asarray(r, dtype=np.float32)
                if buf.dtype != np.float32:
                    log(f"[FIX] casting buf from {buf.dtype} to float32")
                    buf = buf.astype(np.float32)

                all_wt[name] = buf + align_relevance(arr, buf.shape)

        # — Activation: sum children's relevance —
        if layer == "Activation":
            for c in children:
                add_rel(get_relevance_from_child(c, name, all_wt, node_io))
            continue

        # — MLP (linear) —
        if layer == "MLP_Layer":
            if func == "addmm":
                hp = info["layer_hyperparams"]
                print(f"hp: {hp}")
                bias, mat1, mat2 = info["input_values"]

                # Convert all to numpy
                X = process_input_for_eval(mat1)
                W = process_input_for_eval(mat2).T  # 🔄 Transpose for consistency with FC layer
                B = None if isinstance(bias, bool) else process_input_for_eval(bias)

                if DEBUG:
                    log(f"  [addmm] X={X.shape}, W={W.shape}, B={'none' if B is None else B.shape}")

                for c in children:
                    R = get_relevance_from_child(c, name, all_wt, node_io)
                    if DEBUG:
                        log(f"relevance from child: {np.sum(R):.8f}, shape: {R.shape}") 
                    if R is not None:
                        impl = get_layer_implementation("MLP_Layer")
                        if DEBUG:
                            log(f"Using {impl} implementation for MLP layer {name}")
                        δ = UD2.launch_linear(impl, R, X, W, B, activation_master[activation_dict[name]])
                        if DEBUG: 
                            log(f"relevance at {name}: {np.sum(δ):.8f}, shape: {δ.shape}") 
                        add_rel(δ)
                continue
            
            else:
                hp = info["layer_hyperparams"]
                W  = hp["weight"].detach().cpu().numpy()
                B  = None if isinstance(hp["bias"], bool) else hp["bias"].detach().cpu().numpy()
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
                            log(f"Using {impl} implementation for MLP layer {name}")
                        δ = UD2.launch_linear(impl, R, X, W, B, activation_master[activation_dict[name]]) 
                        if DEBUG:
                            log(f"relevance at {name}: {np.sum(δ):.8f}, shape: {δ.shape}") 
                        add_rel(δ)
                continue

        # — Conv2d —
        if layer == "DL_Layer" and func == "conv2d":
            hp = info["layer_hyperparams"]
            W = hp["weight"].detach().cpu().numpy()
            B = None if isinstance(hp["bias"], bool) or hp["bias"] is None else hp["bias"].detach().cpu().numpy()
            stride, pad = hp["stride"], hp["padding"]
            X = process_input_for_eval(info["input_values"])
            
            if DEBUG:
                log(f"  [Conv2d] X={X.shape}, W={W.shape}, B={None if B is None else B.shape}, pad={pad}, stride={stride}, activation_master: {activation_master[activation_dict[name]]}")
            
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

            vals = info.get("input_values", [])
            n = len(vals)
            if n == 4:
                Q, K, V, masked_fill = (tensor_to_numpy(v) for v in vals)
                if DEBUG:
                    log(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}, masked_fill: {masked_fill.shape}")
            elif n == 3:
                # fallback: mask wasn’t recorded as an input, grab it from hyperparams
                Q, K, V = (tensor_to_numpy(v) for v in vals)
                # grab the mask out of layer_hyperparams (or None if missing)
                raw_mask = info.get("layer_hyperparams", {}).get("attn_mask", None)
                masked_fill = None if raw_mask is None else tensor_to_numpy(raw_mask)
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
                X, Y = tensor_to_numpy(vals[0]), tensor_to_numpy(vals[1])
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

                    # If input_sources is accidentally stringified, parse it
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

                    # Utility to detect weight node
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
                        add_rel([np.zeros_like(X), R])  # Relevance only to Y
                    elif is_Y_weight and not is_X_weight:
                        if DEBUG:
                            log(f"Y is a weight node '{parent_names[1]}', discarding its relevance.")
                        add_rel([R, np.zeros_like(Y)])  # Relevance only to X
                    else:

                        impl = get_layer_implementation("Mathematical_Operation_mul")
                        if DEBUG:
                            log(f"Using {impl} implementation for mul operation {name}")
                        # Neither parent is a weight node → split relevance normally
                        Rx, Ry = UD2.launch_wt_mul(impl, R)
                        if DEBUG:
                            log(f"R: {np.sum(R):.8f}, shape: {R.shape}")
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
                X, Y = tensor_to_numpy(vals[0]), tensor_to_numpy(vals[1])
                if DEBUG:
                    log(f"X: {X.shape}, Y: {Y.shape}")

                if isinstance(vals, (list, tuple)):
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

        # — Vector operations: view/reshape, permute, transpose, squeeze, unsqueeze,
        #    slice, select, expand, mean, cat, contiguous/to, fallback —
        if layer == "Vector_Operation":
            vals = info.get("input_values", [])
            # log(f"vals: {vals}")
            if not isinstance(vals, (list, tuple)):
                vals = [vals]    # Ensures vals is a list of tensors/arrays
                # log(f"(list, tuple): vals: {vals}")
            tensor_inputs = [v for v in vals if isinstance(v, (torch.Tensor, np.ndarray))]
            
            if DEBUG:
                for idx, item in enumerate(tensor_inputs):
                    log(f"idx: {idx}, item: {item.shape}")

            # if no valid tensors are found, simply collect relevance from children and skip to the next node
            if not tensor_inputs:
                for c in children:
                    add_rel(get_relevance_from_child(c, name, all_wt, node_io))
                continue

            base  = tensor_inputs[0]
            inp   = tensor_to_numpy(base)
            shape = inp.shape
            hp = info.get("layer_hyperparams", {})
            
            if DEBUG:
                log(f"base: {base.shape}")
                log(f"inp: {inp.shape}")
                log(f"shape: {shape}")
                log(f"hp: {hp}") 

            # Loop over child nodes and pulls their relevance to backpropagate 
            for c in children:
                R = get_relevance_from_child(c, name, all_wt, node_io)
                if R is None:
                    continue 
                
                R = np.asarray(R, dtype=np.float32)
                if DEBUG:
                    log(f"R: {np.sum(R):.8f}, shape: {R.shape}")
                log(f"    before VecOp '{func}', R={R.shape}")

                # common Vector ops…
                if func == "mean":
                    dims = hp.get("dim", None) or hp.get("dims", None)
                    if isinstance(dims, int):
                        dims = (dims,)

                    # Auto-fix dims if shape mismatch indicates different reduction
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

                    # Then divide to spread uniformly
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

                        tmp = np.zeros(shape, dtype=R.dtype)
                        reg = tmp[tuple(slicer)]

                        # Attempt to align R to reg shape
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

                            # Handle dimension size mismatch by truncating or padding
                            if R.shape != reg.shape and R.ndim == reg.ndim:
                                if DEBUG:
                                    log(f"[DEBUG][slice] Handling dimension mismatch: {R.shape} → {reg.shape}")
                                
                                # Check if we can align by truncating or padding
                                aligned = True
                                new_R = np.zeros(reg.shape, dtype=R.dtype)
                                
                                for i in range(min(R.ndim, reg.ndim)):
                                    if R.shape[i] > reg.shape[i]:
                                        # Truncate R to fit reg
                                        if i == 0:
                                            new_R = R[:reg.shape[i]]
                                        elif i == 1:
                                            new_R = R[:, :reg.shape[i]]
                                        else:
                                            # For higher dimensions, use slicing
                                            slices = [slice(None)] * R.ndim
                                            slices[i] = slice(0, reg.shape[i])
                                            new_R = R[tuple(slices)]
                                        R = new_R
                                        break
                                    elif R.shape[i] < reg.shape[i]:
                                        # Pad R to match reg
                                        if i == 0:
                                            new_R[:R.shape[i]] = R
                                        elif i == 1:
                                            new_R[:, :R.shape[i]] = R
                                        else:
                                            # For higher dimensions, use slicing
                                            slices = [slice(None)] * reg.ndim
                                            slices[i] = slice(0, R.shape[i])
                                            new_R[tuple(slices)] = R
                                        R = new_R
                                        break
                                
                                if DEBUG:
                                    log(f"[DEBUG][slice] After alignment: {R.shape}")

                            if R.shape != reg.shape:
                                raise ValueError(f"[slice] ❌ Cannot align R shape {R.shape} to reg shape {reg.shape}")

                        tmp[tuple(slicer)] = R.reshape(reg.shape)
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
                    tmp = np.zeros(shape, dtype=R.dtype)
                    reg = tmp[tuple(sl)]
                    R2  = R.reshape(reg.shape) if R.shape!=reg.shape else R
                    tmp[tuple(sl)] = R2
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
                                axes_to_sum = tuple(i for i, (o, e) in enumerate(zip(shape, R.shape)) if o == 1 and e > 1)
                                if len(shape) == R.ndim and axes_to_sum:
                                    R = np.sum(R, axis=axes_to_sum, keepdims=True).reshape(shape)
                                elif R.size == np.prod(shape):
                                    R = R.reshape(shape)
                                else:
                                    log(f"[WARN] Expand op {name!r} requires custom handling. R_out={R.shape}, input={shape}")
                            except Exception as e_fallback:
                                if DEBUG:
                                    log(f"[ERROR] Expand fallback failed: {e_fallback}")
                                raise ValueError(f"[{name}] ❌ Failed to handle expand relevance reduction properly.")

                        if DEBUG:
                            log(f"R--- shape: {R.shape}")
                        log(f"R ---  value: {np.sum(R):.8f},  shape: {R.shape}")      

                elif func == "cat":
                    dim   = hp.get("dim", 0)
                    sizes = [tensor_to_numpy(t).shape[dim] for t in tensor_inputs]
                    splits= np.cumsum(sizes)[:-1]
                    parts = np.split(R, splits, axis=dim)
                    if DEBUG:
                        log(f"    cat → {len(parts)} parts")

                    # Optional: Clamp, shift if necessary
                    parts = [np.maximum(p, 0) for p in parts]

                    # Normalize to preserve total sum
                    total = np.sum(R)
                    current_sum = sum(np.sum(p) for p in parts)
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

    return all_wt


# ═══════════════════════════════════════════════════════════════════════════════
# GPU-NATIVE PROPAGATION — all data stays as CUDA tensors throughout
# ═══════════════════════════════════════════════════════════════════════════════

def to_gpu_tensor(x, device=None):
    """Convert any input to a CUDA float32 tensor. No-op if already there."""
    if device is None:
        device = torch.device("cuda")
    if isinstance(x, torch.Tensor):
        if x.device.type != 'cuda':
            return x.to(device=device, dtype=torch.float32)
        if x.dtype != torch.float32:
            return x.to(dtype=torch.float32)
        return x
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float32, device=device)
    if isinstance(x, (int, float)):
        return torch.tensor(x, dtype=torch.float32, device=device)
    raise TypeError(f"Cannot convert type {type(x)} to GPU tensor")


def to_gpu_long(x, device=None):
    """Convert to a CUDA int64 tensor (for token IDs)."""
    if device is None:
        device = torch.device("cuda")
    if isinstance(x, torch.Tensor):
        if x.device.type != 'cuda' or x.dtype != torch.long:
            return x.to(device=device, dtype=torch.long)
        return x
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.long, device=device)
    raise TypeError(f"Cannot convert type {type(x)} to GPU long tensor")


def align_relevance_gpu(r, target_shape):
    """
    GPU counterpart of align_relevance(). Aligns tensor `r` to `target_shape`
    using torch operations — no CPU transfer.
    """
    original_total = r.sum()

    # Collapse extra leading dims
    while r.ndim > len(target_shape):
        r = torch.linalg.norm(r, dim=0)

    # Sum-out axes where target_dim==1 but r_dim>1
    for ax, (dr, dt) in enumerate(zip(r.shape, target_shape)):
        if dt == 1 and dr != 1:
            r = r.sum(dim=ax, keepdim=True)

    # Reduce axes where dr % dt == 0
    for ax, (dr, dt) in enumerate(zip(r.shape, target_shape)):
        if dt > 1 and dr != dt and dr % dt == 0:
            _tgt_numel = 1
            for s in target_shape:
                _tgt_numel *= s
            if r.numel() >= _tgt_numel:
                factor = dr // dt
                new_shape = list(r.shape)
                new_shape[ax] = dt
                new_shape.insert(ax + 1, factor)
                try:
                    r = r.reshape(new_shape).sum(dim=ax + 1)
                except Exception:
                    pass

    # Expand axes where target_dim > dr and dt % dr == 0 (upsample)
    for ax, (dr, dt) in enumerate(zip(r.shape, target_shape)):
        if dt > dr and dt % dr == 0:
            factor = dt // dr
            r = r.repeat_interleave(factor, dim=ax)

    # Final broadcast
    if r.shape != target_shape:
        try:
            if r.ndim < len(target_shape):
                missing = len(target_shape) - r.ndim
                r = r.reshape(r.shape + (1,) * missing)
            r = r.expand(target_shape).contiguous()
        except Exception as e:
            raise ValueError(
                f"Cannot align relevance: incompatible shape {r.shape} → {target_shape}: {e}"
            )

    # Normalize relevance total
    final_total = r.sum()
    if not torch.isclose(final_total, original_total, rtol=1e-3) and final_total > 0:
        scale = original_total / (final_total + 1e-8)
        r = r * scale

    return r


def get_relevance_from_child_gpu(child, parent, all_wt, node_io):
    """GPU version of get_relevance_from_child — returns CUDA tensor or None."""
    if child not in all_wt:
        return None
    r = all_wt[child]
    if isinstance(r, list):
        sources = node_io[child]["input_sources"]
        try:
            idx = sources.index(parent)
        except ValueError:
            return None
        if idx >= len(r):
            return None
        return r[idx]
    return r


def assign_embedding_relevance_gpu(node, info, all_wt, node_io, device):
    """GPU-native version of assign_embedding_relevance."""
    R_out = None
    for c in info["output_children"]:
        rc = get_relevance_from_child_gpu(c, node, all_wt, node_io)
        if rc is None:
            continue
        if isinstance(rc, list):
            summed = None
            for x in rc:
                t = to_gpu_tensor(x, device) if not isinstance(x, torch.Tensor) else x
                summed = t if summed is None else (summed + t)
            rc = summed
        elif not isinstance(rc, torch.Tensor):
            rc = to_gpu_tensor(rc, device)
        R_out = rc if R_out is None else (R_out + rc)

    if R_out is None:
        return

    # Extract token_ids
    token_ids = None
    for val in info.get("input_values", []):
        if isinstance(val, torch.Tensor) and not torch.is_floating_point(val):
            token_ids = val if val.is_cuda else val.to(device)
            break

    if token_ids is None:
        for key in ['input_ids', 0, 'ids']:
            val = node_io.get(key)
            if isinstance(val, torch.Tensor) and not torch.is_floating_point(val):
                token_ids = val if val.is_cuda else val.to(device)
                break

    if token_ids is None:
        return

    target_shape = token_ids.shape

    # Retrieve previous relevance
    prev = all_wt.get(node)
    if isinstance(prev, list):
        summed = None
        for x in prev:
            t = to_gpu_tensor(x, device)
            if t.shape == R_out.shape:
                summed = t if summed is None else (summed + t)
        prev = summed
    elif prev is not None and not isinstance(prev, torch.Tensor):
        prev = to_gpu_tensor(prev, device)

    # Fast path
    if prev is not None and R_out.shape == prev.shape:
        all_wt[node] = prev + R_out
        return

    # Align dimensions
    if R_out.shape != target_shape:
        while R_out.ndim > len(target_shape):
            R_out = R_out.sum(dim=-1)
        if R_out.shape != target_shape:
            R_out = align_relevance_gpu(R_out, target_shape)

    if prev is None:
        all_wt[node] = R_out.clone()
    else:
        if prev.shape != R_out.shape:
            R_out = R_out.expand(prev.shape).contiguous()
        all_wt[node] = prev + R_out


def run_evaluation_gpu(
    node_io,
    activation_master,
    mode="default",
    start_wt=None,
    multiplier=100.0,
    scaler=1.0,
    thresholding=0.5,
    task="binary-classification",
    target_token_ids=None,
    get_layer_implementation=None,
):
    """
    GPU-native relevance propagation — all data stays as CUDA tensors.
    Only converts to numpy at the very end when returning results.
    """
    device = torch.device("cuda")
    all_wt = {}
    activation_dict = {}

    if get_layer_implementation is None:
        get_layer_implementation = lambda x: "cuda"

    # --- Step 1: seed the 'output' node ---
    raw_out = node_io["output"]["input_values"]
    if isinstance(raw_out, (list, tuple)):
        raw_out = raw_out[0]
    # calculate_start_wt needs numpy
    out_np = tensor_to_numpy(raw_out)

    seed = UD2.calculate_start_wt(
        out_np,
        scaler=scaler,
        task=task,
        target_indices=target_token_ids,
        thresholding=thresholding,
    )

    seed_np = tensor_to_numpy(seed)
    if seed_np.size == 0 or np.all(seed_np == 0):
        seed_np = np.ones_like(out_np, dtype=np.float32)

    # Move seed to GPU — this is the last numpy→GPU transfer needed
    all_wt["output"] = torch.tensor(seed_np * np.float32(multiplier), dtype=torch.float32, device=device)

    if DEBUG:
        log(f"seed output → {all_wt['output'].shape}, relevance: {all_wt['output'].sum().item():.8f}")

    # --- Step 2: zero-initialize all other nodes (on GPU) ---
    for name in reversed(list(node_io.keys())):
        if name == "output":
            continue

        info = node_io[name]
        layer = info["layer_name"]
        parents = info.get("input_sources", [])

        if layer in ("DL_Layer", "MLP_Layer"):
            activation_dict[name] = "None"

        inp_vals = info.get("input_values", [])
        if not isinstance(inp_vals, (list, tuple)):
            inp_vals = [inp_vals]

        func_name = info["func_name"]
        tensor_inputs = []
        for p, v in zip(parents, inp_vals):
            lt = node_io[p].get("layer_type", "")
            if lt in ("Weight", "Bias", "bn_running_mean", "bn_running_var", "bn_num_batches_tracked"):
                continue
            if func_name == "addmm" and lt == 'future_use':
                continue
            if isinstance(v, (torch.Tensor, np.ndarray)):
                tensor_inputs.append(v)

        if not tensor_inputs:
            out_val = tensor_to_numpy(info["output_values"])
            zeros = [torch.zeros(out_val.shape, dtype=torch.float32, device=device)]
        else:
            zeros = []
            for v in tensor_inputs:
                if isinstance(v, torch.Tensor):
                    zeros.append(torch.zeros(v.shape, dtype=torch.float32, device=device))
                else:
                    zeros.append(torch.zeros(v.shape, dtype=torch.float32, device=device))

        all_wt[name] = zeros if len(zeros) > 1 else zeros[0]

    # --- Step 3: backpropagate relevance (GPU-native) ---
    for name in tqdm(list(node_io.keys())[::-1], desc="Backtracing (GPU)"):
        if name == "output":
            continue

        symbolic_keywords = ("sym_size", "sym_int", "symbolic", "shape", "size", "dim")
        if any(k in name for k in symbolic_keywords):
            continue

        info = node_io[name]
        layer = info["layer_name"]
        func = info.get("func_name", "")
        parents = info.get("input_sources", [])
        children = info.get("output_children", [])

        def add_rel_gpu(r, parent_idx=None):
            """Accumulate relevance r into all_wt[name] as CUDA tensors."""
            if r is None:
                return
            if name.startswith("p_model_") or "weight" in name.lower():
                return

            buf = all_wt[name]

            if isinstance(buf, list):
                parts = r if isinstance(r, (list, tuple)) else [r]
                if parent_idx is not None:
                    if parent_idx < len(buf):
                        t = to_gpu_tensor(r, device) if not isinstance(r, torch.Tensor) else r
                        buf[parent_idx] = buf[parent_idx] + align_relevance_gpu(t, buf[parent_idx].shape)
                else:
                    for i in range(len(buf)):
                        if i < len(parts) and parts[i] is not None:
                            t = to_gpu_tensor(parts[i], device) if not isinstance(parts[i], torch.Tensor) else parts[i]
                            buf[i] = buf[i] + align_relevance_gpu(t, buf[i].shape)
                all_wt[name] = buf
            else:
                t = to_gpu_tensor(r, device) if not isinstance(r, torch.Tensor) else r
                all_wt[name] = buf + align_relevance_gpu(t, buf.shape)

        # — Activation: sum children's relevance —
        if layer == "Activation":
            for c in children:
                add_rel_gpu(get_relevance_from_child_gpu(c, name, all_wt, node_io))
            continue

        # — MLP (linear)
        if layer == "MLP_Layer":
            if func == "addmm":
                hp = info["layer_hyperparams"]
                bias, mat1, mat2 = info["input_values"]

                X = to_gpu_tensor(mat1, device)
                W = to_gpu_tensor(mat2, device).T
                B = None if isinstance(bias, bool) else to_gpu_tensor(bias, device)

                for c in children:
                    R = get_relevance_from_child_gpu(c, name, all_wt, node_io)
                    if R is not None:
                        δ = UD2.launch_linear_gpu(R, X, W, B, activation_master[activation_dict[name]])
                        add_rel_gpu(δ)
                continue

            else:
                hp = info["layer_hyperparams"]
                W = to_gpu_tensor(hp["weight"], device)
                B = None if isinstance(hp["bias"], bool) else to_gpu_tensor(hp["bias"], device)
                inp_vals_raw = info["input_values"]
                if isinstance(inp_vals_raw, (list, tuple)):
                    inp_vals_raw = inp_vals_raw[0]
                X = to_gpu_tensor(inp_vals_raw, device)

                for c in children:
                    R = get_relevance_from_child_gpu(c, name, all_wt, node_io)
                    if R is not None:
                        δ = UD2.launch_linear_gpu(R, X, W, B, activation_master[activation_dict[name]])
                        add_rel_gpu(δ)
                continue

        # — Conv2d — (still uses existing launcher since not yet tensor-native)
        if layer == "DL_Layer" and func == "conv2d":
            hp = info["layer_hyperparams"]
            W = hp["weight"].detach().cpu().numpy()
            B = None if isinstance(hp["bias"], bool) or hp["bias"] is None else hp["bias"].detach().cpu().numpy()
            stride, pad = hp["stride"], hp["padding"]
            X = process_input_for_eval(info["input_values"])

            for c in children:
                R = get_relevance_from_child_gpu(c, name, all_wt, node_io)
                if R is not None:
                    # Conv2d launcher expects numpy — convert for this layer only
                    R_np = R.cpu().numpy()
                    impl = get_layer_implementation("DL_Layer")
                    δ = UD2.launch_conv2d(impl, R_np, X, W, B, pad, stride, activation_master[activation_dict[name]])
                    add_rel_gpu(to_gpu_tensor(δ, device))
            continue

        # — Scaled dot-product attention — kept on GPU
        if layer == "Attention" and func == "scaled_dot_product_attention":
            vals = info.get("input_values", [])
            n = len(vals)

            if n == 4:
                Q, K, V, masked_fill = (to_gpu_tensor(v, device) for v in vals)
            elif n == 3:
                Q, K, V = (to_gpu_tensor(v, device) for v in vals)
                raw_mask = info.get("layer_hyperparams", {}).get("attn_mask", None)
                masked_fill = None if raw_mask is None else to_gpu_tensor(raw_mask, device)
            else:
                raise RuntimeError(f"[{name}] expected 3 or 4 inputs for attention, got {n}")

            for c in children:
                R = get_relevance_from_child_gpu(c, name, all_wt, node_io)
                if R is not None:
                    result = UD2.launch_self_attention_gpu(R, Q, K, V, masked_fill)
                    RQ, RK, RV, R_masked_fill = result
                    add_rel_gpu([RQ, RK, RV, R_masked_fill])
            continue

        # — Embedding — kept on GPU
        if layer == "NLP_Embedding" and func == "embedding":
            assign_embedding_relevance_gpu(name, info, all_wt, node_io, device)
            continue

        # — Elementwise multiply —
        if layer == "Mathematical_Operation" and func in ("mul", "mul_"):
            vals = info.get("input_values", [])
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            if len(vals) < 2:
                for c in children:
                    add_rel_gpu(get_relevance_from_child_gpu(c, name, all_wt, node_io))
            else:
                X = to_gpu_tensor(vals[0], device)
                Y = to_gpu_tensor(vals[1], device)

                for c in children:
                    R = get_relevance_from_child_gpu(c, name, all_wt, node_io)
                    if R is None:
                        continue

                    raw_parents = info.get("input_sources", [])
                    if isinstance(raw_parents, str):
                        try:
                            parent_names = ast.literal_eval(raw_parents)
                        except Exception:
                            continue
                    else:
                        parent_names = raw_parents

                    if len(parent_names) < 2:
                        continue

                    def is_weight(n):
                        return "weight" in n.lower()

                    is_X_weight = is_weight(parent_names[0])
                    is_Y_weight = is_weight(parent_names[1])

                    if is_X_weight and not is_Y_weight:
                        add_rel_gpu([torch.zeros_like(X), R])
                    elif is_Y_weight and not is_X_weight:
                        add_rel_gpu([R, torch.zeros_like(Y)])
                    else:
                        # Split relevance using GPU-native pytorch implementation
                        Rx, Ry = UD2.launch_wt_mul_gpu(R)
                        add_rel_gpu([Rx, Ry])
            continue

        # — Residual connections (add) —
        if layer == "Mathematical_Operation" and func == "add":
            vals = info.get("input_values", [])
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            if len(vals) < 2:
                for c in children:
                    add_rel_gpu(get_relevance_from_child_gpu(c, name, all_wt, node_io))
            else:
                X_t = to_gpu_tensor(vals[0], device)
                Y_t = to_gpu_tensor(vals[1], device)
                inp_t = [X_t, Y_t]
                for c in children:
                    R = get_relevance_from_child_gpu(c, name, all_wt, node_io)
                    if R is not None:
                        result = UD2.launch_wt_add_equal_gpu(R, inp_t)
                        add_rel_gpu(result)
            continue

        # — Norm / Indexing — passthrough
        if layer in {"Normalization", "Indexing_Operation"} or (
           layer == "Mathematical_Operation" and func not in ("mul", "mul_")):
            for c in children:
                add_rel_gpu(get_relevance_from_child_gpu(c, name, all_wt, node_io))
            continue

        # — Vector operations — all using torch ops
        if layer == "Vector_Operation":
            vals = info.get("input_values", [])
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            tensor_inputs = [v for v in vals if isinstance(v, (torch.Tensor, np.ndarray))]

            if not tensor_inputs:
                for c in children:
                    add_rel_gpu(get_relevance_from_child_gpu(c, name, all_wt, node_io))
                continue

            base = tensor_inputs[0]
            if isinstance(base, torch.Tensor):
                shape = base.shape
            else:
                shape = base.shape
            hp = info.get("layer_hyperparams", {})

            for c in children:
                R = get_relevance_from_child_gpu(c, name, all_wt, node_io)
                if R is None:
                    continue

                if not isinstance(R, torch.Tensor):
                    R = to_gpu_tensor(R, device)

                if func == "mean":
                    dims = hp.get("dim", None) or hp.get("dims", None)
                    if isinstance(dims, int):
                        dims = (dims,)
                    _target_numel = 1
                    for s in shape:
                        _target_numel *= s
                    if R.numel() != _target_numel:
                        dims = tuple(i for i, (r, s) in enumerate(zip(R.shape, shape)) if r == 1 and s > 1)
                    tot = 1.0
                    for d in dims:
                        tot *= shape[d]
                    R = R.expand(shape).contiguous() / tot

                elif func in {"view", "reshape", "flatten", "unflatten"}:
                    R = R.reshape(shape)

                elif func == "permute":
                    fwd = hp.get("dims", [])
                    inv = [0] * len(fwd)
                    for i, d in enumerate(fwd):
                        inv[d] = i
                    R = R.permute(inv)

                elif func == "transpose":
                    d0, d1 = info.get("method_args", (None, None))
                    R = R.transpose(d0, d1)

                elif func == "squeeze":
                    d = hp.get("dim")
                    if d is not None and shape[d] == 1:
                        R = R.unsqueeze(d)

                elif func == "unsqueeze":
                    d = hp.get("dim")
                    if d is not None:
                        R = R.squeeze(d)

                elif func == "slice":
                    dim = hp.get("dim", 0)
                    start = hp.get("start", 0)
                    end = hp.get("end", None)
                    step = hp.get("step", 1)

                    if all(x is not None for x in [dim, start, end]):
                        slicer = [slice(None)] * len(shape)
                        slicer[dim] = slice(start, end, step)
                        tmp = torch.zeros(shape, dtype=R.dtype, device=device)
                        reg = tmp[tuple(slicer)]

                        if R.shape != reg.shape:
                            if R.ndim == reg.ndim + 1:
                                R = torch.linalg.norm(R, dim=-1)
                            if R.shape[0] != reg.shape[0] and reg.shape[0] == 1:
                                R = R[:1]
                            elif R.shape[0] != reg.shape[0] and R.shape[0] == 1 and reg.shape[0] > 1:
                                R = R.expand(reg.shape).contiguous()
                            if R.shape != reg.shape and R.ndim == reg.ndim:
                                R = align_relevance_gpu(R, reg.shape)

                        tmp[tuple(slicer)] = R.reshape(reg.shape)
                        R = tmp

                elif func == "select":
                    dim, idx = hp.get("dim", 0), hp.get("index", 0)
                    if isinstance(idx, torch.Tensor):
                        idx = int(idx)
                    sl = [slice(None)] * len(shape)
                    sl[dim] = idx
                    tmp = torch.zeros(shape, dtype=R.dtype, device=device)
                    reg = tmp[tuple(sl)]
                    R2 = R.reshape(reg.shape) if R.shape != reg.shape else R
                    tmp[tuple(sl)] = R2
                    R = tmp

                elif func == "expand":
                    sizes = hp.get("sizes") or hp.get("size") or hp.get("shape")
                    if sizes:
                        exp = tuple(sizes)
                        try:
                            resolved_exp = tuple(
                                s if isinstance(s, int) and s > 0 else R.shape[i]
                                for i, s in enumerate(exp)
                            )
                            if R.shape != resolved_exp:
                                R = R.reshape(resolved_exp)
                            for ax, (o, e) in enumerate(zip(shape, resolved_exp)):
                                if isinstance(o, int) and isinstance(e, int):
                                    if o == 1 and e > 1:
                                        R = R.sum(dim=ax, keepdim=True)
                            R = R.reshape(shape)
                        except Exception:
                            try:
                                axes_to_sum = tuple(
                                    i for i, (o, e) in enumerate(zip(shape, R.shape))
                                    if o == 1 and e > 1
                                )
                                if len(shape) == R.ndim and axes_to_sum:
                                    R = R.sum(dim=axes_to_sum, keepdim=True).reshape(shape)
                                else:
                                    _tgt = 1
                                    for s in shape:
                                        _tgt *= s
                                    if R.numel() == _tgt:
                                        R = R.reshape(shape)
                            except Exception as e_fallback:
                                raise ValueError(f"[{name}] Failed to handle expand: {e_fallback}")

                elif func == "cat":
                    dim_cat = hp.get("dim", 0)
                    # Resolve the actual cat-input shapes from parent nodes,
                    # because input_values may be stored as a single tuple.
                    parent_names = info.get("input_sources", [])
                    if isinstance(parent_names, str):
                        try:
                            parent_names = ast.literal_eval(parent_names)
                        except Exception:
                            parent_names = []
                    cat_inputs = []
                    for pn in parent_names:
                        pinfo = node_io.get(pn, {})
                        pout = pinfo.get("output_values", None)
                        if pout is not None and isinstance(pout, torch.Tensor):
                            cat_inputs.append(pout)
                    # Fallback to tensor_inputs if parent lookup failed
                    if not cat_inputs:
                        cat_inputs = tensor_inputs
                    sizes = [
                        t.shape[dim_cat] for t in cat_inputs
                    ]
                    parts = torch.split(R, sizes, dim=dim_cat)
                    parts = [torch.clamp(p, min=0) for p in parts]
                    total = R.sum()
                    current_sum = sum(p.sum() for p in parts)
                    if current_sum > 0:
                        scale = total / current_sum
                        parts = [p * scale for p in parts]
                    for i, p in enumerate(parts):
                        add_rel_gpu(p, parent_idx=i)
                    continue

                elif func in {"contiguous", "to"}:
                    pass

                else:
                    _tgt = 1
                    for s in shape:
                        _tgt *= s
                    if R.numel() == _tgt:
                        R = R.reshape(shape)

                add_rel_gpu(R)
            continue

        # — Fallback pass-through —
        for c in children:
            add_rel_gpu(get_relevance_from_child_gpu(c, name, all_wt, node_io))

    # --- Step 4: convert all results to numpy (single bulk GPU→CPU transfer) ---
    result = {}
    for name, val in all_wt.items():
        if isinstance(val, torch.Tensor):
            result[name] = val.detach().cpu().numpy()
        elif isinstance(val, list):
            result[name] = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in val]
        else:
            result[name] = val

    return result

class RelevancePropagator:
    """Encapsulates relevance propagation logic with memory optimization."""
    def __init__(self, graph, node_io, activation_master, get_layer_implementation=None):
        self.graph = graph
        self.node_io = node_io
        self.activation_master = activation_master
        self.get_layer_implementation = get_layer_implementation or (lambda x: "original")

    def _uses_gpu(self):
        """Check if any layer is configured to use CUDA."""
        # Test a representative set of layer types
        for layer_type in ["MLP_Layer", "Attention", "NLP_Embedding", "DL_Layer"]:
            if self.get_layer_implementation(layer_type) == "cuda":
                return True
        return False

    def propagate(
        self,
        start_wt=None,
        mode="default",
        multiplier=100.0,
        scaler=1.0,
        thresholding=0.5,
        task="binary-classification",
        target_token_ids=None,
        debug=False,
    ):
        global DEBUG
        DEBUG = debug

        eval_fn = run_evaluation_gpu if self._uses_gpu() else run_evaluation

        return eval_fn(
            self.node_io,
            self.activation_master,
            mode=mode,
            start_wt=start_wt,
            multiplier=multiplier,
            scaler=scaler,
            thresholding=thresholding,
            task=task,
            target_token_ids=target_token_ids,
            get_layer_implementation=self.get_layer_implementation,
        )

