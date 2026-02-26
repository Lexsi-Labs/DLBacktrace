"""
Compiled relevance propagation — pre-analyzes graph structure once,
then executes LRP backpropagation with minimal Python overhead.

Key optimizations vs run_evaluation_gpu():
  1. Integer-indexed buffers instead of dict (all_wt[name] → buffers[idx])
  2. Pre-computed child-to-parent slot mappings (no .index() at runtime)
  3. Integer op-code dispatch (no string comparison chains)
  4. No tqdm progress bar
  5. Returns GPU tensors directly (no numpy conversion)
  6. Pre-classified weight/bias nodes (no string matching at runtime)
"""

import ast
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any

from dl_backtrace.pytorch_backtrace.dlbacktrace.utils import default_v2 as UD2
from .relevance_propagation import (
    tensor_to_numpy,
    to_gpu_tensor,
    align_relevance_gpu,
    assign_embedding_relevance_gpu,
)

# ─── Op codes ───
OP_SKIP         = 0
OP_PASSTHROUGH  = 1
OP_LINEAR_ADDMM = 2
OP_LINEAR_OTHER = 3
OP_ATTENTION    = 4
OP_EMBEDDING    = 5
OP_MUL          = 6
OP_ADD          = 7
OP_VECTOR       = 8

_WEIGHT_LT = frozenset({"Weight", "Bias", "bn_running_mean", "bn_running_var", "bn_num_batches_tracked"})
_SYM_KEYS = ("sym_size", "sym_int", "symbolic", "shape", "size", "dim")


# ═══════════════════════════════════════════════════════════════════════
#  Buffer-dict view (thin wrapper for embedding handler compatibility)
# ═══════════════════════════════════════════════════════════════════════

class _BufferView:
    """Dict-like wrapper so assign_embedding_relevance_gpu can use buffers."""
    __slots__ = ('_buf', '_n2i')

    def __init__(self, buffers, name_to_idx):
        self._buf = buffers
        self._n2i = name_to_idx

    def __getitem__(self, name):
        return self._buf[self._n2i[name]]

    def __setitem__(self, name, value):
        self._buf[self._n2i[name]] = value

    def __contains__(self, name):
        idx = self._n2i.get(name)
        return idx is not None and self._buf[idx] is not None

    def get(self, name, default=None):
        idx = self._n2i.get(name)
        if idx is not None:
            val = self._buf[idx]
            return val if val is not None else default
        return default


# ═══════════════════════════════════════════════════════════════════════
#  PropagationSchedule — built once, reused per token
# ═══════════════════════════════════════════════════════════════════════

class PropagationSchedule:
    """Pre-analyzed graph structure for fast relevance propagation."""

    def __init__(self, node_io: dict, activation_master: dict):
        names = list(node_io.keys())[::-1]
        n = len(names)

        self.node_names: List[str] = names
        self.name_to_idx: Dict[str, int] = {nm: i for i, nm in enumerate(names)}
        self.n_nodes: int = n

        # Per-node arrays
        self.op_codes    = [OP_SKIP] * n
        self.skip_mask   = [True]    * n
        self.func_names  = ['']      * n
        self.is_list_buf = [False]   * n

        # child_slots[idx] = [(child_idx, slot), ...]
        # slot = index of this node in child's input_sources (-1 if scalar buf)
        self.child_slots: List[List[Tuple[int, int]]] = [[] for _ in range(n)]

        # For MUL nodes: (is_X_weight, is_Y_weight) pre-computed
        self.mul_wt_flags: List[Optional[Tuple[bool, bool]]] = [None] * n

        # Weight/parameter nodes: skip accumulation (matches original add_rel_gpu guard)
        self.is_weight_node = [False] * n

        # Activation dict for MLP/DL nodes
        self.act_dict: Dict[str, str] = {}

        # Input masks: for zero-init, which (parent, value) pairs are non-weight?
        # input_mask[idx] = list of bool, aligned with parents + input_values
        self.input_masks: List[Optional[List[bool]]] = [None] * n

        self._build(node_io)

    def _build(self, node_io: dict):
        names = self.node_names
        n = self.n_nodes
        n2i = self.name_to_idx

        for idx in range(n):
            name = names[idx]
            if name == "output":
                continue
            if any(k in name for k in _SYM_KEYS):
                continue

            self.skip_mask[idx] = False

            # Mark weight/parameter nodes (matches original add_rel_gpu guard)
            if name.startswith("p_model_") or "weight" in name.lower():
                self.is_weight_node[idx] = True

            info = node_io[name]
            layer = info["layer_name"]
            func  = info.get("func_name", "")
            self.func_names[idx] = func
            parents  = info.get("input_sources", [])
            children = info.get("output_children", [])

            # Activation tracking
            if layer in ("DL_Layer", "MLP_Layer"):
                self.act_dict[name] = "None"

            # Classify
            if layer == "Activation":
                self.op_codes[idx] = OP_PASSTHROUGH
            elif layer == "MLP_Layer":
                self.op_codes[idx] = OP_LINEAR_ADDMM if func == "addmm" else OP_LINEAR_OTHER
            elif layer == "Attention" and func == "scaled_dot_product_attention":
                self.op_codes[idx] = OP_ATTENTION
            elif layer == "NLP_Embedding" and func == "embedding":
                self.op_codes[idx] = OP_EMBEDDING
            elif layer == "Mathematical_Operation":
                if func in ("mul", "mul_"):
                    self.op_codes[idx] = OP_MUL
                    if len(parents) >= 2:
                        self.mul_wt_flags[idx] = (
                            "weight" in parents[0].lower(),
                            "weight" in parents[1].lower(),
                        )
                elif func == "add":
                    self.op_codes[idx] = OP_ADD
                else:
                    self.op_codes[idx] = OP_PASSTHROUGH
            elif layer in ("Normalization", "Indexing_Operation"):
                self.op_codes[idx] = OP_PASSTHROUGH
            elif layer == "Vector_Operation":
                self.op_codes[idx] = OP_VECTOR
            else:
                self.op_codes[idx] = OP_PASSTHROUGH

            # Build input mask (which parent values are non-weight tensors)
            inp_vals = info.get("input_values", [])
            if not isinstance(inp_vals, (list, tuple)):
                inp_vals = [inp_vals]
            mask = []
            tensor_count = 0
            for pi, p in enumerate(parents):
                if pi >= len(inp_vals):
                    mask.append(False)
                    continue
                v = inp_vals[pi]
                p_info = node_io.get(p, {})
                lt = p_info.get("layer_type", "")
                if lt in _WEIGHT_LT:
                    mask.append(False)
                    continue
                if func == "addmm" and lt == "future_use":
                    mask.append(False)
                    continue
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    mask.append(True)
                    tensor_count += 1
                else:
                    mask.append(False)
            self.input_masks[idx] = mask
            self.is_list_buf[idx] = tensor_count > 1

            # Resolve child indices
            child_idxs = []
            for c in children:
                ci = n2i.get(c)
                if ci is not None:
                    child_idxs.append(ci)

            # Build child-to-parent slot mapping
            slots = []
            for ci in child_idxs:
                child_name = names[ci]
                child_info = node_io.get(child_name, {})
                child_srcs = child_info.get("input_sources", [])
                try:
                    slot = child_srcs.index(name)
                except ValueError:
                    slot = -1
                slots.append((ci, slot))
            self.child_slots[idx] = slots


# ═══════════════════════════════════════════════════════════════════════
#  Fast propagation runner
# ═══════════════════════════════════════════════════════════════════════

def _get_child_rel(buffers, ci, slot, ci_is_list):
    """Get relevance from child ci at the given slot. Inlined for speed."""
    val = buffers[ci]
    if val is None:
        return None
    if ci_is_list:
        if isinstance(val, list) and 0 <= slot < len(val):
            return val[slot]
        return val if not isinstance(val, list) else None
    return val


def _accum(buffers, idx, r, is_list, device, is_wt=False):
    """Accumulate relevance r into buffers[idx]. Inlined for speed."""
    if r is None:
        return
    # Skip weight/parameter nodes (matches original add_rel_gpu guard)
    if is_wt:
        return
    buf = buffers[idx]
    if is_list and isinstance(buf, list):
        parts = r if isinstance(r, (list, tuple)) else [r]
        for i in range(min(len(buf), len(parts))):
            if parts[i] is not None:
                t = parts[i] if isinstance(parts[i], torch.Tensor) else to_gpu_tensor(parts[i], device)
                buf[i] = buf[i] + align_relevance_gpu(t, buf[i].shape)
    else:
        t = r if isinstance(r, torch.Tensor) else to_gpu_tensor(r, device)
        buffers[idx] = buf + align_relevance_gpu(t, buf.shape)


def run_propagation_compiled(
    schedule: PropagationSchedule,
    node_io: dict,
    activation_master: dict,
    *,
    mode: str = "default",
    start_wt=None,
    multiplier: float = 100.0,
    scaler: float = 1.0,
    thresholding: float = 0.5,
    task: str = "binary-classification",
    target_token_ids=None,
    get_layer_implementation=None,
) -> dict:
    """
    Fast LRP propagation using pre-computed schedule.
    Returns GPU tensors by default (no numpy conversion).
    """
    device = torch.device("cuda")
    n = schedule.n_nodes
    names = schedule.node_names
    n2i = schedule.name_to_idx
    op_codes  = schedule.op_codes
    skip_mask = schedule.skip_mask
    is_list   = schedule.is_list_buf
    child_slots = schedule.child_slots
    func_names  = schedule.func_names

    # ── Step 1: seed output ──
    output_idx = n2i["output"]
    raw_out = node_io["output"]["input_values"]
    if isinstance(raw_out, (list, tuple)):
        raw_out = raw_out[0]
    out_np = tensor_to_numpy(raw_out)

    seed = UD2.calculate_start_wt(
        out_np, scaler=scaler, task=task,
        target_indices=target_token_ids, thresholding=thresholding,
    )
    seed_np = tensor_to_numpy(seed)
    if seed_np.size == 0 or np.all(seed_np == 0):
        seed_np = np.ones_like(out_np, dtype=np.float32)

    buffers = [None] * n
    buffers[output_idx] = torch.tensor(
        seed_np * np.float32(multiplier), dtype=torch.float32, device=device
    )

    # ── Step 2: zero-init buffers (using pre-computed input masks) ──
    for idx in range(n):
        if idx == output_idx:
            continue
        if skip_mask[idx]:
            buffers[idx] = torch.zeros(1, dtype=torch.float32, device=device)
            continue

        name = names[idx]
        info = node_io[name]
        mask = schedule.input_masks[idx]
        parents = info.get("input_sources", [])
        inp_vals = info.get("input_values", [])
        if not isinstance(inp_vals, (list, tuple)):
            inp_vals = [inp_vals]

        zeros = []
        if mask is not None:
            for pi, keep in enumerate(mask):
                if keep and pi < len(inp_vals):
                    v = inp_vals[pi]
                    shape = v.shape if isinstance(v, torch.Tensor) else np.asarray(v).shape
                    zeros.append(torch.zeros(shape, dtype=torch.float32, device=device))

        if not zeros:
            ov = info["output_values"]
            shape = ov.shape if isinstance(ov, torch.Tensor) else np.asarray(ov).shape
            buffers[idx] = torch.zeros(shape, dtype=torch.float32, device=device)
        elif len(zeros) == 1:
            buffers[idx] = zeros[0]
        else:
            buffers[idx] = zeros

    # ── Step 3: propagation loop (integer dispatch, no tqdm) ──
    buf_view = _BufferView(buffers, n2i)  # for embedding handler
    wt_flags = schedule.is_weight_node

    for idx in range(n):
        if skip_mask[idx]:
            continue

        op = op_codes[idx]
        name = names[idx]
        slots = child_slots[idx]
        is_wt = wt_flags[idx]  # skip accumulation for weight/param nodes

        try:
            # ── PASSTHROUGH ──
            if op == OP_PASSTHROUGH:
                for ci, slot in slots:
                    R = _get_child_rel(buffers, ci, slot, is_list[ci])
                    if R is not None:
                        _accum(buffers, idx, R, is_list[idx], device, is_wt)
                continue

            info = node_io[name]

            # ── LINEAR (addmm) ──
            if op == OP_LINEAR_ADDMM:
                bias_v, mat1, mat2 = info["input_values"]
                X = to_gpu_tensor(mat1, device)
                W = to_gpu_tensor(mat2, device).T
                B = None if isinstance(bias_v, bool) else to_gpu_tensor(bias_v, device)
                act_key = schedule.act_dict.get(name, "None")
                act = activation_master[act_key]
                for ci, slot in slots:
                    R = _get_child_rel(buffers, ci, slot, is_list[ci])
                    if R is not None:
                        delta = UD2.launch_linear_gpu(R, X, W, B, act)
                        _accum(buffers, idx, delta, is_list[idx], device, is_wt)
                continue

            # ── LINEAR (other) ──
            if op == OP_LINEAR_OTHER:
                hp = info["layer_hyperparams"]
                W = to_gpu_tensor(hp["weight"], device)
                B = None if isinstance(hp["bias"], bool) else to_gpu_tensor(hp["bias"], device)
                inp_raw = info["input_values"]
                if isinstance(inp_raw, (list, tuple)):
                    inp_raw = inp_raw[0]
                X = to_gpu_tensor(inp_raw, device)
                act_key = schedule.act_dict.get(name, "None")
                act = activation_master[act_key]
                for ci, slot in slots:
                    R = _get_child_rel(buffers, ci, slot, is_list[ci])
                    if R is not None:
                        delta = UD2.launch_linear_gpu(R, X, W, B, act)
                        _accum(buffers, idx, delta, is_list[idx], device, is_wt)
                continue

            # ── ATTENTION ──
            if op == OP_ATTENTION:
                vals = info.get("input_values", [])
                nv = len(vals)
                if nv == 4:
                    Q, K, V, mf = (to_gpu_tensor(v, device) for v in vals)
                elif nv == 3:
                    Q, K, V = (to_gpu_tensor(v, device) for v in vals)
                    raw_mask = info.get("layer_hyperparams", {}).get("attn_mask", None)
                    mf = None if raw_mask is None else to_gpu_tensor(raw_mask, device)
                else:
                    raise RuntimeError(f"[{name}] expected 3 or 4 inputs for attention, got {nv}")
                for ci, slot in slots:
                    R = _get_child_rel(buffers, ci, slot, is_list[ci])
                    if R is not None:
                        RQ, RK, RV, Rmf = UD2.launch_self_attention_gpu(R, Q, K, V, mf)
                        _accum(buffers, idx, [RQ, RK, RV, Rmf], is_list[idx], device, is_wt)
                continue

            # ── EMBEDDING ──
            if op == OP_EMBEDDING:
                assign_embedding_relevance_gpu(name, info, buf_view, node_io, device)
                continue

            # ── MUL ──
            if op == OP_MUL:
                vals = info.get("input_values", [])
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                if len(vals) < 2:
                    for ci, slot in slots:
                        _accum(buffers, idx,
                               _get_child_rel(buffers, ci, slot, is_list[ci]),
                               is_list[idx], device, is_wt)
                else:
                    X = to_gpu_tensor(vals[0], device)
                    Y = to_gpu_tensor(vals[1], device)
                    wt_flags_mul = schedule.mul_wt_flags[idx]
                    for ci, slot in slots:
                        R = _get_child_rel(buffers, ci, slot, is_list[ci])
                        if R is None:
                            continue
                        if wt_flags_mul is not None:
                            is_Xw, is_Yw = wt_flags_mul
                            if is_Xw and not is_Yw:
                                _accum(buffers, idx, [torch.zeros_like(X), R], is_list[idx], device, is_wt)
                            elif is_Yw and not is_Xw:
                                _accum(buffers, idx, [R, torch.zeros_like(Y)], is_list[idx], device, is_wt)
                            else:
                                Rx, Ry = UD2.launch_wt_mul_gpu(R)
                                _accum(buffers, idx, [Rx, Ry], is_list[idx], device, is_wt)
                        else:
                            Rx, Ry = UD2.launch_wt_mul_gpu(R)
                            _accum(buffers, idx, [Rx, Ry], is_list[idx], device, is_wt)
                continue

            # ── ADD ──
            if op == OP_ADD:
                vals = info.get("input_values", [])
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                if len(vals) < 2:
                    for ci, slot in slots:
                        _accum(buffers, idx,
                               _get_child_rel(buffers, ci, slot, is_list[ci]),
                               is_list[idx], device, is_wt)
                else:
                    X_t = to_gpu_tensor(vals[0], device)
                    Y_t = to_gpu_tensor(vals[1], device)
                    for ci, slot in slots:
                        R = _get_child_rel(buffers, ci, slot, is_list[ci])
                        if R is not None:
                            result = UD2.launch_wt_add_equal_gpu(R, [X_t, Y_t])
                            _accum(buffers, idx, result, is_list[idx], device, is_wt)
                continue

            # ── VECTOR ──
            if op == OP_VECTOR:
                func = func_names[idx]
                vals = info.get("input_values", [])
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                tensor_inputs = [v for v in vals if isinstance(v, (torch.Tensor, np.ndarray))]

                if not tensor_inputs:
                    for ci, slot in slots:
                        _accum(buffers, idx,
                               _get_child_rel(buffers, ci, slot, is_list[ci]),
                               is_list[idx], device, is_wt)
                    continue

                base = tensor_inputs[0]
                shape = base.shape
                hp = info.get("layer_hyperparams", {})

                for ci, slot in slots:
                    R = _get_child_rel(buffers, ci, slot, is_list[ci])
                    if R is None:
                        continue
                    if not isinstance(R, torch.Tensor):
                        R = to_gpu_tensor(R, device)

                    R = _apply_vector_op(R, func, shape, hp, info, idx, names, node_io, buffers, is_list, device)
                    if R is not None:
                        _accum(buffers, idx, R, is_list[idx], device, is_wt)
                continue

            # ── Fallback ──
            for ci, slot in slots:
                _accum(buffers, idx,
                       _get_child_rel(buffers, ci, slot, is_list[ci]),
                       is_list[idx], device, is_wt)

        except Exception as e:
            import traceback
            raise RuntimeError(
                f"[compiled_propagation] Error at node '{name}' "
                f"(idx={idx}, op={op}, func='{func_names[idx]}'): {e}\n"
                f"{traceback.format_exc()}"
            ) from e

    # ── Step 4: bulk GPU→CPU transfer + numpy conversion ──
    # Issue all .cpu() calls first, then synchronize once, then convert to numpy.
    # This avoids the per-tensor sync that makes torch.save(GPU tensor) slow.
    torch.cuda.synchronize()

    result = {}
    for idx in range(n):
        val = buffers[idx]
        if val is None:
            continue
        name_out = names[idx]
        if isinstance(val, torch.Tensor):
            result[name_out] = val.detach().cpu().numpy()
        elif isinstance(val, list):
            result[name_out] = [
                v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                for v in val
            ]
        else:
            result[name_out] = val

    # Free buffer array
    del buffers
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Vector operation sub-dispatch
# ═══════════════════════════════════════════════════════════════════════

def _apply_vector_op(R, func, shape, hp, info, idx, names, node_io, buffers, is_list, device):
    """Handle vector operations (mean, view, permute, transpose, etc.)."""

    if func == "mean":
        dims = hp.get("dim", None) or hp.get("dims", None)
        if isinstance(dims, int):
            dims = (dims,)
        tgt_numel = 1
        for s in shape:
            tgt_numel *= s
        if R.numel() != tgt_numel:
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
        dim, sel_idx = hp.get("dim", 0), hp.get("index", 0)
        if isinstance(sel_idx, torch.Tensor):
            sel_idx = int(sel_idx)
        sl = [slice(None)] * len(shape)
        sl[dim] = sel_idx
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
                resolved = tuple(
                    s if isinstance(s, int) and s > 0 else R.shape[i]
                    for i, s in enumerate(exp)
                )
                if R.shape != resolved:
                    R = R.reshape(resolved)
                for ax, (o, e) in enumerate(zip(shape, resolved)):
                    if isinstance(o, int) and isinstance(e, int):
                        if o == 1 and e > 1:
                            R = R.sum(dim=ax, keepdim=True)
                R = R.reshape(shape)
            except Exception:
                try:
                    axes = tuple(
                        i for i, (o, e) in enumerate(zip(shape, R.shape))
                        if o == 1 and e > 1
                    )
                    if len(shape) == R.ndim and axes:
                        R = R.sum(dim=axes, keepdim=True).reshape(shape)
                    else:
                        tgt = 1
                        for s in shape:
                            tgt *= s
                        if R.numel() == tgt:
                            R = R.reshape(shape)
                except Exception as e2:
                    raise ValueError(f"Failed to handle expand: {e2}")

    elif func == "cat":
        dim_cat = hp.get("dim", 0)
        method_args = info.get("method_args", ())
        cat_sizes = []
        if method_args and isinstance(method_args[0], (list, tuple)):
            for key in method_args[0]:
                skey = str(key)
                pinfo = node_io.get(skey, {})
                pout = pinfo.get("output_values", None)
                if pout is not None and isinstance(pout, torch.Tensor):
                    cat_sizes.append(pout.shape[dim_cat])

        vals = info.get("input_values", [])
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        if not cat_sizes and isinstance(vals, (list, tuple)):
            tv = [v for v in vals if isinstance(v, torch.Tensor)]
            if len(tv) > 1:
                cat_sizes = [t.shape[dim_cat] for t in tv]

        if not cat_sizes or sum(cat_sizes) != R.shape[dim_cat]:
            parent_names = info.get("input_sources", [])
            if isinstance(parent_names, str):
                try:
                    parent_names = ast.literal_eval(parent_names)
                except Exception:
                    parent_names = []
            n_parents = max(len(parent_names), 1)
            r_dim = R.shape[dim_cat]
            if r_dim % n_parents == 0:
                cat_sizes = [r_dim // n_parents] * n_parents
            else:
                cat_sizes = [r_dim]

        parts = torch.split(R, cat_sizes, dim=dim_cat)
        parts = [torch.clamp(p, min=0) for p in parts]
        total = R.sum()
        current_sum = sum(p.sum() for p in parts)
        if current_sum > 0:
            scale = total / current_sum
            parts = [p * scale for p in parts]

        name = names[idx]
        buf = buffers[idx]
        if isinstance(buf, list):
            for i, p in enumerate(parts):
                if i < len(buf):
                    t = p if isinstance(p, torch.Tensor) else to_gpu_tensor(p, device)
                    buf[i] = buf[i] + align_relevance_gpu(t, buf[i].shape)
        else:
            if len(parts) == 1:
                t = parts[0] if isinstance(parts[0], torch.Tensor) else to_gpu_tensor(parts[0], device)
                buffers[idx] = buf + align_relevance_gpu(t, buf.shape)
        # Return sentinel to skip normal accumulation
        return None

    elif func in {"contiguous", "to"}:
        pass  # R unchanged

    else:
        tgt = 1
        for s in shape:
            tgt *= s
        if R.numel() == tgt:
            R = R.reshape(shape)

    return R
