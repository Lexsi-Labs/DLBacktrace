#!/usr/bin/env python3
"""
Verify that compiled propagation produces the same relevance as the original.

Runs both paths on the same model+input, compares all node relevances.
"""
import copy
import os
import sys
import time

import numpy as np
import torch

# Deterministic setup (same as benchmark)
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.use_deterministic_algorithms(True)

from dl_backtrace.pytorch_backtrace import DLBacktrace
from dl_backtrace.pytorch_backtrace.dlbacktrace.activation import activation_master
from dl_backtrace.pytorch_backtrace.dlbacktrace.core.relevance_propagation import (
    RelevancePropagator,
    run_evaluation_gpu,
)
from dl_backtrace.pytorch_backtrace.dlbacktrace.core.compiled_propagation import (
    PropagationSchedule,
    run_propagation_compiled,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load model ────────────────────────────────────────────────────────────
print("Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map=DEVICE, token=hf_token
)
model.eval()

# ── Tokenize ──────────────────────────────────────────────────────────────
PROMPT = "Explain the difference between O+ and O- blood type."
tokens = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
input_ids = tokens["input_ids"]
attn_mask = tokens["attention_mask"]
print(f"Prompt: {PROMPT!r}  →  {input_ids.shape[1]} tokens")

# ── Export model ──────────────────────────────────────────────────────────
print("Exporting model...")
from torch.export import Dim, export

seq_dim = Dim("seq", min=1, max=model.config.max_position_embeddings)
ep = export(
    model,
    (input_ids, attn_mask),
    dynamic_shapes={"input_ids": {1: seq_dim}, "attention_mask": {1: seq_dim}},
    strict=False,
)
print("✅ Export done")

# ── Setup DLBacktrace ────────────────────────────────────────────────────
dlbt = DLBacktrace(
            model,
            (input_ids, attention_mask),
            dynamic_shapes=dynamic_shapes,
            device=device,
            verbose=False,
        )
dlbt.build_graph(ep)
node_io = dlbt.predict(input_ids, attn_mask, debug=False)
print(f"✅ Predict done, {len(node_io)} nodes")

# ── Deep-copy node_io so we can run both paths ────────────────────────────
print("Deep-copying node_io for two runs...")
node_io_original = copy.deepcopy(node_io)
node_io_compiled = copy.deepcopy(node_io)

# ── Build propagation schedule ────────────────────────────────────────────
print("Building PropagationSchedule...")
schedule = PropagationSchedule(node_io_compiled, activation_master)
print(f"✅ Schedule built: {schedule.n_nodes} nodes")

# ── Run original (run_evaluation_gpu) ─────────────────────────────────────
print("\n═══ Running ORIGINAL (run_evaluation_gpu) ═══")
t0 = time.time()
result_orig = run_evaluation_gpu(
    node_io_original,
    activation_master,
    mode="default",
    start_wt=[],
    multiplier=100.0,
    scaler=1.0,
    thresholding=0.5,
    task="binary-classification",
    target_token_ids=None,
    get_layer_implementation=dlbt.get_layer_implementation,
)
t_orig = time.time() - t0
print(f"✅ Original done in {t_orig:.2f}s, {len(result_orig)} nodes in result")

# ── Run compiled (run_propagation_compiled) ───────────────────────────────
print("\n═══ Running COMPILED (run_propagation_compiled) ═══")
t0 = time.time()
result_comp = run_propagation_compiled(
    schedule,
    node_io_compiled,
    activation_master,
    mode="default",
    start_wt=[],
    multiplier=100.0,
    scaler=1.0,
    thresholding=0.5,
    task="binary-classification",
    target_token_ids=None,
    get_layer_implementation=dlbt.get_layer_implementation,
)
t_comp = time.time() - t0
print(f"✅ Compiled done in {t_comp:.2f}s, {len(result_comp)} nodes in result")

# ── Compare results ───────────────────────────────────────────────────────
print("\n═══ COMPARISON ═══")

all_keys = set(result_orig.keys()) | set(result_comp.keys())
only_orig = set(result_orig.keys()) - set(result_comp.keys())
only_comp = set(result_comp.keys()) - set(result_orig.keys())
common = set(result_orig.keys()) & set(result_comp.keys())

print(f"Total keys: orig={len(result_orig)}, compiled={len(result_comp)}, common={len(common)}")
if only_orig:
    print(f"⚠️  Only in original ({len(only_orig)}): {list(only_orig)[:10]}...")
if only_comp:
    print(f"⚠️  Only in compiled ({len(only_comp)}): {list(only_comp)[:10]}...")

def to_array(val):
    """Convert to numpy array for comparison."""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    if isinstance(val, np.ndarray):
        return val
    if isinstance(val, (list, tuple)):
        return [to_array(v) for v in val]
    return val

mismatches = []
close_matches = []
exact_matches = 0
total_compared = 0

for key in sorted(common):
    v_orig = to_array(result_orig[key])
    v_comp = to_array(result_comp[key])

    if isinstance(v_orig, list) and isinstance(v_comp, list):
        if len(v_orig) != len(v_comp):
            mismatches.append((key, f"list length: {len(v_orig)} vs {len(v_comp)}"))
            continue
        all_close = True
        max_rdiff = 0.0
        for i, (a, b) in enumerate(zip(v_orig, v_comp)):
            a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
            if a.shape != b.shape:
                mismatches.append((key, f"list[{i}] shape: {a.shape} vs {b.shape}"))
                all_close = False
                break
            denom = np.abs(a).max()
            if denom > 0:
                rdiff = np.abs(a - b).max() / denom
            else:
                rdiff = np.abs(a - b).max()
            max_rdiff = max(max_rdiff, rdiff)
            if not np.allclose(a, b, rtol=1e-3, atol=1e-6):
                all_close = False
        if all_close:
            if max_rdiff < 1e-6:
                exact_matches += 1
            else:
                close_matches.append((key, max_rdiff))
        else:
            mismatches.append((key, f"list values differ, max_rdiff={max_rdiff:.6e}"))
        total_compared += 1
        continue

    v_orig = np.asarray(v_orig, dtype=np.float32)
    v_comp = np.asarray(v_comp, dtype=np.float32)

    if v_orig.shape != v_comp.shape:
        mismatches.append((key, f"shape: {v_orig.shape} vs {v_comp.shape}"))
        total_compared += 1
        continue

    denom = np.abs(v_orig).max()
    if denom > 0:
        rdiff = np.abs(v_orig - v_comp).max() / denom
    else:
        rdiff = np.abs(v_orig - v_comp).max()

    if np.allclose(v_orig, v_comp, rtol=1e-3, atol=1e-6):
        if rdiff < 1e-6:
            exact_matches += 1
        else:
            close_matches.append((key, rdiff))
    else:
        mismatches.append((key, f"rdiff={rdiff:.6e}, max_abs={np.abs(v_orig - v_comp).max():.6e}"))

    total_compared += 1

print(f"\nResults: {total_compared} nodes compared")
print(f"  ✅ Exact matches (rdiff < 1e-6): {exact_matches}")
print(f"  ≈ Close matches (rtol=1e-3):    {len(close_matches)}")
print(f"  ❌ Mismatches:                   {len(mismatches)}")

if close_matches:
    print(f"\nClose matches (top 10 by rdiff):")
    for key, rdiff in sorted(close_matches, key=lambda x: -x[1])[:10]:
        print(f"  {key}: rdiff={rdiff:.6e}")

if mismatches:
    print(f"\n❌ MISMATCHES (top 20):")
    for key, desc in mismatches[:20]:
        print(f"  {key}: {desc}")

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
if not mismatches:
    print(f"✅ ALL {total_compared} nodes match (rtol=1e-3, atol=1e-6)")
    print(f"   Compiled speedup: {t_orig:.2f}s → {t_comp:.2f}s ({t_orig/t_comp:.1f}x)")
else:
    print(f"❌ {len(mismatches)} nodes DO NOT MATCH — needs investigation")
print(f"{'='*60}")
