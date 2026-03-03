"""
Analyze the structure of DLBacktrace relevance output (rel_dict / all_wt)
to find the fastest serialization strategy.

Profiles:
  1. Number and shape distribution of tensors
  2. Total memory footprint  
  3. Dtype distribution
  4. Whether tensors can be concatenated into a flat buffer
  5. Benchmarks: torch.save, torch.save+LZ4, safetensors, raw memcpy
"""

import time
import io
import sys
import gc
import numpy as np
import torch
import os
import torch.nn as nn
from torch.export import Dim
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import random
import string
import json
from typing import Dict, Any, Tuple, List

# ── Setup DLBacktrace and run one predict + backtrace ──
print("=" * 70)
print("  Relevance Output Analysis")
print("=" * 70)

MODEL = "meta-llama/Llama-3.2-1B"
PROMPT = "Explain the difference between O+ and O- blood type."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

print(f"\nLoading model: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, device_map=DEVICE, token=hf_token
)

tokens = tokenizer(PROMPT, return_tensors="pt", padding=True, truncation=True)
input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

seq_len = input_ids.shape[1]
seq_dim = Dim("seq", min=1, max=seq_len)
dynamic_shapes = {
    "input_ids": {0: 1, 1: seq_dim},
    "attention_mask": {0: 1, 1: seq_dim},
}

print(f"Input prompt: {PROMPT}")
print(f"Input IDs shape: {input_ids.shape}")

from dl_backtrace.pytorch_backtrace import DLBacktrace

print("Setting up DL-Backtrace...")
ir = DLBacktrace(
    model,
    (input_ids, attention_mask),
    dynamic_shapes=dynamic_shapes,
    device='cuda',  # or 'cpu'
    verbose=False
)

print("Running predict...")
io_data = ir.predict(input_ids, attention_mask, debug=False)

# Extract logits and get target token
from dl_backtrace.pytorch_backtrace.dlbacktrace.core.dlb_auto_sampler import DLBAutoSampler
logits = DLBAutoSampler._extract_last_logits(io_data)
next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
target_ids = [int(next_token.view(-1)[0].item())]
print(f"Target token: {target_ids} → '{tokenizer.decode(target_ids)}'")

print("Running backtrace...")
t0 = time.perf_counter()
rel_dict = ir.evaluation(
    mode="default", start_wt=[], multiplier=100.0,
    scaler=1.0, thresholding=0.5, task="generation", debug=False,
)
if DEVICE == "cuda":
    torch.cuda.synchronize()
bt_time = time.perf_counter() - t0
print(f"Backtrace time: {bt_time:.2f}s")

# ── Analyze the output ──
print("\n" + "=" * 70)
print("  1. STRUCTURE ANALYSIS")
print("=" * 70)

all_tensors = []
all_keys = []
non_tensor_keys = []
list_keys = []

def collect_tensors(data, prefix=""):
    if isinstance(data, torch.Tensor):
        all_tensors.append(data)
        all_keys.append(prefix)
    elif isinstance(data, np.ndarray):
        t = torch.from_numpy(data)
        all_tensors.append(t)
        all_keys.append(prefix)
    elif isinstance(data, dict):
        for k, v in data.items():
            collect_tensors(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            collect_tensors(v, f"{prefix}[{i}]")
    else:
        non_tensor_keys.append((prefix, type(data).__name__, repr(data)[:80]))

collect_tensors(rel_dict)

print(f"\nTotal keys in rel_dict: {len(rel_dict) if isinstance(rel_dict, dict) else 'N/A'}")
print(f"Total tensors found:   {len(all_tensors)}")
print(f"Non-tensor entries:    {len(non_tensor_keys)}")

if non_tensor_keys:
    print(f"\nNon-tensor entries (first 10):")
    for k, t, v in non_tensor_keys[:10]:
        print(f"  {k}: {t} = {v}")

# ── Shape distribution ──
print("\n" + "=" * 70)
print("  2. SHAPE & SIZE DISTRIBUTION")
print("=" * 70)

shapes = {}
dtypes = {}
total_bytes = 0
total_elements = 0

for t in all_tensors:
    s = tuple(t.shape)
    shapes[s] = shapes.get(s, 0) + 1
    d = str(t.dtype)
    dtypes[d] = dtypes.get(d, 0) + 1
    total_bytes += t.nelement() * t.element_size()
    total_elements += t.nelement()

print(f"\nTotal memory: {total_bytes / 1024**2:.1f} MB ({total_bytes / 1024**3:.2f} GB)")
print(f"Total elements: {total_elements:,}")

print(f"\nDtype distribution:")
for d, count in sorted(dtypes.items(), key=lambda x: -x[1]):
    print(f"  {d}: {count} tensors")

print(f"\nShape distribution (top 20 by count):")
sorted_shapes = sorted(shapes.items(), key=lambda x: -x[1])
for shape, count in sorted_shapes[:20]:
    size_mb = 1
    for d in shape:
        size_mb *= d
    size_mb = size_mb * 4 / 1024**2  # assuming float32
    print(f"  {str(shape):>30s}: {count:4d} tensors  ({size_mb:.3f} MB each)")

print(f"\nShape distribution (top 10 by total size):")
shape_total_size = {}
for t in all_tensors:
    s = tuple(t.shape)
    shape_total_size[s] = shape_total_size.get(s, 0) + t.nelement() * t.element_size()
sorted_by_size = sorted(shape_total_size.items(), key=lambda x: -x[1])
for shape, total in sorted_by_size[:10]:
    print(f"  {str(shape):>30s}: {total / 1024**2:8.1f} MB total  ({shapes[shape]} tensors)")

# ── Can we concatenate? ──
print("\n" + "=" * 70)
print("  3. CONCATENATION FEASIBILITY")
print("=" * 70)

unique_dtypes = set(str(t.dtype) for t in all_tensors)
print(f"\nUnique dtypes: {unique_dtypes}")

all_same_dtype = len(unique_dtypes) == 1
print(f"All same dtype: {all_same_dtype}")

if all_same_dtype:
    # Can we flatten all into one contiguous tensor?
    total_flat = sum(t.nelement() for t in all_tensors)
    print(f"If flattened into one tensor: {total_flat:,} elements = {total_flat * 4 / 1024**2:.1f} MB")
    print(f"This would enable memcpy-based serialization (fastest possible)")

# ── Check if tensors are already contiguous ──
non_contiguous = sum(1 for t in all_tensors if not t.is_contiguous())
print(f"\nNon-contiguous tensors: {non_contiguous} / {len(all_tensors)}")

on_gpu = sum(1 for t in all_tensors if t.is_cuda)
on_cpu = len(all_tensors) - on_gpu
print(f"On GPU: {on_gpu}, On CPU: {on_cpu}")

# ── Benchmark serialization strategies ──
print("\n" + "=" * 70)
print("  4. SERIALIZATION BENCHMARKS")
print("=" * 70)

# First move everything to CPU
cpu_tensors = []
for t in all_tensors:
    ct = t.detach().cpu() if t.is_cuda else t.detach()
    cpu_tensors.append(ct)

cpu_dict = {}
if isinstance(rel_dict, dict):
    def to_cpu_dict(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu() if data.is_cuda else data.detach()
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        if isinstance(data, dict):
            return {k: to_cpu_dict(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(to_cpu_dict(v) for v in data)
        return data
    cpu_dict = to_cpu_dict(rel_dict)

gc.collect()

# Strategy 1: torch.save (baseline)
print("\n  Strategy 1: torch.save (pickle, no compression)")
buf = io.BytesIO()
t0 = time.perf_counter()
torch.save(cpu_dict, buf, pickle_protocol=4)
t1 = time.perf_counter()
pickle_size = buf.tell()
print(f"  Time:  {t1-t0:.3f}s")
print(f"  Size:  {pickle_size / 1024**2:.1f} MB")

# Strategy 2: torch.save + LZ4
try:
    import lz4.frame
    print("\n  Strategy 2: torch.save + LZ4")
    buf2 = io.BytesIO()
    t0 = time.perf_counter()
    torch.save(cpu_dict, buf2, pickle_protocol=4)
    raw = buf2.getvalue()
    t1 = time.perf_counter()
    compressed = lz4.frame.compress(raw)
    t2 = time.perf_counter()
    print(f"  Serialize: {t1-t0:.3f}s")
    print(f"  Compress:  {t2-t1:.3f}s")
    print(f"  Total:     {t2-t0:.3f}s")
    print(f"  Raw size:  {len(raw) / 1024**2:.1f} MB")
    print(f"  LZ4 size:  {len(compressed) / 1024**2:.1f} MB")
    print(f"  Ratio:     {len(compressed)/len(raw)*100:.1f}%")
    del raw, compressed, buf2
except ImportError:
    print("\n  Strategy 2: SKIPPED (lz4 not installed)")

# Strategy 3: Flat tensor + metadata (fastest possible)
print("\n  Strategy 3: Flat tensor concat + raw bytes")
t0 = time.perf_counter()
flat_parts = []
metadata = []
offset = 0
for key, tensor in zip(all_keys, cpu_tensors):
    flat_parts.append(tensor.reshape(-1))
    metadata.append((key, list(tensor.shape), offset, tensor.nelement()))
    offset += tensor.nelement()
flat_tensor = torch.cat(flat_parts)
t1 = time.perf_counter()
# Write raw bytes
raw_bytes = flat_tensor.numpy().tobytes()
t2 = time.perf_counter()
print(f"  Concat:    {t1-t0:.3f}s")
print(f"  To bytes:  {t2-t1:.3f}s")
print(f"  Total:     {t2-t0:.3f}s")
print(f"  Size:      {len(raw_bytes) / 1024**2:.1f} MB")
del flat_parts, flat_tensor, raw_bytes

# Strategy 4: safetensors (if available)
try:
    from safetensors.torch import save as st_save
    print("\n  Strategy 4: safetensors")
    # safetensors needs flat dict of tensors
    st_dict = {}
    for key, tensor in zip(all_keys, cpu_tensors):
        safe_key = key.replace(".", "__DOT__").replace("[", "__LB__").replace("]", "__RB__")
        st_dict[safe_key] = tensor.contiguous()
    
    t0 = time.perf_counter()
    buf_st = io.BytesIO()
    # safetensors save to file, not buffer — use save_file workaround
    from safetensors.torch import save as st_save_dict
    st_bytes = st_save_dict(st_dict)
    t1 = time.perf_counter()
    print(f"  Time:  {t1-t0:.3f}s")
    print(f"  Size:  {len(st_bytes) / 1024**2:.1f} MB")
    del st_dict, st_bytes
except ImportError:
    print("\n  Strategy 4: SKIPPED (safetensors not installed)")

# Strategy 5: torch.save with protocol 5 (out-of-band buffers)
print("\n  Strategy 5: torch.save (pickle protocol 5)")
buf5 = io.BytesIO()
t0 = time.perf_counter()
torch.save(cpu_dict, buf5, pickle_protocol=5)
t1 = time.perf_counter()
p5_size = buf5.tell()
print(f"  Time:  {t1-t0:.3f}s")
print(f"  Size:  {p5_size / 1024**2:.1f} MB")

# Strategy 6: numpy savez_compressed
print("\n  Strategy 6: numpy savez_compressed")
np_dict = {str(i): cpu_tensors[i].numpy() for i in range(len(cpu_tensors))}
buf6 = io.BytesIO()
t0 = time.perf_counter()
np.savez_compressed(buf6, **np_dict)
t1 = time.perf_counter()
np_size = buf6.tell()
print(f"  Time:  {t1-t0:.3f}s")
print(f"  Size:  {np_size / 1024**2:.1f} MB")
del np_dict

# Strategy 7: Flat tensor + LZ4 (no pickle overhead)
print("\n  Strategy 7: Flat tensor concat + LZ4 (no pickle)")
try:
    import lz4.frame
    t0 = time.perf_counter()
    flat_parts = []
    for tensor in cpu_tensors:
        flat_parts.append(tensor.reshape(-1))
    flat_tensor = torch.cat(flat_parts)
    raw = flat_tensor.numpy().tobytes()
    t1 = time.perf_counter()
    compressed = lz4.frame.compress(raw)
    t2 = time.perf_counter()
    print(f"  Concat+tobytes: {t1-t0:.3f}s")
    print(f"  LZ4 compress:   {t2-t1:.3f}s")
    print(f"  Total:          {t2-t0:.3f}s")
    print(f"  Raw size:       {len(raw) / 1024**2:.1f} MB")
    print(f"  LZ4 size:       {len(compressed) / 1024**2:.1f} MB")
    del flat_parts, flat_tensor, raw, compressed
except ImportError:
    print("  SKIPPED (lz4 not installed)")

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)
