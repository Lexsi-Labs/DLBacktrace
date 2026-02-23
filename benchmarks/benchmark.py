#!/usr/bin/env python3
"""
DLBacktrace — Sequence-Scaling Benchmark
=========================================

Measures time and memory at each pipeline stage as input sequence length grows.
Outputs a human-readable table to stdout and a JSON report to disk.

Stages measured per sequence length:
  1. DLBacktrace Init  (torch.export + graph build)
  2. Forward Pass       (ExecutionEngineNoCache)
  3. Backward Pass      (RelevancePropagator — GPU)
  4. Total              (aggregate)

Memory tracked:
  - GPU VRAM via torch.cuda.max_memory_allocated()
  - System RAM via psutil.Process().memory_info().rss

Usage:
  python benchmarks/benchmark.py                             # defaults
  python benchmarks/benchmark.py --seq-lengths 8 32 128 512  # custom
  python benchmarks/benchmark.py --model meta-llama/Llama-3.2-1B --runs 3
"""

import argparse
import gc
import json
import os
import random
import string
import sys
import time
from datetime import datetime
from typing import Any, Dict, List
from dl_backtrace.pytorch_backtrace import DLBacktrace
import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.export import Dim

# ── Deterministic setup ─────────────────────────────────────────────────────
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_LOGS"] = "+dynamic"

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.use_deterministic_algorithms(True)

num_cores = os.cpu_count()
if num_cores:
    torch.set_num_threads(num_cores)


# ── Helpers ─────────────────────────────────────────────────────────────────

def generate_sentence(approx_tokens: int) -> str:
    """Generate a random sentence with approximately *approx_tokens* tokens."""
    words = []
    word_count = int(approx_tokens * 0.75)
    for _ in range(word_count):
        length = random.randint(3, 8)
        words.append("".join(random.choices(string.ascii_lowercase, k=length)))
    return " ".join(words)


def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


class MemTracker:
    """Context manager that tracks peak GPU VRAM and system RAM delta."""

    def __init__(self, device: str):
        self.device = device
        self.ram_before = 0
        self.ram_after = 0
        self.vram_peak = 0.0
        self.process = psutil.Process(os.getpid())

    def __enter__(self):
        gc.collect()
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.ram_before = self.process.memory_info().rss
        return self

    def __exit__(self, *args):
        if self.device == "cuda":
            torch.cuda.synchronize()
            self.vram_peak = bytes_to_mb(torch.cuda.max_memory_allocated())
        self.ram_after = self.process.memory_info().rss

    @property
    def ram_delta_mb(self) -> float:
        return bytes_to_mb(self.ram_after - self.ram_before)

    @property
    def vram_peak_mb(self) -> float:
        return self.vram_peak


# ── Model Wrapper ───────────────────────────────────────────────────────────

class ModelWrapper(nn.Module):
    def __init__(self, model_id: str, token: str):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, token=token
        ).eval()

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits


# ── Benchmark Core ──────────────────────────────────────────────────────────

def benchmark_single(
    model,
    tokenizer,
    seq_len: int,
    device: str,
    run_idx: int = 0,
) -> Dict[str, Any]:
    """Run the full DLBacktrace pipeline for one sequence length and return metrics."""

    sentence = generate_sentence(seq_len)
    tokens = tokenizer(
        [sentence],
        return_tensors="pt",
        padding="max_length",
        max_length=seq_len,
        truncation=True,
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    actual_seq_len = input_ids.shape[1]

    seq_dim = Dim("seq", min=1, max=actual_seq_len)
    dynamic_shapes = {
        "input_ids":      {0: 1, 1: seq_dim},
        "attention_mask": {0: 1, 1: seq_dim},
    }

    result: Dict[str, Any] = {
        "sequence_length": actual_seq_len,
        "run_idx": run_idx,
    }

    # ─── Stage 1: DLBacktrace Init (export + graph build) ───
    with MemTracker(device) as mem:
        t0 = time.perf_counter()
        ir = DLBacktrace(
            model,
            (input_ids, attention_mask),
            dynamic_shapes=dynamic_shapes,
            device=device,
            verbose=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    result["init_time_s"] = t1 - t0
    result["init_vram_mb"] = mem.vram_peak_mb
    result["init_ram_delta_mb"] = mem.ram_delta_mb

    # ─── Stage 2: Forward Pass ──────────────────────────────
    with MemTracker(device) as mem:
        t0 = time.perf_counter()
        ir.predict(input_ids, attention_mask)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    result["forward_time_s"] = t1 - t0
    result["forward_vram_mb"] = mem.vram_peak_mb
    result["forward_ram_delta_mb"] = mem.ram_delta_mb

    # ─── Stage 3: Backward Pass (GPU relevance propagation) ─
    with MemTracker(device) as mem:
        t0 = time.perf_counter()
        ir.evaluation(task="generation", multiplier=100.0, debug=False)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    result["backward_time_s"] = t1 - t0
    result["backward_vram_mb"] = mem.vram_peak_mb
    result["backward_ram_delta_mb"] = mem.ram_delta_mb

    # ─── Aggregates ─────────────────────────────────────────
    result["total_time_s"] = (
        result["init_time_s"] + result["forward_time_s"] + result["backward_time_s"]
    )
    result["throughput_tok_per_s"] = (
        actual_seq_len / result["backward_time_s"]
        if result["backward_time_s"] > 0
        else 0.0
    )

    # Clean up to free memory for next run
    del ir
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


# ── Reporting ───────────────────────────────────────────────────────────────

def print_table(records: List[Dict[str, Any]]):
    """Print a human-readable table of benchmark results."""
    try:
        from tabulate import tabulate
    except ImportError:
        # Fallback: plain print
        for r in records:
            print(r)
        return

    headers = [
        "Seq Len",
        "Run",
        "Init (s)",
        "Forward (s)",
        "Backward (s)",
        "Total (s)",
        "Throughput\n(tok/s)",
        "Init\nVRAM (MB)",
        "Fwd\nVRAM (MB)",
        "Bwd\nVRAM (MB)",
        "Init\nΔRAM (MB)",
        "Fwd\nΔRAM (MB)",
        "Bwd\nΔRAM (MB)",
    ]
    rows = []
    for r in records:
        rows.append([
            r["sequence_length"],
            r["run_idx"],
            f"{r['init_time_s']:.3f}",
            f"{r['forward_time_s']:.3f}",
            f"{r['backward_time_s']:.3f}",
            f"{r['total_time_s']:.3f}",
            f"{r['throughput_tok_per_s']:.1f}",
            f"{r['init_vram_mb']:.1f}",
            f"{r['forward_vram_mb']:.1f}",
            f"{r['backward_vram_mb']:.1f}",
            f"{r['init_ram_delta_mb']:.1f}",
            f"{r['forward_ram_delta_mb']:.1f}",
            f"{r['backward_ram_delta_mb']:.1f}",
        ])

    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))


def print_summary(records: List[Dict[str, Any]]):
    """Print an averaged summary grouped by sequence length."""
    try:
        from tabulate import tabulate
    except ImportError:
        return

    # Group by sequence length
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in records:
        grouped[r["sequence_length"]].append(r)

    headers = [
        "Seq Len",
        "Runs",
        "Avg Init (s)",
        "Avg Forward (s)",
        "Avg Backward (s)",
        "Avg Total (s)",
        "Avg Throughput\n(tok/s)",
        "Peak Fwd\nVRAM (MB)",
        "Peak Bwd\nVRAM (MB)",
    ]
    rows = []
    for seq_len in sorted(grouped.keys()):
        runs = grouped[seq_len]
        n = len(runs)
        rows.append([
            seq_len,
            n,
            f"{np.mean([r['init_time_s'] for r in runs]):.3f}",
            f"{np.mean([r['forward_time_s'] for r in runs]):.3f}",
            f"{np.mean([r['backward_time_s'] for r in runs]):.3f}",
            f"{np.mean([r['total_time_s'] for r in runs]):.3f}",
            f"{np.mean([r['throughput_tok_per_s'] for r in runs]):.1f}",
            f"{max(r['forward_vram_mb'] for r in runs):.1f}",
            f"{max(r['backward_vram_mb'] for r in runs):.1f}",
        ])

    print("\n" + "=" * 70)
    print("  SUMMARY (averaged across runs)")
    print("=" * 70)
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def save_report(records: List[Dict[str, Any]], system_info: Dict[str, Any], output_dir: str):
    """Save a JSON report to *output_dir* with a timestamped filename."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"benchmark_{timestamp}.json")

    report = {
        "timestamp": timestamp,
        "system_info": system_info,
        "results": records,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Report saved to: {path}")
    return path


# ── System Info ─────────────────────────────────────────────────────────────

def gather_system_info(device: str, model_id: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "model_id": model_id,
        "device": device,
        "cpu_count": os.cpu_count(),
        "pytorch_version": torch.__version__,
        "python_version": sys.version,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_vram_total_mb"] = bytes_to_mb(
            torch.cuda.get_device_properties(0).total_memory
        )
        info["cuda_version"] = torch.version.cuda
    return info


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DLBacktrace sequence-scaling benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model ID (default: Llama-3.2-1B)",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512],
        help="Sequence lengths to benchmark (default: 8 16 32 64 128 256 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for layer implementations (default: cuda)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repeat count per sequence length for averaging (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory for JSON report output (default: benchmarks/results)",
    )
    args = parser.parse_args()

    # Validate
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    # ── Load model & tokenizer ──────────────────────────────
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    from transformers import AutoTokenizer

    print(f"📦 Loading model: {args.model}")
    model = ModelWrapper(args.model, hf_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Model loaded\n")

    system_info = gather_system_info(args.device, args.model)

    # ── Print header ────────────────────────────────────────
    print("=" * 70)
    print("  DLBacktrace — Sequence-Scaling Benchmark")
    print("=" * 70)
    print(f"  Model         : {args.model}")
    print(f"  Device        : {args.device}")
    if torch.cuda.is_available():
        print(f"  GPU           : {torch.cuda.get_device_name()}")
    print(f"  Seq lengths   : {args.seq_lengths}")
    print(f"  Runs per len  : {args.runs}")
    print(f"  Output dir    : {args.output_dir}")
    print("=" * 70)

    # ── Run benchmarks ──────────────────────────────────────
    all_records: List[Dict[str, Any]] = []

    for seq_len in args.seq_lengths:
        for run_idx in range(args.runs):
            tag = f"seq={seq_len}, run={run_idx + 1}/{args.runs}"
            print(f"\n▶ {tag}")

            try:
                record = benchmark_single(
                    model=model,
                    tokenizer=tokenizer,
                    seq_len=seq_len,
                    device=args.device,
                    run_idx=run_idx,
                )
                record["success"] = True
                all_records.append(record)

                print(
                    f"  ✅ Init: {record['init_time_s']:.3f}s | "
                    f"Fwd: {record['forward_time_s']:.3f}s | "
                    f"Bwd: {record['backward_time_s']:.3f}s | "
                    f"Throughput: {record['throughput_tok_per_s']:.1f} tok/s"
                )
            except Exception as e:
                print(f"  ❌ Error: {e}")
                all_records.append({
                    "sequence_length": seq_len,
                    "run_idx": run_idx,
                    "success": False,
                    "error": str(e),
                })

    # ── Report ──────────────────────────────────────────────
    successful = [r for r in all_records if r.get("success")]
    if successful:
        print_table(successful)
        if args.runs > 1:
            print_summary(successful)
        save_report(all_records, system_info, args.output_dir)
    else:
        print("\n⚠️  No successful benchmark runs.")

    print("\n🎯 Benchmark complete!")


if __name__ == "__main__":
    main()
