#!/usr/bin/env python3
"""
DLBacktrace — Benchmark Suite

Two benchmark modes:
  1. Sequence-Scaling   — Fixed 1-token output, vary input sequence length
  2. Generation-Scaling — Fixed input, vary max_new_tokens (multi-token output)

Each mode measures time and memory (GPU VRAM + system RAM) per pipeline stage.

Usage:
  python engine.py                                           # both modes, defaults
  python engine.py --mode seq                                # sequence scaling only
  python engine.py --mode gen --gen-tokens 1 5 10 20        # generation scaling only
  python engine.py --seq-lengths 8 32 128 --gen-tokens 1 5  # custom lengths
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.export import Dim

from dl_backtrace.pytorch_backtrace import DLBacktrace

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


# ── Curated question prompts (real sentences, varying token counts) ─────────

PROMPTS = {
    # ~8 tokens
    8:   "What is gravity?",
    # ~16 tokens
    16:  "Can you explain how photosynthesis works in simple terms?",
    # ~32 tokens
    32:  "What are the main differences between machine learning and deep learning, and when should you use one over the other in practice?",
    # ~64 tokens
    64:  ("Explain the concept of transformer architecture in neural networks. "
          "How does self-attention work, and why has it become the dominant approach "
          "for natural language processing tasks? Include a brief comparison with "
          "recurrent neural networks and their limitations."),
    # ~128 tokens
    128: ("You are a computer science professor teaching an advanced course on "
          "artificial intelligence. A student asks you to explain the complete "
          "pipeline of training a large language model from scratch, including "
          "data collection, tokenization, model architecture design, pre-training "
          "objectives, optimization strategies, and fine-tuning techniques. "
          "The student also wants to understand the computational requirements "
          "and the environmental impact of training such models. Please provide "
          "a comprehensive overview covering all these aspects."),
    # ~256 tokens
    256: ("Write a detailed technical analysis of the evolution of neural network "
          "architectures from the early perceptron model to modern transformer-based "
          "large language models. Your analysis should cover the following key "
          "milestones and innovations: the original perceptron and its limitations "
          "with the XOR problem, the development of backpropagation and multi-layer "
          "perceptrons, the introduction of convolutional neural networks for image "
          "recognition tasks, the rise of recurrent neural networks and long short-term "
          "memory networks for sequential data processing, the attention mechanism "
          "and its revolutionary impact on sequence-to-sequence models, the transformer "
          "architecture introduced in the landmark 'Attention Is All You Need' paper, "
          "the scaling laws that govern modern large language models, and the emergence "
          "of techniques like reinforcement learning from human feedback. For each "
          "milestone, explain the key technical innovation, why it was important, "
          "and how it addressed limitations of previous approaches. Also discuss "
          "the computational and data requirements at each stage of this evolution."),
    # ~512 tokens
    512: ("You are an expert in explainable artificial intelligence and interpretability "
          "methods for deep neural networks. Write a comprehensive survey covering the "
          "entire landscape of XAI methods, organized by category. Begin with gradient-based "
          "methods including vanilla gradients, integrated gradients, and gradient-weighted "
          "class activation mapping. Then cover perturbation-based methods such as LIME, "
          "SHAP, and occlusion sensitivity analysis. Discuss attention-based interpretability "
          "including attention rollout, attention flow, and the debate about whether attention "
          "weights provide meaningful explanations. Cover concept-based explanations like "
          "TCAV and network dissection. Examine layer-wise relevance propagation and its "
          "variants including deep Taylor decomposition and the alpha-beta rule. Discuss "
          "counterfactual explanations and their relationship to causal inference. Address "
          "the evaluation of explanation methods, including faithfulness metrics, human "
          "evaluation studies, and the axioms that good explanations should satisfy such "
          "as sensitivity and implementation invariance. Compare the computational costs "
          "and scalability of different methods when applied to modern large language models "
          "with billions of parameters. Discuss the unique challenges of explaining "
          "autoregressive language models compared to classification models, including "
          "the need to explain token-level predictions and the compounding effects of "
          "sequential generation. Address recent advances in mechanistic interpretability, "
          "including circuit discovery, probing classifiers, and sparse autoencoders for "
          "understanding internal representations. Finally, discuss the regulatory landscape "
          "including the EU AI Act requirements for transparency and how current XAI methods "
          "do or do not meet these requirements. For each method or category, provide the "
          "mathematical formulation, key assumptions, known limitations, and practical "
          "recommendations for when to use each approach."),
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_prompt_for_seq_len(target_len: int) -> str:
    """Return the best-matching curated prompt for the target sequence length."""
    # Find closest available prompt
    available = sorted(PROMPTS.keys())
    best = min(available, key=lambda k: abs(k - target_len))
    return PROMPTS[best]


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


class MoEBacktraceWrapper(nn.Module):
    def __init__(self, model_id: str, token: str):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, token=token
        ).eval()

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

# ═══════════════════════════════════════════════════════════════════════════
#  MODE 1: Sequence-Scaling Benchmark
#  Fixed output (1 token), vary input sequence length
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_seq_scaling(
    model,
    tokenizer,
    seq_len: int,
    device: str,
    run_idx: int = 0,
) -> Dict[str, Any]:
    """Run the full DLBacktrace pipeline for one sequence length (1-token output)."""

    prompt = get_prompt_for_seq_len(seq_len)
    tokens = tokenizer(
        [prompt],
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
        "mode": "seq_scaling",
        "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
        "sequence_length": actual_seq_len,
        "run_idx": run_idx,
    }

    # ─── Stage 1: DLBacktrace Init ──────────────────────────
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

    # ─── Stage 3: Backward Pass (relevance propagation) ─────
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

    del ir
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  MODE 2: Generation-Scaling Benchmark
#  Fixed input, vary max_new_tokens (multi-token output via run_task)
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_gen_scaling(
    model,
    tokenizer,
    max_new_tokens: int,
    device: str,
    input_prompt: str = "What is the capital of France?",
    run_idx: int = 0,
    cache_dir: str = None,
    explain_tokens = "all",
) -> Dict[str, Any]:
    """Benchmark multi-token generation using run_task(task='generation').
    
    Uses disk cache policy: all per-step data (relevance, scores, IO)
    is streamed to disk instead of accumulating in RAM.
    """

    tokens = tokenizer(
        [input_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    input_seq_len = input_ids.shape[1]

    seq_dim = Dim("seq", min=1, max=input_seq_len)
    dynamic_shapes = {
        "input_ids":      {0: 1, 1: seq_dim},
        "attention_mask": {0: 1, 1: seq_dim},
    }

    result: Dict[str, Any] = {
        "mode": "gen_scaling",
        "prompt": input_prompt[:80] + ("..." if len(input_prompt) > 80 else ""),
        "input_seq_len": input_seq_len,
        "max_new_tokens": max_new_tokens,
        "run_idx": run_idx,
    }

    # ─── Stage 1: DLBacktrace Init ──────────────────────────
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

    # ─── Stage 2+3: Generation (forward + backward per token) ─
    #     Disk cache policy: streams relevance, scores, IO to disk per step
    os.makedirs(cache_dir, exist_ok=True)
    with MemTracker(device) as mem:
        t0 = time.perf_counter()
        gen_results = ir.run_task(
            task="generation",
            inputs={"input_ids": input_ids, "attention_mask": attention_mask},
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            return_relevance=False,
            return_scores=False,
            debug=False,
            explain_tokens=explain_tokens,
            relevance_cache_policy="disk",
            relevance_cache_dir=cache_dir,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    result["generation_time_s"] = t1 - t0
    result["generation_vram_mb"] = mem.vram_peak_mb
    result["generation_ram_delta_mb"] = mem.ram_delta_mb

    # ─── Extract generation stats ───────────────────────────
    generated_ids = gen_results.get("generated_ids")
    if generated_ids is not None:
        output_seq_len = generated_ids.shape[1]
        actual_new_tokens = output_seq_len - input_seq_len
        generated_text = tokenizer.decode(
            generated_ids[0, input_seq_len:], skip_special_tokens=True
        )
    else:
        actual_new_tokens = 0
        generated_text = ""

    num_steps = len(gen_results.get("relevance_trace", []))

    result["actual_new_tokens"] = actual_new_tokens
    result["generation_steps"] = num_steps
    result["generated_text"] = generated_text[:200]

    # ─── Aggregates ─────────────────────────────────────────
    result["total_time_s"] = result["init_time_s"] + result["generation_time_s"]
    result["time_per_token_s"] = (
        result["generation_time_s"] / actual_new_tokens
        if actual_new_tokens > 0
        else 0.0
    )
    result["throughput_tok_per_s"] = (
        actual_new_tokens / result["generation_time_s"]
        if result["generation_time_s"] > 0
        else 0.0
    )

    del ir
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  MODE 3: MoE Backtrace
# ═══════════════════════════════════════════════════════════════════════════

def moe_backtrace(
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int,
    moe_type: str = None,
    explain_tokens="all",
) -> Dict[str, Any]:
    """Run MoE backtrace generation with auto-detected or explicit model type."""
    from dl_backtrace.moe_pytorch_backtrace import Backtrace

    tokens = tokenizer(
        prompt,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    input_seq_len = input_ids.shape[1]

    result: Dict[str, Any] = {
        "mode": "moe_backtrace",
        "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
        "input_seq_len": input_seq_len,
        "max_new_tokens": max_new_tokens,
    }

    with MemTracker(device) as mem:
        t0 = time.perf_counter()
        backtrace = Backtrace(
            model=model,
            model_type=moe_type,  # None = auto-detect
            device=device,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    result["init_time_s"] = t1 - t0
    result["init_vram_mb"] = mem.vram_peak_mb
    result["init_ram_delta_mb"] = mem.ram_delta_mb

    with MemTracker(device) as mem:
        t0 = time.perf_counter()
        results = backtrace.run_task(
            task="generation",
            inputs={"input_ids": input_ids, "attention_mask": attention_mask},
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            return_relevance=True,
            return_scores=False,
            debug=False,
            explain_tokens=explain_tokens,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    result["generation_time_s"] = t1 - t0
    result["generation_vram_mb"] = mem.vram_peak_mb
    result["generation_ram_delta_mb"] = mem.ram_delta_mb
    
    # Decode output
    generated_ids = results.get("generated_ids")
    output_seq_len = generated_ids.shape[1] if generated_ids is not None else input_seq_len
    actual_new_tokens = max(output_seq_len - input_seq_len, 0)
    generated_text = tokenizer.decode(generated_ids[0, input_seq_len:], skip_special_tokens=True)
    print(f"\n✅ Generated text: {generated_text}")

    result["actual_new_tokens"] = actual_new_tokens
    result["generation_steps"] = len(results.get("relevance_trace", []))
    result["generated_text"] = generated_text[:200]
    result["total_time_s"] = result["init_time_s"] + result["generation_time_s"]
    result["time_per_token_s"] = (
        result["generation_time_s"] / actual_new_tokens
        if actual_new_tokens > 0
        else 0.0
    )
    result["throughput_tok_per_s"] = (
        actual_new_tokens / result["generation_time_s"]
        if result["generation_time_s"] > 0
        else 0.0
    )

    backtrace.clear_intermediates(clear_relevance=True)
    del backtrace, input_ids, attention_mask, tokens, results
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_seq_table(records: List[Dict[str, Any]]):
    """Print sequence-scaling results."""
    try:
        from tabulate import tabulate
    except ImportError:
        for r in records:
            print(r)
        return

    headers = [
        "Seq Len", "Run",
        "Init (s)", "Forward (s)", "Backward (s)", "Total (s)",
        "Throughput\n(tok/s)",
        "Init\nVRAM (MB)", "Fwd\nVRAM (MB)", "Bwd\nVRAM (MB)",
        "Init\nΔRAM (MB)", "Fwd\nΔRAM (MB)", "Bwd\nΔRAM (MB)",
    ]
    rows = []
    for r in records:
        rows.append([
            r["sequence_length"], r["run_idx"],
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

    print("\n" + "=" * 70)
    print("  SEQUENCE SCALING (vary input length, 1-token output)")
    print("=" * 70)
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_gen_table(records: List[Dict[str, Any]]):
    """Print generation-scaling results."""
    try:
        from tabulate import tabulate
    except ImportError:
        for r in records:
            print(r)
        return

    headers = [
        "Max New\nTokens", "Actual\nTokens", "Steps", "Run",
        "Init (s)", "Gen (s)", "Total (s)",
        "Per Token\n(s)", "Throughput\n(tok/s)",
        "Init\nVRAM (MB)", "Gen\nVRAM (MB)",
        "Init\nΔRAM (MB)", "Gen\nΔRAM (MB)",
    ]
    rows = []
    for r in records:
        rows.append([
            r["max_new_tokens"], r["actual_new_tokens"], r["generation_steps"],
            r["run_idx"],
            f"{r['init_time_s']:.3f}",
            f"{r['generation_time_s']:.3f}",
            f"{r['total_time_s']:.3f}",
            f"{r['time_per_token_s']:.3f}",
            f"{r['throughput_tok_per_s']:.2f}",
            f"{r['init_vram_mb']:.1f}",
            f"{r['generation_vram_mb']:.1f}",
            f"{r['init_ram_delta_mb']:.1f}",
            f"{r['generation_ram_delta_mb']:.1f}",
        ])

    print("\n" + "=" * 70)
    print("  GENERATION SCALING (fixed input, vary max_new_tokens)")
    print("=" * 70)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Print generated text for each run
    print("\n  Generated text samples:")
    for r in records:
        print(f"    [{r['max_new_tokens']} tokens] → \"{r.get('generated_text', '')}\"")


def print_moe_table(records: List[Dict[str, Any]]):
    """Print MoE generation benchmark results."""
    try:
        from tabulate import tabulate
    except ImportError:
        for r in records:
            print(r)
        return

    headers = [
        "Max New\nTokens", "Actual\nTokens", "Steps",
        "Init (s)", "Gen (s)", "Total (s)",
        "Per Token\n(s)", "Throughput\n(tok/s)",
        "Init\nVRAM (MB)", "Gen\nVRAM (MB)",
        "Init\nΔRAM (MB)", "Gen\nΔRAM (MB)",
    ]
    rows = []
    for r in records:
        rows.append([
            r["max_new_tokens"], r["actual_new_tokens"], r["generation_steps"],
            f"{r['init_time_s']:.3f}",
            f"{r['generation_time_s']:.3f}",
            f"{r['total_time_s']:.3f}",
            f"{r['time_per_token_s']:.3f}",
            f"{r['throughput_tok_per_s']:.2f}",
            f"{r['init_vram_mb']:.1f}",
            f"{r['generation_vram_mb']:.1f}",
            f"{r['init_ram_delta_mb']:.1f}",
            f"{r['generation_ram_delta_mb']:.1f}",
        ])

    print("\n" + "=" * 70)
    print("  MOE BACKTRACE GENERATION")
    print("=" * 70)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    print("\n  Generated text samples:")
    for r in records:
        print(f"    [{r['max_new_tokens']} tokens] → \"{r.get('generated_text', '')}\"")


def save_report(
    seq_records: List[Dict[str, Any]],
    gen_records: List[Dict[str, Any]],
    moe_records: List[Dict[str, Any]],
    system_info: Dict[str, Any],
    output_dir: str,
):
    """Save a JSON report."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    moe_path = os.path.join(output_dir, f"moe_benchmark_{timestamp}.json")
    
    report = {
        "timestamp": timestamp,
        "system_info": system_info,
        "seq_scaling_results": seq_records,
        "gen_scaling_results": gen_records,
        "moe_results": moe_records,
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
        description="DLBacktrace benchmark suite (sequence + generation scaling)"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model ID (default: Llama-3.2-1B)",
    )
    parser.add_argument(
        "--mode", type=str, default="both", choices=["seq", "gen", "both", "moe"],
        help="Benchmark mode: seq (sequence scaling), gen (generation scaling), both",
    )
    parser.add_argument(
        "--seq-lengths", type=int, nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512],
        help="Input sequence lengths for seq-scaling mode",
    )
    parser.add_argument(
        "--gen-tokens", type=int, nargs="+",
        default=[1, 3, 5, 10, 20],
        help="max_new_tokens values for gen-scaling mode",
    )
    parser.add_argument(
        "--gen-prompt", type=str,
        default="What is the capital of France?",
        help="Input prompt for generation-scaling benchmark",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device for layer implementations (default: cuda)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Repeat count per configuration for averaging (default: 1)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/results",
        help="Directory for JSON report output (default: benchmarks/results)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="benchmarks/cache",
        help="Directory for disk-streamed relevance/scores/IO data (default: benchmarks/cache)",
    )
    parser.add_argument(
        "--view-output", action="store_true", default=False,
        help="Load and print saved .dlbr relevance output after generation (default: off)",
    )
    parser.add_argument(
        "--explain-tokens", nargs="+", default=["all"],
        help='Which tokens to compute DLB relevance for. '
             '"all" (default), "none", an int N (first N), '
             'or specific indices like "0 4 9" (default: all)',
    )
    args = parser.parse_args()

    # Resolve --explain-tokens into the right type for run_task
    et = args.explain_tokens
    if len(et) == 1 and et[0].lower() in ("all", "none"):
        args.explain_tokens_resolved = et[0].lower()
    elif len(et) == 1 and et[0].isdigit():
        args.explain_tokens_resolved = int(et[0])  # first-N
    else:
        args.explain_tokens_resolved = [int(x) for x in et]  # specific indices

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    # ── Load model & tokenizer ──────────────────────────────
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    from transformers import AutoTokenizer

    system_info = gather_system_info(args.device, args.model)

    # ── Print header ────────────────────────────────────────
    print("=" * 70)
    print("  DLBacktrace — Benchmark Suite")
    print("=" * 70)
    print(f"  Model         : {args.model}")
    print(f"  Device        : {args.device}")
    if torch.cuda.is_available():
        print(f"  GPU           : {torch.cuda.get_device_name()}")
    print(f"  Mode          : {args.mode}")
    if args.mode in ("seq", "both"):
        print(f"  Seq lengths   : {args.seq_lengths}")
    if args.mode in ("gen", "both"):
        print(f"  Gen tokens    : {args.gen_tokens}")
        print(f"  Gen prompt    : {args.gen_prompt[:60]}...")
    print(f"  Runs per cfg  : {args.runs}")
    print(f"  Output dir    : {args.output_dir}")
    print("=" * 70)

    seq_records: List[Dict[str, Any]] = []
    gen_records: List[Dict[str, Any]] = []
    moe_records: List[Dict[str, Any]] = []

    # ═══════════════════════════════════════════════════════════
    #  Sequence-Scaling Benchmark
    # ═══════════════════════════════════════════════════════════
    if args.mode in ("seq", "both"):
        print("\n" + "─" * 70)
        print("  📐 SEQUENCE SCALING BENCHMARK")
        print("─" * 70)

        print(f"📦 Loading model: {args.model}")
        model = ModelWrapper(args.model, hf_token)
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Model loaded\n")

        for seq_len in args.seq_lengths:
            for run_idx in range(args.runs):
                tag = f"seq={seq_len}, run={run_idx + 1}/{args.runs}"
                print(f"\n▶ {tag}")

                try:
                    record = benchmark_seq_scaling(
                        model=model,
                        tokenizer=tokenizer,
                        seq_len=seq_len,
                        device=args.device,
                        run_idx=run_idx,
                    )
                    record["success"] = True
                    seq_records.append(record)

                    print(
                        f"  ✅ Init: {record['init_time_s']:.3f}s | "
                        f"Fwd: {record['forward_time_s']:.3f}s | "
                        f"Bwd: {record['backward_time_s']:.3f}s | "
                        f"Throughput: {record['throughput_tok_per_s']:.1f} tok/s"
                    )
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    seq_records.append({
                        "mode": "seq_scaling",
                        "sequence_length": seq_len,
                        "run_idx": run_idx,
                        "success": False,
                        "error": str(e),
                    })

    # ═══════════════════════════════════════════════════════════
    #  Generation-Scaling Benchmark
    # ═══════════════════════════════════════════════════════════
    if args.mode in ("gen", "both"):
        print("\n" + "─" * 70)
        print("  🔄 GENERATION SCALING BENCHMARK")
        print("─" * 70)

        print(f"📦 Loading model: {args.model}")
        model = ModelWrapper(args.model, hf_token)
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Model loaded\n")

        for num_tokens in args.gen_tokens:
            for run_idx in range(args.runs):
                tag = f"max_new_tokens={num_tokens}, run={run_idx + 1}/{args.runs}"
                print(f"\n▶ {tag}")

                try:
                    record = benchmark_gen_scaling(
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=num_tokens,
                        device=args.device,
                        input_prompt=args.gen_prompt,
                        run_idx=run_idx,
                        cache_dir=args.cache_dir,
                        explain_tokens=args.explain_tokens_resolved,
                    )
                    record["success"] = True
                    gen_records.append(record)

                    print(
                        f"  ✅ Init: {record['init_time_s']:.3f}s | "
                        f"Gen: {record['generation_time_s']:.3f}s | "
                        f"Tokens: {record['actual_new_tokens']} | "
                        f"Per-tok: {record['time_per_token_s']:.3f}s | "
                        f"→ \"{record['generated_text'][:50]}\""
                    )
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    gen_records.append({
                        "mode": "gen_scaling",
                        "max_new_tokens": num_tokens,
                        "run_idx": run_idx,
                        "success": False,
                        "error": str(e),
                    })

        # ═══════════════════════════════════════════════════════════
        #  Verify .dlbr Relevance Output (if --view-output)
        # ═══════════════════════════════════════════════════════════
        if args.view_output:
            from dl_backtrace.pytorch_backtrace.dlbacktrace.core.dlb_auto_sampler import DLBAutoSampler
            from pathlib import Path

            cache_path = Path(args.cache_dir)
            dlbr_files = sorted(cache_path.rglob("*.dlbr"))
            if dlbr_files:
                print("\n" + "─" * 70)
                print("  📋 RELEVANCE OUTPUT VERIFICATION (.dlbr)")
                print("─" * 70)
                print(f"  Cache dir: {cache_path}")
                print(f"  Found {len(dlbr_files)} .dlbr file(s)\n")

                for dlbr_file in dlbr_files:
                    file_size_mb = dlbr_file.stat().st_size / (1024 ** 2)
                    t0 = time.perf_counter()
                    rel_data = DLBAutoSampler.load_relevance_step(str(dlbr_file))
                    load_time = time.perf_counter() - t0

                    print(f"  ── {dlbr_file.name} ({file_size_mb:.1f} MB, loaded in {load_time:.3f}s) ──")
                    print(f"     Entries: {len(rel_data)}")

                    # Print like ir.print_all_relevance_info()
                    for key, val in rel_data.items():
                        if isinstance(val, (list, tuple)):
                            for i, v in enumerate(val):
                                if hasattr(v, "shape") and hasattr(v, "sum"):
                                    s = float(v.sum())
                                    print(f"     [{key}][{i}] shape: {v.shape}, sum: {s:.4f}")
                        elif hasattr(val, "shape") and hasattr(val, "sum"):
                            s = float(val.sum())
                            print(f"     [{key}] shape: {val.shape}, sum: {s:.4f}")
                        else:
                            print(f"     [{key}] is not a tensor")

                    del rel_data
                    print()

                print(f"  ✅ All {len(dlbr_files)} .dlbr files loaded and verified successfully")
            else:
                print(f"\n  ⚠️  No .dlbr files found in {cache_path}")

    if args.mode == "moe":
        model = MoEBacktraceWrapper(args.model, hf_token)
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Model loaded\n")

        moe_records = []
        for num_tokens in args.gen_tokens:
            tag = f"max_new_tokens={num_tokens}"
            print(f"\n▶ {tag}")

            try:
                record = moe_backtrace(
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=num_tokens,
                    device=args.device,
                    prompt=args.gen_prompt,
                    explain_tokens=args.explain_tokens_resolved,
                )
                record["success"] = True
                moe_records.append(record)

                print(
                    f"Init: {record['init_time_s']:.3f}s | "
                    f"Gen: {record['generation_time_s']:.3f}s | "
                    f"Tokens: {record['actual_new_tokens']} | "
                    f"Per-tok: {record['time_per_token_s']:.3f}s | "
                    f"→ \"{record['generated_text'][:50]}\""
                )
            except Exception as e:
                print(f"Error: {e}")
                moe_records.append({
                    "mode": "moe_backtrace",
                    "max_new_tokens": num_tokens,
                    "success": False,
                    "error": str(e),
                })
    
    # ═══════════════════════════════════════════════════════════
    #  Report
    # ═══════════════════════════════════════════════════════════
    successful_seq = [r for r in seq_records if r.get("success")]
    successful_gen = [r for r in gen_records if r.get("success")]
    successful_moe = [r for r in moe_records if r.get("success")]

    if successful_seq:
        print_seq_table(successful_seq)
    if successful_gen:
        print_gen_table(successful_gen)
    if successful_moe:
        print_moe_table(successful_moe)

    if successful_seq or successful_gen or successful_moe:
        save_report(seq_records, gen_records, moe_records, system_info, args.output_dir)
    else:
        print("\n⚠️  No successful benchmark runs.")

    print("\n🎯 Benchmark complete!")


if __name__ == "__main__":
    main()
