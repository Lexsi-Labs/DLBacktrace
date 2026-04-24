#!/usr/bin/env python3
"""
DLBacktrace — Visualize Network Graph & Relevance
===================================================
Usage:
    python visualize_graph.py --model Qwen/Qwen3-0.6B --prompt "What is deep learning?"
    python visualize_graph.py --model meta-llama/Llama-3.2-1B --device cuda
"""

import argparse
import gc
import os
import time

import torch
import torch.nn as nn
from torch.export import Dim
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Model Wrapper ──────────────────────────────────────────────────────────

class ModelWrapper(nn.Module):
    """Wraps a HuggingFace CausalLM for DLBacktrace graph tracing."""

    def __init__(self, model_id: str, token: str = None):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, token=token,
        ).eval()

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DLBacktrace Graph Visualization")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model ID (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--prompt", type=str,
                        default="Explain the concept of backpropagation in neural networks.",
                        help="Input prompt to trace")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device (default: cuda)")
    parser.add_argument("--output-dir", type=str, default="./viz_output",
                        help="Directory for output files (default: ./viz_output)")
    parser.add_argument("--threshold", type=int, default=2500,
                        help="Auto-collapse threshold for graph SVG (default: 2500)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

    # ── 1. Load model & tokenizer ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Model  : {args.model}")
    print(f"  Device : {args.device}")
    print(f"  Prompt : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"{'='*70}\n")

    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("📦 Loading model...")
    model = ModelWrapper(args.model, token=hf_token)

    # ── 2. Tokenize ────────────────────────────────────────────────────
    tokens = tokenizer(
        [args.prompt], return_tensors="pt", padding=True, truncation=True,
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    seq_len = input_ids.shape[1]
    print(f"✅ Tokenized: {seq_len} tokens")

    # ── 3. Dynamic shapes for export ───────────────────────────────────
    seq_dim = Dim("seq", min=1, max=seq_len)
    dynamic_shapes = {
        "input_ids":      {0: 1, 1: seq_dim},
        "attention_mask": {0: 1, 1: seq_dim},
    }

    # ── 4. Initialize DLBacktrace ──────────────────────────────────────
    print("🔧 Initializing DLBacktrace (tracing graph)...")
    torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from dl_backtrace.pytorch_backtrace import DLBacktrace

    t0 = time.perf_counter()
    ir = DLBacktrace(
        model,
        (input_ids, attention_mask),
        dynamic_shapes=dynamic_shapes,
        device=args.device,
        verbose=False,
    )
    print(f"✅ DLBacktrace initialized in {time.perf_counter() - t0:.2f}s")

    # ── 5. Forward pass (predict) ──────────────────────────────────────
    print("🚀 Running forward pass (predict)...")
    t0 = time.perf_counter()
    io_data = ir.predict(input_ids, attention_mask)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print(f"✅ Predict completed in {time.perf_counter() - t0:.2f}s")
    print(f"   Nodes in graph: {len(io_data)}")

    # ── 6. Relevance propagation (evaluation) ──────────────────────────
    print("📊 Running relevance propagation (evaluation)...")
    t0 = time.perf_counter()
    ir.evaluation(task="generation", multiplier=100.0, debug=False)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print(f"✅ Evaluation completed in {time.perf_counter() - t0:.2f}s")

    # ── 7. Print relevance summary ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("  RELEVANCE SUMMARY")
    print("─" * 70)
    ir.print_all_relevance_info()

    # ── 8. Visualize network graph (SVG) ───────────────────────────────
    svg_path = os.path.join(args.output_dir, "network_graph")
    print(f"\n🎨 Generating network graph SVG...")
    ir.visualize_dlbacktrace(
        output_path=svg_path,
        engine_auto_threshold=args.threshold,
    )
    print(f"   Saved to: {svg_path}*.svg")

    # ── 9. Visualize input heatmap ─────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    print("🎨 Generating input relevance heatmap...")
    ir.visualize_input_heatmap_for_token(
        [ir.all_wt],
        n=0,
        input_ids=input_ids,
        tokenizer=tokenizer,
    )
    heatmap_path = os.path.join(args.output_dir, "input_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved to: {heatmap_path}")

    # ── 10. Tokenwise relevance map ────────────────────────────────────
    print("🎨 Generating tokenwise relevance map...")
    ir.visualize_tokenwise_relevance_map(
        [ir.all_wt],
        input_ids,
        tokenizer,
    )
    relmap_path = os.path.join(args.output_dir, "tokenwise_relevance.png")
    plt.savefig(relmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved to: {relmap_path}")

    # ── Done ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ✅ All visualizations saved to: {args.output_dir}/")
    print(f"{'='*70}\n")

    # Cleanup
    del ir, io_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
