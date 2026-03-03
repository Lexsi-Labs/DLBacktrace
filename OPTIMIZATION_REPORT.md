# DLBacktrace Optimization Report

**Model:** Llama-3.2-1B (fp16) | **GPU:** Tesla T4 | **Benchmark:** 20 tokens autoregressive generation with full relevance tracing

---

## Baseline (Before Any Changes)

- **Per-token time:** ~35.6s average
- **rel_save (relevance saving):** ~23.8s average (67% of total time) — spiked from 3s to 40s+ after token 3
- **Total for 15 tokens:** ~534s (~9 minutes)
- **Problem:** System became unusable after a few tokens due to GPU memory exhaustion and CPU thrashing

---

## Optimizations Applied

### 1. Disk-Based Streaming
**Before:** All relevance data (~60-110 MB per token) kept in RAM → memory grew unboundedly with each token.
**Change:** Stream each token's relevance, scores, and IO data directly to disk files instead of accumulating in memory.
**After:** Constant RAM usage regardless of number of tokens generated.

### 2. Zero-Relevance Node Filtering
**Before:** Saving all 1390 nodes per token (~4.8 GB on GPU), including 147 weight/buffer nodes that are always zero.
**Change:** Name-based filter removes `p_model_*` and `b_model_*` keys instantly (no GPU computation needed).
**After:** Only 1243 non-zero nodes saved (~60-110 MB), purging ~4.7 GB of useless data per token.

### 3. GPU Memory Cleanup (rel_dict.clear())
**Before:** `rel_dict` was the same Python object as `self.dlb.all_wt` (~4.8 GB on GPU). Old code only freed the filtered subset, leaving 4.7 GB alive during serialization → GPU memory exhaustion by token 3.
**Change:** Call `rel_dict.clear()` in-place after CPU copies are made, which also clears `self.dlb.all_wt`, releasing all GPU tensors before serialization starts.
**After:** GPU VRAM dropped from ~12 GB peak to ~7 GB. No more GPU memory exhaustion.

### 4. LZ4 Compression
**Before:** Uncompressed `torch.save` files (~60-110 MB per token on disk).
**Change:** Added LZ4 frame compression to the serialized output.
**After:** Files are 60-80% of original size (e.g., 110 MB → 89 MB). Compression takes only ~0.1-0.3s per token (LZ4 is extremely fast).

### 5. Flat-Buffer Pipeline (The Big Win)
**Before:** 1243 individual `.cpu()` calls (each a separate `cudaMemcpy` + CPU allocation), then `torch.save` pickling 1243 individual tensor objects. gpu2cpu took 2s → 31s, torch.save took 0.8s → 8.8s — both degraded over time.
**Change:** `torch.cat()` all 1243 tensors into ONE flat GPU tensor → ONE `.cpu()` DMA transfer → `.numpy().tobytes()` raw bytes + JSON metadata header. Completely bypasses Python pickle.
**After:** concat+dma = 0.1s, tobytes = 0.1s, lz4+write = 0.2s. Total rel_save = **~1s per token, constant across all tokens.**

---

## Final Results

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| rel_save (avg per token) | 23.8s | **1.05s** | **22.7×** |
| Per-token total (avg) | 35.6s | **6.9s** | **5.2×** |
| 20 tokens total | ~534s | **138s** | **3.9×** |
| GPU VRAM peak | ~12 GB | **~7 GB** | **-42%** |
| CPU RAM growth | Unbounded | **Constant** | ✅ |
| Timing consistency | Spikes at token 3+ | **Flat across all tokens** | ✅ |

---

## Current Bottleneck Breakdown (After Optimization)

| Stage | Avg Time | Share | Description |
|-------|----------|-------|-------------|
| backtrace | 2.69s | 39% | Relevance propagation (actual computation) |
| predict | 2.12s | 31% | Model forward pass |
| **rel_save** | **1.05s** | **15%** | Relevance saving (optimized) |
| scores_save | 0.53s | 8% | Logit scores to disk |
| cleanup | 0.51s | 7% | Memory cleanup between tokens |

---

## Files Modified

- `dl_backtrace/pytorch_backtrace/dlbacktrace/core/dlb_auto_sampler.py` — Rewrote `_store_relevance_entry()`, added `_flat_to_dict()`, `load_relevance_step()`
- New file format: `.dlbr` (JSON header + LZ4-compressed raw bytes) replaces `.pt.lz4` (pickle-based)
