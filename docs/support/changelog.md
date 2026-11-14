# Changelog

All notable changes to DL-Backtrace are documented here.

---

## Unreleased

### Added

- **Two-step generation: `forward_pass()` and `relevance_pass()`.** `forward_pass(inputs,
  max_new_tokens, temperature, top_k, top_p, eos_token_id, ...)` generates without computing
  relevance and keeps the sequence; `relevance_pass(token_indices=None, ...)` then explains
  any of the generated tokens and returns one `{step_index, token_id, relevance}` entry per
  step, equal to what `run_task("generation")` computes for the same token; `clear_traces()`
  forgets the generation. Only the token ids are kept between the calls (the model runs once
  per explained token in `relevance_pass()`), so memory does not grow with the length of the
  generation. Contributed by @neerajaryaai (PR #13), re-implemented on top of the reworked
  generation.
- **`run_task(task="forward-only-generation")`** generates without relevance and without a
  tokenizer, with greedy, temperature, top-k and top-p decoding and EOS stopping, and returns
  `generated_token_ids` (list) and `complete_sequence` (tensor). Also from PR #13.
- **`visualize_dlbacktrace(show=..., inline_format=...)`**: whether to display the graph inline
  and in which format (`"svg"` or `"png"`); both were fixed to `True` and `"svg"`. Also from PR #13.
- **Compact, filtered and paginated relevance graphs** (from PR #13, by @neerajaryaai,
  re-implemented on top of the `graph_mode` / top-k rework). `visualize_dlbacktrace()` takes
  `layer_types` (draw only nodes of these types, linked through the nodes left out),
  `compact=True` (shortcut for `layer_types=SEMANTIC_LAYER_TYPES`, now exported),
  `rankdir` (`"LR"`, `"TB"`, `"RL"`, `"BT"`), `paginated`, `max_nodes_per_page` and
  `pages_per_row` (pages saved as `{output_path}_pageN.svg`, plus a grid
  `{output_path}_combined.svg`). `layer_types` and `rankdir` also work with `graph_mode="top_k"`
  and `"entire"` and on `visualize_relevance`, `visualize_relevance_fast`,
  `visualize_relevance_subgraph` and `visualize_relevance_auto`; `visualize_relevance_paginated`
  is new. A parent on an earlier page is drawn as a dashed stub on every page, so no edge is
  lost. New `relevance_stat="sum" | "mean"` (see Changed). Needs the Graphviz `dot` program,
  as before.

### Changed

- **Relevance graph labels show the sum, not the mean** (from PR #13). Every node label
  (default top-k graph, entire graph, fast graph, pages) now reads `Sum:` and shows the total
  relevance of the node, and for a node that passes relevance to several inputs the total over
  all of them (it was the average of the per-input totals). The mean divides the total by the
  tensor size, so the same total read 0.223 on a `[1, 7, 64]` tensor and 0.446 on a
  `[1, 7, 32]` one; the sum does not depend on the size and the parts a layer splits its
  relevance into add up to the whole. `relevance_stat="mean"` restores the old numbers.
  `top_k` and `relevance_threshold` on `visualize_relevance` rank and threshold on the sum.
- **Relevance graphs no longer draw parameter, buffer and constant placeholders** (`p_*`,
  `b_*`, `c_*`; they carry no relevance and made up 10 to 40% of the nodes of the tiny models
  checked), and nodes are coloured by their layer type as the colour map intended (linear
  layers blue, activations orange, attention gold, ...) instead of all ATen operations being
  violet. Nothing else was added or removed from the default, top-k and entire graphs.
- **Eager and sdpa attention give the same relevance.** Eager attention (`Q @ K^T`,
  scaling, mask, softmax, `P @ V` as separate ops) now goes through the same relevance
  rule as `scaled_dot_product_attention`; its mask is read off the softmax input.
  Before, the ops were propagated one by one and the scaling constant and the additive
  mask took a share of the relevance at every layer, so a model's input relevance
  depended on its `attn_implementation` (SmolLM2-135M: 1.82 with sdpa, 0.197 with
  eager). Results change for models run with eager attention; sdpa results do not.

- **Attention relevance respects the attention mask.** The attention rule spread
  each head's relevance over every position in proportion to the Q/K/V values, so
  padding and future tokens received a share (on a right-padded BERT input about 30%
  of the token relevance landed on the two padding tokens). Keys and values
  now keep only the relevance of the queries allowed to attend to them, padding
  positions get none, and each head's positive and negative totals are unchanged.
  This applies to `scaled_dot_product_attention` on the CPU, GPU and compiled paths;
  inputs without padding or a causal mask give the same relevance as before.
  **Results change** for padded and causal inputs.
- **Relevance through `matmul`, `bmm` and `mm`.** These ops passed all of their
  relevance to the first operand, so in eager attention (`Q @ K^T`, then
  `P @ V`) it never reached keys or values and stayed at the query token. They
  now use the epsilon rule over the products `A[i, k] * B[k, j]`, split evenly
  between the two operands as for elementwise `mul` (or all to the non-weight
  operand when one is a weight), which conserves relevance. Explanations change
  for models that run eager attention or use `matmul` directly; the
  `scaled_dot_product_attention` path is unchanged.
- **`masked_fill` uses the fill value the model passes.** Previously the fill
  value was read from the layer hyperparameters and defaulted to `0.0`. It is now
  taken from the model's own `masked_fill` call, whether it is a number or a
  tensor. Results change for models whose fill value is not `0.0`, including
  attention masks that fill with `-inf`.
- **`transformers` is now a dependency** (`transformers>=4.52.3,<4.53`). With
  torch 2.6.0, transformers 4.53 and later fail to export causal language models
  (Llama, GPT-2, SmolLM2, Qwen3); 4.52.4 was checked on CUDA.
- **`dlbacktrace-server` binds to `127.0.0.1` by default** instead of `0.0.0.0`.
  The server has no authentication; pass `--host 0.0.0.0` to accept remote
  connections, and only on a trusted network.
- **`dlbacktrace-server` keeps request cache directories under `--cache-root`**
  (default: the directory the server starts in). A `/generate` request whose
  `cache_dir` resolves outside that root, through `..`, an absolute path or a
  symlink, is rejected with HTTP 400.

### Fixed

- **Beam search with `return_relevance=True` raised `UnboundLocalError`.** The beam path read
  the `explain_tokens` gate (`_should_run_dlb`), which was only defined for the non-beam path.
- **Generation steps without relevance used the wrong model for a causal LM passed to
  `DLBacktrace` directly.** `explain_tokens` (and the forward-only paths) run steps that
  skip relevance with the model's own KV-cache forward, which looked at `dlb.model.model`:
  for `LlamaForCausalLM` that is the base model, whose hidden states were taken for logits
  (`IndexError`, or wrong tokens). The causal LM is now found the way the sampler finds it,
  so a bare causal LM and a wrapper that holds one in `.model` both work.
- **GPT-2 relevance with a dynamic sequence length.** GPT-2 slices its causal-mask buffer
  `attn.bias[:, :, :T, :T]`. With a dynamic length, `T` is the slice node's second input rather
  than a hyperparameter, so the relevance was not placed in the buffer's shape ("Cannot align
  relevance (1, 1, 5, 64) -> (1, 1, 64, 64)") and generation with relevance failed on GPT-2.
  The CPU, GPU and compiled paths now read it from the node's inputs.
- **GPU relevance through `addmm` layers.** The CUDA linear kernel reads its weight
  row-major, but `addmm` (GPT-2's `Conv1D` and other `x @ W + b` layers) passed it the
  transposed weight as a view, so on the GPU these layers spread relevance with the
  wrong weights; CPU results were right. The weight is now made contiguous first.
- **Token relevance on `input_ids`.** The embedding's `[B, T, D]` relevance was
  aligned to the `[B, T]` token ids by collapsing the batch axis with a norm, which
  mixed positions together and then tiled the result: the per-token relevance did
  not follow the embedding's per-position relevance (padding tokens got a share even
  with eager attention). When the target shape is a prefix, the trailing axes are now
  summed. **Results change** for token-level relevance of every model.
- **`scaled_dot_product_attention` called with keyword arguments.** The graph
  builder read sdpa's arguments by position only, so `is_causal=True`,
  `enable_gqa=True` or `scale=` passed by keyword (as
  `F.scaled_dot_product_attention(q, k, v, is_causal=True)` and newer
  transformers do) were lost and a float reached the engine as the mask. They
  are now read from the op's schema and passed through. The relevance rules no
  longer fail on a missing mask, build the causal mask that `is_causal=True`
  implies, accept boolean masks, and repeat grouped K/V heads (summing their
  relevance back), on the CPU, GPU and compiled paths.
- **Relevance on the GPU path.** The compiled CUDA linear kernel counted part of
  each input twice when the input size is a multiple of 4 (threads past the
  vectorised range also ran the scalar loop), and every thread raced to store the
  block's sums. It lost 35-45% of the relevance at every linear layer, so CUDA
  explanations came out 10 to 50 times smaller than CPU ones. The kernel now
  matches the CPU rule. The compiled ops are rebuilt for torch 2.13.0 and Python
  3.10 to 3.13 (see the `build-compiled-ops` workflow).
- **Linear relevance with a one-sided neuron.** When all of a neuron's
  contributions had the same sign, the CPU rule replaced the empty sum with 1
  before computing the split, handing relevance to a side with no inputs; it was
  lost. The split is now computed from the real sums, as the CUDA kernel does.
- **Generation past the first token.** `run_task(task="generation")` and
  `DLBAutoSampler.generate` run the traced graph on a longer sequence at every
  step; a graph traced without `dynamic_shapes` (as in the guides) was fixed to
  the sample length, so every step after the first failed. Generation now
  re-exports once with a dynamic sequence length when no `dynamic_shapes` were
  given. The helper `sequence_dynamic_shapes` builds that sample and shapes.
- **One-token prompts on the server.** `/generate` traced with
  `Dim(min=1, max=prompt_len)`, which `torch.export` rejects for a one-token
  prompt (HTTP 500) and which capped the traced length at the prompt. It now
  uses `sequence_dynamic_shapes`.
- **Relevance traces from generation.** With the default `"full"` policy each
  step replaced the previous one, and on CPU the stored entry was empty because
  the CPU path returns numpy arrays. The trace now holds one entry per explained
  token on CPU and GPU.
- **Docs.** The guides used `dlb_tokens_count`, which `run_task` no longer
  accepts; it is `explain_tokens`. The GPT-2 pipeline example passed
  `attention_mask` as GPT-2's second positional argument, which is
  `past_key_values`; it now passes `input_ids` only.
- **GPT-2 and nanoGPT-style eager attention.** The forward pass returned `None`:
  `full` rejected the 0-d size GPT-2 uses for its attention scale and mask value,
  and a buffer whose name contains `bias` or `weight` (GPT-2's causal mask
  `attn.bias`) was treated as a layer bias and dropped. Buffers are no longer
  classified as weights or biases, and `full` accepts empty and zero-sized shapes.
- **`where` relevance.** Relevance now goes to the value each position took
  (`input` where the condition holds, `other` elsewhere); the boolean condition
  gets none. Before, all of it went to the condition, so relevance through a
  `where`-masked attention never reached the scores.
- **CPU use without the compiled CUDA ops.** `import dl_backtrace` and
  `device="cpu"` runs work when the compiled CUDA extensions cannot be loaded on
  the platform. If `device="cuda"` is requested in that situation, DL-Backtrace
  warns with the original load error and uses the CPU implementations, so
  `evaluation()` does not fail part-way through. The notice is also logged
  once, so it stays visible after an earlier instance has silenced warnings.
- **Lifted tensor constants** are resolved when tracing. Tensor constants that
  the exporter lifts into graph inputs are now looked up alongside parameters and
  buffers instead of failing to map.
- **`masked_fill` with a fill value held in a buffer named like a weight or
  bias** (for example `masked_bias`). The value was dropped and `predict()`
  returned `None`; it is now used.
- **`python -m dl_backtrace.cli.server` loads the model.** It stored the CLI
  arguments on the `__main__` module, while uvicorn imports the app from
  `dl_backtrace.cli.server`, so the model was never loaded. The
  `dlbacktrace-server` command was not affected.
- **Causal language models that return their KV cache** (`use_cache=True`, the
  default). The relevance seeded on the logits was also handed to every cached
  key and value tensor in the graph output, so `evaluation()` failed with
  "Cannot align relevance". It now goes to the logits only.
- **`torch.all` / `torch.any` with `dim` and `keepdim`.** `all` had no
  hyperparameter entry and was replayed over dim 0 without `keepdim`; `any` read
  its arguments one position off. transformers builds the SDPA attention mask
  with `all(..., dim=-1, keepdim=True)` on CUDA, so causal language models with
  the default sdpa attention gave wrong logits on GPU (SmolLM2-135M was about 30
  logits off). Forward outputs now match PyTorch.

---

<!-- ## [2.0.0] - 2025-01 (Current)

### 🚀 Major Features

#### Enhanced Execution Engine
- Added support for 100+ PyTorch ATen operations
- Improved memory efficiency by ~30%
- Better device handling (CPU/GPU)
- Enhanced dtype consistency framework

#### Smart Attention Detection
- Auto-detects bidirectional vs causal attention
- Proper handling of BERT, RoBERTa (bidirectional)
- Proper handling of GPT, LLaMA (causal)

#### LLaMA-3.2 Support
- Full support for LLaMA-3.2-1B
- Full support for LLaMA-3.2-3B
- Experimental support for LLaMA-3.2-8B

### 🐛 Critical Bug Fixes

#### Boolean Tensor Handling
- **Fixed**: RuntimeError when processing boolean tensors
- **Impact**: RoBERTa, LLaMA models no longer crash
- **Details**: Added proper dtype checking in debug code

#### Embedding Operation OOM Fix
- **Fixed**: Out-of-memory errors in embedding operations
- **Impact**: Large vocabulary models now work correctly
- **Details**: Removed duplicate embedding processing

#### Dtype Consistency
- **Fixed**: Mixed precision scenarios (float16/float32)
- **Impact**: Better CPU compatibility
- **Details**: Universal dtype consistency framework

#### Comparison Operations
- **Fixed**: Comparison operations (ne, eq, lt, le, gt, ge)
- **Impact**: Exact reproducibility across runs
- **Details**: Proper input validation and consistency

### ⚡ Performance Improvements

- 25-30% faster execution for transformer models
- Reduced peak memory usage by ~30%
- Better CUDA kernel utilization
- Optimized tensor operations

### 📚 Documentation

- Complete MkDocs documentation
- User guides for PyTorch and TensorFlow
- 20+ tutorial notebooks
- Comprehensive API reference
- Developer guide

### 🔄 Breaking Changes

None! All changes are backward compatible.

---

## [1.5.0] - 2024-12

### 🚀 Features

- Initial PyTorch 2.6 support
- ExecutionEngineNoCache improvements
- Basic LLaMA model support
- Enhanced graph tracing

### 🐛 Bug Fixes

- Fixed graph visualization issues
- Improved error messages
- Better handling of dynamic shapes

### ⚡ Performance

- 15% faster tracing for large models
- Reduced memory overhead

---

## [1.4.0] - 2024-10

### 🚀 Features

- Enhanced transformer support
- Improved attention mechanism handling
- Better visualization options

### 🐛 Bug Fixes

- Fixed BERT tracing issues
- Improved error handling
- Better device management

---

## [1.3.0] - 2024-08

### 🚀 Features

- Added Vision Transformer (ViT) support
- Enhanced relevance propagation for attention
- Improved graph building

### 🐛 Bug Fixes

- Fixed memory leaks in execution engine
- Improved TensorFlow backend stability

---

## [1.2.0] - 2024-06

### 🚀 Features

- Added contrastive evaluation mode
- Enhanced visualization capabilities
- Better error messages

### 🐛 Bug Fixes

- Fixed numerical stability issues
- Improved gradient handling

---

## [1.1.0] - 2024-04

### 🚀 Features

- Added support for custom CNN architectures
- Improved pooling layer handling
- Enhanced logging

### 🐛 Bug Fixes

- Fixed shape mismatch errors
- Better handling of edge cases

---

## [1.0.0] - 2024-02

### 🎉 Initial Release

#### Core Features

- Layer-wise relevance propagation
- PyTorch backend with torch.fx
- TensorFlow/Keras backend
- Graph visualization
- Deterministic execution

#### Supported Models

- CNN architectures (ResNet, VGG)
- Basic transformer models (BERT)
- Custom tabular models

#### Supported Operations

- Linear, Conv2d, Pooling
- ReLU, GELU activations
- Basic tensor operations
- Attention mechanisms

---

## Version History Summary

| Version | Release Date | Key Features |
|---------|-------------|--------------|
| **2.0.0** | 2025-01 | Critical fixes, LLaMA support, enhanced engine |
| 1.5.0 | 2024-12 | PyTorch 2.6, ExecutionEngineNoCache |
| 1.4.0 | 2024-10 | Enhanced transformers |
| 1.3.0 | 2024-08 | ViT support |
| 1.2.0 | 2024-06 | Contrastive mode |
| 1.1.0 | 2024-04 | Custom CNNs |
| 1.0.0 | 2024-02 | Initial release |

---

## Upcoming Features

### In Development

- Multi-GPU support
- TorchScript compatibility
- ONNX export
- Interactive visualizations
- Custom operation plugins

### Planned

- Model comparison tools
- Deployment utilities
- Enhanced benchmarking
- Additional evaluation modes

---

## How to Update

### From Previous Versions

```bash
cd DL-Backtrace
git pull origin main
pip install -e . --upgrade
```

If you compiled CUDA kernels:
```bash
./compile_cuda_layers.sh
```

### Check Version

```python
import dl_backtrace
print(dl_backtrace.__version__)
```

---

## Migration Guides

### Migrating from 1.x to 2.0

**No breaking changes!** All existing code continues to work.

**New features you can use:**
```python
# LLaMA support (new in 2.0)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Auto-detected attention type (new in 2.0)
# No configuration needed - works automatically!
```

---

## Deprecations

None at this time. All APIs are stable and supported.

---

## Known Issues

### Current Limitations

1. **Multi-GPU**: Not yet supported (in development)
2. **EmbeddingBag**: Not yet supported
3. **Dynamic Control Flow**: Limited support
4. **Some Custom Ops**: May not be supported

See [Known Issues](known-issues.md) for details and workarounds.

---

## Contributors

### Core Team

- Lexsi Labs Development Team

### Community Contributors

Thank you to all contributors who have helped improve DL-Backtrace through bug reports, code contributions, and documentation improvements!

---

## Acknowledgments

Special thanks to:

- PyTorch team for excellent framework
- Hugging Face for transformer models
- Research community for XAI foundations
- Users for valuable feedback

---

## Stay Updated

### Follow Development

- **GitHub**: [Watch repository](https://github.com/Lexsi-Labs/DLBacktrace)
- **Releases**: [GitHub Releases](https://github.com/Lexsi-Labs/DLBacktrace/releases)
- **Issues**: [Track progress](https://github.com/Lexsi-Labs/DLBacktrace/issues)

### Provide Feedback

- **Feature Requests**: [Open an issue](https://github.com/Lexsi-Labs/DLBacktrace/issues/new)
- **Bug Reports**: [Report bugs](https://github.com/Lexsi-Labs/DLBacktrace/issues/new)
- **Discussions**: [Join discussions](https://github.com/Lexsi-Labs/DLBacktrace/discussions)

---

## License

DL-Backtrace is released under the [Lexsi Labs Source Available License (LSAL) v1.0](license.md).

---

<div align="center">

**Thank you for using DL-Backtrace!**

[View Latest Release →](https://github.com/Lexsi-Labs/DLBacktrace/releases/latest){ .md-button }

</div>


 -->
