# Changelog

All notable changes to DL-Backtrace are documented here.

---

## Unreleased

### Changed

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
