# Changelog

All notable changes to DL-Backtrace are documented here.

---

## Unreleased

### Changed

- **`masked_fill` uses the fill value the model passes.** Previously the fill
  value was read from the layer hyperparameters and defaulted to `0.0`. It is now
  taken from the model's own `masked_fill` call, whether it is a number or a
  tensor. Results change for models whose fill value is not `0.0`, including
  attention masks that fill with `-inf`.
- **`transformers` is now a dependency** (`transformers>=4.52.3,<5`).
- **`dlbacktrace-server` binds to `127.0.0.1` by default** instead of `0.0.0.0`.
  The server has no authentication; pass `--host 0.0.0.0` to accept remote
  connections, and only on a trusted network.
- **`dlbacktrace-server` keeps request cache directories under `--cache-root`**
  (default: the directory the server starts in). A `/generate` request whose
  `cache_dir` resolves outside that root, through `..`, an absolute path or a
  symlink, is rejected with HTTP 400.

### Fixed

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
