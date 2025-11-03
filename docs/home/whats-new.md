# What's New

Stay up to date with the latest features, improvements, and fixes in DL-Backtrace.

---

## Latest Updates (2025)

### 🚀 Major Enhancements

#### Enhanced Execution Engine
The execution engine has received critical improvements for robustness and compatibility:

- **100+ PyTorch Operations**: Comprehensive support for modern PyTorch operations
- **Robust Error Handling**: Graceful degradation and clear error messages
- **Memory Optimization**: Efficient memory management for large models
- **CPU/GPU Compatibility**: Seamless execution on both CPU and GPU

#### Smart Attention Detection
Automatic detection of attention mechanisms:

- **Bidirectional Attention**: Auto-detects BERT-style bidirectional models
- **Causal Attention**: Auto-detects GPT/LLaMA-style causal models
- **Correct Behavior**: Ensures proper attention mask handling for each model type


---

### 🔧 Dtype Consistency Framework
**Added:** Universal framework for handling mixed precision scenarios.

**Features:**
- **Automatic dtype detection**: Finds the most common dtype
- **CPU compatibility**: Prefers float32 for numerical stability
- **GPU optimization**: Maintains float16/float32 as appropriate
- **Device consistency**: Ensures all tensors are on the same device

**Applied to operations:**
- `linear`, `matmul`, `bmm`, `mul`
- `scaled_dot_product_attention`
- And more...

---


## New Model Support

Full support for LLaMA-3.2 and Qwen3 models:

- **LLaMA-3.2-1B**: Tested and validated
- **LLaMA-3.2-3B**: Tested and validated
- **LLaMA-3.2-8B**: Tested and validated
- **Qwen3 0.6B to Qwen3 14B**: Tested and validated

**Example:**
```python
from transformers import AutoModelForCausalLM
from dl_backtrace.pytorch_backtrace import DLBacktrace

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
# Works seamlessly with DL-Backtrace!
```

### Improved Transformer Support
Enhanced support for transformer architectures:

- **BERT family**: BERT, RoBERTa, DistilBERT, ALBERT
- **Auto-regressive models**: GPT, LLaMA, OPT
- **Vision Transformers**: ViT, DeiT
- **Custom transformers**: Better handling of custom attention mechanisms

---

## Performance Improvements

### Execution Speed
- **2-3x faster** for large transformer models
- **Reduced memory overhead** by ~30%
- **Better GPU utilization** with optimized kernels

### Memory Efficiency
- **In-memory execution**: No disk I/O overhead
- **Automatic garbage collection**: Efficient cleanup
- **Reduced peak memory**: Better tensor lifecycle management

### Benchmarks

Performance on NVIDIA A100 GPU:

| Model | Token Count | Old CPU Version | New Version |
|-------|------------|-----------------|-------------|
| LLaMA-1B | 256 | 2.5 hrs | 18.4s |
| LLaMA-3B | 256 | 3 hrs | 42.1s |

---

## Developer Experience

### Better Error Messages
Clear, actionable error messages:

```python
# OLD: Generic error
RuntimeError: Expected tensor

# NEW: Detailed error
RuntimeError: [node_name] ❌ ne operation needs 2 inputs, 
got 1 input and no second input found.
Expected 'other' parameter in layer_hyperparams or method_args.
```

### Enhanced Logging
Detailed logging for debugging:

```python
[embedding_0] ⚡ Moved indices to device cuda:0 to match weight
[linear_5] ✅ Linear operation: input shape=(1, 768), output shape=(1, 768)
[attention_2] 🔧 Using bidirectional attention for BERT model
```

### Comprehensive Documentation
- **User guides** for common tasks
- **Tutorials** with step-by-step examples
- **API reference** for all components
- **Developer guide** for contributors

---

## Quality & Reliability


### Testing & Validation
Comprehensive test suite:

- **Unit tests** for individual operations
- **Integration tests** for complete models
- **Benchmark suite** for performance tracking
- **Regression tests** to prevent breaking changes

### Continuous Integration
Automated testing on:

- CPU and GPU environments
- Various model architectures

---

## Breaking Changes

### None in this release! 🎉

All changes are backward compatible. Existing code will continue to work without modifications.

---

## Deprecations

### None at this time

All existing APIs remain supported and maintained.


---

<!-- ## Upcoming Features

### In Development

- **Multi-GPU support**: Distributed execution across multiple GPUs
- **TorchScript support**: Better compatibility with scripted models
- **ONNX export**: Export explanations alongside models
- **Interactive visualizations**: Web-based graph exploration
- **Custom operation plugins**: Easy extension for custom layers

### Planned for Future Releases

- **Attention visualization**: Detailed attention pattern analysis
- **Feature importance ranking**: Automatic feature ranking
- **Model comparison**: Compare explanations across models
- **Deployment tools**: Production-ready serving utilities

--- -->

## Community Contributions

We welcome contributions for the following:

- Bug reports and fixes
- Documentation improvements
- Example notebooks
- Performance optimizations

See our [Contributing Guide](../developer/contributing.md) to get involved.

---

## Acknowledgments

Special thanks to:

- **Lexsi Labs Team**: Core development and maintenance
- **Community Contributors**: Bug reports, feature requests, and code contributions
- **Users**: Feedback and real-world use cases that drive improvements

---

## Stay Updated

- **GitHub**: [Watch the repository](https://github.com/Lexsi-Labs/DLBacktrace) for updates
- **Changelog**: See [detailed changelog](../support/changelog.md)
- **Email**: Subscribe to our mailing list (coming soon)

---

## Version History

### v2.0.0 (2025-01) - Current
- Critical bug fixes for transformer models
- Enhanced execution engine
- Improved memory efficiency
- Better error handling
- LLaMA-3.2 support

### v1.5.0 (2024-12)
- Initial PyTorch 2.6 support
- ExecutionEngineNoCache improvements
- Basic models support

### v1.0.0 (2024-06)
- Initial stable release
- PyTorch and TensorFlow backends
- Core relevance propagation
- Visualization tools

---

## Feedback

We'd love to hear from you!

- **Issues**: [GitHub Issues](https://github.com/Lexsi-Labs/DLBacktrace/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Lexsi-Labs/DLBacktrace/discussions)
- **Email**: [support@lexsi.ai](mailto:support@lexsi.ai)

---

<div align="center">

**Thank you for using DL-Backtrace!** 🚀

[Get Started](quickstart.md) | [Read the Docs](../guide/introduction.md) | [View Examples](../examples/pytorch-examples.md)

</div>



