# Frequently Asked Questions

Common questions and answers about DL-Backtrace.

---

## General Questions

### What is DL-Backtrace?

DL-Backtrace is an explainable AI (XAI) framework that helps you understand how deep learning models make decisions through layer-wise relevance propagation.

### Which frameworks does it support?

- **PyTorch** 2.6+ (full support)

### Is it free and open source?

DL-Backtrace is source-available under the Lexsi Labs Source Available License (LSAL) v1.0. It is free for personal, academic, and research purposes, but requires a commercial license for commercial use.

### Who develops DL-Backtrace?

DL-Backtrace is developed by [Lexsi Labs](https://lexsi.ai), with contributions from the community.

---

## Installation Questions

### How do I install DL-Backtrace?

```bash
git clone https://github.com/Lexsi-Labs/DLBacktrace.git
cd DL-Backtrace
pip install -r requirements.txt
pip install -e .
```

See the [Installation Guide](../home/installation.md) for details.

### Do I need a GPU?

No, DL-Backtrace works on both CPU and GPU. However, GPU significantly speeds up execution for large models.

### What Python version is required?

Python 3.8 or higher is required.

### Can I install it via pip?

Currently, installation from source is recommended. PyPI distribution is planned for future releases.

---

## Usage Questions

### How do I trace my model?

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace

dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,)
)
```

See the [Quick Start](../home/quickstart.md) guide.

### My model fails to trace. What should I do?

Check:
1. Model is in eval mode: `model.eval()`
2. All operations are supported
3. No data-dependent control flow
4. Input shape matches model expectations

### How do I interpret relevance scores?

- **Positive relevance**: Feature contributed to the prediction
- **Negative relevance**: Feature argued against the prediction
- **Zero relevance**: Feature didn't influence the decision

See [Relevance Propagation](../guide/relevance/overview.md).

### Can I use it with pre-trained models?

Yes! DL-Backtrace works with any PyTorch model, including pre-trained ones from Hugging Face, torchvision, etc.

---

## Performance Questions

### How long does tracing take?

- **Small models**: < 1 second
- **Medium models** (ResNet-50): 1-3 seconds
- **Large models** (BERT): 3-10 seconds
- **Very large** (LLaMA-3B): 10-30 seconds

### How much memory does it use?

Memory usage depends on model size:
- **Small models**: < 1 GB
- **Medium models**: 2-4 GB
- **Large models**: 8-16 GB
- **Very large**: 20+ GB

### Can I speed up execution?

Yes:
1. Use GPU instead of CPU
2. Reduce batch size
3. Use mixed precision (FP16)
4. Ensure model is in eval mode

### Does it slow down inference?

Tracing adds overhead, but you only trace once. Subsequent evaluations reuse the traced graph.

---

## Compatibility Questions

### Which models are supported?

**Vision:**
- CNNs (ResNet, VGG, EfficientNet, etc.)
- Vision Transformers (ViT)
- Custom CNN architectures

**NLP:**
- BERT, RoBERTa, ALBERT, DistilBERT
- LLaMA-3.2
- GPT-style models
- Custom transformers

**Others:**
- Custom architectures using supported operations

### Are all PyTorch operations supported?

DL-Backtrace supports 100+ common operations. See the [operations list](../guide/pytorch/operations.md).

### Can I use custom layers?

If your custom layers use supported operations, they should work. Otherwise, you may need to decompose them or request support.

### Does it work with dynamic graphs?

Yes, but with limitations. Data-dependent control flow may cause issues.

---

## Technical Questions

### What is relevance propagation?

Relevance propagation traces the "importance" of input features by backpropagating relevance scores from output to input.

### How is it different from gradients?

- **Relevance**: Shows which features contributed to the output
- **Gradients**: Show sensitivity to small input changes

Relevance is better for understanding model decisions.

### Can I use it for model debugging?

Yes! DL-Backtrace helps identify:
- Biases in model decisions
- Spurious correlations
- Feature importance
- Layer contributions

### Is execution deterministic?

Yes, DL-Backtrace automatically sets up a deterministic environment for reproducible results.

---

## Error Questions

### I get "CUDA out of memory" errors

Solutions:
1. Reduce batch size
2. Use CPU instead of GPU
3. Close other GPU applications
4. Use a GPU with more memory

### I get "Operation not supported" errors

Check if the operation is in the [supported list](../guide/pytorch/operations.md). If not:
1. Request support on GitHub
2. Decompose into supported operations
3. Modify your model

### Tracing hangs or takes forever

Possible causes:
1. Very large model
2. Infinite loop in model
3. Memory issues

Try:
- Smaller model
- Check for loops in forward()
- Monitor memory usage

### I get dtype mismatch errors

DL-Backtrace should handle this automatically. If not:
1. Ensure model and input have same dtype
2. Convert explicitly: `model.float()`, `input.float()`
3. Report the issue

---

## Licensing Questions

### Can I use it commercially?

Commercial use of DL-Backtrace requires a separate commercial license from Lexsi Labs. Please contact [support@lexsi.ai](mailto:support@lexsi.ai) for commercial licensing inquiries.

### Do I need to cite it?

While not required, we appreciate citations in academic work.

### Can I modify the code?

Yes! You can modify and use the code for personal, academic, or research purposes under the LSAL v1.0. Redistribution of modified versions for commercial purposes requires a commercial license.

### Can I contribute?

Absolutely! Contributions are welcome. See the [Contributing Guide](../developer/contributing.md).

---

## Advanced Questions

### Can I add support for new operations?

Yes! See the [Developer Guide](../developer/contributing.md) for details on adding new operations.

### Can I use it in production?

Yes! DL-Backtrace is production-ready with:
- Deterministic execution
- Error handling
- Performance optimization
- Extensive testing

### Can I export explanations?

Yes, relevance scores are Python dictionaries/tensors that you can save:

```python
import torch
torch.save(relevance, 'relevance.pt')
```

---

## Getting Help

### Where can I find more documentation?

- [User Guide](../guide/introduction.md)
- [Examples](../examples/colab-notebooks.md)
- [Use Cases](../examples/use-cases.md)
- [Developer Guide](../developer/contributing.md)

### Where can I ask questions?

- [GitHub Issues](https://github.com/Lexsi-Labs/DLBacktrace/issues)
- [GitHub Discussions](https://github.com/Lexsi-Labs/DLBacktrace/discussions)
- Email: [support@lexsi.ai](mailto:support@lexsi.ai)

### How do I report bugs?

1. Check [existing issues](https://github.com/Lexsi-Labs/DLBacktrace/issues)
2. Create a new issue with:
   - Description of the problem
   - Steps to reproduce
   - Error messages
   - System information

### How do I request features?

Open a [feature request](https://github.com/Lexsi-Labs/DLBacktrace/issues/new) on GitHub with:
- Description of the feature
- Use case
- Example code (if applicable)

---

## Still Have Questions?

If your question isn't answered here:

1. Search [GitHub Issues](https://github.com/Lexsi-Labs/DLBacktrace/issues)
2. Ask in [GitHub Discussions](https://github.com/Lexsi-Labs/DLBacktrace/discussions)
3. Email us: [support@lexsi.ai](mailto:support@lexsi.ai)

---

**Can't find what you're looking for?**

[Ask on GitHub →](https://github.com/Lexsi-Labs/DLBacktrace/discussions){ .md-button }
