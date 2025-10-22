# TensorFlow Support Deprecation Notice

## ⚠️ Important Announcement

**TensorFlow/Keras support in DL-Backtrace is being deprecated and will be removed in a future release.**

---

## Timeline

- **Current Status**: TensorFlow backend is in maintenance mode
- **Recommendation**: All new projects should use PyTorch backend
- **Future**: TensorFlow backend will be removed in version 3.0

---

## Migration to PyTorch

### Why PyTorch?

The PyTorch backend offers:

- ✅ **Better Performance**: Optimized execution engine with CUDA support
- ✅ **More Features**: 100+ supported operations
- ✅ **Active Development**: All new features are PyTorch-first
- ✅ **Better Tracing**: Advanced graph tracing with `torch.export`
- ✅ **Wider Model Support**: Transformers, LLMs, and custom architectures

### Migration Guide

If you're currently using the TensorFlow backend:

```python
# OLD: TensorFlow backend
from dl_backtrace.tf_backtrace import Backtrace as B
model = tf.keras.applications.ResNet50()
backtrace = B(model=model)
```

**Recommended approach:**

1. **Convert your model to PyTorch** using tools like:
   - [ONNX](https://onnx.ai/) for model conversion
   - Manual reimplementation with PyTorch
   - Use PyTorch-native architectures from `torchvision` or Hugging Face

2. **Use DLBacktraceFX** (PyTorch backend):
   ```python
   # NEW: PyTorch backend
   import torch
   import torchvision.models as models
   from dl_backtrace.pytorch_backtrace import DLBacktraceFX
   
   model = models.resnet50(pretrained=True)
   model.eval()
   
   dummy_input = torch.randn(1, 3, 224, 224)
   dlb = DLBacktraceFX(
       model=model,
       input_for_graph=(dummy_input,),
       layer_implementation="pytorch"
   )
   ```

---

## For Existing TensorFlow Users

### Current Support

The TensorFlow backend (`dl_backtrace.tf_backtrace`) remains available but:

- ❌ No new features
- ❌ Limited bug fixes
- ❌ No optimization updates
- ⚠️ May not work with newest TensorFlow versions

### Alternatives

1. **Switch to PyTorch**: Best option for continued support
2. **Pin TensorFlow version**: Use older TensorFlow versions that are known to work
3. **Fork and maintain**: Fork the project to maintain TensorFlow support yourself

---

## Resources

### Learning PyTorch

- [PyTorch Official Tutorial](https://pytorch.org/tutorials/)
- [PyTorch for TensorFlow Users](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)
- [DL-Backtrace PyTorch Guide](guide/pytorch/overview.md)

### Model Conversion

- [ONNX Model Hub](https://github.com/onnx/models)
- [TensorFlow to PyTorch Conversion](https://github.com/pytorch/pytorch/wiki/PyTorch-vs-TensorFlow)

### DL-Backtrace Documentation

- [Quick Start Guide](home/quickstart.md)
- [PyTorch Backend Overview](guide/pytorch/overview.md)
- [DLBacktraceFX API](guide/pytorch/dlbacktracefx.md)

---

## FAQs

### When will TensorFlow support be removed?

Planned for version 3.0 (date TBD). We will provide advance notice.

### Will old TensorFlow notebooks still work?

They may work with older versions of DL-Backtrace, but we recommend migrating to PyTorch.

### Can I still use TensorFlow in the current version?

Yes, but with limited support. We strongly recommend migrating to PyTorch.

### Why is TensorFlow being deprecated?

- PyTorch has become the dominant framework for research
- PyTorch offers better graph tracing capabilities
- Maintaining two backends is resource-intensive
- Most users prefer PyTorch

---

## Contact

If you have questions or concerns about this deprecation:

- **Email**: [support@aryaxai.com](mailto:support@aryaxai.com)
- **GitHub Issues**: [Open an issue](https://github.com/aryaxai/DL-Backtrace/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aryaxai/DL-Backtrace/discussions)

---

## Summary

- ⚠️ **TensorFlow support is deprecated**
- ✅ **PyTorch is the recommended framework**
- 📚 **Migration resources available**
- 🔄 **Easy transition process**

We appreciate your understanding and are committed to providing the best explainability tools for PyTorch models.

---

<div align="center">

**Ready to switch to PyTorch?**

[PyTorch Quick Start →](home/quickstart.md){ .md-button .md-button--primary }

</div>

