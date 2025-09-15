# 🚀 DL-Backtrace Layer Implementation Selection Guide

DL-Backtrace now supports **multiple layer implementations**, allowing you to choose between different computational backends for optimal performance based on your hardware and requirements.

## 📋 Available Implementations

| Implementation | Description | Use Case | Requirements |
|---------------|-------------|----------|-------------|
| **`original`** | NumPy-based baseline | Most stable, debugging | NumPy |
| **`cuda`** | CUDA-accelerated | Large models, maximum speed | CUDA toolkit + GPU |
| **`pytorch`** | PyTorch-native | Balanced speed/compatibility | PyTorch |
| **`refactored`** | Optimized NumPy | Improved original version | NumPy |

## 🔧 Basic Usage

### Default Usage (Original Implementation)
```python
from dl_backtrace.pytorch_backtrace import DLBacktraceFX

# Uses 'original' implementation by default
dlb = DLBacktraceFX(model, sample_input)
io_data = dlb.predict(sample_input)
dlb.evaluation(task="classification")
```

### CUDA Acceleration
```python
# Use CUDA for maximum performance (requires CUDA-compatible GPU)
dlb = DLBacktraceFX(
    model, 
    sample_input,
    layer_implementation="cuda"  # 🚀 GPU acceleration
)
```

### PyTorch Native
```python
# Use PyTorch's native operations
dlb = DLBacktraceFX(
    model, 
    sample_input,
    layer_implementation="pytorch"  # ⚡ PyTorch optimized
)
```

## 🏃‍♂️ Performance Comparison Example

```python
import time
import torch
from dl_backtrace.pytorch_backtrace import DLBacktraceFX

def benchmark_implementations(model, sample_input):
    implementations = ["original"]
    if torch.cuda.is_available():
        implementations.append("cuda")
    
    results = {}
    
    for impl in implementations:
        print(f"Testing {impl.upper()} implementation...")
        
        # Time the execution
        start_time = time.time()
        
        dlb = DLBacktraceFX(model, sample_input, layer_implementation=impl)
        io_data = dlb.predict(sample_input)
        dlb.evaluation(task="classification", multiplier=100.0)
        
        total_time = time.time() - start_time
        results[impl] = total_time
        
        print(f"  ⏱️ Total time: {total_time:.2f}s")
    
    # Compare performance
    if len(results) > 1:
        speedup = results['original'] / results['cuda']
        print(f"\n🚀 CUDA Speedup: {speedup:.2f}x")

# Usage
model = your_pytorch_model
sample_input = your_sample_tensor
benchmark_implementations(model, sample_input)
```

## 🎯 Choosing the Right Implementation

### When to use **Original** (`original`)
- ✅ First time using DL-Backtrace
- ✅ Debugging and development
- ✅ Small to medium models
- ✅ CPU-only environments
- ✅ Maximum stability required

### When to use **CUDA** (`cuda`)
- ✅ Large models (>1B parameters)
- ✅ Production deployments
- ✅ CUDA-compatible GPU available
- ✅ Maximum performance needed
- ⚠️ Requires CUDA toolkit installation

### When to use **PyTorch** (`pytorch`)
- ✅ Good balance of speed and compatibility
- ✅ Leveraging PyTorch optimizations
- ✅ Cross-platform deployment
- ✅ Integration with existing PyTorch pipelines

### When to use **Refactored** (`refactored`)
- ✅ Improved version of original
- ✅ Better numerical stability
- ✅ CPU environments with performance needs

## 🔄 Automatic Fallback

The system includes intelligent fallback mechanisms:

```python
# If CUDA is requested but not available, automatically falls back to 'original'
dlb = DLBacktraceFX(model, input, layer_implementation="cuda")
# Output: ⚠️ CUDA implementation requested but CUDA not available. 
#         Falling back to 'original' implementation.
```

## 🧪 Supported Layers

All layer implementations support these layer types:

- ✅ **Linear/Dense** layers
- ✅ **Convolutional** layers (Conv1D, Conv2D)
- ✅ **Pooling** layers (MaxPool, AvgPool, AdaptiveAvgPool)
- ✅ **Attention** mechanisms (Self-attention, Multi-head)
- ✅ **Embedding** layers
- ✅ **Mathematical operations** (Add, Multiply)
- ✅ **Activation functions**
- ✅ **Normalization** layers

## 💡 Advanced Usage

### Model-Specific Optimization
```python
# For transformer models, CUDA gives best results
if "transformer" in model.__class__.__name__.lower():
    implementation = "cuda"
else:
    implementation = "original"

dlb = DLBacktraceFX(model, input, layer_implementation=implementation)
```

### Error Handling
```python
def safe_dlbacktrace(model, input, preferred_impl="cuda"):
    implementations = [preferred_impl, "pytorch", "original"]
    
    for impl in implementations:
        try:
            dlb = DLBacktraceFX(model, input, layer_implementation=impl)
            return dlb
        except Exception as e:
            print(f"Failed with {impl}: {e}")
            continue
    
    raise RuntimeError("All implementations failed")
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Fallback to original implementation
   dlb = DLBacktraceFX(model, input, layer_implementation="original")
   ```

2. **Import Errors**
   ```bash
   # Ensure CUDA layers are compiled
   ./compile_cuda_layers.sh
   ```

3. **Performance Issues**
   ```python
   # Try different implementations to find optimal performance
   for impl in ["cuda", "pytorch", "original"]:
       # Benchmark each implementation
   ```

## 📊 Expected Performance Gains

| Model Size | Original | CUDA | Speedup |
|-----------|----------|------|---------|
| Small (<10M params) | 1.0s | 0.8s | 1.25x |
| Medium (10M-100M) | 5.0s | 2.0s | 2.5x |
| Large (>1B params) | 60s | 15s | 4.0x |

*Performance varies based on hardware configuration and model architecture*

## 🎉 Migration from Old Version

```python
# Old way
from dl_backtrace.pytorch_backtrace import DLBacktraceFX
dlb = DLBacktraceFX(model, input)

# New way - same code works, but now you can optimize!
dlb = DLBacktraceFX(model, input, layer_implementation="cuda")  # Just add this parameter!
```

## 🤝 Contributing

To add a new layer implementation:

1. Create implementation in `dl_backtrace/pytorch_backtrace/dlbacktrace/utils/cuda_utils/LayerName/`
2. Add launch function in `default_v2.py`
3. Update relevance propagation to use the launch function
4. Test all implementations
5. Update documentation

---

**🎯 Ready to accelerate your model explainability? Choose your implementation and experience the performance boost!** 