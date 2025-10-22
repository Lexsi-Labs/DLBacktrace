# Troubleshooting

Solutions to common issues when using DL-Backtrace.

---

## Installation Issues

### ImportError: No module named 'dl_backtrace'

**Problem:** Package not installed correctly.

**Solution:**
```bash
cd DL-Backtrace
pip install -e .
```

### CUDA/cuDNN version mismatch

**Problem:** PyTorch CUDA version doesn't match system CUDA.

**Solution:**
```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch version
# For CUDA 12.6:
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### Cannot compile CUDA kernels

**Problem:** Missing CUDA toolkit or compiler.

**Solution:**
1. Install CUDA Toolkit: [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Install C++ compiler:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential
   
   # macOS
   xcode-select --install
   ```
3. Set CUDA_HOME:
   ```bash
   export CUDA_HOME=/usr/local/cuda
   ```

---

## Model Tracing Issues

### RuntimeError: Tracing failed

**Problem:** Model contains unsupported operations or dynamic control flow.

**Solution:**

**Check for data-dependent control flow:**
```python
# ❌ BAD: Data-dependent if statement
def forward(self, x):
    if x.sum() > 0:  # Data-dependent!
        return self.branch_a(x)
    return self.branch_b(x)

# ✅ GOOD: Use tensor operations
def forward(self, x):
    mask = (x.sum() > 0).float()
    return mask * self.branch_a(x) + (1 - mask) * self.branch_b(x)
```

**Check for unsupported operations:**
```python
# Find which operation fails
try:
    dlb = DLBacktraceFX(model, input_for_graph=(dummy,))
except Exception as e:
    print(f"Failed at: {e}")
```

### Model must be in eval mode

**Problem:** Forgot to call `model.eval()`.

**Solution:**
```python
model.eval()  # Always set eval mode!
dlb = DLBacktraceFX(model, input_for_graph=(dummy,))
```

### Shape mismatch errors

**Problem:** Dummy input shape doesn't match model expectations.

**Solution:**
```python
# Make sure dummy input matches real input
# For image models:
dummy_input = torch.randn(1, 3, 224, 224)  # (batch, channels, height, width)

# For text models:
dummy_input = torch.randint(0, vocab_size, (1, seq_len))  # (batch, sequence)

# Check model's expected input
print(model.forward.__code__.co_varnames)
```

---

## Memory Issues

### CUDA out of memory

**Problem:** Model too large for GPU.

**Solutions:**

**1. Reduce batch size:**
```python
# Instead of:
dummy_input = torch.randn(32, 3, 224, 224)

# Use:
dummy_input = torch.randn(1, 3, 224, 224)
```

**2. Use CPU:**
```python
model = model.cpu()
dummy_input = dummy_input.cpu()
```

**3. Clear CUDA cache:**
```python
import torch
torch.cuda.empty_cache()
```

**4. Monitor memory:**
```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Out of RAM (CPU memory)

**Problem:** System running out of RAM.

**Solutions:**

**1. Close other applications**

**2. Reduce model size:**
```python
# Use smaller variants
model = models.resnet18()  # Instead of resnet152
```

**3. Process in batches:**
```python
# Instead of processing all at once
for batch in dataloader:
    node_io = dlb.predict(batch)
    # Process results immediately
    del node_io  # Free memory
```

---

## Execution Issues

### Operation not supported

**Problem:** Model uses unsupported PyTorch operation.

**Solutions:**

**1. Check supported operations:**
See [Supported Operations](../guide/pytorch/operations.md)

**2. Decompose operation:**
```python
# Replace unsupported operation with supported ones
# Example: Replace custom op with standard ops
```

**3. Request support:**
Open an issue on [GitHub](https://github.com/aryaxai/DL-Backtrace/issues)

### Dtype mismatch errors

**Problem:** Tensors have incompatible dtypes.

**Solution:**
```python
# Ensure consistent dtype
model = model.float()  # or .half() for FP16
input_tensor = input_tensor.float()

dlb = DLBacktraceFX(model, input_for_graph=(input_tensor,))
```

### Device mismatch errors

**Problem:** Tensors on different devices (CPU vs GPU).

**Solution:**
```python
# Ensure all on same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
input_tensor = input_tensor.to(device)

dlb = DLBacktraceFX(model, input_for_graph=(input_tensor,))
```

---

## Relevance Evaluation Issues

### Relevance scores are all zero

**Problem:** Incorrect task type or evaluation parameters.

**Solution:**
```python
# Make sure task type matches your model
# For classification:
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="multi-class classification"  # Not "bbox-regression"!
)

# Check output was computed
node_io = dlb.predict(input_tensor)
print("Output computed:", len(node_io) > 0)
```

### Relevance values are very small

**Problem:** Default multiplier might be too small for visualization.

**Solution:**
```python
# Increase multiplier
relevance = dlb.evaluation(
    multiplier=1000.0,  # Increase from 100.0
    task="multi-class classification"
)
```

### NaN or Inf in relevance

**Problem:** Numerical instability in relevance calculation.

**Solution:**
```python
# 1. Check for NaN in model output
output = model(input_tensor)
assert not torch.isnan(output).any(), "Model output contains NaN"

# 2. Use float32 instead of float16
model = model.float()
input_tensor = input_tensor.float()

# 3. Check for zero divisions
# This is usually handled internally, report if it persists
```

---

## Visualization Issues

### Visualization doesn't generate

**Problem:** Graphviz not installed.

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
# Download from https://graphviz.org/download/

# Python package
pip install graphviz
```

### Visualization is too large

**Problem:** Graph has too many nodes.

**Solution:**
```python
# Instead of full graph
dlb.visualize()

# Use top-k visualization
dlb.visualize_dlbacktrace(top_k=15)  # Only show top 15 nodes
```

### Cannot open visualization file

**Problem:** File path issues or permissions.

**Solution:**
```python
import os

# Check if file was created
if os.path.exists('dlbacktrace_graph.png'):
    print("✅ File created successfully")
else:
    print("❌ File not created")

# Specify absolute path
dlb.visualize(filename='/path/to/my_graph')
```

---

## Performance Issues

### Tracing is very slow

**Possible causes and solutions:**

**1. Large model:**
```python
# This is expected for large models (LLaMA-3B+)
# Wait patiently or use smaller model variant
```

**2. CPU vs GPU:**
```python
# Use GPU for faster tracing
model = model.cuda()
dummy_input = dummy_input.cuda()
```

**3. Debug mode enabled:**
```python
# Disable debug logging
import logging
logging.getLogger('dl_backtrace').setLevel(logging.WARNING)
```

### Evaluation is slow

**Solutions:**

**1. Use GPU:**
```python
# Move to GPU
model = model.cuda()
```

**2. Reduce visualization frequency:**
```python
# Don't visualize every prediction
if i % 100 == 0:  # Only every 100th
    dlb.visualize_dlbacktrace(top_k=10)
```

---

## Model-Specific Issues

### Transformer models fail

**Problem:** Attention mechanisms not handled correctly.

**Solution:**
```python
# DL-Backtrace auto-detects attention type
# If it fails, check:

# 1. Model is in eval mode
model.eval()

# 2. Attention mask is provided
dlb.predict(input_ids, attention_mask=attention_mask)

# 3. Model name is recognized
# (BERT, RoBERTa automatically detected)
```

### LLaMA models run out of memory

**Problem:** LLaMA models are very large.

**Solutions:**

**1. Use smaller variant:**
```python
# LLaMA-1B instead of LLaMA-3B
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
```

**2. Use CPU with smaller batch:**
```python
model = model.cpu()
input_ids = input_ids[:1]  # Single example
```

**3. Clear cache frequently:**
```python
torch.cuda.empty_cache()
```

### Custom models fail

**Problem:** Custom architecture uses unsupported patterns.

**Solution:**

**1. Check operations:**
```python
# Print model architecture
print(model)

# Check if operations are supported
```

**2. Simplify model:**
```python
# Test with simpler version first
class SimpleModel(nn.Module):
    # Simplified version for testing
    pass
```

**3. Debug step by step:**
```python
# Trace only part of the model
class PartialModel(nn.Module):
    def forward(self, x):
        return self.encoder(x)  # Test encoder only
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("dl_backtrace")
logger.setLevel(logging.DEBUG)
```

### Inspect Intermediate Results

```python
# Check node I/O
node_io = dlb.predict(input_tensor)

for name, (inputs, output) in node_io.items():
    print(f"{name}:")
    print(f"  Input shapes: {[i.shape for i in inputs]}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Has NaN: {torch.isnan(output).any()}")
```

### Minimal Reproducible Example

Create smallest example that reproduces the issue:

```python
import torch
import torch.nn as nn
from dl_backtrace.pytorch_backtrace import DLBacktraceFX

# Minimal model
class MinimalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

# Test
model = MinimalModel()
model.eval()

dummy = torch.randn(1, 10)

try:
    dlb = DLBacktraceFX(model, input_for_graph=(dummy,))
    print("✅ Minimal example works")
except Exception as e:
    print(f"❌ Error: {e}")
```

---

## Getting Further Help

If issues persist:

### 1. Check Documentation
- [User Guide](../guide/introduction.md)
- [API Reference](../api/pytorch/dlbacktracefx.md)
- [FAQ](faq.md)

### 2. Search Issues
Check if someone else had the same problem:
- [GitHub Issues](https://github.com/aryaxai/DL-Backtrace/issues)

### 3. Create New Issue
Include:
- Python version
- PyTorch/TensorFlow version
- DL-Backtrace version
- Minimal code to reproduce
- Full error message
- System info (OS, GPU, etc.)

### 4. Contact Support
- GitHub: [Create an issue](https://github.com/aryaxai/DL-Backtrace/issues/new)
- Email: [support@aryaxai.com](mailto:support@aryaxai.com)

---

## System Information

To help with debugging, collect system information:

```python
import torch
import sys
import platform

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

<div align="center">

**Still stuck?**

[Ask on GitHub →](https://github.com/aryaxai/DL-Backtrace/issues/new){ .md-button }

</div>



