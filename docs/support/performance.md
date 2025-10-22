# Performance Tips

Optimize DL-Backtrace performance for your use case.

---

## Hardware Optimization

### Use GPU

```python
# Move model and inputs to GPU
model = model.cuda()
input_tensor = input_tensor.cuda()

dlb = DLBacktraceFX(model, input_for_graph=(input_tensor,))
```

**Impact**: 10-50x speedup depending on model size

### Use Mixed Precision

```python
# Enable automatic mixed precision
with torch.cuda.amp.autocast():
    node_io = dlb.predict(input_tensor)
```

**Impact**: 2-3x speedup with minimal accuracy loss

---

## Software Optimization

### Batch Processing

```python
# Process multiple inputs at once
batch = torch.stack([input1, input2, input3])
node_io = dlb.predict(batch)
```

**Impact**: Better GPU utilization

### Disable Debugging

```python
import logging
logging.getLogger('dl_backtrace').setLevel(logging.WARNING)
```

**Impact**: Slight speedup by reducing I/O

---

## Memory Optimization

### Clear Cache

```python
import torch
torch.cuda.empty_cache()
```

### Use No-Cache Engine

ExecutionEngineNoCache is automatically used and is already optimized.

---

## Benchmarking

### Measure Performance

```python
import time

start = time.time()
node_io = dlb.predict(input_tensor)
end = time.time()

print(f"Forward pass: {end - start:.2f}s")
```

---

For more tips, see the [User Guide](../guide/introduction.md).



