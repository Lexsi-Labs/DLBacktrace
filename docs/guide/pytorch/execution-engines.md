# Execution Engines

DL-Backtrace provides optimized execution engines for running traced models.

---

## ExecutionEngineNoCache

The **ExecutionEngineNoCache** is the recommended engine for most use cases.

### Features

✅ **In-memory execution** - No disk I/O overhead  
✅ **Memory efficient** - Automatic tensor cleanup  
✅ **CPU/GPU compatible** - Works on both devices  
✅ **Fast execution** - Optimized for speed  
✅ **100+ operations** - Comprehensive operation support

### How It Works

```python
# Automatically used by default
dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,),
    layer_implementation="pytorch"
)

# Forward pass uses ExecutionEngineNoCache
node_io = dlb.predict(input_tensor)
```

### Memory Management

The engine automatically manages memory:

1. **Tensor Storage**: Keeps tensors in RAM
2. **Automatic Cleanup**: Removes unused tensors
3. **Efficient Reuse**: Reuses memory where possible
4. **No Disk I/O**: Never writes to disk

### Performance Characteristics

**Advantages:**
- Fast execution (no disk I/O)
- Lower memory footprint
- Better for large models
- Simpler implementation

**When to Use:**
- Large transformer models (BERT, LLaMA)
- Production environments
- Memory-constrained systems
- Real-time applications

---

## Execution Process

### 1. Graph Tracing

Model is traced to create a computational graph:

```python
dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,)
)
```

**What's Captured:**
- All operations (conv, linear, etc.)
- Parameter values (weights, biases)
- Tensor shapes and dtypes
- Operation order (topological sort)

### 2. Node Processing

Nodes are executed in topological order:

```
Input → Conv → ReLU → Pool → Linear → Output
  ↓       ↓      ↓      ↓       ↓       ↓
 Save   Save   Save   Save    Save    Save
```

Each node:
1. Retrieves inputs from tensor map
2. Executes operation
3. Stores output
4. Cleans up unused tensors

### 3. Output Collection

Results are collected as `(inputs, output)` pairs:

```python
node_io = {
    "input": ([], tensor([...])),
    "conv1": ([input_tensor], tensor([...])),
    "relu": ([conv_output], tensor([...])),
    # ... more nodes
}
```

---

## Device Support

### Device Consistency

Ensures all tensors are on the same device:

```python
def ensure_tensor_consistency(tensors, target_device=None):
    """Ensures all tensors are on the same device"""
    if target_device is None:
        target_device = tensors[0].device
    
    return [t.to(device=target_device) if t.device != target_device 
            else t for t in tensors]
```

---

## Dtype Handling

### Automatic Dtype Consistency

The engine ensures dtype consistency across operations:

```python
def ensure_dtype_consistency(tensors, target_dtype=None):
    """Ensures all tensors have consistent dtypes"""
    # Automatically converts float16 to float32 on CPU
    # Maintains dtype on GPU when appropriate
```

### CPU vs GPU Dtypes

**CPU:**
- Prefers `float32` for stability
- Converts `float16` → `float32`

**GPU:**
- Supports `float16` and `float32`
- Uses mixed precision when beneficial

---

## Operation Support

### 100+ Supported Operations

The execution engine supports comprehensive PyTorch operations:

**Core Operations:**
- `linear`, `conv2d`, `conv1d`
- `matmul`, `bmm`, `addmm`
- `relu`, `gelu`, `silu`
- `max_pool2d`, `avg_pool2d`
- `layer_norm`, `batch_norm`

**Tensor Operations:**
- `transpose`, `permute`, `reshape`
- `slice`, `cat`, `split`
- `squeeze`, `unsqueeze`
- `index_select`, `gather`

**Advanced Operations:**
- `scaled_dot_product_attention`
- `embedding`
- `softmax`, `log_softmax`
- `dropout` (pass-through in eval mode)

See [Supported Operations](operations.md) for the complete list.

---

## Deterministic Execution

### Environment Setup

DL-Backtrace automatically configures deterministic execution:

```python
# Set in reproducibility.py
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

### Random Seed Control

```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
```

### Reproducible Results

Same input + same model = same output, every time.

---

## Error Handling

### Robust Validation

The engine validates inputs and handles errors gracefully:

```python
# Example: Embedding operation
if weight.device != indices.device:
    indices = indices.to(weight.device)
    logger.info(f"Moved indices to device {weight.device}")
```

### Clear Error Messages

```python
raise RuntimeError(
    f"[{node_name}] ❌ Operation {func_name} expects 2 inputs, "
    f"got {len(layer_in)} inputs"
)
```

---

## Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("dl_backtrace")
```

### Inspect Node I/O

```python
node_io = dlb.predict(input_tensor)

for node_name, (inputs, output) in node_io.items():
    print(f"{node_name}:")
    print(f"  Input shapes: {[inp.shape for inp in inputs]}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
```

---

## Best Practices

!!! tip "Monitor Memory"
    Watch GPU memory usage with `nvidia-smi` or `torch.cuda.memory_summary()`.

!!! tip "Enable Determinism"
    For reproducible results, ensure deterministic mode is enabled.

!!! warning "Large Models"
    Models like LLaMA-3B+ require significant memory. Use GPU with sufficient VRAM.

---

## Troubleshooting

### Out of Memory

**Solution:**
- Use smaller batch sizes
- Run on CPU
- Use gradient checkpointing

### Slow Execution

**Solution:**
- Use GPU instead of CPU
- Ensure model is in eval mode
- Check for disk I/O (shouldn't happen with no-cache)

### Inconsistent Results

**Solution:**
- Enable deterministic mode
- Set random seeds
- Ensure model is in eval mode

---

## Next Steps

- [Supported Operations](operations.md) - See all supported operations
- [Model Tracing](tracing.md) - Learn about graph tracing
- [Examples](../../examples/colab-notebooks.md) - Interactive notebooks



