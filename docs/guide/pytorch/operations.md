# Supported Operations

DL-Backtrace supports 100+ PyTorch ATen operations. This page lists all supported operations.

---

## Core Operations

### Linear Layers

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `linear` | Fully connected layer | ✅ |
| `addmm` | Matrix multiply and add | ✅ |
| `mm` | Matrix multiply | ✅ |

---

### Convolutional Layers

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `conv2d` | 2D convolution | ✅ |
| `conv1d` | 1D convolution | ✅ |
| `conv_transpose2d` | 2D transposed convolution | 🔄 |
| `conv_transpose1d` | 1D transposed convolution | 🔄 |

---

### Pooling Operations

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `max_pool2d` | 2D max pooling | ✅ |
| `avg_pool2d` | 2D average pooling | ✅ |
| `adaptive_avg_pool2d` | Adaptive average pooling 2D | ✅ |
| `adaptive_max_pool2d` | Adaptive max pooling 2D | ✅ |
| `max_pool1d` | 1D max pooling | ✅ |
| `avg_pool1d` | 1D average pooling | ✅ |
| `adaptive_avg_pool1d` | Adaptive average pooling 1D | ✅ |
| `adaptive_max_pool1d` | Adaptive max pooling 1D | ✅ |

---

### Activation Functions

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `relu` | ReLU activation | ✅ |
| `gelu` | GELU activation | ✅ |
| `silu` / `swish` | SiLU/Swish activation | ✅ |
| `sigmoid` | Sigmoid activation | ✅ |
| `tanh` | Tanh activation | ✅ |
| `leaky_relu` | Leaky ReLU | ✅ |
| `elu` | ELU activation | ✅ |
| `softmax` | Softmax activation | ✅ |
| `log_softmax` | Log softmax | ✅ |

---

## Tensor Operations

### Shape Manipulation

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `view` | Reshape tensor | ✅ |
| `reshape` | Reshape tensor | ✅ |
| `flatten` | Flatten tensor | ✅ |
| `squeeze` | Remove dimensions of size 1 | ✅ |
| `unsqueeze` | Add dimension of size 1 | ✅ |
| `transpose` | Transpose dimensions | ✅ |
| `permute` | Permute dimensions | ✅ |
| `contiguous` | Make tensor contiguous | ✅ |

---

### Slicing & Indexing

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `slice` | Slice tensor along dimension | ✅ |
| `index_select` | Select indices | ✅ |
| `gather` | Gather values | ✅ |
| `select` | Select single index | ✅ |
| `narrow` | Narrow tensor | ✅ |

---

### Concatenation & Stacking

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `cat` | Concatenate tensors | ✅ |
| `stack` | Stack tensors | ✅ |
| `split` | Split tensor | ✅ |
| `chunk` | Chunk tensor | ✅ |

---

## Arithmetic Operations

### Basic Arithmetic

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `add` | Addition | ✅ |
| `sub` | Subtraction | ✅ |
| `mul` | Multiplication | ✅ |
| `div` | Division | ✅ |
| `pow` | Power | ✅ |
| `sqrt` | Square root | ✅ |
| `exp` | Exponential | ✅ |
| `log` | Logarithm | ✅ |

---

### Matrix Operations

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `matmul` | Matrix multiplication | ✅ |
| `bmm` | Batch matrix multiplication | ✅ |
| `addmm` | Matrix multiply and add | ✅ |
| `baddbmm` | Batch matrix multiply and add | ✅ |

---

### Comparison Operations

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `eq` | Equal | ✅ |
| `ne` | Not equal | ✅ |
| `lt` | Less than | ✅ |
| `le` | Less than or equal | ✅ |
| `gt` | Greater than | ✅ |
| `ge` | Greater than or equal | ✅ |

---

## Normalization

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `layer_norm` | Layer normalization | ✅ |
| `batch_norm` | Batch normalization | ✅ |
| `group_norm` | Group normalization | ✅ |
| `instance_norm` | Instance normalization | ✅ |
---

## Attention Operations

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `scaled_dot_product_attention` | Scaled dot-product attention | ✅ |
| `softmax` | Softmax for attention weights | ✅ |
| `dropout` | Dropout (pass-through in eval) | ✅ |

---

## Embedding Operations

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `embedding` | Embedding lookup | ✅ |
| `embedding_bag` | Embedding bag | 🔄 |

---

## Utility Operations

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `clone` | Clone tensor | ✅ |
| `detach` | Detach from graph | ✅ |
| `to` | Convert dtype/device | ✅ |
| `type` | Type conversion | ✅ |
| `arange` | Create range | ✅ |
| `zeros` | Create zeros | ✅ |
| `ones` | Create ones | ✅ |
| `full` | Create filled tensor | ✅ |

---

## Special Operations

### Symbolic Operations

| Operation | Description | Supported |
|-----------|-------------|-----------|
| `sym_size` | Symbolic size | ✅ |
| `sym_numel` | Symbolic number of elements | ✅ |

These are used internally during graph tracing.

---

## Negative Indexing Support

All dimension-based operations support PyTorch's negative indexing:

```python
# All of these work!
x.transpose(-1, -2)         # Last two dimensions
x.permute([0, -1, -2, 1])   # Mix of positive and negative
x.unsqueeze(-1)             # Add dimension at end
x.slice(dim=-1, ...)        # Slice last dimension
torch.cat([x, y], dim=-1)   # Concatenate on last dimension
```

---

## Operation Categories

### ✅ Fully Supported
Operations that are fully implemented and tested.

### 🔄 In Development
Operations that are planned but not yet implemented.

### ❌ Not Supported
Operations that are not currently supported.

---

## Custom Operations

### Adding Custom Operations

If you need support for a custom operation, you can:

1. **Request support**: Open an issue on GitHub
2. **Contribute**: Submit a pull request
3. **Workaround**: Decompose into supported operations

---

## Testing Operations

To test if an operation is supported:

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Create simple model using the operation
class TestModel(torch.nn.Module):
    def forward(self, x):
        return your_operation(x)

# Try to trace it
try:
    model = TestModel()
    dlb = DLBacktrace(
        model=model,
        input_for_graph=(dummy_input,)
    )
    print("✅ Operation supported!")
except Exception as e:
    print(f"❌ Operation not supported: {e}")
```

---

## Next Steps

- [Execution Engines](execution-engines.md) - Learn how operations are executed
- [Model Tracing](tracing.md) - Understand graph tracing
- [Examples](../../examples/colab-notebooks.md) - See operations in action
- [Best Practices](../best-practices.md) - Optimization tips



