# Model Tracing

Learn how DL-Backtrace traces PyTorch models to create computational graphs.

---

## What is Model Tracing?

Model tracing is the process of capturing a model's computational graph - the sequence of operations that transform inputs to outputs.

```python
# Your model
Input → Conv → ReLU → Pool → Linear → Output

# Traced graph
Node: input (placeholder)
Node: conv2d (call_function)
Node: relu (call_function)
Node: adaptive_avg_pool2d (call_function)
Node: linear (call_function)
Node: output (output)
```

---

## How Tracing Works

### Step 1: torch.export_for_training

DL-Backtrace uses PyTorch's `torch.export_for_training` to trace models:

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Provide a dummy input for tracing
dummy_input = torch.randn(1, 3, 224, 224)

# DLBacktrace uses torch.export internally
dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,)
)
```

### Step 2: Graph Extraction

The traced graph contains:

- **Nodes**: Individual operations
- **Edges**: Data flow between operations
- **Parameters**: Model weights and biases
- **Metadata**: Operation types and arguments

### Step 3: Graph Building

DL-Backtrace processes the traced graph:

```python
# Internal process
1. Extract all nodes
2. Identify node types (placeholder, call_function, output)
3. Extract operation names (conv2d, linear, etc.)
4. Get hyperparameters (kernel_size, stride, etc.)
5. Build NetworkX graph
6. Perform topological sort
```

---

## Tracing Requirements

### Model Requirements

The model must be:

✅ A `torch.nn.Module` subclass  
✅ In evaluation mode (`model.eval()`)  
✅ Using supported operations  
✅ Traceable (no dynamic control flow that depends on data values)

### Input Requirements

The dummy input must:

✅ Have the correct shape  
✅ Have the correct dtype  
✅ Be on the correct device (CPU/GPU)  
✅ Match the actual input structure

---

## Example: Simple Model

```python
import torch
import torch.nn as nn
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Define model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

# Create model
model = SimpleCNN()
model.eval()

# Prepare dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Trace the model
dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,),
    device="cpu"
)

print("✅ Model traced successfully!")
```

<!-- ---

## Traced Graph Structure

### Node Types

**1. Placeholder Nodes**
- Represent model inputs
- No computation
- Source of data flow

```python
# Example placeholder
Node(name='input_1', op='placeholder', target='input_1')
```

**2. Call Function Nodes**
- Represent operations
- Contain function reference
- Have arguments

```python
# Example call_function
Node(
    name='conv2d_1',
    op='call_function',
    target=<torch.ops.aten.conv2d>,
    args=(...)
)
```

**3. Output Nodes**
- Represent model outputs
- End of data flow

```python
# Example output
Node(name='output', op='output', target='output')
```

### Hyperparameters

Each node stores its hyperparameters:

```python
# Conv2d hyperparameters
{
    'weight': tensor(...),
    'bias': tensor(...),
    'stride': (1, 1),
    'padding': (1, 1),
    'dilation': (1, 1),
    'groups': 1
}
```

---

## Multi-Input Models

### Models with Multiple Inputs

```python
class MultiInputModel(nn.Module):
    def forward(self, image, metadata):
        # Process image
        x = self.image_encoder(image)
        # Process metadata
        y = self.metadata_encoder(metadata)
        # Combine
        return self.classifier(torch.cat([x, y], dim=1))

# Trace with multiple inputs
dummy_image = torch.randn(1, 3, 224, 224)
dummy_metadata = torch.randn(1, 100)

dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_image, dummy_metadata)
)
``` -->

---

## Dynamic Shapes

### Symbolic Dimensions

DL-Backtrace handles dynamic shapes using symbolic dimensions:

```python
# Variable sequence length
dummy_input = torch.randn(1, 128, 768)  # (batch, seq_len, hidden)

# torch.export creates symbolic dimensions
# seq_len becomes a symbol (e.g., s0)

dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,)
)

# Works with different sequence lengths
real_input = torch.randn(1, 256, 768)  # Different seq_len
node_io = dlb.predict(real_input)  # ✅ Works!
```

### Constraints

Some constraints apply:
- Batch dimension often must match
- Some operations require fixed sizes
- Model architecture must support dynamic shapes

---

### Caching Traced Graphs

Trace once, use multiple times:

```python
# Trace model
dlb = DLBacktrace(model=model, input_for_graph=(dummy_input,))

# Use with different inputs
for input_batch in dataloader:
    node_io = dlb.predict(input_batch)
    # Process results
```

---

## Best Practices

!!! tip "Use Representative Inputs"
    Ensure dummy input represents actual use case (shape, dtype, device).

!!! tip "Trace Once"
    Trace the model once, then reuse for multiple predictions.

!!! tip "Check Compatibility"
    Verify all operations are supported before tracing large models.

!!! warning "Eval Mode Required"
    Always use `model.eval()` before tracing.

---

## Performance Considerations

### Memory Usage

Tracing requires memory for:
- Graph structure
- Model parameters
- Dummy input execution

---

## Next Steps

- [Execution Engines](execution-engines.md) - How traced graphs are executed
- [Supported Operations](operations.md) - What can be traced
- [DLBacktrace Guide](dlbacktrace.md) - Complete API
- [Examples](../../examples/colab-notebooks.md) - Tracing examples



