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

### Step 1: torch.export

DL-Backtrace uses PyTorch's `torch.export` to trace models:

```python
from dl_backtrace.pytorch_backtrace import DLBacktraceFX

# Provide a dummy input for tracing
dummy_input = torch.randn(1, 3, 224, 224)

# DLBacktraceFX uses torch.export internally
dlb = DLBacktraceFX(
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
from dl_backtrace.pytorch_backtrace import DLBacktraceFX

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
dlb = DLBacktraceFX(
    model=model,
    input_for_graph=(dummy_input,),
    layer_implementation="pytorch"
)

print("✅ Model traced successfully!")
```

---

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

dlb = DLBacktraceFX(
    model=model,
    input_for_graph=(dummy_image, dummy_metadata)
)
```

---

## Dynamic Shapes

### Symbolic Dimensions

DL-Backtrace handles dynamic shapes using symbolic dimensions:

```python
# Variable sequence length
dummy_input = torch.randn(1, 128, 768)  # (batch, seq_len, hidden)

# torch.export creates symbolic dimensions
# seq_len becomes a symbol (e.g., s0)

dlb = DLBacktraceFX(
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

## Troubleshooting Tracing

### Common Issues

#### 1. Dynamic Control Flow

**Problem:**
```python
def forward(self, x):
    if x.sum() > 0:  # ❌ Data-dependent control flow
        return self.path_a(x)
    else:
        return self.path_b(x)
```

**Solution:**
Use torch operations instead:
```python
def forward(self, x):
    mask = (x.sum() > 0).float()
    return mask * self.path_a(x) + (1 - mask) * self.path_b(x)
```

#### 2. In-place Operations

**Problem:**
```python
def forward(self, x):
    x += bias  # ❌ In-place operation
    return x
```

**Solution:**
Use out-of-place operations:
```python
def forward(self, x):
    x = x + bias  # ✅ Out-of-place
    return x
```

#### 3. Unsupported Operations

**Problem:**
```python
def forward(self, x):
    return torch.special.some_function(x)  # ❌ Not supported
```

**Solution:**
- Check [supported operations](operations.md)
- Decompose into supported operations
- Request support on GitHub

---

## Graph Inspection

### View Traced Nodes

```python
# After tracing
dlb = DLBacktraceFX(model=model, input_for_graph=(dummy_input,))

# Access graph
graph = dlb.graph

# Inspect nodes
for node in graph.nodes():
    print(f"Node: {node}")
    print(f"  Type: {graph.nodes[node].get('type', 'unknown')}")
    print(f"  Operation: {graph.nodes[node].get('node_type', 'unknown')}")
```

### Visualize Graph

```python
# Generate graph visualization
dlb.visualize()

# This creates:
# - dlbacktrace_graph.png
# - dlbacktrace_graph.svg
```

---

## Advanced Tracing

### Custom Tracing Logic

For advanced use cases, you can customize the tracing:

```python
# Access internal graph builder
from dl_backtrace.pytorch_backtrace.dlbacktrace.core import graph_builder

# Customize tracing (advanced)
# See developer documentation for details
```

### Caching Traced Graphs

Trace once, use multiple times:

```python
# Trace model
dlb = DLBacktraceFX(model=model, input_for_graph=(dummy_input,))

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

### Tracing Time

- **Small models**: < 1 second
- **Medium models** (ResNet-50): 1-3 seconds
- **Large models** (BERT-base): 3-10 seconds
- **Very large models** (LLaMA-3B): 10-30 seconds

### Memory Usage

Tracing requires memory for:
- Graph structure
- Model parameters
- Dummy input execution

Typical overhead: 100-500 MB

---

## Next Steps

- [Execution Engines](execution-engines.md) - How traced graphs are executed
- [Supported Operations](operations.md) - What can be traced
- [DLBacktraceFX Guide](dlbacktracefx.md) - Complete API
- [Troubleshooting](../../support/troubleshooting.md) - Fix tracing issues



