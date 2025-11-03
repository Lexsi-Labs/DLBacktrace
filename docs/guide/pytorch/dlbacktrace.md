# DLBacktrace Guide

`DLBacktrace` is the main class for analyzing PyTorch models with DL-Backtrace.

---

## Basic Usage

### Initialization

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace

dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,),
    device="cuda"
)
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` | `nn.Module` | PyTorch model to trace | Required |
| `input_for_graph` | `tuple` | Example inputs for tracing | Required |
| `device` | `str` | Device type | `"cpu"` |

---

## Methods

### `predict()`

Runs forward pass and captures layer outputs.

```python
node_io = dlb.predict(*inputs)
```

**Parameters:**
- `*inputs`: Model inputs (tensors)

**Returns:**
- `dict`: Mapping of node names to (inputs, output) tuples

**Example:**
```python
test_input = torch.randn(1, 3, 224, 224)
node_io = dlb.predict(test_input)

for node_name, (inputs, output) in node_io.items():
    print(f"{node_name}: {output.shape}")
```

---

### `evaluation()`

Calculates relevance propagation.

```python
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="multi-class classification"
)
```

**Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `mode` | `str` | Evaluation mode | `"default"` |
| `multiplier` | `float` | Starting relevance value | `100.0` |
| `task` | `str` | Task type | Required |
| `thresholding` | `float` | Threshold for segmentation | `0.5` |

**Task Types:**
- `"binary-classification"`
- `"multi-class classification"`

**Model Types:**
- `"Encoder"`: Standard encoder models
- `"Encoder_Decoder"`: Seq2seq models

**Returns:**
- `dict`: Mapping of node names to relevance scores

**Example:**
```python
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="multi-class classification"
)

# Find most relevant layers
sorted_rel = sorted(
    relevance.items(),
    key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
    reverse=True
)

print("Top 5 relevant layers:")
for name, score in sorted_rel[:5]:
    print(f"  {name}: {score}")
```

---

### `visualize()`

Generates visualization of the computational graph.

```python
dlb.visualize(
    save_path="my_graph.png"
)


---

### `visualize_dlbacktrace()`

Generates visualization of top-k most relevant nodes.

```python
dlb.visualize_dlbacktrace(
    top_k=15,
    ouput_path="relevance_graph",
    relevance_threshold
)

---

## Complete Example

```python
import torch
import torch.nn as nn
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Define model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        return self.fc(x.flatten(1))

# Initialize
model = CNN()
model.eval()

dummy_input = torch.randn(1, 3, 32, 32)
dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,),
    device="cpu"
)

# Analyze
test_input = torch.randn(1, 3, 32, 32)
node_io = dlb.predict(test_input)

relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="multi-class classification"
)

# Visualize
dlb.visualize()
dlb.visualize_dlbacktrace(top_k=10)

print("Analysis complete!")
```

---

## Tips & Best Practices

!!! tip "Use Evaluation Mode"
    Always set model to eval mode: `model.eval()`

!!! tip "Match Input Shapes"
    Ensure dummy input shape matches real input shape

!!! tip "GPU Memory"
    Monitor memory usage for large models

!!! warning "Unsupported Operations"
    Some custom ops may not be supported. Check error messages.

---

## Next Steps

- [Execution Engines](execution-engines.md) - Learn about execution options
- [Supported Operations](operations.md) - See all supported operations
- [Examples](../../examples/colab-notebooks.md) - Interactive notebooks
- [Best Practices](../best-practices.md) - Tips for effective use



