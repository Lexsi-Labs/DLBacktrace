# DLBacktraceFX Guide

`DLBacktraceFX` is the main class for analyzing PyTorch models with DL-Backtrace.

---

## Basic Usage

### Initialization

```python
from dl_backtrace.pytorch_backtrace import DLBacktraceFX

dlb = DLBacktraceFX(
    model=model,
    input_for_graph=(dummy_input,),
    layer_implementation="pytorch"
)
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` | `nn.Module` | PyTorch model to trace | Required |
| `input_for_graph` | `tuple` | Example inputs for tracing | Required |
| `layer_implementation` | `str` | Implementation type | `"pytorch"` |

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
| `model_type` | `str` | Model architecture type | `"Encoder"` |

**Task Types:**
- `"binary-classification"`
- `"multi-class classification"`
- `"bbox-regression"`
- `"binary-segmentation"`

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
    filename="my_graph",
    format="png"
)
```

**Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `filename` | `str` | Output filename | `"dlbacktrace_graph"` |
| `format` | `str` | Output format | `"png"` |

**Supported Formats:**
- `"png"`: Raster image
- `"svg"`: Vector graphics
- `"pdf"`: PDF document

**Example:**
```python
# Save as PNG
dlb.visualize()

# Save as SVG
dlb.visualize(filename="model_graph", format="svg")
```

---

### `visualize_dlbacktrace()`

Generates visualization of top-k most relevant nodes.

```python
dlb.visualize_dlbacktrace(
    top_k=15,
    filename="relevance_graph",
    format="png"
)
```

**Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `top_k` | `int` | Number of top nodes to show | `15` |
| `filename` | `str` | Output filename | `"dlbacktrace_topk"` |
| `format` | `str` | Output format | `"png"` |

**Example:**
```python
# Show top 10 most relevant nodes
dlb.visualize_dlbacktrace(top_k=10)

# Save with custom filename
dlb.visualize_dlbacktrace(
    top_k=20,
    filename="important_layers",
    format="svg"
)
```

---

## Complete Example

```python
import torch
import torch.nn as nn
from dl_backtrace.pytorch_backtrace import DLBacktraceFX

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
dlb = DLBacktraceFX(
    model=model,
    input_for_graph=(dummy_input,),
    layer_implementation="pytorch"
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

## Advanced Usage

### Multiple Inputs

For models with multiple inputs:

```python
# Model with two inputs
model = MyMultiInputModel()
dummy_input1 = torch.randn(1, 3, 224, 224)
dummy_input2 = torch.randn(1, 100)

# Initialize
dlb = DLBacktraceFX(
    model=model,
    input_for_graph=(dummy_input1, dummy_input2),
    layer_implementation="pytorch"
)

# Predict with real inputs
node_io = dlb.predict(real_input1, real_input2)
```

### Custom Device

Specify device for execution:

```python
# Use GPU
model = model.cuda()
dummy_input = torch.randn(1, 3, 224, 224).cuda()

dlb = DLBacktraceFX(
    model=model,
    input_for_graph=(dummy_input,),
    layer_implementation="pytorch"
)

# Predictions will run on GPU
node_io = dlb.predict(test_input.cuda())
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
- [API Reference](../../api/pytorch/dlbacktracefx.md) - Detailed API docs
- [Tutorials](../../tutorials/vision/resnet.md) - Step-by-step examples



