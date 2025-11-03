# Best Practices

Guidelines for effective use of DL-Backtrace.

---

## Model Preparation

### Always Use Eval Mode

```python
model.eval()  # Critical!
dlb = DLBacktrace(model, input_for_graph=(dummy,))
```

### Match Input Shapes

```python
# Dummy input shape must match real input
dummy_input = torch.randn(1, 3, 224, 224)
real_input = torch.randn(1, 3, 224, 224)  # Same shape!
```

### Test on Small Models First

Start with simple models to understand the workflow before scaling up.

---

## Performance

### Use GPU When Available

```python
model = model.cuda()
input_tensor = input_tensor.cuda()
```

### Reuse Traced Graph

Trace once, predict many times:

```python
# Trace once
dlb = DLBacktrace(model, input_for_graph=(dummy,))

# Use multiple times
for batch in dataloader:
    node_io = dlb.predict(batch)
```

---

## Relevance Evaluation

### Choose Correct Task Type

Match task type to your model:
- Classification: `"multi-class classification"`
- Detection: `"bbox-regression"`
- Segmentation: `"binary-segmentation"`

### Adjust Multiplier for Visualization

```python
# Default (good starting point)
relevance = dlb.evaluation(multiplier=100.0)

# Increase if values too small
relevance = dlb.evaluation(multiplier=1000.0)
```

---

## Debugging

### Enable Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Intermediate Outputs

```python
node_io = dlb.predict(input_tensor)

for name, (inputs, output) in node_io.items():
    print(f"{name}: {output.shape}")
```

---

## Memory Management

### Clear Cache for Large Models

```python
import torch
torch.cuda.empty_cache()
```

### Use Batch Size 1 for Tracing

```python
# For tracing, use single example
dummy_input = torch.randn(1, 3, 224, 224)
```

---

## Interpretation

### Compare Relative Scores

Focus on relative differences, not absolute values.

### Validate Results

Check if important features make sense for your domain.

### Use Multiple Examples

Test with various inputs to understand model behavior.

---

See [User Guide](introduction.md) for comprehensive documentation.



