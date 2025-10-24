# Relevance Propagation Overview

Relevance propagation is the core technique DL-Backtrace uses to explain model predictions.

---

## What is Relevance Propagation?

**Relevance propagation** traces the "importance" or "contribution" of each input feature to the model's output by propagating relevance scores backward through the network.
We use Layer-specific algorithms (Linear, Convolutional, Attention) that distribute relevance based on each layer's mathematical properties and activation patterns
It distributes relevance scores across layers, providing insights into feature importance, information flow, and bias, enabling better model interpretation and validation without external dependencies.

### Key Concept

Starting with the output (which has 100% relevance to itself), we ask:

> "Which neurons in the previous layer contributed to this output, and how much?"

We repeat this question layer by layer until we reach the input, resulting in a relevance score for each input feature.

---

## How It Works in DL-Backtrace

### Step 1: Forward Pass

Run input through the model and capture all activations:

```python
node_io = dlb.predict(input_tensor)
```

This stores:
- Input to each layer
- Output from each layer
- Intermediate activations

### Step 2: Initialize Output Relevance

Start with 100% relevance at the output:

```python
relevance = dlb.evaluation(
    multiplier=100.0,  # Starting relevance
    task="multi-class classification"
)
```

### Step 3: Backward Propagation

Propagate relevance backward through each layer:

```
Output (R=100) → Linear (R=?) → ReLU (R=?) → Conv (R=?) → Input (R=?)
```

Each layer distributes its received relevance to its inputs based on their contributions.

### Step 4: Relevance Scores

The result is a relevance score for each:
- Input feature
- Intermediate layer
- Network node

---

## Example: Simple Network

Consider a simple network:

```python
Input (x) → Linear (W·x + b) → ReLU → Output (y)
```

### Forward Pass

```python
# Input
x = [1.0, 2.0, 3.0]

# Linear layer (simplified)
W = [[0.5, 0.3, 0.2],
     [0.1, 0.4, 0.5]]
b = [0.1, 0.2]

# Intermediate
z = W @ x + b = [2.4, 2.4]

# ReLU
a = relu(z) = [2.4, 2.4]

# Output relevance (start here)
R_output = [100, 0]  # Class 0 predicted
```

### Backward Pass

```python
# Propagate through ReLU (pass-through for positive values)
R_z = [100, 0]

# Propagate through Linear
# R_input[i] = sum_j (W[j,i] * x[i] / (W @ x)[j]) * R_z[j]
R_x[0] = (0.5 * 1.0 / 2.4) * 100 = 20.8
R_x[1] = (0.3 * 2.0 / 2.4) * 100 = 25.0
R_x[2] = (0.2 * 3.0 / 2.4) * 100 = 25.0

# Input relevances
# x[0]=1.0 → 20.8% relevance
# x[1]=2.0 → 25.0% relevance
# x[2]=3.0 → 25.0% relevance
```

---

## Visualization

### Input Heatmaps

For images, relevance creates heatmaps showing important regions:

```python
# After evaluation
relevance = dlb.evaluation(...)

# Get input relevance
input_relevance = relevance['input']  # Shape: (1, 3, 224, 224)

# Visualize as heatmap
import matplotlib.pyplot as plt
plt.imshow(input_relevance.sum(1).squeeze(), cmap='hot')
plt.title('Input Relevance Heatmap')
plt.colorbar()
```

### Layer Importance

Identify which layers contribute most:

```python
# Sort layers by relevance
sorted_layers = sorted(
    relevance.items(),
    key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
    reverse=True
)

print("Most relevant layers:")
for name, score in sorted_layers[:10]:
    print(f"  {name}: {score:.2f}")
```

---

## Use Cases

### 1. Feature Importance

**Question:** Which input features matter most?

```python
# For tabular data
input_relevance = relevance['input']
feature_importance = input_relevance.squeeze()

# Rank features
top_features = feature_importance.argsort(descending=True)
```

### 2. Image Saliency

**Question:** Which pixels influenced the prediction?

```python
# For images
pixel_relevance = relevance['input']  # (1, 3, H, W)

# Create saliency map
saliency = pixel_relevance.abs().sum(1)  # Sum over channels
```

### 3. Token Attribution

**Question:** Which words/tokens were most important?

```python
# For text
token_relevance = relevance['embedding']  # (1, seq_len, hidden)

# Get per-token importance
token_scores = token_relevance.sum(dim=-1)  # Sum over hidden dim
```

### 4. Layer Analysis

**Question:** Which layers contribute most to the decision?

```python
# Aggregate relevance by layer type
layer_contributions = {}
for name, rel in relevance.items():
    layer_type = name.split('_')[0]  # e.g., 'conv' from 'conv_1'
    layer_contributions[layer_type] = layer_contributions.get(layer_type, 0) + rel
```

---

<!-- ## Evaluation Modes

DL-Backtrace supports different evaluation modes:

### Default Mode

Standard relevance propagation:

```python
relevance = dlb.evaluation(mode="default")
```

### Contrastive Mode

Compare relevance for different classes:

```python
relevance = dlb.evaluation(mode="contrastive") -->
```

---

## Properties

### 1. Conservation

Total relevance is conserved:

```python
# At output
total_output = sum(output_relevance) = 100

# At each layer
total_layer = sum(layer_relevance) ≈ 100

# At input
total_input = sum(input_relevance) ≈ 100
```

### 2. Selectivity

Only relevant features get high scores:

- Important features: High relevance
- Irrelevant features: Low/zero relevance

### 3. Completeness

All contributions are accounted for:

- Positive contributions: Positive relevance
- Negative contributions: Negative relevance

---

## Interpretation Guidelines

### Positive Relevance

**Meaning:** This feature contributed to the predicted class.

**Example:** For predicting "cat", pixels showing cat features have high positive relevance.

### Negative Relevance

**Meaning:** This feature argued against the predicted class (or for other classes).

**Example:** For predicting "cat", pixels showing dog features might have negative relevance.

### Zero Relevance

**Meaning:** This feature didn't influence the prediction.

**Example:** Background pixels unrelated to the object.

---

## Best Practices

!!! tip "Choose Right Task"
    Select the appropriate task type (classification, detection, etc.) for accurate relevance.

!!! tip "Use Appropriate Multiplier"
    The multiplier (typically 100) scales the starting relevance. Higher values can help with visualization.

!!! tip "Interpret in Context"
    Relevance scores are relative. Compare features against each other, not absolute values.

!!! warning "Not Gradients"
    Relevance ≠ gradients. Relevance shows contribution; gradients show sensitivity.

---

## Next Steps

- [Parameters](parameters.md) - Tune relevance calculation
- [Examples](../../examples/colab-notebooks.md) - See relevance in action

---


