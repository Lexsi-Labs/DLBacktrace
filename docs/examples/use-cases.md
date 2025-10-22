# Use Cases

Real-world applications of DL-Backtrace.

---

## Model Debugging

Identify spurious correlations and biases:

```python
# Analyze multiple inputs
for image, label in test_set:
    relevance = dlb.evaluation(...)
    # Check if model focuses on correct features
```

---

## Feature Importance

Rank input features by importance:

```python
input_relevance = relevance['input']
feature_scores = input_relevance.squeeze()
top_features = feature_scores.argsort(descending=True)
```

---

## Model Comparison

Compare explanations across models:

```python
# Model A explanations
relevance_a = dlb_a.evaluation(...)

# Model B explanations
relevance_b = dlb_b.evaluation(...)

# Compare which features each model uses
```

---

## Regulatory Compliance

Provide evidence for model decisions for audits and compliance.

---

## Research

Understand architecture choices and guide improvements.

---

See [Examples](colab-notebooks.md) for detailed implementations.



