# Visualization

Visualize computational graphs and relevance scores with DL-Backtrace.

---

## Graph Visualization

### Full Graph

Visualize the complete computational graph:

```python
dlb.visualize()
```

Output: `dlbacktrace_graph.png` and `dlbacktrace_graph.svg`

### Top-K Relevance

Show only the most relevant nodes:

```python
dlb.visualize_dlbacktrace(top_k=15)
```

Output: `dlbacktrace_topk.png` and `dlbacktrace_topk.svg`

---

## Customization

### Custom Filename

```python
dlb.visualize(filename="my_model_graph")
```

### Custom Format

```python
dlb.visualize(format="svg")  # or "png", "pdf"
```

---

## Input Heatmaps

For image models, create relevance heatmaps:

```python
import matplotlib.pyplot as plt

# Get input relevance
input_relevance = relevance['input']  # Shape: (1, 3, 224, 224)

# Sum across channels
heatmap = input_relevance.sum(1).squeeze()

# Visualize
plt.imshow(heatmap, cmap='hot')
plt.colorbar()
plt.title('Input Relevance Heatmap')
plt.savefig('relevance_heatmap.png')
```

---

## Token Attribution

For text models, visualize token importance:

```python
# Get token relevance
token_relevance = relevance['embedding']

# Sum over hidden dimension
token_scores = token_relevance.sum(dim=-1).squeeze()

# Visualize
import matplotlib.pyplot as plt
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

plt.figure(figsize=(12, 4))
plt.bar(range(len(tokens)), token_scores)
plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.ylabel('Relevance')
plt.title('Token Attribution')
plt.tight_layout()
plt.savefig('token_attribution.png')
```

---

## Installation

Visualization requires graphviz:

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Python package
pip install graphviz
```

---

See [Relevance Overview](relevance/overview.md) for more on interpretation.



