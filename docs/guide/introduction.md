# Introduction to DL-Backtrace

Welcome to the DL-Backtrace user guide! This guide will help you understand and effectively use DL-Backtrace for explainable AI and model interpretability.

---

## What is DL-Backtrace?

DL-Backtrace is an **explainable AI (XAI) framework** that helps you understand how deep learning models make decisions. It provides layer-wise relevance propagation and comprehensive model tracing to reveal which parts of your input contribute most to the model's predictions.

### Core Concept: Layer-wise Relevance Propagation

At its heart, DL-Backtrace uses **relevance propagation** - a technique that traces the "importance" of each input feature backward through the network:

```
Input → Layer 1 → Layer 2 → ... → Output
  ↑       ↑         ↑              ↑
  R₀  ←   R₁    ←   R₂    ←  ...  R_final (100%)
```

Starting with the output (100% relevance), we trace backward to see how this relevance distributes across layers and ultimately back to the input features.

---

## Why Use DL-Backtrace?

### 1. **Model Understanding**
Gain deep insights into your model's decision-making process:

- Which input features drive predictions?
- How do different layers contribute to the output?
- What patterns does the model learn?

### 2. **Model Debugging**
Identify issues before deployment:

- Detect spurious correlations
- Find bias in model decisions
- Validate model behavior on edge cases

### 3. **Regulatory Compliance**
Meet explainability requirements:

- Provide evidence for model decisions
- Document decision-making processes
- Satisfy audit requirements

### 4. **Research & Development**
Advance model architectures:

- Understand architecture choices
- Compare different models
- Guide architecture improvements

---

## How DL-Backtrace Works

### Step 1: Graph Tracing

DL-Backtrace first traces your model's computational graph:

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace

dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,)
)
```

This captures:
- Every operation in your model
- Parameter values (weights, biases)
- Tensor shapes and connections
- Execution order

### Step 2: Forward Execution

Run your input through the traced graph:

```python
node_io = dlb.predict(input_tensor)
```

This produces:
- Layer-wise activations
- Intermediate outputs
- Complete execution trace

### Step 3: Relevance Propagation

Calculate how relevance flows backward:

```python
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="multi-class classification"
)
```

This generates:
- Relevance scores for each layer
- Input feature importance
- Decision attribution

### Step 4: Visualization

Visualize the results:

```python
dlb.visualize()  # Full graph
dlb.visualize_dlbacktrace(top_k=15)  # Top contributors
```

---

## Key Concepts

### Relevance

**Relevance** measures how much each neuron/layer contributes to the final prediction. It's a value that:

- Starts at 100% at the output
- Distributes backward through the network
- Conserves total relevance (∑R = 100%)

### Execution Engines

DL-Backtrace provides optimized execution engines:

**ExecutionEngineNoCache**
- In-memory execution
- Fast and memory-efficient
- No disk I/O overhead

Both engines support:
- CPU and GPU execution
- 100+ PyTorch operations
- Deterministic execution

### Task Types

Different tasks require different evaluation approaches:

- **Classification**: Binary or multi-class
- **Regression**: Continuous outputs
- **Segmentation**: Pixel-level predictions
- **Generation**: Autoregressive models

---

## Supported Framework

### PyTorch

Full support for PyTorch 2.6+ models:

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace

dlb = DLBacktrace(
    model=pytorch_model,
    input_for_graph=(dummy_input,),
    device="cuda"
)
```

**Features:**
- Native PyTorch integration
- CUDA acceleration
- Dynamic graph tracing
- Custom operation support
- 100+ ATen operations

---

## Typical Workflow

Here's a typical DL-Backtrace workflow:

### 1. Prepare Your Model

```python
import torch
import torchvision.models as models

# Load your model
model = models.resnet18(pretrained=True)
model.eval()
```

### 2. Initialize DL-Backtrace

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Create dummy input for tracing
dummy_input = torch.randn(1, 3, 224, 224)

# Initialize
dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,),
    device="cuda"
)
```

### 3. Prepare Real Input

```python
from PIL import Image
from torchvision import transforms

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

image = Image.open('cat.jpg')
input_tensor = transform(image).unsqueeze(0)
```

### 4. Run Analysis

```python
# Forward pass
node_io = dlb.predict(input_tensor)

# Relevance propagation
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="multi-class classification"
)
```

### 5. Interpret Results

```python
# Get prediction
output = node_io[list(node_io.keys())[-1]][1]
predicted_class = output.argmax()

print(f"Predicted class: {predicted_class}")
print(f"Number of nodes: {len(relevance)}")

# Find most relevant layers
sorted_relevance = sorted(
    relevance.items(),
    key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
    reverse=True
)

print("\nTop 5 most relevant layers:")
for node_name, rel_score in sorted_relevance[:5]:
    print(f"  {node_name}: {rel_score}")
```

### 6. Visualize

```python
# Save visualizations
dlb.visualize()  # Full graph
dlb.visualize_dlbacktrace(top_k=15)  # Top 15 nodes
```

---

## Advanced Capabilities

DL-Backtrace includes powerful features for advanced use cases:

### High-Level Pipeline Interface

Simplify your workflow with the Pipeline interface:

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline

# One-line setup
pipeline = DLBacktracePipeline.create_simple(
    model_name="bert-base",
    device="cuda"
)

# One-line analysis
results = pipeline.run_simple_analysis(text, label)
```

**[Learn more about Pipeline →](pytorch/pipeline.md)**

### MoE Model Support

Analyze Mixture of Experts models with expert-level tracking:

- **JetMoE**, **OLMoE**, **Qwen MoE**, **GPT-OSS**
- Track which experts contribute most
- Understand expert routing patterns
- CUDA-accelerated MoE layer implementations

**[Learn more about MoE Models →](pytorch/moe-models.md)**

### DLB Auto Sampler

Advanced text generation with explainability:

- Multiple sampling strategies (greedy, temperature, top-k, top-p, beam search)
- Token-level relevance tracking
- HuggingFace-compatible generation API
- Full control over generation parameters

**[Learn more about Auto Sampler →](pytorch/auto-sampler.md)**

### Temperature Scaling

Control generation diversity and model confidence:

- Adjust prediction confidence without retraining
- Fine-tune randomness in text generation
- Improve probability calibration
- Zero-overhead implementation

**[Learn more about Temperature Scaling →](pytorch/temperature-scaling.md)**

---

## What's Next?

Now that you understand the basics, dive deeper into specific topics:

### Learn by Framework

=== "PyTorch Users"
    - [PyTorch Overview](pytorch/overview.md)
    - [DLBacktrace Guide](pytorch/dlbacktrace.md)
    - [Pipeline Interface](pytorch/pipeline.md) - High-level workflows
    - [MoE Models](pytorch/moe-models.md) - Expert-level analysis
    - [Auto Sampler](pytorch/auto-sampler.md) - Advanced generation
    - [Temperature Scaling](pytorch/temperature-scaling.md) - Control diversity
    - [Execution Engines](pytorch/execution-engines.md)

### Learn by Topic

- **[Relevance Propagation](relevance/overview.md)** - Understand the theory

### Learn by Example

- **[Colab Notebooks](../examples/colab-notebooks.md)** - Interactive examples

---

## Best Practices

!!! tip "Start Simple"
    Begin with small models to understand the workflow before moving to large transformers.

!!! tip "Use Evaluation Mode"
    Always set your model to evaluation mode: `model.eval()`

!!! tip "Match Input Shapes"
    Ensure your dummy input shape matches your real input shape.

!!! warning "GPU Memory"
    Large models (LLaMA-3B+) require significant memory. Monitor GPU usage.

!!! warning "Custom Layers"
    Some custom operations may not be supported. Check the operation list.

---

## Getting Help

If you run into issues:

1. Check the [FAQ](../support/faq.md)
2. Search [GitHub Issues](https://github.com/Lexsi-Labs/DLBacktrace/issues)
3. Ask in [GitHub Discussions](https://github.com/Lexsi-Labs/DLBacktrace/discussions)
4. Email [support@lexsi.ai](mailto:support@lexsi.ai)

---

## Contributing

DL-Backtrace is open source! Contributions are welcome:

- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

See the [Contributing Guide](../developer/contributing.md) to get started.

---

**Ready to make your models explainable?**

[Quick Start →](../home/quickstart.md){ .md-button .md-button--primary }




