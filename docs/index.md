##

![DL-Backtrace Logo](assets/images/dlb_logo.png)

[![License](https://img.shields.io/badge/License-LSAL%20v1.0-yellow.svg)](https://github.com/Lexsi-Labs/DLBacktrace/blob/main/LICENSE.md)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-red.svg)](https://pytorch.org)

[Get Started](home/quickstart.md){ .md-button .md-button--primary }

[View on GitHub](https://github.com/Lexsi-Labs/DLBacktrace){ .md-button }

---

## Overview

DLBacktrace is a model-agnostic explainability framework developed by Lexsi Labs. It provides comprehensive layer-wise importance values (relevance) and model tracing capabilities across a wide range of model architectures — including transformers, LLMs, MoEs, and more — as well as diverse task types such as tabular, vision, and text. The framework is designed for robust and efficient execution on both CPU and GPU environments.

---

## Why DL-Backtrace?

### 🔍 **Deep Model Interpretability**
Gain comprehensive insights into your AI models using advanced relevance propagation algorithms. Understand which features and layers contribute most to your model's predictions.

### ⚡ **High Performance**
Optimized execution engine with CUDA acceleration and deterministic tracing. Choose between CPU and GPU execution based on your needs.

### 🏗️ **Architecture Agnostic**
Support for CNN, RNN, Transformer, and custom architectures. Works seamlessly with popular models like ResNet, BERT, LLaMA, and more.

### 🎯 **Multi-Task Support**
Binary/multi-class classification, segmentation, and text generation - all supported out of the box.

### 🛡️ **Production Ready**
Deterministic execution environment with comprehensive error handling. Battle-tested on real-world models and datasets.

---

## Quick Example

```python
import torch
import torch.nn as nn
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

# Initialize model and DL-Backtrace
model = MyModel()
x = torch.randn(1, 3, 224, 224)

dlb = DLBacktrace(
    model=model,
    input_for_graph=(x,),
    device="cuda"
)

# Get layer-wise outputs
node_io = dlb.predict(x)

# Calculate relevance propagation
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="multi-class classification"
)

# Visualize the graph
dlb.visualize()
```

---

## Supported Models

DL-Backtrace has been extensively tested with:

### Vision Models
- ResNet, VGG, DenseNet, EfficientNet, MobileNet
- Vision Transformer (ViT)
- Custom CNNs

### NLP Models
- BERT, ALBERT, RoBERTa, DistilBERT
- ELECTRA, XLNet
- LLaMA-3.2 (1B, 3B), Qwen3
- MoE's like JetMoE, OLMoE, GPT-oss, Qwen MoE

### Tasks
- Binary & Multi-class Classification
- Object Detection
- Semantic Segmentation
- Text Generation
- Sentiment Analysis

---

## Key Capabilities

### Layer-wise Relevance Propagation
It distributes relevance scores across layers, providing insights into feature importance, information flow, and bias, enabling better model interpretation and validation without external dependencies.

### Graph Tracing & Visualization
Automatically trace your model's computational graph and visualize the architecture with relevance scores. Supports both full graph and top-k relevance visualization.

### Deterministic Execution
Ensures reproducible results across runs with automatic environment configuration.

---

## Community & Support

- **Documentation**: You're reading it! 📚
- **Examples**: Check out our [example notebooks](examples/colab-notebooks.md)
- **Issues**: [GitHub Issues](https://github.com/Lexsi-Labs/DLBacktrace/issues)
- **Email**: [support@lexsi.ai](mailto:support@lexsi.ai)

---

## Next Steps

### 🚀 Quick Start
Get up and running in minutes with our quick start guide.

→ [Getting Started](home/quickstart.md)

### 📚 User Guide
Learn about features, APIs, and best practices.

→ [Read the Guide](guide/introduction.md)

### 💻 Examples
Interactive notebooks and real-world use cases.

→ [Explore Examples](examples/colab-notebooks.md)

### 🤝 Developer Guide
Contributing and extending DL-Backtrace.

→ [Developer Docs](developer/contributing.md)

---


DL-Backtrace - Making AI Transparent and Explainable 🚀

Built with ❤️ by [Lexsi Labs](https://lexsi.ai/)

