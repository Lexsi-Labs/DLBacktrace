<p align="center">
  <img src="assets/images/dlb_logo.png" alt="DLB Logo" width="300px"/>
</p>

<h2 align="center">DLBacktrace: Model-Agnostic Explainability for Any Deep Learning Model</h2>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.6%2B-red.svg"></a>
  <a href="https://dlbacktrace.lexsi.ai/"><img src="https://img.shields.io/badge/docs-available-green.svg"></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-Lexsi%20Labs%20Source%20Available-orange"></a>
</p>

<p align="center">
  <i>Making AI Transparent and Explainable — Developed by <a href="https://lexsi.ai/">Lexsi Labs</a></i> 🚀
</p>

---

## Overview

DLBacktrace is a model-agnostic explainability framework developed by *Lexsi Labs*. It provides comprehensive layer-wise importance values (relevance) and model tracing capabilities across a wide range of model architectures — including transformers, LLMs, MoEs, and more — as well as diverse task types such as tabular, vision, and text. The framework is designed for robust and efficient execution on both CPU and GPU environments.

## Key Features

### Core Capabilities
- **🔍 Deep Model Interpretability:** Gain comprehensive insights into your AI models using advanced relevance propagation algorithms
- **🎯 Multi-Task Support:** Binary/multi-class classification, object detection, segmentation, and text generation
- **🏗️ Architecture Agnostic:** Support for CNN, RNN, Transformer, and custom architectures including Mixture of Experts (MoE)
- **⚡ High Performance:** Optimized execution engine with CUDA acceleration and deterministic tracing
- **🔧 Robust Operations:** Full support for negative indexing and complex tensor operations
- **📊 Comprehensive Tracing:** Layer-wise activation and relevance analysis with detailed execution tracking
- **🛡️ Production Ready:** Deterministic execution environment with comprehensive error handling

### Advanced Features
- **🚀 High-Level Pipeline Interface:** Simplified API for text/image classification and generation with automatic model loading and configuration
- **🎲 DLB Auto Sampler:** Advanced text generation with multiple sampling strategies (greedy, temperature, top-k, top-p, beam search) and token-level relevance tracking
- **🧠 Mixture of Experts (MoE) Support:** Built-in support for MoE architectures (JetMoE, OLMoE, Qwen3-MoE, GPT-OSS) with expert-level relevance analysis
- **🌡️ Temperature Scaling:** Control generation diversity and model confidence with flexible temperature parameters
- **🔄 Enhanced Execution Engine:** Critical fixes for RoBERTa, LLaMA, and other transformer models

---

## ⚡ Performance Improvements in DLB v2

DLB v2 introduces **major architectural upgrades** to the explainability engine — resulting in *orders of magnitude faster performance* compared to v1.

> 📘 **Note:** All benchmarks below were conducted on the **LLaMA 3.2 – 3 B parameter model** using the MMLU dataset on a NVIDIA RTX 4090.

---

### 📊 Benchmark Summary

| Metric | DLB v1 | DLB v2 | Improvement |
|:-------|:-------|:-------|:-------------|
| ⏱️ **Explainability Time** | Up to 29,000 s | 🕒 20 s | 🔥 **1000× faster** |
| 🚀 **Throughput** | ~0.03 tokens/s | 🧩 75 tokens/s | ⚡ **2000× higher** |
| 📈 **Scalability** | Degrades sharply | Scales linearly | ✅ **Stable & predictable** |

---

### 📈 Performance Graphs

| Metric | Comparison Plot |
|:--------|:----------------|
| 🕒 **Total Time vs Sequence Length** | <img src="assets/images/total_dlb_time_comparison.png" width="500"/> |
| 🚀 **Token Throughput** | <img src="assets/images/throughput_comparison.png" width="500"/> |
| ⚙️ **Speedup (v2 / v1)** | <img src="assets/images/speedup_comparison.png" width="500"/> |

---

> 💡 **DLB v2** achieves up to **1000× speedup** using fused GPU kernels.

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/Lexsi-Labs/DLBacktrace.git
cd DLBacktrace
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 2.6+ (with CUDA 12.6 support recommended)
- Additional dependencies: transformers, matplotlib, seaborn, graphviz, joblib, zstandard

See `requirements.txt` for the complete list of dependencies.

### Hugging Face Setup

For accessing models from Hugging Face Hub (required for BERT, RoBERTa, LLaMA, etc.):

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face (required for gated models)
huggingface-cli login
```

You'll need a Hugging Face account and access token. Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Quick Start

### PyTorch Models (Recommended)

```python
import torch
import torch.nn as nn
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

# Initialize model and DL-Backtrace
model = MyModel()
x = torch.randn(1, 10)  # Example input

# Create DL-Backtrace instance
dlb = DLBacktrace(
    model=model,
    input_for_graph=(x,),
    layer_implementation="pytorch"
)

# Get layer-wise outputs
node_io = dlb.predict(x)

# Calculate relevance propagation
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="binary-classification"
)
```

## Advanced Features

### 🚀 High-Level Pipeline Interface
Simplified API for common ML tasks with automatic model loading and configuration:

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace import DLBPipeline

# Text classification with relevance
pipeline = DLBPipeline(task="text-classification", model_name="bert-base-uncased")
result = pipeline("Sample text", return_relevance=True)

# Text generation with Auto Sampler
gen_pipeline = DLBPipeline(task="text-generation", model_name="gpt2")
output = gen_pipeline("Once upon a time", max_length=50, sampling_strategy="top_p")
```

### 🎲 DLB Auto Sampler
Advanced text generation with multiple sampling strategies and token-level relevance tracking:

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.core.dlb_auto_sampler import DLBAutoSampler

sampler = DLBAutoSampler(model, tokenizer)

# Greedy sampling
output = sampler.generate("Prompt", strategy="greedy", max_length=50)

# Top-k and Top-p sampling
output = sampler.generate("Prompt", strategy="top_k", top_k=50, temperature=0.8)
output = sampler.generate("Prompt", strategy="top_p", top_p=0.9, temperature=0.8)

# Beam search
output = sampler.generate("Prompt", strategy="beam_search", num_beams=5)

# Access token-level relevance
print(output['relevance_scores'])
```

### 🧠 Mixture of Experts (MoE) Support
Built-in support for MoE architectures with expert-level relevance analysis:

```python
from dl_backtrace.moe_pytorch_backtrace.backtrace import Backtrace

# Supported MoE models: JetMoE, OLMoE, Qwen3-MoE, GPT-OSS
backtrace = Backtrace(
    model=moe_model,
    model_type="jetmoe",  # or "olmoe", "qwen", "gpt_oss"
    input_text="Sample input",
    tokenizer=tokenizer
)

# Get expert-level relevance
expert_relevance = backtrace.all_layer_expert_relevance()
```

### 🌡️ Temperature Scaling
Control generation diversity and model confidence:

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace import DLBacktrace

dlb = DLBacktrace(model, input_for_graph=(input_tensor,))

# Apply temperature scaling for generation
output = dlb.generate_with_temperature(
    input_ids,
    temperature=0.7,  # Lower = more focused, Higher = more diverse
    max_length=100
)
```

### ⚡ Execution Engines
DL-Backtrace provides optimized execution engines:

#### ExecutionEngineNoCache 
- **Memory-efficient**: Runs entirely in RAM for faster execution
- **Enhanced Operations**: Supports 100+ PyTorch operations with robust error handling
- **Recent Improvements**: Critical fixes for transformer models (RoBERTa, LLaMA, BERT)

### 🛡️ Deterministic Execution Environment
DL-Backtrace automatically sets up a deterministic environment for consistent results:
- ✅ CUDA memory management and synchronization
- ✅ Deterministic algorithms and cuDNN settings
- ✅ Random seed control and environment variables
- ✅ Warning suppression for cleaner output

### 🔧 Robust Tensor Operations
Full support for PyTorch's negative indexing and complex operations:
- ✅ `transpose(-1, -2)`, `permute([-1, -2, 0])`
- ✅ `unsqueeze(-1)`, `squeeze(-1)`
- ✅ `slice(dim=-1, ...)`, `cat(tensors, dim=-1)`
- ✅ `index_select(dim=-1, ...)`

### Evaluation Parameters

| Parameter    | Description | Values |
|--------------|-------------|--------|
| `mode`       | Evaluation algorithm mode | `default`, `contrastive` |
| `multiplier` | Starting relevance at output layer | Float (default: 100.0) |
| `scaler`     | Relevance scaling factor | Float (default: 1.0) |
| `thresholding` | Pixel selection threshold for segmentation | Float (default: 0.5) |
| `task`       | Model task type | `binary-classification`, `multi-class classification` |

## Example Notebooks

| Name        | Task        | Link                          |
|-------------|-------------|-------------------------------|
| Custom Tabular Model | Binary Classification | [Colab Link](https://colab.research.google.com/drive/1TqgeeBqQ1G9UGWfHV0MUloCccalpsRCh?usp=sharing)|
| VGG Model | Multi-Class Classification | [Colab Link](https://colab.research.google.com/drive/1iJJZ0ApWHltTjnbGRKhJDrHlKTlm1koD?usp=sharing) |
| ResNet Model | Multi-Class Classification | [Colab Link](https://colab.research.google.com/drive/1mpo--AD8vNqm6Y05rb46Yzjx6VhLwXZh?usp=sharing) |
| ViT Model | Multi-Class Classification | [Colab Link](https://colab.research.google.com/drive/1BhzIw7Pf9-g1tqndaijwZ5FLaDUpjBaR?usp=sharing) |
| DenseNet Model | Multi-Class Classification | [Colab Link](https://colab.research.google.com/drive/1CE2XBBGd5VSQuipJTyyRcb7mu5RSG6K5?usp=sharing) |
| EfficientNet Model | Multi-Class Classification | [Colab Link](https://colab.research.google.com/drive/1O-MyvIKWoADG2RrF43p2k8mUYpF9N_8m?usp=sharing) |
| MobileNet Model | Multi-Class Classification | [Colab Link](https://colab.research.google.com/drive/1BzsID9U3HndLrh67nPWRw_bm7UWLwLOH?usp=sharing) |
| BERT-Base Model | Sentiment Classification | [Colab Link](https://colab.research.google.com/drive/1ANZPjaAxl2oF2WHj23f87AR9-ZDIMBm9?usp=sharing) |
| ALBERT Model | Sentiment Classification | [Colab Link](https://colab.research.google.com/drive/1RuAW0FgtWqKkdVbc97VXf9z4oDXmA1ms?usp=sharing) |
| RoBERTa Model | Sentiment Classification | [Colab Link](https://colab.research.google.com/drive/1Nw6lTSQKJvGU9JBZeXnA7EXboU7mE282?usp=sharing) |
| DistilBERT Model | Sentiment Classification | [Colab Link](https://colab.research.google.com/drive/13_hqUC2vaJWfF-UheggHJU5RmWS5A2u3?usp=sharing) |
| Electra Model | Sentiment Classification | [Colab Link](https://colab.research.google.com/drive/1sht3uLej8g-4hMtaHm7VwwUuGAmAqpH_?usp=sharing) |
| XLNeT Model | Sentiment Classification | [Colab Link](https://colab.research.google.com/drive/1ZmVusCPgeXLGnbt7NzM-3SJuiGRgTzBa?usp=sharing) |
| LLaMA-3.2-1B Model | Text Generation | [Colab Link](https://colab.research.google.com/drive/1i_CKoCfKdY4fcWyFdzuc_0e868jux12h?usp=sharing) |
| LLaMA-3.2-3B Model | Text Generation | [Colab Link](https://colab.research.google.com/drive/1ki8kcc4ez8-kdvdlhtoq7Sed9v5hiaNs?usp=sharing) |

For more detailed examples and use cases, check out our documentation.

## Supported Layers

### PyTorch

**Core Operations:**
- [x] **Linear (Fully Connected) Layer**
- [x] **Convolutional Layer** (Conv2D)
- [x] **Reshape & Flatten Layers**
- [x] **Pooling Layers** (AdaptiveAvgPool2d, MaxPool2d, AvgPool2d, AdaptiveMaxPool2d)
- [x] **1D Pooling Layers** (AvgPool1d, MaxPool1d, AdaptiveAvgPool1d, AdaptiveMaxPool1d)
- [x] **Concatenate & Add Layers**
- [x] **LSTM Layer**
- [x] **Dropout Layer**
- [x] **Embedding Layer**

**Advanced Operations:**
- [x] **Tensor Manipulation** (transpose, permute, unsqueeze, squeeze, slice, cat, index_select)
- [x] **Negative Indexing Support** (all operations support PyTorch's negative indexing)
- [x] **Layer Normalization**
- [x] **Batch Normalization**
- [x] **View & Reshape Operations**

## Testing & Validation

### Supported Models
DL-Backtrace has been extensively tested with:
- **Vision Models**: ResNet, VGG, DenseNet, EfficientNet, MobileNet, ViT
- **NLP Models**: BERT, ALBERT, RoBERTa, DistilBERT, ELECTRA, XLNet, LLaMA-3.2
- **MoE Models**: JetMoE, OLMoE (Open Language Model with Experts), Qwen3-MoE, GPT-OSS
- **Tasks**: Classification, Object Detection, Segmentation, Text Generation, Expert-Level Analysis

## Getting Started

If you're new to DL-Backtrace:

1. **📖 Read the Documentation**: [https://dlbacktrace.lexsi.ai/](https://dlbacktrace.lexsi.ai/)
2. **🚀 Try the Quick Start**: See examples above for PyTorch models
3. **💻 Explore Notebooks**: Check out our comprehensive example notebooks for various use cases
4. **🧪 Run Tests**: Validate your installation with the benchmark scripts

For advanced features like the Pipeline Interface, Auto Sampler, MoE models, and Temperature Scaling, refer to the full documentation.

## Contributing

We welcome contributions from the community! Please follow our contribution guidelines and submit pull requests for any improvements.

## License

This project is licensed under a custom License - see the [LICENSE](LICENSE.md) file for details.

## Recent Updates & New Features

### Latest Release (2025)

**New Features:**
- 🚀 **High-Level Pipeline Interface**: Simplified API for text/image classification and generation
- 🎲 **DLB Auto Sampler**: Advanced text generation with multiple sampling strategies
- 🧠 **MoE Model Support**: Built-in support for Mixture of Experts architectures (JetMoE, OLMoE, Qwen3-MoE, GPT-OSS)
- 🌡️ **Temperature Scaling**: Flexible control over generation diversity and model confidence

**Critical Fixes & Improvements:**
- 🔧 Enhanced execution engine with robust handling of complex tensor operations
- ⚡ Deterministic environment setup for consistent, reproducible results
- 🛡️ Comprehensive error handling for production use
- 🚨 Critical fixes for transformer models (RoBERTa, LLaMA, BERT)
- 🧠 Smart attention detection for bidirectional vs causal attention
- 💾 Memory optimization and improved OOM error handling

## Contact

For any inquiries, support, or collaboration opportunities:

- **Email**: [support@lexsi.ai](mailto:support@lexsi.ai)
- **Website**: [https://lexsi.ai/](https://lexsi.ai/)
- **GitHub Issues**: [https://github.com/Lexsi-Labs/DLBacktrace/issues](https://github.com/Lexsi-Labs/DLBacktrace/issues)
- **Documentation**: [https://dlbacktrace.lexsi.ai/](https://dlbacktrace.lexsi.ai/)

## Citations
```
@misc{sankarapu2024dlbacktracemodelagnosticexplainability,
      title={DLBacktrace: A Model Agnostic Explainability for any Deep Learning Models}, 
      author={Vinay Kumar Sankarapu and Chintan Chitroda and Yashwardhan Rathore and Neeraj Kumar Singh and Pratinav Seth},
      year={2024},
      eprint={2411.12643},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.12643}, 
}
```
---

<p align="center"> <b>DLBacktrace</b> — Bridging Performance and Explainability 🔍<br> <a href="https://lexsi.ai/">Lexsi Labs</a> | <a href="mailto:support@lexsi.ai">support@lexsi.ai</a> </p> 
