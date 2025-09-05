# DL-Backtrace
A patent-pending explainable AI (XAI) framework for deep learning model interpretability using TensorFlow and PyTorch

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://tensorflow.org)

## Overview

DL-Backtrace is a powerful and patent-pending explainable AI framework developed by AryaXAI for enhancing the interpretability of deep learning models. It provides comprehensive layer-wise relevance propagation and model tracing capabilities across various architectures and tasks.

## Key Features

- **🔍 Deep Model Interpretability:** Gain comprehensive insights into your AI models using advanced relevance propagation algorithms
- **🎯 Multi-Task Support:** Binary/multi-class classification, object detection, segmentation, and text generation
- **🏗️ Architecture Agnostic:** Support for CNN, RNN, Transformer, and custom architectures
- **⚡ High Performance:** Optimized execution engine with CUDA acceleration and deterministic tracing
- **🔧 Robust Operations:** Full support for negative indexing and complex tensor operations
- **📊 Comprehensive Tracing:** Layer-wise activation and relevance analysis with detailed execution tracking
- **🛡️ Production Ready:** Deterministic execution environment with comprehensive error handling

## Installation

Install DL-Backtrace using pip:

```bash
pip install dl-backtrace
```

## Quick Start

### PyTorch Models (Recommended)

```python
import torch
import torch.nn as nn
from dl_backtrace.pytorch_backtrace import DLBacktraceFX

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
dlb = DLBacktraceFX(
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

### TensorFlow-Keras Models

```python
from dl_backtrace.tf_backtrace import Backtrace as B

# Initialize with your Keras model
backtrace = B(model=keras_model)

# Get layer outputs
layer_outputs = backtrace.predict(test_data[0])

# Calculate relevance
relevance = backtrace.eval(
    layer_outputs,
    mode='default',
    scaler=1,
    thresholding=0.5,
    task="binary-classification"
)
```

## Advanced Features

### Deterministic Execution Environment
DL-Backtrace automatically sets up a deterministic environment for consistent results:
- ✅ CUDA memory management and synchronization
- ✅ Deterministic algorithms and cuDNN settings
- ✅ Random seed control and environment variables
- ✅ Warning suppression for cleaner output

### Robust Tensor Operations
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
| `task`       | Model task type | `binary-classification`, `multi-class classification`, `bbox-regression`, `binary-segmentation` |
| `model-type` | Model architecture type | `Encoder`, `Encoder_Decoder` |

## Example Notebooks : 

### Tensorflow-Keras : 

| Name        | Task        | Link                          |
|-------------|-------------|-------------------------------|
| Backtrace Loan Classification Tabular Dataset | Binary Classification | [Colab Link](https://colab.research.google.com/drive/1H5jaryVPEAQuqk9XPP71UIL4cemli98K?usp=sharing) |
| Backtrace Image FMNIST Dataset | Multi-Class Classification | [Colab Link](https://colab.research.google.com/drive/1BZsdo7IWYGhdy0Pg_m8r7c3COczuW_tG?usp=sharing)  |
| Backtrace CUB Bounding Box Regression Image Dataset | Single Object Detection | [Colab Link](https://colab.research.google.com/drive/15mmJ2aGt-_Ho7RdPWjNEEoFXE9mu9HLV?usp=sharing) |
| Backtrace Next Word Generation Textual Dataset | Next Word Generation | [Colab Link](https://colab.research.google.com/drive/14R3DuDLjvgowA2ucsoccpyN7Lp-ZOAz4?usp=sharing) |
| Backtrace ImDB Sentiment Classification Textual Dataset | Sentiment Classification | [Colab Link](https://colab.research.google.com/drive/1Kgthc7rbaNsSqLuH7RPm_vRIPB98uoCW?usp=sharing)|
| Backtrace Binary Classification Textual Dataset | Binary Classification | [Colab Link](https://colab.research.google.com/drive/1C1M2uNXi1WjpC1N74wl3bbQOIFNm57No?usp=sharing) |
| Backtrace Multi-Class NewsGroup20 Classification Textual Dataset | Multi-Class Classification | [Colab Link](https://colab.research.google.com/drive/1xqBuix5qk0mDSxMScubO4ENeMb8F9IgE?usp=sharing) |
| Backtrace CVC-ClinicDB Colonoscopy Binary Segmentation | Organ Segmentation | [Colab Link](https://colab.research.google.com/drive/1cUNUao7fahDgndVI-cpn2iSByTiWaB4j?usp=sharing) | 
| Backtrace CamVid Road Car Binary Segmentation | Binary Segmentation | [Colab Link](https://colab.research.google.com/drive/1OAY7aAraKq_ucyVt5AYPBD8LkQOIuy1C?usp=sharing) |
| Backtrace Transformer Encoder for Sentiment Analysis | Binary Classification | [Colab Link](https://colab.research.google.com/drive/1H7-4ox3YWMtoH0vptYGXaN63PRJFbTrX?usp=sharing) |
| Backtrace Transformer Encoder-Decoder Model for Neural Machine Translation | Neural Machine Translation | [Colab Link](https://colab.research.google.com/drive/1NApbrd11TEqlrqGCBYPmgMvBbZBJhpWD?usp=sharing) |
| Backtrace Transformer Encoder-Decoder Model for Text Summarization | Text Summarization | [Colab Link](https://colab.research.google.com/drive/18CPNnEJzGlCPJ2sSXX4mArAzK1NLe9Lj?usp=sharing) |

### Pytorch :  
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

### TensorFlow-Keras

- [x] **Dense (Fully Connected) Layer**
- [x] **Convolutional Layers** (Conv2D, Conv1D)
- [x] **Transpose Convolutional Layers** (Conv2DTranspose, Conv1DTranspose)
- [x] **Reshape & Flatten Layers**
- [x] **Pooling Layers** (Global Max/Average, Max/Average Pooling 2D & 1D)
- [x] **Concatenate & Add Layers**
- [x] **LSTM Layer**
- [x] **Dropout Layer**
- [x] **Embedding Layer**
- [x] **TextVectorization Layer**
- [x] **Attention Layers** (Self-Attention, Cross-Attention)
- [x] **Feed-Forward & Pooler Layers**
- [x] **Decoder LM Head**
- [ ] Other Custom Layers

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

**Planned Support:**
- [ ] EmbeddingBag Layer
- [ ] 1D Convolution Layer (Conv1d)
- [ ] Transpose Convolution Layers (ConvTranspose2d, ConvTranspose1d)
- [ ] Custom Layer Support


## Performance & Reliability

### Recent Improvements
- **🔧 Enhanced Execution Engine:** Robust handling of complex tensor operations with comprehensive error handling
- **⚡ Deterministic Environment:** Automatic setup for consistent, reproducible results across runs
- **🛡️ Error Resilience:** Comprehensive validation and graceful error handling for production use
- **📊 Better Debugging:** Detailed logging and execution tracking for troubleshooting

### Benchmarks
DL-Backtrace has been tested on various model architectures:
- **Vision Models:** ResNet, VGG, DenseNet, EfficientNet, MobileNet, ViT
- **NLP Models:** BERT, ALBERT, RoBERTa, DistilBERT, ELECTRA, XLNet, LLaMA
- **Tasks:** Classification, Object Detection, Segmentation, Text Generation

## Getting Started

If you're new to DL-Backtrace, check out our comprehensive example notebooks above. For detailed documentation and advanced usage, visit our documentation portal.

## Contributing

We welcome contributions from the community! Please follow our contribution guidelines and submit pull requests for any improvements.

## License

This software is the confidential and proprietary information of AryaXAI. 
You may not copy, modify, distribute, or disclose any part of this software without express written permission from AryaXAI.

This code is not open-source. It is made available solely as part of a hosted service provided by AryaXAI. All rights reserved.

## Contact

For any inquiries, support, or collaboration opportunities, please contact [AryaXAI Support](mailto:support@aryaxai.com).

---

**DL-Backtrace** - Making AI Transparent and Explainable 🚀
