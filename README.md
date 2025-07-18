# AryaXai-Backtrace
Backtrace module for Generating Explainability on Deep learning models using TensorFlow / Pytorch

# Backtrace Module
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

The Backtrace Module is a powerful and patent-pending algorithm developed by AryaXAI for enhancing the explainability of AI models, particularly in the context of complex techniques like deep learning.

## Features

- **Explainability:** Gain deep insights into your AI models by using the Backtrace algorithm, providing multiple explanations for their decisions.

- **Consistency:** Ensure consistent and accurate explanations across different scenarios and use cases.

- **Mission-Critical Support:** Tailored for mission-critical AI use cases where transparency is paramount.

## Installation

To integrate the Backtrace Module into your project, follow these simple steps:

```bash
pip install dl-backtrace
```

## Usage 

### Tensoflow-Keras based models

```python
from dl_backtrace.tf_backtrace import Backtrace as B
```

### Pytorch based models

```python
from dl_backtrace.pytorch_backtrace import Backtrace as B
```

### Evalauting using Backtrace:

1. Step - 1: Initialize a Backtrace Object using your Model
```python
backtrace = B(model=model)
```

2. Step - 2: Calculate layer-wise output using a data instance

```python
layer_outputs = backtrace.predict(test_data[0])
```

3. Step - 3: Calculate layer-wise Relevance using Evaluation 
```python
relevance = backtrace.eval(layer_outputs,mode='default',scaler=1,thresholding=0.5,task="binary-classification")
```

#### Depending on Task we have several attributes for Relevance Calculation in Evalaution:

| Attribute    | Description | Values |
|--------------|-------------|--------|
| mode         | evaluation mode of algorithm | { default, contrastive}|
| scaler       | Total / Starting Relevance at the Last Layer | Integer ( Default: None, Preferred: 1)|
| thresholding | Thresholding Model Prediction in Segemntation Task to select Pixels predicting the actual class. (Only works in Segmentation Tasks) |  Default:0.5      |
| task         | The task of the Model | { binary-classification, multi-class classification, bbox-regression, binary-segmentation} |
| model-type   | Type of the Model | {Encoder/ Encoder_Decoder} |

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

## Supported Layers and Future Work :

### Tensorflow-Keras:

- [x] Dense (Fully Connected) Layer
- [x] Convolutional Layer (Conv2D,Conv1D)
- [x] Transpose Convolutional Layer (Conv2DTranspose,Conv1DTranspose)
- [x] Reshape Layer
- [x] Flatten Layer
- [x] Global Max Pooling (2D & 1D) Layer
- [x] Global Average Pooling (2D & 1D) Layer
- [x] Max Pooling (2D & 1D) Layer
- [x] Average Pooling (2D & 1D) Layer
- [x] Concatenate Layer
- [x] Add Layer
- [x] Long Short-Term Memory (LSTM) Layer
- [x] Dropout Layer
- [x] Embedding Layer
- [x] TextVectorization Layer
- [x] Self-Attention Layer
- [x] Cross-Attention Layer
- [x] Feed-Forward Layer
- [x] Pooler Layer
- [x] Decoder LM (Language Model) Head
- [ ] Other Custom Layers 

### Pytorch :

(Note: Currently we only Support Binary and Multi-Class Classification in Pytorch, Segmentation and Single Object Detection will be supported in the next release.)

- [x] Linear (Fully Connected) Layer
- [x] Convolutional Layer (Conv2D)
- [x] Reshape Layer
- [x] Flatten Layer
- [x] Global Average Pooling 2D Layer (AdaptiveAvgPool2d)
- [x] Max Pooling 2D Layer (MaxPool2d)
- [x] Average Pooling 2D Layer (AvgPool2d)
- [x] Concatenate Layer
- [x] Add Layer
- [x] Long Short-Term Memory (LSTM) Layer
- [x] Dropout Layer
- [x] Embedding Layer
- [ ] EmbeddingBag Layer
- [ ] 1d Convolution Layer (Conv1d)
- [x] 1d Pooling Layers (AvgPool1d,MaxPool1d,AdaptiveAvgPool1d,AdaptiveMaxPool1d)
- [ ] Transpose Convolution Layers (ConvTranspose2d,ConvTranspose1d)
- [x] Global Max Pooling 2D Layer (AdaptiveMaxPool2d)
- [ ] Other Custom Layers


## Getting Started
If you are new to Backtrace, head over to our Getting Started Guide to quickly set up and use the module in your projects.

## Contributing
We welcome contributions from the community. To contribute, please follow our Contribution Guidelines.

## License
This software is the confidential and proprietary information of AryaXAI. 
You may not copy, modify, distribute, or disclose any part of this software without express written permission from AryaXAI.

This code is not open-source. It is made available solely as part of a hosted service provided by AryaXAI. All rights reserved.

## Contact
For any inquiries or support, please contact AryaXAI Support.
