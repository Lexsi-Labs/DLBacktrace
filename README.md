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
| Backtrace Tabular Dataset | Binary Classification | [Colab Link](https://colab.research.google.com/drive/1_r-IS7aIuATSvGNRLk8VDVVLkDSaKCpD?usp=sharing)|
| Backtrace Image Dataset | Multi-Class Classification | [Colab Link](https://colab.research.google.com/drive/1v2XajWtIbf7Vt31Z1fnKnAjyiDzPxwnU?usp=sharing) |

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
- [ ] Dropout Layer
- [ ] Embedding Layer
- [ ] EmbeddingBag Layer
- [ ] 1d Convolution Layer (Conv1d)
- [ ] 1d Pooling Layers (AvgPool1d,MaxPool1d,AdaptiveAvgPool1d,AdaptiveMaxPool1d)
- [ ] Transpose Convolution Layers (ConvTranspose2d,ConvTranspose1d)
- [ ] Global Max Pooling 2D Layer (AdaptiveMaxPool2d)
- [ ] Other Custom Layers


## Getting Started
If you are new to Backtrace, head over to our Getting Started Guide to quickly set up and use the module in your projects.

## Contributing
We welcome contributions from the community. To contribute, please follow our Contribution Guidelines.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any inquiries or support, please contact AryaXAI Support.
