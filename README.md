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

usage for Tensoflow based models

```python
from dl_backtrace.tf_backtrace import Backtrace as B
from dl_backtrace.tf_backtrace import contrast as UC
from dl_backtrace.tf_backtrace import prop as UP
from dl_backtrace.tf_backtrace import activation_master
```

usage for Pytorch based models

```python
from dl_backtrace.pytorch_backtrace import Backtrace as B
from dl_backtrace.pytorch_backtrace import contrast as UC
from dl_backtrace.pytorch_backtrace import prop as UP
from dl_backtrace.pytorch_backtrace import activation_master
```

## Example Notebooks

| Name        | Link                          |
|-------------|-------------------------------|
| Tensorflow Backtrace Tabular Dataset | [Colab Link](https://colab.research.google.com/drive/1A4J-wgShD7M_pUmsqbnI8BD3hE43dT8o?usp=sharing)  |
| Tensorflow Backtrace Textual Dataset | [Colab Link](https://colab.research.google.com/drive/1zT_K8mHdzyfQe_LG576qwiBqw8o6LRQH?usp=sharing)  |
| Tensorflow Backtrace Image Dataset | [Colab Link](https://colab.research.google.com/drive/1KbLtcjYDrPQvG6oJj1wmHdiWxRrtKNrV?usp=sharing)  |
| Pytorch Backtrace Tabular Dataset | [Colab Link](https://colab.research.google.com/drive/1Z4UJNFd83dwXBMM0cmiNYEjh6xhRtQA_?usp=sharing) |
| Pytorch Backtrace Image Dataset | [Colab Link](https://colab.research.google.com/drive/14XKwCsS9IZep2AlDDYfavnVRNz8_b-jM?usp=sharing) |


For more detailed examples and use cases, check out our documentation.

## Supported Layers and Future Work
- [x] Dense (Fully Connected) Layer
- [x] Convolutional Layer (Conv2D)
- [x] Reshape Layer
- [x] Flatten Layer
- [x] Global Average Pooling 2D Layer
- [x] Max Pooling 2D Layer
- [x] Average Pooling 2D Layer
- [x] Concatenate Layer
- [x] Add Layer
- [x] Long Short-Term Memory (LSTM) Layer
- [x] Batch Normalisation Layer
- [x] Dropout Layer
- [ ] Embedding Layer
- [ ] Other Custom Layers


## Getting Started
If you are new to Backtrace, head over to our Getting Started Guide to quickly set up and use the module in your projects.

## Contributing
We welcome contributions from the community. To contribute, please follow our Contribution Guidelines.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any inquiries or support, please contact AryaXAI Support.
