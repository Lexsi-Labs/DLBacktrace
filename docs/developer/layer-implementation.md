# Layer Implementation Guide

Guide for implementing custom layer relevance rules.

---

## Overview

DL-Backtrace allows custom relevance propagation rules for layers.

---

## Layer Structure

Each layer can have multiple implementations:

```
layers/MyLayer/
├── original_version.py      # NumPy baseline
├── pytorch_version.py        # PyTorch implementation
├── refactored_version.py     # Optimized version
└── cuda_version/             # CUDA implementation (optional)
    ├── kernel.cu
    ├── ops.cpp
    └── setup.py
```

---

## Implementation Template

```python
# pytorch_version.py
import torch

def launch_mylayer(
    inputs,
    weights,
    layer_params,
    version="pytorch"
):
    """
    Calculate relevance for MyLayer.
    
    Args:
        inputs: Input tensors
        weights: Layer weights
        layer_params: Layer hyperparameters
        version: Implementation version
    
    Returns:
        Relevance tensor
    """
    # Your relevance calculation here
    relevance = ...
    
    return relevance
```

---

## Adding to Framework

1. Create layer directory
2. Implement versions
3. Add launcher in `utils/default_v2.py`
4. Test with examples
5. Document in operations list

---

See [Developer Guide](DEVELOPER_GUIDE.md) from dev_notes for complete details.



