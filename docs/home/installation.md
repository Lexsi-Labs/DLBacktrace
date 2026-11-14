# Installation

This guide covers how to install DLBacktrace and its dependencies.

---

## Requirements

### Python Version
- **Python 3.10 to 3.13** is required

### Framework Requirements

- **PyTorch 2.13.0**, installed with the package (the PyPI build, which uses CUDA 13)
- A GPU is optional

### System Requirements

**For GPU Support:**
- NVIDIA GPU of compute capability 7.5 or newer (Turing to Blackwell) and driver 580 or newer, for the
  bundled CUDA kernels. They are built for the PyPI build of PyTorch 2.13.0; with another PyTorch build
  DL-Backtrace falls back to its PyTorch implementations.

**For CPU-only:**
- No special requirements

---

## Installation Methods

### From Source (Recommended)

This is the recommended method for getting the latest features and updates.

```bash
# Clone the repository
git clone https://github.com/Lexsi-Labs/DLBacktrace.git
cd DLBacktrace

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

!!! tip "Development Mode"
    Installing with `-e` (editable mode) allows you to modify the source code and see changes immediately without reinstalling.

### Dependencies

The main dependencies are automatically installed from `requirements.txt`:

**Core Dependencies:**
```txt
torch==2.13.0
transformers>=4.52.3,<4.53
numpy>=1.21.0
networkx>=2.6
matplotlib>=3.4.0
seaborn>=0.11.0
```

**Optional Dependencies:**
```txt
# For visualization (also needs the Graphviz program `dot`: `apt install graphviz`, `brew install graphviz`)
graphviz>=0.16

# For caching and compression
joblib>=1.0.0
zstandard>=0.15.0
```

---

## Framework-Specific Setup

### PyTorch Setup

For the best experience with PyTorch, install with CUDA support:

=== "CUDA 12.6 (Recommended)"
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```

=== "CPU Only"
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

---

## Hugging Face Setup

For using pre-trained transformer models (BERT, RoBERTa, LLaMA, etc.), you need to set up Hugging Face:

### Install Hugging Face CLI

```bash
pip install huggingface_hub
```

### Login to Hugging Face

Required for accessing gated models like LLaMA:

```bash
huggingface-cli login
```

You'll be prompted to enter your access token. Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## Verification

### Verify Installation

Test your installation with this simple script:

```python
import torch
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Check if PyTorch is installed
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check if DLBacktrace is installed
print("DLBacktrace imported successfully!")

# Simple test
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
x = torch.randn(1, 10)

dlb = DLBacktrace(
    model=model,
    input_for_graph=(x,),
    layer_implementation="pytorch"
)

print("DLBacktrace initialized successfully!")
```

### Run Benchmark Tests

Test with a real model:

```bash
# Test PyTorch backend with a transformer model
python benchmarks/trace_RoBERTa.py

# Test with a simple linear layer
python benchmarks/benchmark_linear.py
```

---

## Troubleshooting

### Common Issues

??? question "ImportError: No module named 'dl_backtrace'"
    Make sure you've installed the package:
    ```bash
    cd DLBacktrace
    pip install -e .
    ```

??? question "CUDA out of memory"
    Try using CPU or reducing batch size:
    ```python
    # Force CPU execution
    import torch
    torch.cuda.is_available = lambda: False
    ```

??? question "Cannot compile CUDA kernels"
    Check that:
    - CUDA Toolkit is installed: `nvcc --version`
    - C++ compiler is available: `g++ --version` (Linux) or `cl` (Windows)
    - CUDA_HOME is set: `echo $CUDA_HOME`

??? question "Hugging Face authentication error"
    Login again with your token:
    ```bash
    huggingface-cli login
    ```

### Getting Help

If you encounter issues:

1. Check the [FAQ](../support/faq.md)
2. Search [GitHub Issues](https://github.com/Lexsi-Labs/DLBacktrace/issues)
3. Create a new issue with details about your setup
4. Email support: [support@lexsi.ai](mailto:support@lexsi.ai)

---

## What's Next?

Now that you have DLBacktrace installed:

- [Quick Start Guide](quickstart.md) - Build your first explainable model
- [User Guide](../guide/introduction.md) - Learn the concepts
- [Examples](../examples/colab-notebooks.md) - Interactive notebooks
- [Developer Guide](../developer/contributing.md) - Contributing and extending

---

## Updating DLBacktrace

To update to the latest version:

```bash
cd DLBacktrace
git pull origin main
pip install -e . --upgrade
```
