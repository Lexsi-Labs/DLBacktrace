# Installation

This guide covers how to install DL-Backtrace and its dependencies.

---

## Requirements

### Python Version
- **Python 3.8 or higher** is required

### Framework Requirements

- **PyTorch 2.6+** (recommended for best compatibility)
- CUDA 12.6+ (optional, for GPU acceleration)

### System Requirements

**For GPU Support:**
- NVIDIA GPU with CUDA capability 7.0+
- CUDA Toolkit 12.6 (for PyTorch) or 11.x (for TensorFlow)
- cuDNN compatible with your CUDA version

**For CPU-only:**
- No special requirements

---

## Installation Methods

### From Source (Recommended)

This is the recommended method for getting the latest features and updates.

```bash
# Clone the repository
git clone https://github.com/aryaxai/DL-Backtrace.git
cd DL-Backtrace

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
torch>=2.6.0
transformers>=4.30.0
numpy>=1.21.0
networkx>=2.6
matplotlib>=3.4.0
seaborn>=0.11.0
```

**Optional Dependencies:**
```txt
# For visualization
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

=== "CUDA 11.8"
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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

!!! warning "Gated Models"
    Some models like LLaMA require you to accept their terms of use on Hugging Face before you can download them.

---

## CUDA Layers Compilation (Optional)

For maximum performance, you can compile custom CUDA kernels:

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed and in PATH
- C++ compiler (g++ on Linux, MSVC on Windows)

### Compilation

```bash
cd DL-Backtrace

# Make the script executable
chmod +x compile_cuda_layers.sh

# Compile all CUDA layers
./compile_cuda_layers.sh
```

This will compile custom CUDA kernels for:
- Linear layers
- Conv2D layers
- Embedding layers
- Self-attention layers
- And more...

!!! note "Compilation Time"
    Compiling CUDA kernels can take several minutes. You only need to do this once after installation.

---

## Verification

### Verify Installation

Test your installation with this simple script:

```python
import torch
from dl_backtrace.pytorch_backtrace import DLBacktraceFX

# Check if PyTorch is installed
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check if DL-Backtrace is installed
print("DL-Backtrace imported successfully!")

# Simple test
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
x = torch.randn(1, 10)

dlb = DLBacktraceFX(
    model=model,
    input_for_graph=(x,),
    layer_implementation="pytorch"
)

print("DL-Backtrace initialized successfully!")
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
    cd DL-Backtrace
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
2. Search [GitHub Issues](https://github.com/aryaxai/DL-Backtrace/issues)
3. Create a new issue with details about your setup
4. Email support: [support@aryaxai.com](mailto:support@aryaxai.com)

---

## Docker Installation (Coming Soon)

We're working on official Docker images for easy deployment:

```bash
# Pull the Docker image (coming soon)
docker pull aryaxai/dl-backtrace:latest

# Run with GPU support
docker run --gpus all -it aryaxai/dl-backtrace:latest
```

---

## What's Next?

Now that you have DL-Backtrace installed:

- [Quick Start Guide](quickstart.md) - Build your first explainable model
- [User Guide](../guide/introduction.md) - Learn the concepts
- [Tutorials](../tutorials/vision/resnet.md) - Follow detailed examples
- [API Reference](../api/pytorch/dlbacktracefx.md) - Explore the API

---

## Updating DL-Backtrace

To update to the latest version:

```bash
cd DL-Backtrace
git pull origin main
pip install -e . --upgrade
```

If you've compiled CUDA kernels, you may need to recompile:

```bash
./compile_cuda_layers.sh
```



