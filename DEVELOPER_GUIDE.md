# Developer Guide: DL-Backtrace

Welcome to the DL-Backtrace project! This guide is intended for new developers who want to contribute to the codebase. It covers the project's architecture, setup instructions, and contribution guidelines.

## 1. Project Overview

DL-Backtrace is a framework designed to trace and analyze computations and relevance within deep learning models built with PyTorch. The core idea is to provide different implementations for common neural network layers, including:

-   **Original**: A baseline implementation, typically in NumPy.
-   **Refactored**: An optimized or clearer version of the original.
-   **PyTorch**: An implementation using native PyTorch operations.
-   **CUDA**: A high-performance version written in CUDA C++ for NVIDIA GPUs.

The framework allows for benchmarking these different versions to compare performance and correctness, which is especially useful for developing and optimizing custom layer implementations.

## 2. Repository Structure

The project is organized to separate different concerns, from the core backtracing logic to layer implementations and benchmarks.

```
DL-Backtrace/
├── dl_backtrace/
│   └── pytorch_backtrace/
│       └── backtrace/
│           ├── refactored_utils/
│           │   ├── layers/         # All layer implementations reside here
│           │   │   ├── Linear/
│           │   │   ├── Conv2D/
│           │   │   └── ...
│           │   ├── benchmarks/     # Scripts to benchmark layer performance
│           │   └── prop.py         # Launch functions for refactored layers
│           └── utils/              # Original utility functions
|           └── backtrace.py        # Main Backtrace code
├── tests/                          # Test scripts
├── compile_cuda_layers.sh          # Script to compile all custom CUDA kernels
└── DEVELOPER_GUIDE.md              # This guide
```

-   `dl_backtrace/pytorch_backtrace/backtrace/refactored_utils/layers/`: This is the most critical directory for layer development. Each subdirectory corresponds to a neural network layer (e.g., `Linear`, `Conv2D`). Inside each layer's directory, you will find the different implementations (`original_version.py`, `pytorch_version.py`, `cuda_version/`, etc.).

-   `dl_backtrace/pytorch_backtrace/backtrace/refactored_utils/benchmarks/`: Contains scripts for performance testing of the various layer implementations.

-   `dl_backtrace/pytorch_backtrace/backtrace/refactored_utils/prop.py`: This file acts as a dispatcher. It contains `launch_*` functions that select the correct layer implementation (`original`, `pytorch`, `cuda`, etc.) at runtime.

-   `compile_cuda_layers.sh`: A utility script to automate the compilation of all CUDA kernels.

## 3. Setup and Installation

To get started with development, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd DL-Backtrace
    ```

2.  **Set up a Python environment:**
    It is recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Compile CUDA Layers:**
    If you have a compatible NVIDIA GPU and the CUDA toolkit installed, you can compile the custom CUDA kernels.
    ```bash
    chmod +x compile_cuda_layers.sh
    ./compile_cuda_layers.sh
    ```
    This script will iterate through all `setup.py` files in the `layers` directory and run `python setup.py develop` for each.

## 4. Running Benchmarks

To evaluate the performance of the layer implementations, you can run the scripts located in the `dl_backtrace/pytorch_backtrace/backtrace/refactored_utils/benchmarks/` directory.

For example, to benchmark the `Linear` layer:
```bash
python dl_backtrace/pytorch_backtrace/backtrace/refactored_utils/benchmarks/benchmark_linear.py
```
Each benchmark script will typically compare the execution time and output correctness of the `original`, `refactored`, `pytorch`, and `cuda` versions.

## 5. Adding a New Refactored Layer

To contribute a new layer, follow this structured approach:

1.  **Create the Layer Directory:**
    Add a new directory for your layer under `dl_backtrace/pytorch_backtrace/backtrace/refactored_utils/layers/`. For example, for a new `Dropout` layer:
    ```
    mkdir dl_backtrace/pytorch_backtrace/backtrace/refactored_utils/layers/Dropout
    ```

2.  **Add Implementations:**
    Inside your new layer directory, add files for each implementation you are providing. For example:
    -   `original_version.py`
    -   `pytorch_version.py`
    -   For a CUDA implementation, create a `cuda_version/` subdirectory containing the `.cu`, `.cpp`, and `setup.py` files required for compilation.

3.  **Create a Benchmark:**
    Add a new benchmark script (e.g., `benchmark_dropout.py`) in the `benchmarks` directory to test your new layer. Follow the existing benchmark scripts for structure.

4.  **Add a Launch Function:**
    Finally, add a new `launch_dropout()` function to `dl_backtrace/pytorch_backtrace/backtrace/refactored_utils/prop.py`. This function will import your new layer's implementations and select one based on the `version` argument. This allows the rest of the codebase to easily switch between different versions of your layer.

## 6. Using Layers Across Branches

The refactored layers are designed to be modular and "plug-and-play." This means you can integrate them into other branches of the DL-Backtrace project, such as `llama_3.2` or `llama2_export`, to leverage the performance improvements.

To do this:
1.  Identify the corresponding layer implementation in the target branch that you wish to replace.
2.  Replace the existing layer logic with a call to the appropriate `launch_*` function from `refactored_utils/prop.py`.
3.  You can refer to `backtrace.py` for examples of how the different layer versions (especially CUDA) are invoked and integrated into the main workflow.
