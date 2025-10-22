# CUDA Development

Guide for developing CUDA kernels for DL-Backtrace.

---

## Overview

Custom CUDA kernels can significantly speed up relevance calculations.

---

## Structure

```
layers/MyLayer/cuda_version/
├── kernel.cu                 # CUDA kernel
├── ops.cpp                   # C++ bindings
├── setup.py                  # Build script
└── include/
    └── utils.h               # Helper functions
```

---

## Example Kernel

```cuda
// kernel.cu
__global__ void my_kernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}
```

---

## Python Bindings

```cpp
// ops.cpp
#include <torch/extension.h>

torch::Tensor my_operation(torch::Tensor input) {
    // Launch kernel
    // Return result
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_operation", &my_operation);
}
```

---

## Building

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_cuda_ops',
    ext_modules=[
        CUDAExtension('my_cuda_ops', [
            'kernel.cu',
            'ops.cpp',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

---

## Compilation

```bash
python setup.py install
```

---

See [compile_cuda_layers.sh](../../compile_cuda_layers.sh) for automated compilation.



