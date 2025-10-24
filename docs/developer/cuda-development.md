# CUDA Development

Guide for developing CUDA kernels for DL-Backtrace.

---

## Overview

DL-Backtrace uses **inline CUDA compilation** via PyTorch's `load_inline()` for custom kernels that significantly accelerate relevance calculations. This approach embeds CUDA code as Python strings and compiles at runtime, eliminating separate build steps.

---

## Architecture

### File Structure

```
dl_backtrace/pytorch_backtrace/dlbacktrace/utils/cuda_utils/
└── Linear_v3/
    └── cuda_v3.py              # Complete inline CUDA implementation
```

Each CUDA utility is self-contained in a single Python file with:
- CUDA kernel source as a raw string
- C++ function declarations
- Python wrapper function
- Dynamic compilation configuration

---

## Inline CUDA Implementation

### 1. Define CUDA Kernel Source

Embed the complete CUDA kernel code as a Python raw string:

```python
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Device helper functions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Main kernel
__global__ void my_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}

// Launcher function
torch::Tensor launch_my_kernel(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    
    auto output = torch::zeros_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    my_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel()
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}
"""
```

### 2. Define C++ Function Declaration

Declare the launcher function signature:

```python
cpp_declaration = r"""
torch::Tensor launch_my_kernel(
    const torch::Tensor& input
);
"""
```

### 3. Configure Compilation Flags

```python
def get_cuda_arch_flags():
    """Generate NVCC flags for current device architecture."""
    if not torch.cuda.is_available():
        return []
    
    major, minor = torch.cuda.get_device_capability()
    arch_flag = f"--generate-code=arch=compute_{major}{minor},code=sm_{major}{minor}"
    return [arch_flag]

extra_flags = [
    '-O3',                    # Maximum optimization
    '--use_fast_math',        # Fast math operations
]
extra_flags.extend(get_cuda_arch_flags())
```

### 4. Compile Inline

```python
from torch.utils.cpp_extension import load_inline

my_cuda_ops = load_inline(
    name="my_cuda_ops",
    cpp_sources=cpp_declaration,
    cuda_sources=cuda_source,
    functions=["launch_my_kernel"],
    extra_cuda_cflags=extra_flags,
    verbose=True
)
```

### 5. Create Python Wrapper

```python
def my_operation_cuda(input_tensor):
    """
    CUDA-accelerated operation with Python-side preprocessing.
    
    Args:
        input_tensor: Input PyTorch tensor
        
    Returns:
        Output tensor with same shape as input
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    # Move to CUDA
    input_cuda = input_tensor.to('cuda')
    
    # Call compiled CUDA function
    result = my_cuda_ops.launch_my_kernel(input_cuda)
    
    return result
```

---

## Optimization Techniques

### Memory Access Patterns

**Vectorized Loads (float4):**
```cuda
// Load 4 floats at once for coalesced memory access
float4 vec = *reinterpret_cast<const float4*>(&input[idx * 4]);

#pragma unroll
for (int j = 0; j < 4; j++) {
    float val = ((float*)&vec)[j];
    // Process val
}
```

**Restrict Pointers:**
```cuda
const float* __restrict__ input  // Tells compiler no aliasing
```

### Warp-Level Primitives

**Warp Reduction:**
```cuda
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

**Block Reduction:**
```cuda
__device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    // Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Write to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Final reduction
    if (threadIdx.x < blockDim.x / 32) {
        val = shared[threadIdx.x];
    } else {
        val = 0.0f;
    }
    
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}
```

### Fast Math Intrinsics

```cuda
__fdividef(a, b)        // Fast division
__frcp_rn(x)           // Fast reciprocal
__expf(x)              // Fast exponential
__fmaf_rn(a, b, c)     // Fused multiply-add
```

### Grid-Stride Loops

```cuda
// Process multiple elements per block for better occupancy
for (int idx = blockIdx.x; idx < num_elements; idx += gridDim.x) {
    for (int i = threadIdx.x; i < element_size; i += blockDim.x) {
        // Process element
    }
    __syncthreads();
}
```

### Shared Memory

```cuda
// Declare shared memory (sized for max warps)
__shared__ float shared_data[32];
__shared__ float result_shared;

// Thread 0 computes, stores in shared memory
if (threadIdx.x == 0) {
    result_shared = compute_value();
}
__syncthreads();

// All threads read shared value
float result = result_shared;
```

---

## Best Practices

### Input Validation

Always validate tensors in the launcher function:

```cuda
torch::Tensor launch_function(...) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    
    // Ensure contiguous memory layout
    auto input_c = input.contiguous();
    
    // ... launch kernel
}
```

### Error Checking

```cuda
my_kernel<<<grid, block>>>(...);

cudaError_t err = cudaGetLastError();
TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

// Optional: Wait for completion
cudaDeviceSynchronize();
```

### Memory Management

```cuda
// Allocate output with same properties as input
auto output = torch::zeros_like(input);

// Or create with explicit options
auto output = torch::zeros({size}, 
    torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device()));
```

---

## Example: Linear Layer Relevance Propagation

See the complete implementation in [`cuda_v3.py`](../../dl_backtrace/pytorch_backtrace/dlbacktrace/utils/cuda_utils/Linear_v3/cuda_v3.py) for a production example featuring:

- **430+ lines** of optimized CUDA code
- **Warp and block reductions** for parallel summation
- **Vectorized memory access** with float4
- **Multiple activation functions** via device functions
- **Grid-stride loops** for variable-sized batches
- **Atomic operations** for result accumulation
- **Dynamic architecture flags** for compatibility

---

## Compilation Details

### First-Time Compilation

On first run, PyTorch compiles the CUDA code and caches it:

```
Emitting ninja build file /tmp/torch_extensions/linear_layer_cuda_v3/build.ninja...
Building extension module linear_layer_cuda_v3...
```

### Cache Location

Compiled extensions are cached in:
- Linux: `~/.cache/torch_extensions/`
- Windows: `%LOCALAPPDATA%\torch_extensions\`

### Recompilation

Recompilation occurs when:
- Source code changes (Python string content)
- PyTorch version changes
- CUDA version changes
- Compiler flags change

---

## Debugging

### Enable Verbose Output

```python
my_cuda_ops = load_inline(
    ...,
    verbose=True  # Shows compilation commands and output
)
```

### CUDA Error Checking

```python
import torch
torch.cuda.synchronize()  # Wait for all kernels to complete

# Check for errors
if torch.cuda.is_available():
    print(f"CUDA errors: {torch.cuda.get_device_name()}")
```

### Print from Kernel (Debug Only)

```cuda
__global__ void debug_kernel(...) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Debug value: %f\n", some_value);
    }
}
```

---

## Performance Tips

1. **Profile first**: Use `nvprof` or Nsight Compute
2. **Coalesce memory access**: Align reads/writes to warp size
3. **Minimize atomics**: Use when necessary, prefer reductions
4. **Tune block size**: Test 128, 256, 512 threads per block
5. **Use fast math**: `-use_fast_math` for faster operations
6. **Avoid divergence**: Minimize conditional branches in warps
7. **Reuse compiled code**: Cache compiled modules across runs

---

## See Also

- [Complete Linear Layer Implementation](../../dl_backtrace/pytorch_backtrace/dlbacktrace/utils/cuda_utils/Linear_v3/cuda_v3.py)
- [PyTorch C++ Extensions Documentation](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)



