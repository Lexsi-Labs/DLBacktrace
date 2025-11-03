# Known Issues

Current limitations and known issues in DL-Backtrace.

---

## Platform-Specific Issues

### Windows

**Issue**: CUDA kernel compilation may fail on Windows

**Workaround**: Use pre-compiled binaries or WSL2

**Status**: Investigating cross-platform build system

---

## Model-Specific Issues

### Very Large Models (>10B parameters)

**Issue**: May run out of memory even on high-end GPUs

**Workaround**: Use CPU execution or model quantization

**Status**: Working on memory optimization

### Dynamic Models

**Issue**: Models with data-dependent control flow may fail to trace

**Workaround**: Rewrite using tensor operations

**Status**: Limited by PyTorch's tracing capabilities

---

## Operation-Specific Issues

### EmbeddingBag

**Issue**: Not yet supported

**Status**: Planned for future release

### Custom Operations

**Issue**: Some custom operations may not be recognized

**Workaround**: Decompose into supported operations

**Status**: Ongoing - add support as needed

---

## Feature Limitations

### Multi-GPU

**Issue**: Multi-GPU inference not yet supported

**Status**: In development for future release

---

## Workarounds

Report new issues on [GitHub](https://github.com/Lexsi-Labs/DLBacktrace/issues).



