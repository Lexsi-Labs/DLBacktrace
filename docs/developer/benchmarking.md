# Benchmarking Guide

Performance benchmarking for DL-Backtrace.

---

## Running Benchmarks

```bash
# Linear layer benchmark
python benchmarks/benchmark_linear.py

# Transformer models
python benchmarks/trace_RoBERTa.py
python benchmarks/trace_llama3_2_1B_selective.py
```

---

## Creating Benchmarks

```python
import time
import torch
from dl_backtrace.pytorch_backtrace import DLBacktraceFX

def benchmark_model(model, input_tensor, n_runs=100):
    """Benchmark model tracing and evaluation."""
    # Warmup
    dlb = DLBacktraceFX(model, input_for_graph=(input_tensor,))
    node_io = dlb.predict(input_tensor)
    
    # Benchmark tracing
    times = []
    for _ in range(n_runs):
        start = time.time()
        node_io = dlb.predict(input_tensor)
        end = time.time()
        times.append(end - start)
    
    print(f"Average time: {sum(times)/len(times):.4f}s")
    print(f"Std dev: {torch.tensor(times).std().item():.4f}s")
```

---

## Metrics

Track:
- Execution time
- Memory usage
- GPU utilization
- Accuracy vs baseline

---

## Results

Store results in `Benchmark results/` directory.

---

See existing benchmarks for examples.



