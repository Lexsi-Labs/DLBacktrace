# Testing Guide

Guidelines for testing DL-Backtrace.

---

## Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_operations.py

# With coverage
pytest --cov=dl_backtrace tests/
```

---

## Writing Tests

### Unit Tests

```python
# tests/test_operations.py
import pytest
import torch
from dl_backtrace.pytorch_backtrace import DLBacktrace

def test_linear_layer():
    """Test linear layer tracing."""
    model = torch.nn.Linear(10, 5)
    model.eval()
    
    input_tensor = torch.randn(1, 10)
    
    dlb = DLBacktrace(model, input_for_graph=(input_tensor,))
    node_io = dlb.predict(input_tensor)
    
    assert len(node_io) > 0
```

### Integration Tests

```python
def test_full_workflow():
    """Test complete workflow."""
    model = create_test_model()
    model.eval()
    
    input_tensor = torch.randn(1, 3, 32, 32)
    
    dlb = DLBacktrace(model, input_for_graph=(input_tensor,))
    node_io = dlb.predict(input_tensor)
    relevance = dlb.evaluation(
        mode="default",
        task="multi-class classification"
    )
    
    assert relevance is not None
    assert len(relevance) > 0
```

---

## Benchmark Tests

Located in `benchmarks/`:

```bash
python benchmarks/trace_RoBERTa.py
python benchmarks/benchmark_linear.py
```

---

## Test Coverage

Check coverage:

```bash
pytest --cov=dl_backtrace --cov-report=html tests/
# View htmlcov/index.html
```

---

See [Contributing Guide](contributing.md) for submission requirements.



