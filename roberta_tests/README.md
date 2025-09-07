# RoBERTa Testing Suite

This folder contains comprehensive tests for diagnosing RoBERTa tracing issues in DL-Backtrace.

## Test Files Overview

### Individual Operation Tests
- `test_attention.py` - Tests scaled_dot_product_attention operation
- `test_layernorm.py` - Tests LayerNorm operation
- `test_embedding.py` - Tests embedding operations
- `test_cumsum.py` - Tests cumsum operation (position embeddings)
- `test_slice.py` - Tests slice operations on buffer tensors
- `test_add.py` - Tests addition operations (residual connections)
- `test_ne.py` - Tests not-equal operation (attention masks)
- `test_boolean_tensors.py` - Tests boolean tensor handling

### Comprehensive Analysis Tests
- `test_tensor_shapes.py` - Analyzes tensor shapes throughout execution
- `test_numerical_divergence.py` - Finds where numerical divergence starts
- `test_weight_synchronization.py` - Verifies weight consistency
- `test_graph_execution.py` - Analyzes graph execution order

### Full Model Tests
- `test_roberta_full.py` - Complete RoBERTa model test
- `test_roberta_minimal.py` - Minimal RoBERTa test with simple inputs

### Diagnostic Tools
- `run_all_tests.py` - Runs all tests and generates summary report
- `compare_operations.py` - Compares direct vs traced execution for specific operations

## Usage

```bash
# Run all tests
python roberta_tests/run_all_tests.py

# Run specific test
python roberta_tests/test_attention.py

# Run diagnostic comparison
python roberta_tests/compare_operations.py
```

## Test Results Interpretation

- ✅ **Max difference: 0.000000** = Perfect match
- ⚠️ **Max difference: < 1e-5** = Acceptable precision difference  
- ❌ **Max difference: > 1e-5** = Significant issue requiring investigation
