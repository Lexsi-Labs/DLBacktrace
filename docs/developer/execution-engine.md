# Execution Engine Development

Understanding and extending the execution engine.

---

## Overview

The execution engine (`execution_engine_noncache.py`) runs the traced computational graph.

---

## Key Components

### 1. Node Processing

Each node is processed in topological order:

```python
for node in topological_order:
    if node.op == "call_function":
        output = execute_aten_operation(node, ...)
    elif node.op == "placeholder":
        output = handle_input(node)
    # Store output for dependent nodes
    tensor_map[node.name] = output
```

### 2. Operation Dispatch

Operations are dispatched by name:

```python
def execute_aten_operation(node, ...):
    func_name = get_func_name(node.target)
    
    if func_name == "linear":
        return handle_linear(...)
    elif func_name == "conv2d":
        return handle_conv2d(...)
    # ... more operations
```

---

## Adding New Operations

1. Identify operation name
2. Add handler in `execute_aten_operation`
3. Extract hyperparameters
4. Execute operation
5. Return output

Example:

```python
elif func_name == "my_new_op":
    # Get inputs
    layer_in = [node_io[p]['output_values'] for p in parents]
    
    # Get parameters
    param = layer_hyperparams.get('param', default_value)
    
    # Execute
    output = aten_op(layer_in, param)
    
    return output
```

---

See [EXECUTION_ENGINE_CRITICAL_FIXES.md](../../dev_notes/EXECUTION_ENGINE_CRITICAL_FIXES.md) for important implementation details.



