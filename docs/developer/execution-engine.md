# Execution Engine Development

Understanding and extending the non-cached execution engine for DL-Backtrace.

---

## Overview

The execution engine (`execution_engine_noncache.py`) is the **core runtime system** that executes traced computational graphs. It processes **3,982 lines** of highly optimized code supporting **100+ PyTorch operations** with comprehensive consistency checks, precision handling, and debugging capabilities.

**Key Features:**
- Deterministic execution with consistency enforcement
- Support for complex models (transformers, CNNs, hybrid architectures)
- Device-aware computation (CPU/GPU)
- Precision-preserving operations (fp16, fp32, int64)
- Comprehensive error handling and debugging
- Zero-copy optimization where possible

---

## Architecture

### 1. Centralized Logging System

**Class: `LoggingManager`** (Lines 10-101)

Singleton pattern for consistent logging across execution:

```python
# Configure logging
logger = setup_logging(debug=True, log_level="DEBUG")

# Use in code
logger.debug("Processing node: linear_0")
logger.warning("Device mismatch detected")
logger.error("Operation failed: invalid shape")
```

**Features:**
- Console and file logging (when debug=True)
- Timestamps and structured formatting
- Log file saved to `dlbacktrace_logs/`
- Prevents duplicate handlers
- Automatic log flushing

### 2. Consistency Enforcement Functions

**Critical Functions** (Lines 145-554):

**`enforce_precision_consistency`** - Ensures real tensors (not FakeTensors):
```python
# Converts FakeTensor to real tensor while preserving values
tensor = enforce_precision_consistency(tensor)
```

**`ensure_dtype_consistency`** - Handles mixed precision (fp16/fp32):
```python
# Converts all tensors to consistent dtype for CPU compatibility
tensors = ensure_dtype_consistency(tensors, target_dtype=torch.float32)
```

**`ensure_tensor_consistency`** - Comprehensive dtype and device alignment:
```python
# Ensures all tensors have same dtype and are on same device
a, b = ensure_tensor_consistency([a, b])
```

**`ensure_embedding_consistency`** - Special handling for embeddings:
```python
# Ensures weight is float, indices are long, same device
weight, indices = ensure_embedding_consistency(weight, indices, node_name)
```

### 3. Deterministic Environment Setup

**Function: `setup_consistent_environment`** (Lines 292-384)

Sets up deterministic execution:
- Disables gradients
- Enables deterministic algorithms
- Configures cuDNN for reproducibility
- Sets CUDA workspace config
- Manages memory for consistency

```python
setup_consistent_environment(model)
# ✅ Warnings suppressed
# ✅ Gradients disabled
# ✅ Deterministic cuDNN enabled
# ✅ CUDA memory cleared
```

---

## Operation Execution

### Main Dispatcher: `execute_aten_operation`

**Function** (Lines 690-3541) - **2,851 lines of operation handlers**

Dispatches to specialized handlers for 100+ operations:

```python
def execute_aten_operation(func_name, aten_op, layer_in, layer_hyperparams, 
                          method_args, parents, node_io, node_name, 
                          tensor_map, children=None, model_dtype=None):
    """
    Execute any ATen operation with full consistency checks
    """
    # Infer model compute dtype
    model_compute_dtype = infer_dtype(layer_in, model_dtype)
    
    # Enforce precision consistency
    layer_in = enforce_precision_consistency(layer_in)
    
    # Dispatch to operation-specific handler
    if func_name == "linear":
        # ... 30 lines of linear operation handling
    elif func_name == "scaled_dot_product_attention":
        # ... 100 lines of attention handling
    # ... 100+ more operations
```

### Supported Operation Categories

**1. Linear Algebra** (Lines 762-794, 2029-2060):
- `linear` - Fully connected layers with dtype consistency
- `matmul` - Matrix multiplication with broadcasting
- `bmm` - Batch matrix multiplication
- `mm` - 2D matrix multiplication
- `addmm` - Matrix multiply-add (A + B @ C)
- `baddbmm` - Batch matrix multiply-add
- `addmv` - Matrix-vector multiply-add
- `einsum` - Einstein summation

**2. Convolution** (Lines 795-837):
- `conv1d` - 1D convolution for sequences
- `conv2d` - 2D convolution for images
- Device and dtype consistency enforced

**3. Pooling** (Lines 839-871):
- `max_pool2d` - Maximum pooling
- `adaptive_avg_pool2d` - Adaptive average pooling
- `avg_pool2d` - Average pooling

**4. Normalization** (Lines 1037-1156, 3424-3456):
- `layer_norm` - Layer normalization with validation
- `batch_norm` - Batch normalization
- `native_layer_norm` - Optimized layer norm

**5. Attention** (Lines 1158-1259):
- `scaled_dot_product_attention` - Transformer attention
  - Automatic causal/bidirectional detection
  - Mask handling and reshaping
  - Model type inference (BERT vs GPT)

**6. Activations** (Lines 881-924, 2405-2411):
- `relu`, `relu_` - ReLU activation
- `gelu` - Gaussian Error Linear Unit
- `tanh` - Hyperbolic tangent
- `silu` - Sigmoid Linear Unit
- `sigmoid` - Sigmoid function

**7. Shape Operations** (Lines 925-997, 1327-1454, 2294-2403):
- `view` - Reshape with -1 inference
- `reshape` - Reshape tensor
- `transpose` - Swap dimensions
- `permute` - Reorder dimensions
- `unsqueeze` - Add dimension
- `squeeze` - Remove dimension
- `flatten`, `unflatten` - Flatten/unflatten
- `expand` - Broadcast tensor
- `select` - Select along dimension
- `slice` - Slice tensor
- All support negative indexing

**8. Element-wise** (Lines 1749-2003):
- `add`, `sub`, `mul`, `div` - Arithmetic
- `pow` - Power
- `neg` - Negation
- `rsqrt` - Reciprocal square root
- Broadcasting with shape alignment

**9. Comparison** (Lines 1894-2003, 2645-2708):
- `gt`, `ge`, `lt`, `le`, `eq`, `ne` - Comparisons
- Scalar and tensor variants
- Automatic broadcasting

**10. Tensor Creation** (Lines 2063-2142, 2413-2414):
- `full` - Create filled tensor
- `zeros` - Create zero tensor
- `arange` - Create range
- Symbolic size resolution

**11. Indexing & Gathering** (Lines 3178-3288, 3398-3422):
- `cat` - Concatenate tensors
- `split` - Split tensor
- `chunk` - Split into equal chunks
- `stack` - Stack along new dimension
- `gather` - Gather values
- `index` - Advanced indexing
- `index_select` - Select indices

**12. Masking** (Lines 1285-1325, 2460-2511):
- `masked_fill` - Fill masked positions
- `where` - Conditional selection
- Preserves -inf values correctly

**13. Other Operations**:
- `embedding` - Lookup embeddings (Lines 1713-1747)
- `softmax` - Softmax activation (Lines 1695-1704)
- `dropout` - Dropout (always disabled in eval) (Lines 873-879)
- `clone` - Clone tensor (Lines 2826-2844)
- `contiguous` - Make contiguous (Lines 2513-2519)
- `to`, `_to_copy` - Type/device conversion (Lines 1603-1647, 2521-2535)

---

## Node Processing Pipeline

### Main Function: `run_execution_nocache`

**Function** (Lines 3542-3935) - **393 lines**

Executes the entire graph in topological order:

```python
def run_execution_nocache(graph, layer_stack, model, extracted_weights, 
                          inputs, tracer, exported_program=None):
    """
    Main execution loop - processes nodes in dependency order
    
    Args:
        graph: NetworkX graph with node metadata
        layer_stack: Topologically sorted node names
        model: Original PyTorch model
        extracted_weights: Dictionary of model parameters
        inputs: Input tensors
        tracer: FX tracer instance
        exported_program: ExportedProgram for weight mapping
        
    Returns:
        node_io: Dictionary mapping node names to execution results
    """
    # Setup deterministic environment
    setup_consistent_environment(model)
    
    # Initialize tracking structures
    tensor_map = {}  # Current node outputs
    node_io = {}     # Complete execution history
    
    # Process each node in topological order
    for node_name in layer_stack:
        node_data = graph.nodes[node_name]
        
        # Skip if parents not ready
        if any(p not in tensor_map for p in parents):
            continue
        
        # Get inputs from parent nodes
        layer_in = [tensor_map[p] for p in parents]
        
        # Process by node type
        if layer_type == "Placeholder":
            output = handle_placeholder(...)
        elif layer_type == "Model_Layer":
            output = execute_submodule(...)
        elif layer_type == "ATen_Operation":
            output = execute_aten_operation(...)
        
        # Store output for children
        tensor_map[node_name] = output
        node_io[node_name] = {
            "input_values": layer_in,
            "output_values": output,
            "layer_type": layer_type,
            # ... more metadata
        }
    
    return node_io
```

**Key Steps:**

1. **Environment Setup** (Lines 3547-3602):
   - Deterministic configuration
   - Model consistency checks
   - Weight synchronization
   - Device and dtype detection

2. **Input Processing** (Lines 3607-3647):
   - Input validation
   - Type-specific casting (input_ids → long)
   - Contiguity enforcement
   - Initial tensor_map population

3. **Node Iteration** (Lines 3649-3839):
   - Topological processing
   - Parent dependency checking
   - Type-specific execution
   - Output validation

4. **Error Handling** (Lines 3836-3838):
   - Graceful degradation
   - Detailed error logging
   - Fallback strategies

---

## Node Types and Handling

### 1. Placeholder Nodes

**Lines 3689-3715**

Handle model inputs and parameters:

```python
if layer_type == "Placeholder":
    output = extracted_weights.get(node_name, inp_map.get(node_name))
    
    # Classify by name patterns
    if "weight" in node_name:
        layer_type = "Weight"
    elif "bias" in node_name:
        layer_type = "Bias"
    elif "running_mean" in node_name:
        layer_type = "bn_running_mean"
    # ... more types
```

### 2. Model_Layer Nodes

**Lines 3717-3753**

Execute PyTorch submodules:

```python
elif layer_type == "Model_Layer":
    # Resolve nested module (e.g., "encoder.layer.0.attention")
    resolved_layer = model
    for attr in layer.split("."):
        resolved_layer = getattr(resolved_layer, attr)
    
    # Ensure consistency
    layer_in = [enforce_precision_consistency(x) for x in layer_in]
    resolved_layer.eval()
    resolved_layer.requires_grad_(False)
    
    # Execute
    output = resolved_layer(*layer_in)
```

### 3. ATen Operation Nodes

**Lines 3761-3828**

Execute PyTorch operations:

```python
elif layer_type in ("ATen_Operation", "Operation"):
    # Collect inputs (filter out weights/biases for most ops)
    layer_in = []
    for node_x in parents:
        if node_io[node_x]['layer_type'] not in excluded_types:
            layer_in.append(node_io[node_x]['output_values'])
    
    # Process inputs
    layer_in, _ = _process_layer_input(layer_in)
    
    # Execute operation
    output = execute_aten_operation(
        func_name, aten_op, layer_in, layer_hyperparams,
        method_args, parents, node_io, node_name,
        tensor_map, children=children, model_dtype=model_dtype
    )
```

---

## Adding New Operations

### Step-by-Step Guide

**1. Identify Operation Name**

Check FX graph tracing output or error messages:
```
🚨 UNHANDLED OPERATION: `my_new_op` - needs implementation!
```

**2. Add Handler in `execute_aten_operation`**

Add new `elif` block around line 3518:

```python
elif func_name == "my_new_op":
    # 🔧 NEW OPERATION: Brief description
    logger.debug(f"[{node_name}] 🔧 my_new_op: input type={type(layer_in)}")
    
    # Step 1: Validate and unwrap inputs
    if isinstance(layer_in, (list, tuple)):
        layer_in = layer_in[0]
    layer_in = enforce_precision_consistency(layer_in)
    
    if not isinstance(layer_in, torch.Tensor):
        raise TypeError(f"[{node_name}] ❌ my_new_op: expected tensor, got {type(layer_in)}")
    
    # Step 2: Extract hyperparameters
    param1 = layer_hyperparams.get("param1", default_value)
    param2 = layer_hyperparams.get("param2")
    
    if param2 is None:
        raise RuntimeError(f"[{node_name}] ❌ my_new_op: missing required parameter 'param2'")
    
    # Step 3: Handle negative indexing (if applicable)
    if isinstance(param1, int) and param1 < 0:
        param1 = layer_in.dim() + param1
    
    # Step 4: Validate parameters
    if param1 < 0 or param1 >= layer_in.dim():
        raise ValueError(f"[{node_name}] ❌ my_new_op: param1 out of range")
    
    # Step 5: Execute operation with error handling
    try:
        output = aten_op(layer_in, param1, param2)
        logger.debug(f"[{node_name}] ✅ my_new_op: output shape={output.shape}")
        return output
    except Exception as e:
        raise RuntimeError(f"[{node_name}] ❌ my_new_op failed: {e}")
```

**3. Test the Operation**

```python
# Test with a simple model
model = nn.Sequential(
    nn.Linear(10, 20),
    MyNewOp(),  # Your operation
    nn.Linear(20, 10)
)

dlb = DLBacktrace(model, input_for_graph=(torch.randn(1, 10),))
result = dlb.predict(torch.randn(1, 10))
```

---

## Best Practices

### 1. Always Enforce Consistency

```python
# ✅ Good - enforce consistency
layer_in = enforce_precision_consistency(layer_in)

# ❌ Bad - skip consistency checks
output = aten_op(layer_in)  # May have FakeTensors!
```

### 2. Handle List Inputs Properly

```python
# ✅ Good - check and unwrap
if isinstance(layer_in, list):
    if len(layer_in) == 1:
        layer_in = layer_in[0]
    elif len(layer_in) == 0:
        raise RuntimeError("Empty input list")

# ❌ Bad - assume structure
layer_in = layer_in[0]  # May crash!
```

### 3. Validate Inputs and Parameters

```python
# ✅ Good - comprehensive validation
if not isinstance(layer_in, torch.Tensor):
    raise TypeError(f"Expected tensor, got {type(layer_in)}")

if param is None:
    raise RuntimeError("Missing required parameter")

# ❌ Bad - no validation
output = aten_op(layer_in, param)  # May crash later
```

### 4. Use Descriptive Logging

```python
# ✅ Good - structured logging
logger.debug(f"[{node_name}] 🔧 linear: input shape={layer_in.shape}")
logger.debug(f"[{node_name}] ✅ linear: output shape={output.shape}")

# ❌ Bad - minimal logging
print("doing linear")
```

### 5. Handle Device Consistency

```python
# ✅ Good - ensure same device
target_device = layer_in.device
if weight.device != target_device:
    weight = weight.to(device=target_device)

# ❌ Bad - ignore device mismatches
output = aten_op(layer_in, weight)  # May fail on CPU/GPU mismatch
```

### 6. Support Negative Indexing

```python
# ✅ Good - handle negative indices
tensor_rank = layer_in.dim()
if dim < 0:
    dim = tensor_rank + dim

# ❌ Bad - don't handle negative indices
output = aten_op(layer_in, dim)  # May crash with dim=-1
```

---

## Debugging

### Enable Debug Logging

```python
# Create engine with debug enabled
engine = ExecutionEngineNoCache(
    model, weights, graph, stack, tracer, exported_program,
    debug=True,          # Enable file logging
    log_level="DEBUG"    # Verbose output
)

# Run execution
node_io = engine.run(inputs)

# Check log file
# dlbacktrace_logs/dlbacktrace_debug_YYYYMMDD_HHMMSS.log
```

### Common Issues

**1. Device Mismatch**
```
Error: Expected all tensors to be on the same device
```
**Fix:** Add device consistency check before operation

**2. Dtype Mismatch**
```
Error: expected scalar type Float but found Half
```
**Fix:** Use `ensure_tensor_consistency()` or `ensure_dtype_consistency()`

**3. Shape Mismatch**
```
Error: mat1 and mat2 shapes cannot be multiplied
```
**Fix:** Add broadcasting or shape validation

**4. None Input**
```
Error: Cannot perform operation on None
```
**Fix:** Add None check and graceful fallback

---

## Performance Tips

1. **Avoid Unnecessary Copies**: Use `.detach()` instead of `.clone()` when possible
2. **Reuse Tensors**: Don't create new tensors in hot loops
3. **Batch Operations**: Process multiple items together when possible
4. **Profile First**: Use logging to identify bottlenecks
5. **CUDA Synchronization**: Minimize `torch.cuda.synchronize()` calls

---

## Execution Engine Class

**Class: `ExecutionEngineNoCache`** (Lines 3937-3982)

Wrapper class for the execution engine:

```python
class ExecutionEngineNoCache:
    def __init__(self, model, extracted_weights, fx_graph, layer_stack, 
                 tracer, exported_program, debug=False, log_level="INFO"):
        self.model = model
        self.extracted_weights = extracted_weights
        self.graph = fx_graph
        self.layer_stack = layer_stack
        self.tracer = tracer
        self.exported_program = exported_program
        self.debug = debug
        self.log_level = log_level
        
        # Setup logging
        self.logger = setup_logging(debug=debug, log_level=log_level)
    
    def run(self, inputs, debug=None, log_level=None):
        """Execute graph and return node_io dictionary"""
        # Update logging if parameters changed
        if debug is not None or log_level is not None:
            self.logger = setup_logging(debug or self.debug, log_level or self.log_level)
        
        # Run execution
        result = run_execution_nocache(
            graph=self.graph,
            layer_stack=self.layer_stack,
            model=self.model,
            extracted_weights=self.extracted_weights,
            inputs=inputs,
            tracer=self.tracer,
            exported_program=self.exported_program,
        )
        
        # Flush logs
        flush_logs()
        
        return result
```

---

## See Also

- [CUDA Development](cuda-development.md) - Writing CUDA kernels for custom operations
- [Architecture](architecture.md) - Overall system architecture
- [Contributing](contributing.md) - Development guidelines



