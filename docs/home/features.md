# Key Features

DL-Backtrace provides a comprehensive set of features for explainable AI and model interpretability. Here's a detailed look at what makes it powerful.

---

## 🔍 Deep Model Interpretability

### Layer-wise Relevance Propagation
Understand how relevance flows through your model from outputs back to inputs.

- **Multiple Evaluation Modes**: Default and contrastive explanation modes
- **Layer-by-layer Analysis**: Track relevance at each layer of your model

### Relevance Attribution
Identify which input features contribute most to predictions:

- Pixel-level attribution for images
- Token-level attribution for text

---

## 🏗️ Architecture Agnostic

### Supported Architectures

=== "Convolutional Networks"
    - Standard CNNs (VGG, ResNet, DenseNet)
    - Modern architectures (EfficientNet, MobileNet)
    - Custom convolutional architectures
    
=== "Transformers"
    - BERT family (BERT, RoBERTa, DistilBERT, ALBERT)
    - SoTa models (LLaMA, Qwen)
    - Vision Transformers (ViT)
    - Custom transformer architectures

=== "Mixture of Experts"
    - Qwen3 MoE
    - GPT oss
    - JetMoE
    - OLMoE

=== "Recurrent Networks"
    - LSTM networks
    - Bidirectional RNNs

### Framework Support

- **PyTorch**: Full support for PyTorch 2.6+ models with comprehensive operation coverage

---

## ⚡ High Performance

### Optimized Execution Engines

**ExecutionEngineNoCache**
- In-memory execution for maximum speed
- Memory-efficient tensor management
- Enhanced operation support

**CUDA Acceleration**
- Custom CUDA kernels for critical operations
- Mixed precision support (FP16/FP32)
- Efficient memory management
- Automatic device placement

### Benchmarks

Example performance on NVIDIA A100 GPU:

| Model | Size | Trace Time | Evaluation Time |
|-------|------|-----------|-----------------|
| ResNet-18 | 11M params | 2.3s | 1.5s |
| BERT-base | 110M params | 5.7s | 3.2s |
| LLaMA-3.2-1B | 1B params | 18.4s | 12.1s |
| LLaMA-3.2-3B | 3B params | 42.1s | 28.6s |

---

## 🔧 Robust Operations

### Comprehensive PyTorch Operation Support

**100+ Supported Operations**

=== "Basic Operations"
    - Linear layers
    - Convolutional layers (Conv1d, Conv2d)
    - Pooling layers (Max, Average, Adaptive)
    - Activation functions (ReLU, GELU, SiLU, etc.)
    
=== "Tensor Operations"
    - Reshape, view, flatten
    - Transpose, permute
    - Squeeze, unsqueeze
    - Concatenate, stack
    - Slice, index_select
    
=== "Advanced Operations"
    - Layer normalization
    - Batch normalization
    - Attention mechanisms
    - Embedding layers
    - Dropout

### Negative Indexing Support

Full support for PyTorch's negative indexing:

```python
# All of these work seamlessly
x.transpose(-1, -2)
x.permute([-1, -2, 0])
x.unsqueeze(-1)
x.slice(dim=-1, start=0, end=-1)
torch.cat([x, y], dim=-1)
```

### Error Handling

- Comprehensive validation
- Graceful error messages
- Automatic dtype handling
- Shape mismatch detection

---

## 📊 Comprehensive Tracing

### Graph Capture

Automatically trace your model's computational graph:

- **Node-level tracking**: Every operation is traced
- **Parameter extraction**: Automatic weight and bias extraction
- **Topology sorting**: Correct execution order
- **Dynamic shapes**: Support for variable-length inputs

### Execution Tracking

Monitor execution in detail:

```python
# Get layer-wise outputs
node_io = dlb.predict(input_data)

# Access intermediate activations
for node_name, (inputs, output) in node_io.items():
    print(f"{node_name}: {output.shape}")
```

### Metadata Storage

- Layer hyperparameters
- Operation types
- Input/output shapes
- Execution statistics

---

## 🛡️ Production Ready

### Deterministic Execution Environment

Automatic setup for consistent results:

- ✅ CUDA memory management and synchronization
- ✅ Deterministic algorithms (when available)
- ✅ Random seed control
- ✅ Environment variable configuration

### Error Resilience

- Comprehensive validation at each step
- Graceful degradation when possible
- Detailed error messages and stack traces
- Debugging utilities and logging

### Testing & Validation

- Extensive test suite
- Benchmark suite for performance tracking
- Validation against known models
- Continuous integration

---

## 💾 Memory Efficient

### Memory Management Features

**ExecutionEngineNoCache**
- Runs entirely in RAM (no disk I/O)
- Automatic tensor cleanup
- Memory-efficient intermediate storage
- Garbage collection optimization

**Mixed Precision**
- FP16 support for reduced memory
- Automatic dtype conversion
- CPU/GPU dtype consistency

### Memory Optimization Tips

```python

dlb = DLBacktraceFX(
    model=model,
    input_for_graph=(x,),
    device="cuda"
)

# Enable mixed precision if supported
with torch.cuda.amp.autocast():
    node_io = dlb.predict(x)
```

---

## 📈 Visualization

### Graph Visualization

Generate beautiful visualizations of your model:

```python
# Visualize full computational graph
dlb.visualize()

# Visualize top-k most relevant nodes
dlb.visualize_dlbacktrace(top_k=15)
```

### Supported Formats

- PNG images
- SVG vector graphics
- Interactive graphs (via networkx)

### Customization

- Node coloring by relevance
- Edge thickness by flow
- Hierarchical layouts
- Customizable styling

---

<!-- ## 🚀 Recent Improvements (2025)

### Critical Fixes

- **🔧 Enhanced Execution Engine**: Robust handling of complex tensor operations
- **🚨 Boolean Tensor Handling**: Fixed crashes in RoBERTa/LLaMA models
- **🧠 Smart Attention Detection**: Auto-detects bidirectional vs causal attention
- **💾 Memory Optimization**: Fixed OOM errors in embedding operations
- **🔄 Dtype Consistency**: Universal framework for mixed precision

### New Features

- Support for LLaMA-3.2 models (1B, 3B)
- Enhanced CUDA kernels
- Improved error messages
- Better logging and debugging

--- -->



---

## Next Steps

- [Installation Guide](installation.md) - Get DL-Backtrace installed
- [Quick Start](quickstart.md) - Build your first explainable model
- [User Guide](../guide/introduction.md) - Learn the details
- [Examples](../examples/colab-notebooks.md) - Interactive notebooks and use cases



