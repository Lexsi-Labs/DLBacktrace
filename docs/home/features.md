# Key Features

DLBacktrace provides a comprehensive set of features for explainable AI and model interpretability. Here's a detailed look at what makes it powerful.

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

## 🚀 High-Level Pipeline Interface

### Unified `run_task()` Method

Simplified, unified interface for running explainability analysis with a single method call:

- **🎯 Unified API**: Single method for all task types (classification, generation)
- **🔧 Auto Task Detection**: Automatically detect task from inputs
- **📊 Built-in Relevance**: Automatic layer-wise propagation
- **�  Generation Support**: Greedy, sampling, and beam search with tracing
- **⚙️ Flexible Configuration**: Comprehensive parameter control
- **💾 Structured Results**: Easy access to predictions, relevance, and traces

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prepare input
text = "This product is amazing!"
tokens = tokenizer(text, return_tensors="pt")

# Initialize DLBacktrace
dlb = DLBacktrace(
    model=model,
    input_for_graph=(tokens["input_ids"], tokens["attention_mask"]),
    device="cuda"
)

# Run analysis with one call
results = dlb.run_task(
    task="text-classification",  # or "auto" for automatic detection
    inputs={'input_ids': tokens["input_ids"], 'attention_mask': tokens["attention_mask"]}
)

# Access results
print(f"Prediction: {results['predictions'].argmax()}")
print(f"Token relevance: {results['relevance']['input_ids'].shape}")
```

[Learn more about Pipeline →](../guide/pytorch/pipeline.md)

---

## 🎨 Advanced Text Generation

### DLB Auto Sampler

Native text generation with explainability built-in via `run_task()`:

- **🎯 Multiple Sampling Strategies**: Greedy, temperature, top-k, top-p, beam search
- **🔍 Token-Level Relevance**: Track relevance for each generated token
- **🤝 HuggingFace Compatible**: Familiar generation parameters
- **🎛️ Flexible Control**: Full parameter customization
- **📊 Generation Tracing**: Optional scores, relevance, and layer output traces

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Prepare input
prompt = "The future of AI is"
tokens = tokenizer(prompt, return_tensors="pt")

# Initialize DLBacktrace
dlb = DLBacktrace(
    model=model,
    input_for_graph=(tokens["input_ids"],),
    device="cuda"
)

# Generate with relevance tracing
results = dlb.run_task(
    task="generation",
    inputs={'input_ids': tokens["input_ids"]},
    tokenizer=tokenizer,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
    return_relevance=True,
    return_scores=True
)

# Decode and analyze
generated_text = tokenizer.decode(results['generated_ids'][0], skip_special_tokens=True)
print(f"Generated: {generated_text}")
print(f"Relevance trace: {len(results['relevance_trace'])} steps")
```

[Learn more about Auto Sampler →](../guide/pytorch/auto-sampler.md)

### Temperature Scaling

Control generation diversity and model confidence:

- **🎛️ Confidence Control**: Adjust prediction confidence without retraining
- **🎨 Generation Diversity**: Fine-tune randomness in text generation
- **⚖️ Calibration**: Improve probability calibration
- **⚡ Zero Overhead**: Efficient implementation

```python
# Generate with temperature scaling
node_io = dlb.predict(
    input_data,
    temperature=0.8  # Control confidence/diversity
)
```

[Learn more about Temperature Scaling →](../guide/pytorch/temperature-scaling.md)

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

=== "Mixture-of-Experts"
    - **JetMoE**: Efficient MoE with sparse activation
    - **OLMoE**: Open Language MoE
    - **Qwen MoE**: Advanced routing with grouped query attention
    - **GPT-OSS**: Configurable expert architecture with sliding window
    
    **Expert-Level Tracking:**
    - Track relevance at expert level
    - Understand expert routing patterns
    - Analyze expert contributions
    - CUDA-accelerated MoE layers
    
    [Learn more about MoEs Support →](../guide/pytorch/moe-models.md)

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

dlb = DLBacktrace(
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

- [Installation Guide](installation.md) - Get DLBacktrace installed
- [Quick Start](quickstart.md) - Build your first explainable model
- [User Guide](../guide/introduction.md) - Learn the details
- [Examples](../examples/colab-notebooks.md) - Interactive notebooks and use cases



