# DL-Backtrace Pipeline

The **DL-Backtrace Pipeline** provides a high-level, batteries-included interface for running explainability analysis on PyTorch models. It handles model loading, input preprocessing, execution, and result management automatically.

---

## Overview

The Pipeline simplifies the DL-Backtrace workflow by providing:

- **🔧 Automatic Model Loading**: Seamless integration with HuggingFace and TorchVision models
- **🎯 Multi-Modal Support**: Text classification, image classification, and text generation
- **⚙️ Flexible Configuration**: Comprehensive configuration system for all parameters
- **📊 Built-in Relevance Analysis**: Automatic layer-wise relevance propagation
- **🎨 Visualization Support**: Optional graph and relevance visualization
- **💾 Result Management**: Automatic saving of results, timing, and metrics

---

## Quick Start

### Basic Usage

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline

# Create a simple pipeline
pipeline = DLBacktracePipeline.create_simple(
    model_name="bert-base",
    device="cpu"
)

# Run analysis
results = pipeline.run_simple_analysis(
    "This product is amazing!",
    label="positive"
)

print(f"Prediction: {results['predictions'][0]}")
print(f"Execution time: {results['timing']['total_time']:.2f}s")
```

---

## Supported Tasks

### 1. Text Classification

Analyze sentiment, intent, or any text classification task with transformer models.

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline

# Initialize pipeline
pipeline = DLBacktracePipeline.create_simple(
    model_name="bert-base",
    device="cpu"
)

# Batch text classification
texts = [
    "This movie is fantastic!",
    "Terrible experience, very disappointed.",
    "It was okay, nothing special."
]
labels = ["positive", "negative", "neutral"]

results = pipeline.run_text_classification(texts, labels)

# Access results
for i, text in enumerate(texts):
    print(f"Text: {text}")
    print(f"Predicted: {results['predictions'][i]}")
    print(f"Relevance computed: {results['relevance_computed']}")
    print("---")
```

**Supported Models:**
- BERT (`bert-base`)
- RoBERTa (`roberta`)
- ALBERT (`albert`)
- ELECTRA (`electra`)
- XLNet (`xlnet`)
- DistilBERT (`distilbert`)

### 2. Image Classification

Analyze image classifications with CNN or Vision Transformer models.

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline
from PIL import Image

# Initialize pipeline
pipeline = DLBacktracePipeline.create_simple(
    model_name="resnet",
    device="cpu"
)

# Load and classify images
images = [
    Image.open("cat.jpg"),
    Image.open("dog.jpg")
]
labels = ["cat", "dog"]

results = pipeline.run_image_classification(images, labels)

# Access predictions and relevance
for i, label in enumerate(labels):
    print(f"Image {i+1}: Predicted class {results['predictions'][i]}")
```

**Supported Models:**
- ResNet-50 (`resnet`)
- VGG-16 (`vgg`)
- Vision Transformer (`vit`)
- DenseNet-121 (`densenet`)
- EfficientNet-B0 (`efficientnet`)
- MobileNet-V2 (`mobilenet`)

### 3. Text Generation

Generate text with relevance analysis for LLMs and MoE models.

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline

# Initialize pipeline for generation
pipeline = DLBacktracePipeline.create_simple(
    model_name="llama3.2-1b",
    device="cuda"  # Recommended for generation
)

# Generate text with custom parameters
prompts = ["The future of artificial intelligence is"]

results = pipeline.run_text_generation(
    prompts,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    return_relevance=True
)

# Access generated text and relevance
print(f"Generated: {results['generated_texts'][0]}")
print(f"Relevance computed: {results['relevance_computed']}")
```

**Supported Models:**
- Llama 3.2 1B (`llama3.2-1b`)
- Llama 3.2 3B (`llama3.2-3b`)
- Qwen 2.5 0.5B (`qwen2.5-0.5b`)
- And many more
- MoE Models (see [MoE Support](moe-models.md))

---

## Configuration

### Pipeline Configuration

The `PipelineConfig` class provides comprehensive control over the pipeline behavior.

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import PipelineConfig

config = PipelineConfig(
    # Model configuration
    model_name="bert-base",
    device="cpu",
    
    # Input configuration
    max_length=512,
    batch_size=4,
    
    # Relevance parameters
    mode="default",
    multiplier=100.0,
    
    # Output configuration
    save_results=True,
    save_visualization=False,
    output_dir="my_results"
)

pipeline = DLBacktracePipeline(config)
```

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | **Required** | Model identifier from registry |
| `device` | str | `"cpu"` | Device: `"cpu"` or `"cuda"` |
| `batch_size` | int | `1` | Batch size for processing |
| `verbose` | bool | `False` | Enable verbose logging |
| `debug` | bool | `False` | Enable debug mode |

### Input Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_length` | int | `512` | Maximum sequence length (text) |
| `image_size` | Tuple | `(224, 224)` | Image dimensions |
| `labels` | List[str] | `None` | Ground truth labels |

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_new_tokens` | int | `50` | Maximum tokens to generate |
| `temperature` | float | `1.0` | Sampling temperature |
| `top_k` | int | `50` | Top-k sampling |
| `top_p` | float | `0.9` | Nucleus sampling threshold |
| `num_beams` | int | `1` | Beam search width |
| `return_relevance` | bool | `True` | Compute relevance |

### Relevance Analysis

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | `"default"` | Relevance propagation mode |
| `multiplier` | float | `100.0` | Relevance multiplier |
| `scaler` | float | `1.0` | Relevance scaler |
| `thresholding` | float | `0.5` | Relevance threshold |

### Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_results` | bool | `True` | Save results to disk |
| `save_visualization` | bool | `False` | Generate visualizations |
| `output_dir` | str | `"backtrace_results"` | Output directory |

---

## Advanced Usage

### Custom Model Loading

```python
import torch
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import PipelineConfig, DLBacktracePipeline

# Load your own model
model = torch.load("my_model.pt")

config = PipelineConfig(
    model_name="bert-base",  # Use for tokenizer/preprocessing
    device="cuda"
)

pipeline = DLBacktracePipeline(config)
pipeline.model = model  # Override with custom model
pipeline.initialize_dlbacktrace()
```

### Batch Processing

```python
# Process multiple inputs efficiently
texts = ["text1", "text2", "text3", "text4"]
labels = ["label1", "label2", "label3", "label4"]

# Configure batch size
config = PipelineConfig(
    model_name="bert-base",
    batch_size=2,  # Process 2 at a time
    device="cuda"
)

pipeline = DLBacktracePipeline(config)
results = pipeline.run_text_classification(texts, labels)
```

### Custom Relevance Analysis

```python
config = PipelineConfig(
    model_name="resnet",
    mode="default",
    multiplier=200.0,  # Increase relevance magnitude
    scaler=1.5,
    thresholding=0.3
)

pipeline = DLBacktracePipeline(config)
results = pipeline.run_image_classification(images, labels)

# Access relevance for each layer
relevance_data = results['relevance']
```

---

## Result Structure

The pipeline returns a dictionary with the following structure:

```python
{
    'predictions': [...],           # Model predictions
    'generated_texts': [...],       # Generated text (if applicable)
    'relevance': {...},            # Layer-wise relevance values
    'relevance_computed': bool,    # Whether relevance was computed
    'timing': {
        'model_load_time': float,
        'inference_time': float,
        'relevance_time': float,
        'total_time': float
    },
    'config': {...},               # Configuration used
    'metadata': {...}              # Additional metadata
}
```

---

## Model Registry

The Pipeline includes a comprehensive model registry. List available models:

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import ModelRegistry

# List all available models
models = ModelRegistry.list_models()
for model in models:
    print(f"{model}: {ModelRegistry.get_model_info(model).model_type}")

# Get model information
info = ModelRegistry.get_model_info("bert-base")
print(f"Model: {info.base_model_path}")
print(f"Type: {info.model_type}")
print(f"Modality: {info.modality}")
```

---

## Best Practices

### 1. Device Selection

- Use `device="cuda"` for faster execution with GPU acceleration
- The device parameter automatically selects optimal layer implementations
- For large models or generation tasks, CUDA is strongly recommended

### 2. Batch Size Optimization

- Increase `batch_size` for better throughput when processing multiple inputs
- Monitor GPU memory usage and adjust accordingly
- For generation tasks, typically use `batch_size=1`

### 3. Result Management

```python
config = PipelineConfig(
    model_name="bert-base",
    save_results=True,
    output_dir="./results/experiment_1"
)
```

Results are automatically saved with timestamps, making it easy to track experiments.

### 4. Memory Management

For large models or long sequences:

```python
import torch

# Clear cache between runs
torch.cuda.empty_cache()

# Use smaller batch sizes
config.batch_size = 1
```

---

## Troubleshooting

### Model Loading Issues

```python
# Check available models
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import ModelRegistry
print(ModelRegistry.list_models())

# Verify model exists
info = ModelRegistry.get_model_info("your-model-name")
```

### Memory Errors

- Reduce `batch_size`
- Reduce `max_length` for text models
- Use `device="cpu"` if GPU memory is limited
- Enable `debug=False` to reduce overhead

### Performance Optimization

- Use CUDA when available: `device="cuda"`
- Disable visualization for faster execution: `save_visualization=False`
- Use appropriate batch sizes for your hardware
- Consider disabling relevance for inference-only: `return_relevance=False`

---

## Next Steps

- Learn about [MoE Model Support](moe-models.md)
- Explore [DLB Auto Sampler](auto-sampler.md) for advanced text generation
- Check [Temperature Scaling](temperature-scaling.md) for controlled generation
- See [Examples](../../examples/colab-notebooks.md) for complete use cases

