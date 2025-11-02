# Multi-Modal DL-Backtrace Pipeline

A comprehensive, high-level pipeline interface for running DL-Backtrace on PyTorch models across multiple modalities and tasks. Supports text classification, image classification, and text generation with automated model loading, execution, and result management.

## 🚀 Features

- **🔤 Text Classification**: BERT, RoBERTa, ALBERT, XLNet, ELECTRA for sentiment analysis
- **🖼️ Image Classification**: ResNet, DenseNet, ViT, VGG, EfficientNet, MobileNet for image recognition  
- **📝 Text Generation**: Llama, Qwen, GPT-2 for text generation with relevance analysis
- **🔧 Automatic Model Loading**: Seamless integration with HuggingFace and TorchVision models
- **⚙️ Flexible Configuration**: Comprehensive configuration system for all task types
- **📊 Relevance Analysis**: Layer-wise relevance propagation for all supported models
- **🎯 Batch Processing**: Efficient processing of multiple inputs
- **💾 Result Management**: Automatic saving of results, timing, and relevance values
- **📈 Visualization Support**: Optional graph and relevance visualization

## 🚀 Quick Start

### Text Classification

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline

# Create pipeline for sentiment analysis
pipeline = DLBacktracePipeline.create_simple(
    model_name="bert-base",
    device="cpu"
)

# Run text classification with labels
texts = ["This movie is great!", "I hate this film."]
labels = ["positive", "negative"]
results = pipeline.run_text_classification(texts, labels)
```

### Image Classification

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline
from PIL import Image

# Create pipeline for image classification
pipeline = DLBacktracePipeline.create_simple(
    model_name="resnet",
    device="cpu"
)

# Load images and run classification
images = [Image.open("cat.jpg"), Image.open("dog.jpg")]
labels = ["cat", "dog"]
results = pipeline.run_image_classification(images, labels)
```

### Text Generation

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline

# Create pipeline for text generation
pipeline = DLBacktracePipeline.create_simple(
    model_name="llama3.2-1b",
    device="cpu"
)

# Generate text with custom parameters
prompts = ["The future of AI is"]
results = pipeline.run_text_generation(
    prompts,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
    return_relevance=True
)
```

### Simple Analysis (Auto-detect Task)

```python
# Single input analysis - automatically detects task type
pipeline = DLBacktracePipeline.create_simple("bert-base")
results = pipeline.run_simple_analysis("This is amazing!", label="positive")
```

## Configuration

### PipelineConfig Parameters

#### Core Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | **Required** | Model identifier from registry |
| `device` | str | "cpu" | Device: "cpu" or "cuda" (auto-selects layer implementations) |
| `batch_size` | int | 1 | Batch size for processing |
| `verbose` | bool | False | Enable verbose logging |
| `debug` | bool | False | Enable debug mode |

#### Input Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_length` | int | 512 | Maximum sequence length (text models) |
| `image_size` | Tuple[int, int] | (224, 224) | Image size for image models |
| `labels` | List[str] | None | Ground truth labels for classification |

#### Generation Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_new_tokens` | int | 50 | Maximum new tokens to generate |
| `temperature` | float | 1.0 | Temperature for sampling |
| `top_k` | int | 50 | Top-k sampling parameter |
| `top_p` | float | 0.9 | Top-p (nucleus) sampling parameter |
| `num_beams` | int | 1 | Number of beams for beam search |
| `num_return_sequences` | int | 1 | Number of sequences to return |
| `early_stopping` | bool | True | Enable early stopping |
| `return_scores` | bool | False | Return generation scores |
| `return_relevance` | bool | True | Compute relevance for generation |

#### Relevance Analysis Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | "default" | Relevance propagation mode |
| `multiplier` | float | 100.0 | Relevance multiplier |
| `scaler` | float | 1.0 | Relevance scaler |
| `thresholding` | float | 0.5 | Relevance threshold |
| `task` | str | "binary-classification" | Task type for relevance |

#### Output Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_results` | bool | True | Save results to disk |
| `save_visualization` | bool | False | Generate visualizations |
| `output_dir` | str | "backtrace_results" | Output directory |

**Device Configuration**: The `device` parameter automatically selects optimal layer implementations:
- `"cpu"`: Refactored implementations for most layers, original for critical layers
- `"cuda"`: CUDA implementations where available (Linear, Embedding, Attention, Conv2D)

## 📦 Supported Models

### Image Classification Models
- **VGG**: `vgg` - VGG-16 for image classification
- **ResNet**: `resnet` - ResNet-50 for image classification
- **Vision Transformer**: `vit` - ViT base model for image classification
- **DenseNet**: `densenet` - DenseNet-121 for image classification
- **EfficientNet**: `efficientnet` - EfficientNet-B0 for image classification
- **MobileNet**: `mobilenet` - MobileNet-V2 for image classification

### Text Classification Models
- **BERT**: `bert-base` - BERT base model for text classification
- **ALBERT**: `albert` - ALBERT base model for text classification
- **RoBERTa**: `roberta` - RoBERTa base model for text classification
- **DistilBERT**: `distilbert` - DistilBERT base model for text classification
- **ELECTRA**: `electra` - ELECTRA base model for text classification
- **XLNet**: `xlnet` - XLNet base model for text classification

### Text Generation Models
- **Llama 3.2**: `llama3.2-1b`, `llama3.2-3b` - Meta Llama 3.2 models for text generation
- **Qwen 3**: `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `qwen3-32b` - Qwen 3 models for text generation

### Adding Custom Models

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline.model_registry import register_custom_model

# Register custom text classification model
register_custom_model(
    model_name="my-sentiment-model",
    base_model_path="cardiffnlp/twitter-roberta-base-sentiment-latest",
    model_type="text_classification",
    modality="text",
    description="Custom sentiment analysis model"
)

# Register custom generation model  
register_custom_model(
    model_name="my-llama-model",
    base_model_path="meta-llama/Llama-2-7b-hf",
    model_type="generation",
    modality="text",
    description="Custom Llama model"
)

# Use in pipeline
pipeline = DLBacktracePipeline.create_simple("my-sentiment-model")
```

## Pipeline Workflow

1. **Model Loading**: Automatically loads model and tokenizer based on configuration
2. **Input Preprocessing**: Tokenizes and prepares input text
3. **DL-Backtrace Initialization**: Sets up computation graph and tracing
4. **Forward Pass**: Executes model prediction
5. **Relevance Propagation**: Computes relevance values
6. **Result Collection**: Gathers timing, outputs, and relevance data
7. **Result Saving**: Saves results to disk (if configured)
8. **Visualization**: Generates graphs and visualizations (if configured)

## Examples

See `examples/pipeline_example.py` for complete working examples including:

- Basic usage
- Advanced configuration
- Multi-model batch processing
- Custom model registration

## Output Structure

The pipeline generates the following outputs:

```
backtrace_results/
├── bert-sst2_20240101_120000.json
├── bert-sst2_graph.png
└── bert-sst2_relevance/
    ├── graph.png
    └── ...
```

### Result JSON Structure

```json
{
  "model_name": "bert-sst2",
  "model_info": {
    "base_model_path": "...",
    "model_type": "classification",
    "description": "..."
  },
  "inputs": ["..."],
  "input_shape": {...},
  "timing": {
    "total_time": 10.5,
    "predict_time": 5.2,
    "eval_time": 5.3
  },
  "node_io": {...},
  "relevance": {...},
  "timestamp": "..."
}
```

## Advanced Usage

### Custom Model Classes

```python
config = PipelineConfig(
    model_name="custom-model",
    model_kwargs={"custom_param": "value"},
    tokenizer_kwargs={"use_fast": True}
)
```

### Custom Input Handling

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model-path")
inputs = pipeline.prepare_inputs(["Your text here"])
```

### Accessing Internal Components

```python
# Access model
model = pipeline.model

# Access tokenizer
tokenizer = pipeline.tokenizer

# Access DL-Backtrace instance
dlbt = pipeline.dlbt

# Access results
results = pipeline.results
```

## Error Handling

The pipeline includes comprehensive error handling:

- Model loading failures
- Input preprocessing errors
- DL-Backtrace initialization issues
- Execution errors
- Result saving errors

All errors are logged with detailed messages to help diagnose issues.

## Performance Tips

1. **Use CPU for smaller models**: CPU mode is sufficient for most transformer models
2. **Batch processing**: Process multiple inputs in a single batch
3. **Disable visualization**: Set `save_visualization=False` for faster execution
4. **Reduce verbose output**: Set `verbose=False` for batch processing
5. **Tune sequence length**: Reduce `max_length` to improve performance

## Troubleshooting

### Common Issues

1. **Model not found**: Check if model is registered in `ModelRegistry`
2. **CUDA out of memory**: Reduce `batch_size` or switch to CPU
3. **Export errors**: Some models may require custom dynamic shapes
4. **Tokenization errors**: Ensure tokenizer is compatible with model

For more help, see the main DL-Backtrace documentation.

