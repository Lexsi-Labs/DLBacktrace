# DL-Backtrace Pipeline

A high-level pipeline interface for running DL-Backtrace on PyTorch models with automated model loading, execution, and result management.

## Features

- **Automatic Model Loading**: Supports various model types (classification, causal LM, masked LM)
- **Configurable Execution**: Customize device, batch size, input length, and more
- **Result Management**: Automatic saving of results, timing, and relevance values
- **Visualization Support**: Optional graph and relevance visualization
- **Batch Processing**: Process multiple models or inputs efficiently

## Quick Start

### Basic Usage

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    model_name="bert-sst2",  # Mandatory field
    device="cpu",
    verbose=True
)

# Create and run pipeline
pipeline = DLBacktracePipeline(config)
results = pipeline.run(["This movie is fantastic!", "I didn't like it."])
```

### Using Configuration Dictionary

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline, PipelineConfig

config_dict = {
    "model_name": "bert-sst2",
    "device": "cpu",
    "batch_size": 2,
    "max_length": 128,
    "verbose": True
}

pipeline = DLBacktracePipeline.from_config_dict(config_dict)
results = pipeline.run(["Your input text here"])
```

### Using YAML Configuration

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline

# Create pipeline from YAML config file
pipeline = DLBacktracePipeline.from_yaml("config.yaml")
results = pipeline.run(["Your input text here"])
```

## Configuration

### PipelineConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | **Required** | Model identifier from registry |
| `device` | str | "cpu" | Device to use: "cpu" or "cuda" |
| `batch_size` | int | 1 | Batch size for processing |
| `max_length` | int | 512 | Maximum sequence length |
| `mode` | str | "default" | Relevance propagation mode |
| `multiplier` | float | 100.0 | Relevance multiplier |
| `scaler` | float | 1.0 | Relevance scaler |
| `thresholding` | float | 0.5 | Relevance threshold |
| `task` | str | "binary-classification" | Task type |
| `temperature` | float | 1.0 | Temperature scaling for logits |
| `save_results` | bool | True | Save results to disk |
| `save_visualization` | bool | False | Generate visualizations |
| `output_dir` | str | "backtrace_results" | Output directory |
| `verbose` | bool | False | Enable verbose logging |
| `debug` | bool | False | Enable debug mode |

## Supported Models

The pipeline includes a model registry with pre-configured models:

- **Classification Models**: bert-sst2, roberta-base, distilbert
- **Causal LM Models**: llama-3.2-1b, phi-4, gpt2

### Adding Custom Models

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline.model_registry import ModelRegistry, ModelInfo

# Register a new model
new_model = ModelInfo(
    model_name="my-custom-model",
    model_type="classification",
    base_model_path="path/to/model",
    default_batch_size=1,
    default_max_length=512,
    description="My custom model"
)

ModelRegistry.register_model(new_model)

# Use in pipeline
config = PipelineConfig(model_name="my-custom-model")
pipeline = DLBacktracePipeline(config)
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

