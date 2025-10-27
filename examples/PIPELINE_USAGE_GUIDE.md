# DL-Backtrace Pipeline Usage Guide

## Overview

The DL-Backtrace Pipeline provides a simple, high-level interface for running explainability analysis on PyTorch models. It automates the entire workflow from model loading to result saving.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DL-Backtrace Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ PipelineConfig│  │  ModelRegistry│  │ DLBacktraceFX │    │
│  │ (Mandatory)   │→ │  (Model Info) │→ │ (Core Engine) │     │
│  │ model_name    │  │  base_path    │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│                        Main Workflow:                        │
│                                                              │
│  1. Load Model & Tokenizer                                   │
│  2. Prepare Inputs (Tokenization)                            │
│  3. Initialize DL-Backtrace (Graph Tracing)                 │
│  4. Run Forward Pass (Prediction)                           │
│  5. Run Relevance Propagation                               │
│  6. Collect & Save Results                                  │
│  7. Generate Visualizations (Optional)                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. PipelineConfig
- **Location**: `dl_backtrace/pytorch_backtrace/dlbacktrace/pipeline/config.py`
- **Purpose**: Configuration management with validation
- **Mandatory Field**: `model_name` (string)
- **Key Parameters**:
  - `model_name`: Model identifier from registry (REQUIRED)
  - `device`: "cpu" or "cuda"
  - `batch_size`: Batch size for processing
  - `max_length`: Maximum sequence length
  - `mode`: Relevance propagation mode
  - `multiplier`: Starting relevance value
  - `save_results`: Whether to save results
  - `save_visualization`: Whether to generate visualizations

### 2. ModelRegistry
- **Location**: `dl_backtrace/pytorch_backtrace/dlbacktrace/pipeline/model_registry.py`
- **Purpose**: Repository of supported models
- **Pre-configured Models**:
  - `bert-sst2`: BERT fine-tuned on SST-2
  - `roberta-base`: RoBERTa base model
  - `distilbert`: DistilBERT model
  - `llama-3.2-1b`: Meta Llama 3.2 1B
  - `phi-4`: Microsoft Phi-4
  - `gpt2`: GPT-2 base model

### 3. DLBacktracePipeline
- **Location**: `dl_backtrace/pytorch_backtrace/dlbacktrace/pipeline/pipeline.py`
- **Purpose**: Main orchestration class
- **Key Methods**:
  - `load_model()`: Load model and tokenizer
  - `prepare_inputs(text)`: Tokenize input text
  - `run(text)`: Execute full pipeline
  - `print_relevance_summary()`: Display relevance information

### 4. Integration with Core Components
- **DLBacktraceFX** (`dlbacktrace.py`): Main execution engine
- **Graph Builder** (`core/graph_builder.py`): Graph construction
- **Execution Engine** (`core/execution_engine_noncache.py`): Forward pass
- **Relevance Propagation** (`core/relevance_propagation.py`): Relevance computation
- **Utils** (`utils/`): Layer implementations (Linear, Conv2D, Attention, etc.)

## Quick Start

### Method 1: Command Line

```bash
# Basic usage
python examples/run_dlbacktrace_pipeline.py --model-name bert-sst2 --input "This movie is great!"

# With custom settings
python examples/run_dlbacktrace_pipeline.py \
  --model-name bert-sst2 \
  --input "Your text here" \
  --device cpu \
  --batch-size 1 \
  --max-length 128 \
  --mode default \
  --multiplier 100.0 \
  --verbose
```

### Method 2: Python Script

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline, PipelineConfig

# Create configuration (model_name is mandatory)
config = PipelineConfig(
    model_name="bert-sst2",  # REQUIRED
    device="cpu",
    verbose=True
)

# Create and run pipeline
pipeline = DLBacktracePipeline(config)
results = pipeline.run(["This movie is fantastic!", "I didn't like it."])

# View results
print(f"Total time: {results['timing']['total_time']}s")
pipeline.print_relevance_summary()
```

### Method 3: YAML Configuration

```yaml
# config.yaml
model_name: "bert-sst2"  # MANDATORY
device: "cpu"
batch_size: 2
max_length: 128
verbose: true
```

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline

pipeline = DLBacktracePipeline.from_yaml("config.yaml")
results = pipeline.run(["Your text here"])
```

## Detailed Usage

### Basic Configuration

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline, PipelineConfig

# Minimal configuration
config = PipelineConfig(
    model_name="bert-sst2",  # MANDATORY
    device="cpu",
    verbose=True
)

pipeline = DLBacktracePipeline(config)
results = pipeline.run(["Input text"])
```

### Advanced Configuration

```python
config = PipelineConfig(
    model_name="llama-3.2-1b",  # MANDATORY
    device="cuda",
    batch_size=1,
    max_length=256,
    mode="default",
    multiplier=100.0,
    scaler=1.0,
    thresholding=0.5,
    task="generation",
    temperature=1.0,
    save_results=True,
    save_visualization=True,
    output_dir="results",
    verbose=True,
    debug=True
)

pipeline = DLBacktracePipeline(config)
results = pipeline.run(["Long input text here"])
```

### Batch Processing

```python
models = ["bert-sst2", "roberta-base", "distilbert"]

for model_name in models:
    config = PipelineConfig(
        model_name=model_name,
        device="cpu",
        verbose=False
    )
    
    pipeline = DLBacktracePipeline(config)
    results = pipeline.run(["Test input"])
    
    print(f"{model_name}: {results['timing']['total_time']}s")
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | **Required** | Model identifier (MANDATORY) |
| `device` | str | "cpu" | Device: "cpu" or "cuda" |
| `batch_size` | int | 1 | Batch size for processing |
| `max_length` | int | 512 | Maximum sequence length |
| `mode` | str | "default" | Relevance propagation mode |
| `multiplier` | float | 100.0 | Starting relevance value |
| `scaler` | float | 1.0 | Relevance scaler |
| `thresholding` | float | 0.5 | Relevance threshold |
| `task` | str | "binary-classification" | Task type |
| `temperature` | float | 1.0 | Temperature scaling |
| `save_results` | bool | True | Save results to disk |
| `save_visualization` | bool | False | Generate visualizations |
| `output_dir` | str | "backtrace_results" | Output directory |
| `verbose` | bool | False | Enable verbose logging |
| `debug` | bool | False | Enable debug mode |
| `strict_cpu` | bool | True | Strict CPU determinism |

## Output Structure

```
backtrace_results/
├── bert-sst2_20240101_120000.json     # Results JSON
├── bert-sst2_graph.png                 # Graph visualization
└── bert-sst2_relevance/              # Relevance visualizations
    ├── graph.png
    └── ...
```

### Result JSON Structure

```json
{
  "model_name": "bert-sst2",
  "model_info": {
    "base_model_path": "textattack/bert-base-uncased-SST-2",
    "model_type": "classification",
    "description": "BERT model fine-tuned on SST-2"
  },
  "inputs": ["This movie is fantastic!"],
  "input_shape": {
    "input_ids": [2, 128],
    "attention_mask": [2, 128]
  },
  "timing": {
    "total_time": 10.5,
    "predict_time": 5.2,
    "eval_time": 5.3
  },
  "node_io": {
    "output": {
      "layer_type": "Output",
      "output_shape": "torch.Size([2, 2])"
    }
  },
  "relevance": {
    "output": "shape: (2, 2), sum: 0.0000"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

## Integration with Core Components

### How the Pipeline Uses Core Components

```
User Input
    ↓
DLBacktracePipeline
    ↓
DLBacktraceFX (dlbacktrace.py)
    ├── Graph Building (graph_builder.py)
    │   ├── Extract placeholders (trace_utils.py)
    │   ├── Map placeholders to weights (trace_utils.py)
    │   └── Build computation graph (graph_builder.py)
    │
    ├── Forward Pass (execution_engine_noncache.py)
    │   ├── Execute graph
    │   ├── Capture activations
    │   └── Store node I/O
    │
    └── Relevance Propagation (relevance_propagation.py)
        ├── Compute relevance scores
        ├── Propagate backward
        └── Generate explanation
    ↓
Results
```

### Layer Implementations

The pipeline supports various layer implementations from `utils/cuda_utils/`:
- **Linear**: CUDA-accelerated linear layers
- **Embedding**: CUDA-accelerated embeddings
- **Attention**: CUDA-accelerated attention mechanisms
- **Conv2D**: CUDA-accelerated convolutions
- **MaxPool2D**: CUDA-accelerated pooling
- And many more...

## Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   ValueError: Model 'my-model' not found in registry
   ```
   **Solution**: Use a model from ModelRegistry or register your model

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce `batch_size` or use `device="cpu"`

3. **Model Export Failed**
   ```
   RuntimeError: Export failed
   ```
   **Solution**: Check model compatibility or set `verbose=True` for details

### Debug Mode

Enable detailed debugging:

```python
config = PipelineConfig(
    model_name="bert-sst2",
    verbose=True,
    debug=True
)
```

## Adding Custom Models

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline.model_registry import ModelRegistry, ModelInfo

# Register a new model
new_model = ModelInfo(
    model_name="my-custom-model",
    model_type="classification",
    base_model_path="huggingface/path/to/model",
    default_batch_size=1,
    default_max_length=512,
    description="My custom model"
)

ModelRegistry.register_model(new_model)

# Use in pipeline
config = PipelineConfig(model_name="my-custom-model")
pipeline = DLBacktracePipeline(config)
```

## Best Practices

1. **Start with small models**: Use CPU for development
2. **Use appropriate batch sizes**: Balance speed vs. memory
3. **Enable verbose mode**: For debugging and monitoring
4. **Save results**: For post-analysis and visualization
5. **Batch process efficiently**: Group multiple inputs together
6. **Monitor timing**: Check performance metrics
7. **Use appropriate task types**: Match your use case

## Examples

See `examples/` directory for complete examples:
- `pipeline_quick_start.py`: Minimal usage
- `pipeline_example.py`: Comprehensive examples
- `pipeline_config_example.yaml`: YAML configuration
- `run_dlbacktrace_pipeline.py`: Command-line entry point

## References

- Main DL-Backtrace documentation: `docs/`
- Pipeline API: `dl_backtrace/pytorch_backtrace/dlbacktrace/pipeline/`
- Core components: `dl_backtrace/pytorch_backtrace/dlbacktrace/core/`
- Layer implementations: `dl_backtrace/pytorch_backtrace/dlbacktrace/utils/`

