# DL-Backtrace Pipeline - Summary

## Overview

The DL-Backtrace Pipeline provides a high-level, easy-to-use interface for running explainability analysis on PyTorch models using DL-Backtrace. The pipeline automates model loading, input preprocessing, execution, and result management.

## Architecture

```
DL-Backtrace Pipeline
├── PipelineConfig      # Configuration management
├── ModelRegistry       # Model repository and metadata
├── DLBacktracePipeline # Main pipeline orchestration
└── Integration with DLBacktraceFX
```

## Key Components

### 1. PipelineConfig (`pipeline/config.py`)

Manages all configuration parameters with validation.

**Mandatory Field:**
- `model_name`: Model identifier (must be from ModelRegistry)

**Key Features:**
- Device configuration (CPU/CUDA)
- Batch size and sequence length control
- Relevance propagation parameters
- Temperature scaling
- Output management
- Debug and verbose modes

### 2. ModelRegistry (`pipeline/model_registry.py`)

Pre-configured model database.

**Supported Models:**
- Classification: bert-sst2, roberta-base, distilbert
- Causal LM: llama-3.2-1b, phi-4, gpt2

**Features:**
- Model metadata (type, path, default settings)
- Support for dynamic shapes
- Easy model registration
- Automatic model class selection

### 3. DLBacktracePipeline (`pipeline/pipeline.py`)

Main orchestration class.

**Workflow:**
1. Load model and tokenizer
2. Prepare inputs
3. Initialize DL-Backtrace FX
4. Run forward pass (prediction)
5. Run relevance propagation
6. Collect and save results
7. Generate visualizations (optional)

**Key Methods:**
- `load_model()`: Load model and tokenizer
- `prepare_inputs(text)`: Tokenize input text
- `initialize_dlbt(inputs)`: Set up DL-Backtrace
- `run(text)`: Execute full pipeline
- `print_relevance_summary()`: Display relevance info

### 4. Integration with Core Components

The pipeline integrates with:
- `DLBacktraceFX`: Main execution engine
- `core/graph_builder.py`: Graph construction
- `core/execution_engine_noncache.py`: Forward pass execution
- `core/relevance_propagation.py`: Relevance computation
- `utils/`: Layer implementations

## Usage Patterns

### Pattern 1: Minimal Configuration

```python
config = PipelineConfig(model_name="bert-sst2")
pipeline = DLBacktracePipeline(config)
results = pipeline.run(["Your text here"])
```

### Pattern 2: Custom Configuration

```python
config = PipelineConfig(
    model_name="llama-3.2-1b",
    device="cuda",
    batch_size=2,
    max_length=256,
    multiplier=200.0,
    save_visualization=True
)
pipeline = DLBacktracePipeline(config)
results = pipeline.run(["Long input text here"])
```

### Pattern 3: Batch Processing

```python
models = ["bert-sst2", "roberta-base", "distilbert"]
for model_name in models:
    config = PipelineConfig(model_name=model_name)
    pipeline = DLBacktracePipeline(config)
    results = pipeline.run(["Input text"])
    print(f"Completed {model_name}")
```

### Pattern 4: YAML Configuration

```yaml
# config.yaml
model_name: "bert-sst2"
device: "cpu"
batch_size: 2
max_length: 128
```

```python
pipeline = DLBacktracePipeline.from_yaml("config.yaml")
results = pipeline.run(["Input text"])
```

## File Structure

```
dl_backtrace/pytorch_backtrace/dlbacktrace/
├── pipeline/
│   ├── __init__.py           # Module exports
│   ├── config.py             # Configuration management
│   ├── model_registry.py     # Model registry
│   ├── pipeline.py           # Main pipeline class
│   └── README.md             # Detailed documentation
└── ...

examples/
├── pipeline_example.py        # Comprehensive examples
├── pipeline_config_example.yaml  # Configuration template
└── pipeline_quick_start.py   # Quick start guide
```

## Output Structure

```
backtrace_results/
├── bert-sst2_20240101_120000.json  # Results JSON
├── bert-sst2_graph.png             # Graph visualization
└── bert-sst2_relevance/            # Relevance visualizations
    ├── graph.png
    └── ...
```

## Key Features

### 1. Automatic Model Management
- Loads models from Hugging Face Hub
- Handles different model types automatically
- Manages tokenizers and special tokens

### 2. Flexible Configuration
- YAML configuration support
- Dictionary-based configuration
- Programmatic configuration

### 3. Comprehensive Results
- Timing information (total, prediction, evaluation)
- Node I/O data
- Relevance values
- Input/output shapes
- Model metadata

### 4. Result Management
- Automatic saving to JSON
- Optional visualization generation
- Timestamped outputs
- Organized directory structure

### 5. Error Handling
- Model loading validation
- Input preprocessing checks
- DL-Backtrace initialization handling
- Execution error recovery

## Integration with Core DL-Backtrace

The pipeline wraps the core DL-Backtrace functionality:

```
User Code
    ↓
DLBacktracePipeline
    ↓
DLBacktraceFX (dlbacktrace.py)
    ↓
Core Components:
    - build_graph()          (graph_builder.py)
    - ExecutionEngineNoCache (execution_engine_noncache.py)
    - RelevancePropagator    (relevance_propagation.py)
    - trace_utils           (trace_utils.py)
    ↓
Utils Layer:
    - Layer implementations (utils/cuda_utils/)
```

## Extension Points

### Adding New Models

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline.model_registry import ModelRegistry, ModelInfo

new_model = ModelInfo(
    model_name="my-model",
    model_type="classification",
    base_model_path="path/to/model",
    default_batch_size=1,
    default_max_length=512,
    description="My custom model"
)

ModelRegistry.register_model(new_model)
```

### Custom Pipeline Steps

```python
class CustomPipeline(DLBacktracePipeline):
    def custom_step(self):
        # Your custom logic here
        pass
    
    def run(self, input_text):
        # Call parent run
        results = super().run(input_text)
        # Add custom processing
        results['custom_data'] = self.custom_step()
        return results
```

## Best Practices

1. **Start Simple**: Use basic configuration first
2. **Monitor Performance**: Check timing information
3. **Batch Processing**: Process multiple inputs together
4. **Memory Management**: Use CPU for large models
5. **Error Handling**: Check results for errors
6. **Visualization**: Enable for debugging
7. **Configuration**: Use YAML for complex setups

## Troubleshooting

### Common Issues

1. **Model Not Found**: Register model in ModelRegistry
2. **Memory Errors**: Reduce batch_size or use CPU
3. **Export Failures**: Check model compatibility
4. **Slow Performance**: Tune batch size and sequence length

### Debug Mode

```python
config = PipelineConfig(
    model_name="bert-sst2",
    verbose=True,
    debug=True
)
```

## Future Enhancements

Potential improvements:
- Support for more model types
- Distributed processing
- Real-time visualization
- Interactive debugging tools
- Advanced caching strategies
- Multi-GPU support

## See Also

- `pipeline/README.md` - Detailed pipeline documentation
- `examples/pipeline_example.py` - Working examples
- `examples/pipeline_quick_start.py` - Quick start guide

