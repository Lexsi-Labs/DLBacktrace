# DL-Backtrace Pipeline - Quick Start

## Overview

The DL-Backtrace Pipeline provides a simple, high-level interface for running explainability analysis on PyTorch models. **`model_name` is a mandatory field** that you must provide.

## Structure

```
DL-Backtrace Pipeline
│
├── Core Components (in dlbacktrace/)
│   ├── dlbacktrace.py       # Main DLBacktraceFX class
│   ├── core/                 # Core functionality
│   │   ├── graph_builder.py              # Graph construction
│   │   ├── execution_engine_noncache.py  # Forward pass execution
│   │   ├── relevance_propagation.py      # Relevance calculation
│   │   ├── trace_utils.py                # Trace utilities
│   │   ├── config.py                     # Configuration
│   │   ├── visualization.py             # Visualization
│   │   └── reproducibility.py            # Deterministic execution
│   │
│   └── utils/               # Layer implementations
│       └── cuda_utils/     # CUDA-accelerated layers
│           ├── Linear/      # Linear layer implementations
│           ├── Conv2D/      # Conv2D implementations
│           ├── Embedded/    # Embedding implementations
│           ├── Attention/   # Attention implementations
│           └── ...          # And many more
│
└── Pipeline (in pipeline/)
    ├── pipeline.py         # DLBacktracePipeline class
    ├── config.py           # PipelineConfig with model_name (mandatory)
    ├── model_registry.py   # Model registry
    └── __init__.py         # Exports
```

## Quick Usage

### 1. Simple Python Script

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline, PipelineConfig

# model_name is MANDATORY
config = PipelineConfig(
    model_name="bert-sst2",  # REQUIRED
    device="cpu",
    verbose=True
)

# Create and run pipeline
pipeline = DLBacktracePipeline(config)
results = pipeline.run(["This movie is fantastic!"])

# View results
print(f"Total time: {results['timing']['total_time']}s")
pipeline.print_relevance_summary()
```

### 2. Command Line

```bash
# Basic usage
python examples/run_dlbacktrace_pipeline.py --model-name bert-sst2 --input "This movie is great!"

# With options
python examples/run_dlbacktrace_pipeline.py \
  --model-name bert-sst2 \
  --input "Your text" \
  --device cpu \
  --verbose \
  --save-viz
```

### 3. YAML Configuration

```yaml
# config.yaml
model_name: "bert-sst2"  # MANDATORY
device: "cpu"
batch_size: 2
max_length: 128
verbose: true
```

```python
pipeline = DLBacktracePipeline.from_yaml("config.yaml")
results = pipeline.run(["Your text"])
```

## Mandatory Fields

### model_name (Required)

**The `model_name` parameter is MANDATORY** and must be one of the pre-configured models in the ModelRegistry:

- **Classification Models**:
  - `bert-sst2`: BERT fine-tuned on SST-2 sentiment classification
  - `roberta-base`: RoBERTa base model
  - `distilbert`: DistilBERT model

- **Causal LM Models**:
  - `llama-3.2-1b`: Meta Llama 3.2 1B model
  - `phi-4`: Microsoft Phi-4 model
  - `gpt2`: GPT-2 base model

## Key Files

### Main Pipeline Files

1. **`pipeline.py`**: Main pipeline orchestrator
   - Loads models from ModelRegistry
   - Prepares inputs
   - Executes forward pass
   - Runs relevance propagation
   - Saves results

2. **`config.py`**: Configuration management
   - Validates `model_name` (mandatory)
   - Device configuration
   - Input parameters
   - Evaluation parameters

3. **`model_registry.py`**: Model repository
   - Pre-configured models
   - Model metadata
   - Automatic model class selection

### Core Components

1. **`dlbacktrace.py`**: Main DLBacktraceFX class
   - Model tracing with `torch.export`
   - Forward pass execution
   - Relevance propagation

2. **`graph_builder.py`**: Graph construction
   - Builds computation graph
   - Extracts layer metadata
   - Topological sorting

3. **`execution_engine_noncache.py`**: Forward pass
   - Executes graph
   - Captures activations
   - Stores node I/O

4. **`relevance_propagation.py`**: Relevance calculation
   - Computes relevance scores
   - Propagates backward
   - Generates explanations

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    DL-Backtrace Pipeline                     │
│                                                              │
│  1. Load Model                                               │
│     ├── Get model info from registry (uses model_name)     │
│     ├── Load tokenizer                                      │
│     └── Load model weights                                  │
│                                                              │
│  2. Prepare Inputs                                           │
│     ├── Tokenize input text                                 │
│     ├── Add special tokens                                  │
│     └── Pad/truncate to max_length                          │
│                                                              │
│  3. Initialize DL-Backtrace                                  │
│     ├── Trace model with torch.export                       │
│     ├── Build computation graph                             │
│     ├── Extract weights from graph                          │
│     └── Create execution engine                             │
│                                                              │
│  4. Run Forward Pass                                         │
│     ├── Execute computation graph                           │
│     ├── Capture activations                                 │
│     ├── Store node I/O data                                 │
│     └── Apply temperature scaling                           │
│                                                              │
│  5. Run Relevance Propagation                                │
│     ├── Initialize relevance at output                      │
│     ├── Propagate backward through layers                   │
│     ├── Calculate relevance scores                         │
│     └── Apply conservation rules                            │
│                                                              │
│  6. Collect & Save Results                                  │
│     ├── Gather timing information                           │
│     ├── Serialize node I/O                                  │
│     ├── Serialize relevance                                 │
│     └── Save to JSON                                        │
│                                                              │
│  7. Generate Visualizations (optional)                      │
│     ├── Graph visualization                                │
│     └── Relevance visualization                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Examples

### Basic Example

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    model_name="bert-sst2",  # MANDATORY
    device="cpu",
    verbose=True
)

# Create and run
pipeline = DLBacktracePipeline(config)
results = pipeline.run([
    "This movie is fantastic!",
    "I really didn't enjoy it."
])

# Results automatically saved to 'backtrace_results/'
```

### Advanced Example

```python
config = PipelineConfig(
    model_name="llama-3.2-1b",  # MANDATORY
    device="cuda",
    batch_size=1,
    max_length=256,
    mode="default",
    multiplier=100.0,
    task="generation",
    temperature=1.0,
    save_visualization=True,
    verbose=True,
    debug=True
)

pipeline = DLBacktracePipeline(config)
results = pipeline.run(["Long input text here"])
```

### Adding Custom Models

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline.model_registry import ModelRegistry, ModelInfo

# Register custom model
ModelRegistry.register_model(
    ModelInfo(
        model_name="my-model",
        model_type="classification",
        base_model_path="huggingface/model-path",
        default_batch_size=1,
        default_max_length=512,
        description="My custom model"
    )
)

# Use it
config = PipelineConfig(model_name="my-model")
pipeline = DLBacktracePipeline(config)
results = pipeline.run(["Your text"])
```

## Output

Results are saved to the configured output directory (default: `backtrace_results/`):

```
backtrace_results/
├── bert-sst2_20240101_120000.json
├── bert-sst2_graph.png
└── bert-sst2_relevance/
    ├── graph.png
    └── ...
```

## Documentation

- **Pipeline Usage**: `examples/PIPELINE_USAGE_GUIDE.md`
- **Pipeline Summary**: `examples/PIPELINE_SUMMARY.md`
- **Pipeline README**: `dl_backtrace/pytorch_backtrace/dlbacktrace/pipeline/README.md`
- **Examples**: `examples/pipeline_example.py`, `examples/pipeline_quick_start.py`

## Installation

The pipeline is part of the DL-Backtrace package. No additional installation needed after installing DL-Backtrace.

## Support

For issues or questions:
1. Check the examples in `examples/`
2. See the pipeline documentation
3. Review the main DL-Backtrace documentation

