# DL-Backtrace Pipeline - Complete Guide

## ✅ Pipeline Status: Ready to Use

The DL-Backtrace Pipeline is **fully functional** and ready to use. It provides a high-level interface for running explainability analysis on PyTorch models.

## 📋 Quick Reference

### Mandatory Field
- **`model_name`**: Required parameter that identifies which model to use from the ModelRegistry

### Key Files Location
```
dl_backtrace/pytorch_backtrace/dlbacktrace/
├── pipeline/              # Pipeline implementation
│   ├── pipeline.py        # Main DLBacktracePipeline class
│   ├── config.py          # PipelineConfig (model_name is mandatory)
│   ├── model_registry.py  # Model repository
│   └── README.md          # Detailed documentation
│
├── dlbacktrace.py          # DLBacktraceFX class
│
├── core/                   # Core functionality
│   ├── graph_builder.py
│   ├── execution_engine_noncache.py
│   ├── relevance_propagation.py
│   ├── trace_utils.py
│   ├── config.py
│   ├── visualization.py
│   └── reproducibility.py
│
└── utils/                  # Layer implementations
    └── cuda_utils/        # CUDA-accelerated layers
        ├── Linear/
        ├── Conv2D/
        ├── Embedded/
        ├── Attention/
        └── ... (many more)
```

## 🚀 Usage Examples

### Example 1: Basic Usage

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
results = pipeline.run(["This movie is fantastic!"])

# Results automatically saved to 'backtrace_results/'
print(f"Total time: {results['timing']['total_time']}s")
pipeline.print_relevance_summary()
```

### Example 2: Command Line

```bash
# Simple usage
python examples/run_dlbacktrace_pipeline.py --model-name bert-sst2 --input "This movie is great!"

# With options
python examples/run_dlbacktrace_pipeline.py \
  --model-name llama-3.2-1b \
  --input "Your text here" \
  --device cuda \
  --batch-size 1 \
  --max-length 256 \
  --verbose \
  --save-viz
```

### Example 3: YAML Configuration

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

### Example 4: Advanced Configuration

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline, PipelineConfig

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

## 📚 Available Models

### Pre-configured in ModelRegistry

#### Classification Models
- `bert-sst2`: BERT fine-tuned on SST-2 sentiment classification
- `roberta-base`: RoBERTa base model  
- `distilbert`: DistilBERT model

#### Causal LM Models
- `llama-3.2-1b`: Meta Llama 3.2 1B model
- `phi-4`: Microsoft Phi-4 model
- `gpt2`: GPT-2 base model

### Registering Custom Models

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline.model_registry import ModelRegistry, ModelInfo

# Register a new model
ModelRegistry.register_model(
    ModelInfo(
        model_name="my-custom-model",
        model_type="classification",  # or "causal_lm", "masked_lm", "seq2seq"
        base_model_path="huggingface/model-path",
        supports_dynamic_shapes=False,
        default_batch_size=1,
        default_max_length=512,
        description="My custom model"
    )
)

# Use it
config = PipelineConfig(model_name="my-custom-model")
pipeline = DLBacktracePipeline(config)
results = pipeline.run(["Your text"])
```

## 🔧 Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **`model_name`** | **str** | **Required** | Model identifier (MANDATORY) |
| `device` | str | "cpu" | Device: "cpu" or "cuda" |
| `batch_size` | int | 1 | Batch size |
| `max_length` | int | 512 | Max sequence length |
| `mode` | str | "default" | Relevance mode |
| `multiplier` | float | 100.0 | Starting relevance |
| `scaler` | float | 1.0 | Relevance scaler |
| `thresholding` | float | 0.5 | Threshold |
| `task` | str | "binary-classification" | Task type |
| `temperature` | float | 1.0 | Temperature scaling |
| `save_results` | bool | True | Save results |
| `save_visualization` | bool | False | Generate visualizations |
| `output_dir` | str | "backtrace_results" | Output directory |
| `verbose` | bool | False | Verbose logging |
| `debug` | bool | False | Debug mode |

## 📁 Output Structure

Results are saved to the configured output directory (default: `backtrace_results/`):

```
backtrace_results/
├── bert-sst2_20240101_120000.json       # Results JSON
├── bert-sst2_graph.png                   # Graph visualization
└── bert-sst2_relevance/                  # Relevance visualizations
    ├── graph.png
    └── ...
```

### Result JSON

```json
{
  "model_name": "bert-sst2",
  "model_info": {
    "base_model_path": "textattack/bert-base-uncased-SST-2",
    "model_type": "classification",
    "description": "BERT model fine-tuned on SST-2 sentiment classification"
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
  "node_io": {...},
  "relevance": {...},
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🏗️ Pipeline Architecture

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    DL-Backtrace Pipeline                     │
│                                                              │
│  1. Load Model (using model_name from registry)            │
│     ↓                                                        │
│  2. Prepare Inputs (tokenization)                          │
│     ↓                                                        │
│  3. Initialize DL-Backtrace (graph tracing)                │
│     ↓                                                        │
│  4. Run Forward Pass (prediction)                          │
│     ↓                                                        │
│  5. Run Relevance Propagation (explainability)            │
│     ↓                                                        │
│  6. Collect & Save Results                                 │
│     ↓                                                        │
│  7. Generate Visualizations (optional)                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Integration Points

```
User Code
    ↓
DLBacktracePipeline
    ↓
DLBacktraceFX (dlbacktrace.py)
    ├── Graph Building (graph_builder.py)
    │   ├── trace_utils.py
    │   └── config.py
    │
    ├── Forward Pass (execution_engine_noncache.py)
    │   └── Compute activations
    │
    └── Relevance (relevance_propagation.py)
        ├── utils/cuda_utils/
        └── Generate explanations
```

## 📝 Example Files

All examples are in the `examples/` directory:

1. **`pipeline_quick_start.py`**: Minimal usage example
2. **`pipeline_example.py`**: Comprehensive examples
3. **`run_dlbacktrace_pipeline.py`**: Command-line entry point
4. **`pipeline_config_example.yaml`**: YAML configuration template
5. **`PIPELINE_README.md`**: This guide
6. **`PIPELINE_USAGE_GUIDE.md`**: Detailed usage guide
7. **`PIPELINE_SUMMARY.md`**: Architecture summary

## 🐛 Troubleshooting

### Issue: Model Not Found
```
ValueError: Model 'my-model' not found in registry
```
**Solution**: Use a pre-configured model or register your model:
```python
ModelRegistry.register_model(ModelInfo(...))
```

### Issue: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `batch_size` or use `device="cpu"`:
```python
config = PipelineConfig(model_name="...", device="cpu", batch_size=1)
```

### Issue: Export Failed
```
RuntimeError: Export failed
```
**Solution**: Enable verbose mode for details:
```python
config = PipelineConfig(model_name="...", verbose=True, debug=True)
```

## 🎯 Next Steps

1. **Try the quick start**: `python examples/pipeline_quick_start.py`
2. **Run command line**: `python examples/run_dlbacktrace_pipeline.py --help`
3. **Explore examples**: Read `examples/pipeline_example.py`
4. **Read documentation**: Check `docs/` for detailed guides

## 📚 Additional Resources

- **Pipeline API Docs**: `dl_backtrace/pytorch_backtrace/dlbacktrace/pipeline/README.md`
- **Core Documentation**: `docs/`
- **Examples**: `examples/`
- **Developer Notes**: `dev_notes/`

## ✨ Key Features

1. **Simple API**: Just provide `model_name` and run
2. **Automatic Model Loading**: From Hugging Face Hub
3. **Flexible Configuration**: YAML, dict, or programmatic
4. **Comprehensive Results**: Timing, relevance, visualizations
5. **Error Handling**: Automatic recovery strategies
6. **Extensible**: Easy to add custom models and features

## 🔑 Key Points

1. **`model_name` is mandatory** - You must specify which model to use
2. **Pipeline is ready** - Fully functional and tested
3. **Core components work** - Graph building, execution, relevance propagation
4. **Utils layer complete** - All layer implementations available
5. **Easy to use** - Simple API with comprehensive examples

---

**The DL-Backtrace Pipeline is complete and ready to use!** 🎉

Start with: `python examples/pipeline_quick_start.py`

