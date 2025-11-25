# DLBacktrace Pipeline

The **DLBacktrace Pipeline** provides a unified, high-level interface for running explainability analysis on PyTorch models through the `run_task()` method. It handles model execution, relevance propagation, and result management automatically in a single call.

---

## Overview

The `run_task()` method simplifies the DLBacktrace workflow by providing:

- **🎯 Unified Interface**: Single method for all task types (classification, generation)
- **🔧 Automatic Task Detection**: Auto-detect task type from inputs
- **📊 Built-in Relevance Analysis**: Automatic layer-wise relevance propagation
- **🚀 Generation Support**: Greedy, sampling, and beam search with tracing
- **⚙️ Flexible Configuration**: Comprehensive parameters for all use cases
- **💾 Structured Results**: Easy access to predictions, relevance, and traces

---

## Quick Start

### Basic Usage

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

# Prepare input
text = "This product is amazing!"
tokens = tokenizer(text, return_tensors="pt")

# Initialize DLBacktrace
dlb = DLBacktrace(
    model=model,
    input_for_graph=(tokens["input_ids"], tokens["attention_mask"]),
    device="cuda"  # or "cpu"
)

# Run analysis with run_task()
results = dlb.run_task(
    task="text-classification",  # or "auto" for automatic detection
    inputs={'input_ids': tokens["input_ids"], 'attention_mask': tokens["attention_mask"]}
)

print(f"Prediction: {results['predictions'].argmax()}")
print(f"Relevance computed: {results['relevance_computed']}")
```

---

## Supported Tasks

### 1. Text Classification

Analyze sentiment, intent, or any text classification task with transformer models.

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

# Prepare input
text = "This movie is fantastic!"
tokens = tokenizer(text, return_tensors="pt")

# Initialize DLBacktrace
dlb = DLBacktrace(
    model=model,
    input_for_graph=(tokens["input_ids"], tokens["attention_mask"]),
    device="cuda"
)

# Run text classification
results = dlb.run_task(
    task="text-classification",
    inputs={'input_ids': tokens["input_ids"], 'attention_mask': tokens["attention_mask"]},
    mode="default",
    multiplier=100.0,
    debug=False
)

# Access results
print(f"Predicted class: {results['predictions'].argmax()}")
print(f"Relevance keys: {list(results['relevance'].keys())}")

# Get token relevance
token_relevance = results['relevance']['input_ids']
print(f"Token relevance shape: {token_relevance.shape}")
```

**Supported Models:**
- BERT (all variants)
- RoBERTa
- ALBERT
- ELECTRA
- XLNet
- DistilBERT
- And any HuggingFace sequence classification model

### 2. Image Classification

Analyze image classifications with CNN or Vision Transformer models.

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn
import torch

# Load MobileNet model
class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

model = MobileNetClassifier(num_classes=10)
model.eval()

# Prepare input (batch_size=1, channels=3, height=224, width=224)
image_tensor = torch.randn(1, 3, 224, 224)

# Initialize DLBacktrace
dlb = DLBacktrace(
    model=model,
    input_for_graph=image_tensor,
    device="cuda"
)

# Run image classification
results = dlb.run_task(
    task="image-classification",
    inputs=image_tensor,
    debug=False
)

# Access predictions and relevance
print(f"Predicted class: {results['predictions'].argmax()}")
print(f"Input relevance shape: {results['relevance']['x'].shape}")
```

**Supported Models:**
- ResNet (all variants)
- VGG (all variants)
- Vision Transformer (ViT)
- DenseNet
- EfficientNet
- MobileNet-V2
- And any TorchVision or custom CNN model

### 3. Text Generation

Generate text with relevance analysis for LLMs and MoEs models.

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Prepare input
prompt = "The future of artificial intelligence is"
tokens = tokenizer(prompt, return_tensors="pt")

# Initialize DLBacktrace
dlb = DLBacktrace(
    model=model,
    input_for_graph=(tokens["input_ids"], tokens["attention_mask"]),
    device="cuda"  # Recommended for generation
)

# Run text generation with relevance tracing
results = dlb.run_task(
    task="generation",
    inputs={'input_ids': tokens["input_ids"], 'attention_mask': tokens["attention_mask"]},
    tokenizer=tokenizer,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    return_relevance=True,
    return_scores=True,
    debug=False
)

# Access generated text and traces
generated_text = tokenizer.decode(results['generated_ids'][0], skip_special_tokens=True)
print(f"Generated: {generated_text}")
print(f"Relevance trace steps: {len(results['relevance_trace'])}")
print(f"Scores trace steps: {len(results['scores_trace'])}")
```

**Supported Models:**
- GPT-2 (all sizes)
- Llama (all versions)
- Qwen (all versions)
- Mistral
- And any HuggingFace causal LM model
- **MoEs Models**: JetMoE, OLMoE, Qwen-MoE, GPT-Oss (see [MoEs Support](moe-models.md))

---

## Configuration

### `run_task()` Parameters

The `run_task()` method accepts comprehensive parameters for all use cases.

```python
results = dlb.run_task(
    task="auto",                    # Task type
    inputs=None,                    # Input data
    tokenizer=None,                 # Required for generation
    mode="default",                 # Relevance mode
    multiplier=100.0,               # Relevance multiplier
    scaler=1.0,                     # Relevance scaler
    thresholding=0.5,               # Relevance threshold
    temperature=1.0,                # Temperature for predict()
    return_scores=False,            # Return scores trace (generation)
    return_relevance=False,         # Return relevance trace (generation)
    return_layerwise_output=False,  # Return layer outputs (generation)
    debug=False,                    # Enable debug logging
    **generation_kwargs             # Additional generation parameters
)
```

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | str | `"auto"` | Task type: `"auto"`, `"text-classification"`, `"image-classification"`, or `"generation"` |
| `inputs` | dict/tuple | **Required** | Input data (dict with keys or tuple of tensors) |
| `tokenizer` | Tokenizer | `None` | HuggingFace tokenizer (required for generation) |
| `debug` | bool | `False` | Enable debug logging |

### Relevance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | `"default"` | Relevance propagation mode |
| `multiplier` | float | `100.0` | Starting relevance value |
| `scaler` | float | `1.0` | Relevance scaling factor |
| `thresholding` | float | `0.5` | Relevance threshold |
| `temperature` | float | `1.0` | Temperature for predict() call (logit scaling) |

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_new_tokens` | int | `50` | Maximum tokens to generate |
| `temperature` | float | `1.0` | Sampling temperature (in generation_kwargs) |
| `top_k` | int | `50` | Top-k sampling |
| `top_p` | float | `0.9` | Nucleus sampling threshold |
| `num_beams` | int | `1` | Beam search width (>1 enables beam search) |
| `length_penalty` | float | `1.0` | Beam search length penalty |
| `early_stopping` | bool | `False` | Stop when all beams finish |
| `return_scores` | bool | `False` | Return scores trace per generation step |
| `return_relevance` | bool | `False` | Return relevance trace per generation step |
| `return_layerwise_output` | bool | `False` | Return layer outputs per generation step |

### Device Configuration

Device is configured during `DLBacktrace` initialization:

| Device | Description |
|--------|-------------|
| `"cpu"` | Use optimized CPU implementations for all layers |
| `"cuda"` | Use CUDA-accelerated implementations where available (Linear, Embedding, Attention, Conv2D, etc.) |

---

## Advanced Usage

### Auto Task Detection

Let DLBacktrace automatically detect the task type:

```python
# For text classification
results = dlb.run_task(
    task="auto",  # Automatically detects text-classification
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask}
)

# For image classification
results = dlb.run_task(
    task="auto",  # Automatically detects image-classification
    inputs=image_tensor
)

# For generation (requires tokenizer)
results = dlb.run_task(
    task="auto",  # Automatically detects generation
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
    tokenizer=tokenizer,
    max_new_tokens=50
)
```

### Generation with Tracing

Track relevance, scores, and layer outputs during generation:

```python
results = dlb.run_task(
    task="generation",
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
    tokenizer=tokenizer,
    max_new_tokens=20,
    return_relevance=True,      # Track relevance per step
    return_scores=True,          # Track logits per step
    return_layerwise_output=True # Track layer outputs per step
)

# Access traces
for step_idx, relevance_data in enumerate(results['relevance_trace']):
    print(f"Step {step_idx}: {len(relevance_data)} nodes with relevance")

for step_idx, scores in enumerate(results['scores_trace']):
    print(f"Step {step_idx} logits shape: {scores.shape}")
```

### Custom Relevance Analysis

Fine-tune relevance propagation parameters:

```python
results = dlb.run_task(
    task="image-classification",
    inputs=image_tensor,
    mode="default",
    multiplier=200.0,    # Increase relevance magnitude
    scaler=1.5,          # Scale relevance values
    thresholding=0.3,    # Lower threshold for more relevance
    debug=True           # Show detailed logs
)

# Access layer-specific relevance
for node_name, relevance_values in results['relevance'].items():
    print(f"{node_name}: {relevance_values.shape}")
```

### Beam Search Generation

Use beam search for better generation quality:

```python
results = dlb.run_task(
    task="generation",
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
    tokenizer=tokenizer,
    max_new_tokens=50,
    num_beams=4,              # Use 4 beams
    length_penalty=1.0,       # Neutral length penalty
    early_stopping=True,      # Stop when all beams finish
    return_relevance=True
)

generated_text = tokenizer.decode(results['generated_ids'][0], skip_special_tokens=True)
print(f"Generated (beam search): {generated_text}")
```

---

## Result Structure

### Classification Tasks

For `text-classification` and `image-classification`:

```python
{
    'task': str,                    # Task type ("text-classification" or "image-classification")
    'node_io': dict,                # Layer-wise outputs from predict()
    'relevance': dict,              # Layer-wise relevance values
    'predictions': np.ndarray,      # Model predictions (logits)
    'scores_trace': list,           # (Optional) Single-step scores
    'relevance_trace': list,        # (Optional) Single-step relevance
    'layerwise_output_trace': list  # (Optional) Single-step layer outputs
}
```

**Example:**
```python
results = dlb.run_task(task="text-classification", inputs=tokens)

# Access predictions
predicted_class = results['predictions'].argmax()

# Access token relevance
token_relevance = results['relevance']['input_ids']

# Access layer outputs
layer_output = results['node_io']['decoder_feed_forward_0']['output_values']
```

### Generation Tasks

For `generation` task:

```python
{
    'task': str,                    # "generation"
    'node_io': dict,                # Layer-wise outputs from last step
    'relevance': dict,              # Layer-wise relevance from last step
    'generated_ids': torch.Tensor,  # Generated token IDs (shape: [1, total_length])
    'scores_trace': list,           # (Optional) Logits per generation step
    'relevance_trace': list,        # (Optional) Relevance per generation step
    'layerwise_output_trace': list  # (Optional) Layer outputs per generation step
}
```

**Example:**
```python
results = dlb.run_task(
    task="generation",
    inputs=tokens,
    tokenizer=tokenizer,
    max_new_tokens=10,
    return_relevance=True,
    return_scores=True
)

# Decode generated text
generated_text = tokenizer.decode(results['generated_ids'][0], skip_special_tokens=True)

# Access per-step relevance
for step_idx, step_data in enumerate(results['relevance_trace']):
    token_rel = step_data['input_ids']  # Token relevance at this step
    print(f"Step {step_idx}: Token relevance shape {token_rel.shape}")

# Access per-step scores (logits)
for step_idx, logits in enumerate(results['scores_trace']):
    print(f"Step {step_idx}: Logits shape {logits.shape}")
```

---

## Visualization

DLBacktrace provides built-in visualization methods:

### Visualize Computation Graph

```python
# Visualize the full computation graph
dlb.visualize(save_path="graph.png")

# Visualize with relevance overlay
dlb.visualize_dlbacktrace(
    output_path="backtrace_graph",
    engine_auto_threshold=1500
)
```

### Visualize Token Relevance (Generation)

For generation tasks, visualize token-wise relevance:

```python
# Run generation with relevance tracing
results = dlb.run_task(
    task="generation",
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
    tokenizer=tokenizer,
    max_new_tokens=10,
    return_relevance=True
)

# Visualize token relevance map (heatmap)
dlb.visualize_tokenwise_relevance_map(
    timewise_relevance_out=results['relevance_trace'],
    input_ids=input_ids,
    tokenizer=tokenizer,
    generated_ids=results['generated_ids'],
    figsize=(12, 6)
)

# Visualize relevance for specific generated token
dlb.visualize_input_heatmap_for_token(
    timewise_relevance_out=results['relevance_trace'],
    n=0,  # First generated token
    input_ids=input_ids,
    tokenizer=tokenizer,
    generated_ids=results['generated_ids'],
    figsize=(10, 3)
)
```

---

## Best Practices

### 1. Device Selection

Configure device during initialization for optimal performance:

```python
# CPU mode: optimized CPU implementations
dlb = DLBacktrace(model, input_for_graph, device="cpu")

# CUDA mode: GPU-accelerated implementations
dlb = DLBacktrace(model, input_for_graph, device="cuda")
```

- Use `device="cuda"` for faster execution with GPU acceleration
- CUDA implementations available for: Linear, Embedding, Attention, Conv2D layers
- For large models or generation tasks, CUDA is strongly recommended

### 2. Generation Best Practices

For text generation tasks:

```python
# Use appropriate max_new_tokens
results = dlb.run_task(
    task="generation",
    inputs=tokens,
    tokenizer=tokenizer,
    max_new_tokens=50,  # Adjust based on use case
    temperature=0.7,    # Lower for more focused output
    top_p=0.9,          # Nucleus sampling
    debug=False         # Enable for troubleshooting
)
```

- Start with smaller `max_new_tokens` for faster iteration
- Use `debug=True` to see generation progress
- Enable tracing only when needed (adds overhead)

### 3. Memory Management

For large models or long sequences:

```python
import torch

# Clear CUDA cache between runs
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Process shorter sequences
tokens = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
```

### 4. Relevance Analysis Tuning

Adjust relevance parameters for better insights:

```python
# For stronger relevance signals
results = dlb.run_task(
    task="text-classification",
    inputs=tokens,
    multiplier=200.0,    # Increase from default 100.0
    scaler=1.5,          # Scale up relevance
    thresholding=0.3     # Lower threshold
)

# For more conservative relevance
results = dlb.run_task(
    task="text-classification",
    inputs=tokens,
    multiplier=50.0,     # Decrease from default
    thresholding=0.7     # Higher threshold
)
```

---

## Troubleshooting

### Model Export Issues

If model export fails during initialization:

```python
# Enable verbose mode to see export details
dlb = DLBacktrace(
    model=model,
    input_for_graph=sample_input,
    device="cuda",
    verbose=True  # Shows export strategies
)

# Check model compatibility
dlb.diagnose_model_compatibility()
```

### Memory Errors

- Reduce sequence length for text models
- Use `device="cpu"` if GPU memory is limited
- Clear CUDA cache: `torch.cuda.empty_cache()`
- For generation, reduce `max_new_tokens`

### Generation Issues

If generation produces unexpected results:

```python
# Enable debug mode
results = dlb.run_task(
    task="generation",
    inputs=tokens,
    tokenizer=tokenizer,
    max_new_tokens=10,
    debug=True  # Shows step-by-step progress
)

# Check execution differences
dlb.debug_execution_differences(input_ids, attention_mask)
```

### Performance Optimization

- Use CUDA when available: `device="cuda"` in initialization
- Use greedy decoding (default) instead of beam search for speed
- Enable `debug=False` to reduce logging overhead

### Task Detection Issues

If auto-detection fails, specify task explicitly:

```python
# Instead of task="auto"
results = dlb.run_task(
    task="text-classification",  # Explicit task type
    inputs=tokens
)
```

---

## Complete Examples

### Example 1: BERT Sentiment Analysis

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn

# Load model
class BERTSentiment(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
    
    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

model = BERTSentiment("textattack/bert-base-uncased-SST-2")
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

# Prepare input
text = "This movie is fantastic!"
tokens = tokenizer(text, return_tensors="pt")

# Initialize and run
dlb = DLBacktrace(model, (tokens["input_ids"], tokens["attention_mask"]), device="cuda")
results = dlb.run_task(
    task="text-classification",
    inputs={'input_ids': tokens["input_ids"], 'attention_mask': tokens["attention_mask"]}
)

# Results
print(f"Predicted: {results['predictions'].argmax()}")
print(f"Token relevance: {results['relevance']['input_ids'].shape}")
```

### Example 2: MobileNet Image Classification

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace
from torchvision.models import mobilenet_v2
import torch

# Load model
model = mobilenet_v2(pretrained=True)
model.eval()

# Prepare input
image = torch.randn(1, 3, 224, 224)

# Initialize and run
dlb = DLBacktrace(model, image, device="cuda")
results = dlb.run_task(task="image-classification", inputs=image)

# Results
print(f"Predicted class: {results['predictions'].argmax()}")
print(f"Input relevance: {results['relevance']['x'].shape}")
```

### Example 3: GPT-2 Text Generation with Tracing

```python
from dl_backtrace.pytorch_backtrace import DLBacktrace
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Prepare input
prompt = "The future of AI is"
tokens = tokenizer(prompt, return_tensors="pt")

# Initialize and run
dlb = DLBacktrace(model, (tokens["input_ids"],), device="cuda")
results = dlb.run_task(
    task="generation",
    inputs={'input_ids': tokens["input_ids"]},
    tokenizer=tokenizer,
    max_new_tokens=20,
    temperature=0.7,
    return_relevance=True,
    return_scores=True
)

# Results
generated = tokenizer.decode(results['generated_ids'][0], skip_special_tokens=True)
print(f"Generated: {generated}")
print(f"Relevance trace: {len(results['relevance_trace'])} steps")
```

---

## Next Steps

- Learn about [MoEs Model Support](moe-models.md) for JetMoE, OLMoE, Qwen-MoE, GPT-Oss 
- Explore [DLB Auto Sampler](auto-sampler.md) for advanced text generation
- Check [Temperature Scaling](temperature-scaling.md) for controlled generation
- See [Examples](../../examples/colab-notebooks.md) for complete notebooks

