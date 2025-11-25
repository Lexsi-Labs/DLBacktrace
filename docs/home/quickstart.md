# Quick Start

Get started with DLBacktrace in minutes! This guide will walk you through your first explainability analysis.

---

## Prerequisites

Make sure you have DLBacktrace installed. If not, see the [Installation Guide](installation.md).

---

## Your First Example

Let's start with a simple PyTorch model and make it explainable.

### Step 1: Import and Define Model

```python
import torch
import torch.nn as nn
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)

# Create model instance
model = SimpleCNN(num_classes=10)
model.eval()  # Set to evaluation mode
```

### Step 2: Initialize DLBacktrace

```python
# Create dummy input for graph tracing
dummy_input = torch.randn(1, 3, 32, 32)

# Initialize DLBacktrace
dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,),
    device="cuda"
)

print("✅ DLBacktrace initialized successfully!")
```

### Step 3: Run Forward Pass

```python
# Prepare your actual input
test_input = torch.randn(1, 3, 32, 32)

# Get layer-wise outputs
node_io = dlb.predict(test_input)

print(f"✅ Traced {len(node_io)} nodes in the computational graph")
```

### Step 4: Calculate Relevance

```python
# Calculate relevance propagation
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="multi-class classification"
)

print("✅ Relevance propagation completed!")
print(f"Number of nodes with relevance: {len(relevance)}")
```

### Step 5: Visualize (Optional)

```python
# Visualize the full computational graph
dlb.visualize()

# Visualize top 10 most relevant nodes
dlb.visualize_dlbacktrace(top_k=10)

print("✅ Visualizations saved!")
```

---

## Complete Example

Here's the complete code in one block:

```python
import torch
import torch.nn as nn
from dl_backtrace.pytorch_backtrace import DLBacktrace

# 1. Define Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)

# 2. Initialize
model = SimpleCNN(num_classes=10)
model.eval()

dummy_input = torch.randn(1, 3, 32, 32)
dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,),
    device="cpu"
)

# 3. Run Analysis
test_input = torch.randn(1, 3, 32, 32)
node_io = dlb.predict(test_input)
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="multi-class classification"
)

# 4. Visualize
dlb.visualize()
dlb.visualize_dlbacktrace(top_k=10)

print("✅ Analysis complete!")
```

---

## Working with Pre-trained Models

### ResNet Example

```python
import torch
import torchvision.models as models
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Load pre-trained ResNet
model = models.resnet18(pretrained=True)
model.eval()

# Initialize DLBacktrace
dummy_input = torch.randn(1, 3, 224, 224)
dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,),
    device="cuda"
)

# Load and preprocess image
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

image = Image.open('cat.jpg')
input_tensor = transform(image).unsqueeze(0)

# Analyze
node_io = dlb.predict(input_tensor)
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="multi-class classification"
)

# Visualize
dlb.visualize_dlbacktrace(top_k=20)
```

### BERT Example

```python
import torch
from transformers import AutoTokenizer, AutoModel
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Load pre-trained BERT
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Prepare input
text = "DLBacktrace makes AI explainable!"
inputs = tokenizer(text, return_tensors="pt", padding=True)

# Initialize DLBacktrace
dlb = DLBacktrace(
    model=model,
    input_for_graph=(inputs['input_ids'], inputs['attention_mask']),
    device="cuda"
)

# Analyze
node_io = dlb.predict(inputs['input_ids'], inputs['attention_mask'])
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="multi-class classification"
)

print("✅ BERT analysis complete!")
```

---

## Simplified `run_task()` API

For even faster setup, use the streamlined **`run_task()`** method that combines prediction and relevance evaluation in a single call:

### Text Classification with `run_task()`

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-SST-2"
).eval()
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

# Tokenize input
sentences = ["This movie is fantastic!"]
tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

# Initialize DLBacktrace
dlb = DLBacktrace(
    model,
    (input_ids, attention_mask),
    device="cuda"
)

# Run text classification - ONE CALL!
results = dlb.run_task(
    task="text-classification",  # or "auto" for automatic detection
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
    debug=True
)

# Access results
print(f"Task: {results['task']}")
print(f"Predictions: {results['predictions']}")
print(f"Relevance keys: {list(results['relevance'].keys())}")

# Get token-level relevance
relevance_input_ids = results['relevance']["input_ids"]
```

### Image Classification with `run_task()`

```python
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Load pre-trained model
model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()

# Prepare input image (3, 224, 224)
test_image = torch.randn(1, 3, 224, 224)

# Initialize DLBacktrace
dlb = DLBacktrace(
    model,
    input_for_graph=test_image,
    device="cuda"
)

# Run image classification - ONE CALL!
results = dlb.run_task(
    task="image-classification",  # or "auto"
    inputs=test_image,
    debug=True
)

# Access results
print(f"Predicted class: {results['predictions'].argmax()}")
print(f"Number of nodes: {len(results['relevance'])}")

# Visualize
dlb.visualize_dlbacktrace(output_path="mobilenet")
```

### Text Generation with `run_task()`

For standard language models (non-MoE) like Qwen, LLaMA, GPT-2:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dl_backtrace.pytorch_backtrace import DLBacktrace

# Load language model
model_id = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize input
prompt = "What is the capital of France?"
tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

# Initialize DLBacktrace
dlb = DLBacktrace(
    model,
    (input_ids, attention_mask),
    device="cuda"
)

# Run generation - ONE CALL!
results = dlb.run_task(
    task="generation",
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
    tokenizer=tokenizer,
    max_new_tokens=5,
    temperature=0.7,        # Optional: sampling
    top_p=0.9,              # Optional: nucleus sampling
    return_relevance=True,  # Track token relevance per step
    return_scores=True,     # Return generation scores
    debug=True
)

# Access results
generated_text = tokenizer.batch_decode(results['generated_ids'], skip_special_tokens=True)
print(f"Generated: {generated_text[0]}")
print(f"Relevance trace steps: {len(results['relevance_trace'])}")

# Visualize token-wise relevance
dlb.visualize_tokenwise_relevance_map(
    results['relevance_trace'],
    input_ids,
    tokenizer,
    generated_ids=results['generated_ids']
)
```

**Generation Modes Supported:**
- 🎯 Greedy decoding (default)
- 🎲 Sampling with temperature
- 🔝 Top-k and top-p (nucleus) sampling
- 🌟 Beam search with `num_beams` parameter

### Text Generation with MoE Models

For Mixture of Experts (MoE) models like Qwen3-MoE, GPT-OSS, JetMoE, and OLMoE:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dl_backtrace.moe_pytorch_backtrace import Backtrace

# Load MoE model
model_id = "Qwen/Qwen3-30B-A3B"  # or "openai/gpt-oss-20b"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Initialize MoE Backtrace
bt = Backtrace(
    model=model,
    model_type='qwen3_moe',  # or 'gpt_oss', 'jetmoe', 'olmoe'
    device="cuda"
)

# Tokenize input
prompt = "What is the capital of France?"
tokens = tokenizer(prompt, return_tensors="pt")
input_ids = tokens["input_ids"].to("cuda")
attention_mask = tokens["attention_mask"].to("cuda")

# Run generation with relevance tracing - ONE CALL!
results = bt.run_task(
    task="generation",
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
    tokenizer=tokenizer,
    max_new_tokens=10,
    temperature=0.7,        # Optional: sampling
    top_p=0.9,              # Optional: nucleus sampling
    return_relevance=True,  # Track token & expert relevance
    return_scores=True,     # Return generation scores
    debug=True
)

# Access results
generated_text = tokenizer.decode(results['generated_ids'][0], skip_special_tokens=True)
print(f"Generated: {generated_text}")
print(f"Relevance trace steps: {len(results['relevance_trace'])}")

# Access expert routing information
expert_relevance = bt.all_layer_expert_relevance
print(f"Expert layers tracked: {len(expert_relevance)}")
```

**MoE Generation Features:**
- 🎯 Automatic expert routing analysis
- 📊 Per-token relevance tracking across generation
- 🔍 Expert-level contribution scores
- 🚀 Supports greedy, sampling, and beam search

### Supported Tasks

The `run_task()` method supports:

- `"text-classification"` - Sentiment analysis, text categorization
- `"image-classification"` - Image recognition, object classification
- `"generation"` - Text generation (both standard LMs and MoE models)
- `"auto"` - Automatic task detection based on model output

**Generation Task Features:**
- Works with both `DLBacktrace` (standard models) and `Backtrace` (MoE models)
- Automatic relevance tracing per generation step
- Support for greedy, sampling (temperature, top-k, top-p), and beam search
- Per-token relevance visualization
- Expert routing analysis (MoE models only)

### `run_task()` Benefits

- ✨ **Single call**: Combines `predict()` + `evaluation()` into one
- 🎯 **Task-aware**: Automatically configures evaluation parameters
- 📦 **Structured output**: Consistent return format across tasks
- 🔍 **Easy access**: Direct access to predictions and relevance
- 🚀 **Cleaner code**: Less boilerplate, more readable


---

## Understanding the Output

### Node I/O Dictionary

The `predict()` method returns a dictionary mapping node names to their inputs and outputs:

```python
node_io = dlb.predict(test_input)

for node_name, (inputs, output) in node_io.items():
    print(f"Node: {node_name}")
    print(f"  Input shapes: {[inp.shape for inp in inputs]}")
    print(f"  Output shape: {output.shape}")
```

### Relevance Dictionary

The `evaluation()` method returns relevance scores for each node:

```python
relevance = dlb.evaluation(...)

for node_name, rel_score in relevance.items():
    print(f"{node_name}: {rel_score}")
```

### Visualization Output

The visualization methods save files to your current directory:

- `dlbacktrace_graph.png` - Full computational graph
- `dlbacktrace_top_k.png` - Top-k relevant nodes
- `.svg` versions of both

---

## Common Parameters

### DLBacktrace Initialization

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | PyTorch model to trace | Required |
| `input_for_graph` | Tuple of example inputs | Required |
| `device` | Device type | `"cpu"` |

### Evaluation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mode` | Evaluation mode | `"default"` |
| `multiplier` | Starting relevance value | `100.0` |
| `task` | Task type | Required |
| `thresholding` | Threshold for segmentation | `0.5` |

### Task Types

- `"text-classification"` / `"binary-classification"` / `"multi-class classification"`
- `"image-classification"`
- `"generation"`
- `"bbox-regression"`
- `"binary-segmentation"`
- `"auto"` - Automatic detection

### Generation Parameters (for `task="generation"`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `tokenizer` | HuggingFace tokenizer | Required |
| `max_new_tokens` | Maximum tokens to generate | `20` |
| `temperature` | Sampling temperature (None = greedy) | `None` |
| `top_k` | Top-k sampling | `None` |
| `top_p` | Nucleus sampling threshold | `None` |
| `num_beams` | Number of beams for beam search | `1` |
| `return_relevance` | Return per-step relevance trace | `False` |
| `return_scores` | Return per-step generation scores | `False` |
| `return_layerwise_output` | Return per-step layer outputs | `False` |

---

## Advanced Features

### Temperature Scaling

Control generation diversity and prediction confidence:

```python
# For classification - adjust confidence
node_io = dlb.predict(test_input, temperature=0.8)

# For generation - control randomness
from dl_backtrace.pytorch_backtrace.dlbacktrace.core.dlb_auto_sampler import DLBAutoSampler

sampler = DLBAutoSampler(dlb=dlb, tokenizer=tokenizer)
output = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=1.2  # Higher = more creative
)
```

[Learn more →](../guide/pytorch/temperature-scaling.md)

### MoEs Models

Analyze Mixture-of-Experts models with expert-level tracking:

```python
from dl_backtrace.moe_pytorch_backtrace.backtrace import Backtrace

# Supported: JetMoE, OLMoE, Qwen MoE, GPT-OSS
backtrace = Backtrace(
    model=moe_model,
    model_type='jetmoe',
    device="cuda"
)

# Track expert contributions
relevance = backtrace.eval(all_in, all_out, device="cuda")
expert_relevance = backtrace.all_layer_expert_relevance
```

[Learn more →](../guide/pytorch/moe-models.md)

### DLB Auto Sampler

Advanced text generation with multiple sampling strategies:

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.core.dlb_auto_sampler import DLBAutoSampler

sampler = DLBAutoSampler(dlb=dlb, tokenizer=tokenizer)

# Greedy decoding
output_greedy = sampler.generate(input_ids, temperature=None)

# Nucleus sampling
output_nucleus = sampler.generate(
    input_ids,
    temperature=0.8,
    top_p=0.9,
    top_k=50
)

# Beam search
output_beam = sampler.generate(
    input_ids,
    num_beams=5,
    early_stopping=True
)
```

[Learn more →](../guide/pytorch/auto-sampler.md)

---

## Next Steps

Now that you've run your first example, dive deeper:

### Learn the Concepts
- [Introduction to DLBacktrace](../guide/introduction.md)
- [Pipeline Interface](../guide/pytorch/pipeline.md) - High-level workflows
- [Understanding Relevance Propagation](../guide/relevance/overview.md)
- [Execution Engine Explained](../guide/pytorch/execution-engines.md)

### Explore Examples
- [Google Colab Notebooks](../examples/colab-notebooks.md)
- [Use Cases](../examples/use-cases.md)

### Best Practices
- [Best Practices Guide](../guide/best-practices.md)

### Get Help
- [FAQ](../support/faq.md)

---

## Tips for Success

!!! tip "Start Simple"
    Begin with small models to understand the workflow before moving to large transformers.

!!! tip "Use GPU When Available"
    DLBacktrace benefits significantly from GPU acceleration for large models.

!!! tip "Check Model Compatibility"
    Some custom operations might not be supported yet. Check the [supported operations list](../guide/pytorch/operations.md).

!!! warning "Memory Usage"
    Large models (like LLaMA-8B) require significant memory. This requires a lot of RAM.

---

## Troubleshooting Quick Start

??? question "Model fails to trace"
    Make sure your model is in evaluation mode and the input shape matches what the model expects:
    ```python
    model.eval()
    # Check input shape matches model's expected input
    ```

??? question "CUDA out of memory"
    Try using CPU or reduce model size:
    ```python
    # Force CPU
    model = model.cpu()
    test_input = test_input.cpu()
    ```

??? question "Visualization doesn't appear"
    Check that graphviz is installed:
    ```bash
    # Ubuntu/Debian
    sudo apt-get install graphviz
    
    # macOS
    brew install graphviz
    
    # Python package
    pip install graphviz
    ```



