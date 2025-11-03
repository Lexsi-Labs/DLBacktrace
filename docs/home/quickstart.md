# Quick Start

Get started with DL-Backtrace in minutes! This guide will walk you through your first explainability analysis.

---

## Prerequisites

Make sure you have DL-Backtrace installed. If not, see the [Installation Guide](installation.md).

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

### Step 2: Initialize DL-Backtrace

```python
# Create dummy input for graph tracing
dummy_input = torch.randn(1, 3, 32, 32)

# Initialize DL-Backtrace
dlb = DLBacktrace(
    model=model,
    input_for_graph=(dummy_input,),
    device="cuda"
)

print("✅ DL-Backtrace initialized successfully!")
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

# Initialize DL-Backtrace
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
text = "DL-Backtrace makes AI explainable!"
inputs = tokenizer(text, return_tensors="pt", padding=True)

# Initialize DL-Backtrace
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

## Simplified Pipeline Approach

For even faster setup, use the high-level **Pipeline** interface:

### Text Classification with Pipeline

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.pipeline import DLBacktracePipeline

# Create pipeline with one line
pipeline = DLBacktracePipeline.create_simple(
    model_name="bert-base",
    device="cpu"
)

# Run analysis with one line
results = pipeline.run_simple_analysis(
    "This product exceeded my expectations!",
    label="positive"
)

print(f"Prediction: {results['predictions'][0]}")
print(f"Relevance computed: {results['relevance_computed']}")
```

### Image Classification with Pipeline

```python
from PIL import Image

# Create pipeline
pipeline = DLBacktracePipeline.create_simple(
    model_name="resnet",
    device="cuda"
)

# Load image and classify
image = Image.open("cat.jpg")
results = pipeline.run_simple_analysis(image, label="cat")

print(f"Prediction: {results['predictions'][0]}")
```

### Text Generation with Pipeline

```python
# Create pipeline for generation
pipeline = DLBacktracePipeline.create_simple(
    model_name="llama3.2-1b",
    device="cuda"
)

# Generate text with relevance analysis
results = pipeline.run_text_generation(
    prompts=["The future of AI is"],
    max_new_tokens=50,
    temperature=0.8,
    return_relevance=True
)

print(f"Generated: {results['generated_texts'][0]}")
```

**Pipeline Benefits:**
- 🔧 Automatic model loading
- ⚙️ Sensible defaults
- 📊 Built-in result management
- 🎯 Task-specific methods

[Learn more about Pipeline →](../guide/pytorch/pipeline.md)

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

- `"binary-classification"`
- `"multi-class classification"`
- `"bbox-regression"`
- `"binary-segmentation"`

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

### MoE Models

Analyze Mixture of Experts models with expert-level tracking:

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
- [Introduction to DL-Backtrace](../guide/introduction.md)
- [Pipeline Interface](../guide/pytorch/pipeline.md) - High-level workflows
- [Understanding Relevance Propagation](../guide/relevance/overview.md)
- [Execution Engines Explained](../guide/pytorch/execution-engines.md)

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
    DL-Backtrace benefits significantly from GPU acceleration for large models.

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



