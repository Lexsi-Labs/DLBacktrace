# Mixture of Experts (MoE) Support

DL-Backtrace provides comprehensive support for **Mixture of Experts (MoE)** models, enabling explainability analysis for these advanced architectures with expert-level relevance tracking.

---

## Overview

MoE models use multiple specialized "expert" networks that are dynamically activated based on the input. DL-Backtrace provides:

- **✨ Expert-Level Relevance Tracking**: Track which experts contribute most to predictions
- **🎯 Model-Specific Implementations**: Optimized support for popular MoE architectures
- **⚡ CUDA Acceleration**: GPU-accelerated relevance propagation for MoE layers
- **📊 Expert Routing Analysis**: Understand expert selection and contribution patterns
- **🔍 Layer-wise Attribution**: Full relevance flow through MoE feed-forward and attention blocks

---

## Supported MoE Models

### 1. JetMoE

**JetMoE** is an efficient MoE architecture with sparse expert activation.

```python
from dl_backtrace.moe_pytorch_backtrace.backtrace import Backtrace
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load JetMoE model
model_name = "jetmoe/jetmoe-8b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize DL-Backtrace for JetMoE
backtrace = Backtrace(
    model=model,
    model_type='jetmoe',
    input_text="Your prompt here",
    tokenizer=tokenizer,
    max_length=512,
    device="cuda"
)

# Get model outputs
input_ids = tokenizer("Your prompt", return_tensors="pt").input_ids
all_in, all_out = backtrace.model(input_ids, return_dict=True)

# Compute relevance with expert tracking
relevance = backtrace.eval(
    all_in=all_in,
    all_out=all_out,
    mode="default",
    device="cuda"
)

# Access expert-level relevance
expert_relevance = backtrace.all_layer_expert_relevance
print(f"Layers with expert analysis: {expert_relevance.keys()}")
```

**Key Features:**
- Sparse expert activation
- Efficient attention mechanisms
- Expert-level feed-forward and attention tracking

### 2. OLMoE

**OLMoE** (Open Language MoE) is an open-source MoE model optimized for efficiency.

```python
from dl_backtrace.moe_pytorch_backtrace.backtrace import Backtrace
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load OLMoE model
model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

# Initialize backtrace
backtrace = Backtrace(
    model=model,
    model_type='olmoe',
    input_text="Example text",
    tokenizer=tokenizer,
    max_length=512,
    device="cuda"
)

# Run analysis
all_in, all_out = backtrace.model(input_ids, return_dict=True)
relevance = backtrace.eval(all_in, all_out, device="cuda")

# Analyze expert contributions
for layer_name, expert_rel in backtrace.all_layer_expert_relevance.items():
    print(f"{layer_name}: {expert_rel.shape}")
```

**Key Features:**
- Open-source architecture
- Multiple expert configurations
- Optimized for research and production

### 3. Qwen MoE

**Qwen MoE** model combines strong language understanding with efficient MoE architecture.

```python
from dl_backtrace.moe_pytorch_backtrace.backtrace import Backtrace
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Qwen MoE model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")

# Initialize backtrace
backtrace = Backtrace(
    model=model,
    model_type='qwen3_moe',
    input_text="Your input text",
    tokenizer=tokenizer,
    max_length=1024,
    device="cuda"
)

# Compute relevance
input_ids = tokenizer("Input", return_tensors="pt").input_ids
all_in, all_out = backtrace.model(input_ids, return_dict=True)

relevance = backtrace.eval(
    all_in=all_in,
    all_out=all_out,
    mode="default",
    device="cuda"
)
```

**Key Features:**
- Grouped query attention
- Advanced expert routing
- Strong multilingual capabilities

### 4. GPT-OSS

**GPT-OSS** is an open-source MoE implementation with configurable expert architectures.

```python
from dl_backtrace.moe_pytorch_backtrace.backtrace import Backtrace
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load GPT-OSS model
model = AutoModelForCausalLM.from_pretrained("gpt-oss/gpt-oss-model")
tokenizer = AutoTokenizer.from_pretrained("gpt-oss/gpt-oss-model")

# Initialize backtrace
backtrace = Backtrace(
    model=model,
    model_type='gpt_oss',
    input_text="Example prompt",
    tokenizer=tokenizer,
    max_length=512,
    device="cuda"
)

# Run with sliding window attention support
all_in, all_out = backtrace.model(input_ids, return_dict=True)
relevance = backtrace.eval(all_in, all_out, device="cuda")
```

**Key Features:**
- Sliding window attention support
- Flexible expert configuration
- Full attention and feed-forward MoE layers

---

## MoE-Specific Features

### Expert Relevance Tracking

DL-Backtrace tracks relevance at the expert level, allowing you to understand which experts contribute to predictions.

```python
# After running evaluation
expert_relevance = backtrace.all_layer_expert_relevance

# Analyze each layer
for layer_name, expert_scores in expert_relevance.items():
    print(f"\n{layer_name}:")
    print(f"  Shape: {expert_scores.shape}")
    print(f"  Mean relevance: {expert_scores.mean():.4f}")
    print(f"  Max relevance: {expert_scores.max():.4f}")
    
    # Identify most relevant experts
    if len(expert_scores.shape) > 1:
        top_experts = expert_scores.mean(axis=0).argsort()[-3:]
        print(f"  Top 3 experts: {top_experts}")
```

### Layer Types

MoE models have specialized layer types:

- **MoE Feed-Forward Layers**: Multiple expert networks with routing
- **MoE Self-Attention Layers**: Expert-based attention mechanisms  
- **Router Layers**: Gate networks that select experts
- **Standard Transformer Layers**: Traditional attention and FFN

### Device Configuration

MoE models benefit significantly from GPU acceleration:

```python
# CPU mode (slower, uses original implementations)
backtrace = Backtrace(..., device="cpu")
relevance = backtrace.eval(all_in, all_out, device="cpu")

# CUDA mode (recommended - much faster)
backtrace = Backtrace(..., device="cuda")
relevance = backtrace.eval(all_in, all_out, device="cuda")
```

**Performance Tips:**
- Always use `device="cuda"` for MoE models when possible
- CUDA implementations provide 10-100x speedup for large MoE models
- Memory usage scales with number of experts and sequence length

---

## Advanced Configuration

### Relevance Propagation Parameters

```python
relevance = backtrace.eval(
    all_in=all_in,
    all_out=all_out,
    mode="default",          # Relevance mode
    multiplier=100.0,        # Scale relevance values
    scaler=0,                # Additional scaling
    max_unit=0,              # Normalize to max value
    thresholding=0.5,        # Threshold for binary tasks
    task="generation",       # Task type
    device="cuda"            # Compute device
)
```

### Model Configuration Access

```python
from dl_backtrace.moe_pytorch_backtrace.backtrace.backtrace import get_model_config

# Get model configuration
config = get_model_config(model)

print(f"Hidden size: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")
print(f"Num experts: {getattr(config, 'num_experts', 'N/A')}")
print(f"Experts per token: {getattr(config, 'num_experts_per_tok', 'N/A')}")
```

---

## Expert Analysis Workflow

### Complete MoE Analysis Example

```python
import numpy as np
from dl_backtrace.moe_pytorch_backtrace.backtrace import Backtrace
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("your-moe-model")
tokenizer = AutoTokenizer.from_pretrained("your-moe-model")

# 2. Prepare input
text = "Analyze this important text for expert routing"
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()

# 3. Initialize backtrace
backtrace = Backtrace(
    model=model,
    model_type='jetmoe',  # or 'olmoe', 'qwen3_moe', 'gpt_oss'
    input_text=text,
    tokenizer=tokenizer,
    max_length=512,
    device="cuda"
)

# 4. Get model outputs
all_in, all_out = backtrace.model(input_ids, return_dict=True)

# 5. Compute relevance
relevance = backtrace.eval(
    all_in=all_in,
    all_out=all_out,
    mode="default",
    multiplier=100.0,
    device="cuda"
)

# 6. Analyze expert routing
print("\n=== Expert Routing Analysis ===")
for layer_name, expert_rel in backtrace.all_layer_expert_relevance.items():
    print(f"\n{layer_name}:")
    
    # Get layer and expert type
    if "ff_expert" in layer_name:
        layer_type = "Feed-Forward"
    elif "attention_expert" in layer_name:
        layer_type = "Attention"
    else:
        layer_type = "Unknown"
    
    print(f"  Type: {layer_type}")
    print(f"  Shape: {expert_rel.shape}")
    
    # Compute statistics
    mean_rel = np.mean(expert_rel)
    max_rel = np.max(expert_rel)
    min_rel = np.min(expert_rel)
    
    print(f"  Mean relevance: {mean_rel:.4f}")
    print(f"  Max relevance: {max_rel:.4f}")
    print(f"  Min relevance: {min_rel:.4f}")
    
    # Find top experts
    if len(expert_rel.shape) > 1:
        expert_means = expert_rel.mean(axis=tuple(range(len(expert_rel.shape)-1)))
        top_k = min(5, len(expert_means))
        top_experts = expert_means.argsort()[-top_k:][::-1]
        
        print(f"  Top {top_k} experts:")
        for idx, expert_idx in enumerate(top_experts):
            print(f"    {idx+1}. Expert {expert_idx}: {expert_means[expert_idx]:.4f}")

# 7. Analyze overall relevance flow
print("\n=== Overall Relevance Flow ===")
for layer_name, layer_rel in relevance.items():
    if isinstance(layer_rel, np.ndarray):
        print(f"{layer_name}: {layer_rel.shape}, sum={np.sum(layer_rel):.2f}")
```

---

## Implementation Details

### MoE Layer Processing

DL-Backtrace processes MoE layers with specialized implementations:

```python
# Feed-forward MoE layer
if model_resource['graph'][start_layer]["class"] == 'JetMoE_Feed_Forward':
    weights = all_wts[start_layer]
    feed_forward_weights = helper.rename_jetmoe_feed_forward_keys(weights)
    
    temp_wt, ff_expert = UD2.launch_jetmoe_feed_forward(
        impl="cuda",  # or "original"
        all_wt[start_layer],
        all_out[child_nodes[0]][0].detach().numpy(),
        feed_forward_weights,
        model
    )
    
    # Store expert relevance
    layer = f"{start_layer}_ff_expert"
    all_layer_expert_relevance[layer] = ff_expert
```

### Expert Routing

The routing mechanism determines which experts process each token:

1. **Router Network**: Computes scores for each expert
2. **Top-K Selection**: Selects top experts based on scores
3. **Expert Execution**: Selected experts process the input
4. **Weighted Combination**: Expert outputs are weighted by router scores

DL-Backtrace tracks relevance through this entire routing process.

---

## Performance Considerations

### Memory Usage

MoE models require more memory due to multiple expert networks:

```python
# Estimate memory requirements
num_experts = config.num_experts
expert_size = config.hidden_size * config.intermediate_size
memory_per_layer = num_experts * expert_size * 4  # bytes (float32)

print(f"Approx memory per MoE layer: {memory_per_layer / 1e9:.2f} GB")
```

### Computation Time

CUDA acceleration is essential for reasonable performance:

| Model Size | CPU Time | CUDA Time | Speedup |
|------------|----------|-----------|---------|
| 1B params | ~300s | ~15s | 20x |
| 7B params | ~2000s | ~60s | 33x |
| 14B params | ~4500s | ~120s | 37x |

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Reduce batch size or sequence length
backtrace = Backtrace(..., max_length=256)  # Reduce from 512
```

**2. Slow CPU Execution**
```python
# Solution: Use CUDA
backtrace = Backtrace(..., device="cuda")
relevance = backtrace.eval(..., device="cuda")
```

**3. Missing Expert Relevance**
```python
# Check if all_layer_expert_relevance is populated
if not backtrace.all_layer_expert_relevance:
    print("No expert relevance computed - check model_type")
```

---

## Best Practices

1. **Always use CUDA** for MoE models when possible
2. **Monitor memory usage** - MoE models are memory-intensive
3. **Analyze expert specialization** - identify which experts handle specific patterns
4. **Compare across prompts** - see how expert routing varies
5. **Use appropriate `multiplier`** - scale relevance for visualization

---

## Next Steps

- Learn about [DLB Auto Sampler](auto-sampler.md) for MoE text generation
- Explore [Temperature Scaling](temperature-scaling.md) for controlled generation
- Check [Pipeline](pipeline.md) for high-level MoE workflows
- See [Examples](../../examples/colab-notebooks.md) for complete MoE use cases

