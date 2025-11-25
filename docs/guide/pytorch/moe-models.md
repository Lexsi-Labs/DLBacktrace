# Mixture-of-Experts (MoEs) Support

DLBacktrace provides comprehensive support for **Mixture-of-Experts (MoEs)** models, enabling explainability analysis for these advanced architectures with expert-level relevance tracking.

---

## Overview

MoEs models use multiple specialized "expert" networks that are dynamically activated based on the input. DLBacktrace provides:

- **✨ Expert-Level Relevance Tracking**: Track which experts contribute most to predictions
- **🎯 Model-Specific Implementations**: Optimized support for popular MoEs architectures
- **⚡ CUDA Acceleration**: GPU-accelerated relevance propagation for MoEs layers
- **📊 Expert Routing Analysis**: Understand expert selection and contribution patterns
- **🔍 Layer-wise Attribution**: Full relevance flow through MoEs feed-forward and attention blocks

---

## Supported MoEs Models

### 1. JetMoE

**JetMoE** is an efficient MoEs architecture with sparse expert activation.

```python
from dl_backtrace.moe_pytorch_backtrace import Backtrace
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load JetMoE model
model_name = "jetmoe/jetmoe-8b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize DLBacktrace for JetMoE
bt = Backtrace(
    model=model,
    model_type='jetmoe',
    device='cuda'  # or 'cpu'
)

# Prepare input
prompt = "What is the capital of France?"
tokens = tokenizer(prompt, return_tensors="pt")
input_ids = tokens["input_ids"].cuda()
attention_mask = tokens["attention_mask"].cuda()

# Run generation with run_task() - ONE CALL!
results = bt.run_task(
    task="generation",
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
    tokenizer=tokenizer,
    max_new_tokens=10,
    return_relevance=True,
    return_scores=True,
    debug=False
)

# Access generated text and relevance
generated_text = tokenizer.decode(results['generated_ids'][0], skip_special_tokens=True)
print(f"Generated: {generated_text}")

# Access expert-level relevance
expert_relevance = bt.all_layer_expert_relevance
print(f"Expert routing across {len(expert_relevance)} layers")

# Access token and expert relevance per step
for step_data in results['relevance_trace']:
    token_rel = step_data['all_wt']  # Token relevance
    expert_rel = step_data['expert_relevance']  # Expert relevance
```

**Key Features:**
- Sparse expert activation
- Efficient attention mechanisms
- Expert-level feed-forward and attention tracking

### 2. OLMoE

**OLMoE** (Open Language MoE) is an open-source MoEs model optimized for efficiency.

```python
from dl_backtrace.moe_pytorch_backtrace import Backtrace
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load OLMoE model
model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125-Instruct")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125-Instruct")

# Initialize backtrace
bt = Backtrace(
    model=model,
    model_type='olmoe',
    device='cuda'  # or 'cpu'
)

# Prepare input
prompt = "What is the capital of France?"
tokens = tokenizer(prompt, return_tensors="pt")
input_ids = tokens["input_ids"].cuda()
attention_mask = tokens["attention_mask"].cuda()

# Run generation with run_task() - ONE CALL!
results = bt.run_task(
    task="generation",
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
    tokenizer=tokenizer,
    max_new_tokens=10,
    return_relevance=True,
    return_scores=True,
    debug=False
)

# Access generated text
generated_text = tokenizer.decode(results['generated_ids'][0], skip_special_tokens=True)
print(f"Generated: {generated_text}")

# Analyze expert contributions
expert_relevance = bt.all_layer_expert_relevance
for layer_name, expert_rel in expert_relevance.items():
    print(f"{layer_name}: {expert_rel.shape}")
```

**Key Features:**
- Open-source architecture
- Multiple expert configurations
- Optimized for research and production

### 3. Qwen MoE

**Qwen MoE** model combines strong language understanding with efficient MoEs architecture.

```python
from dl_backtrace.moe_pytorch_backtrace import Backtrace
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Qwen MoE model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

# Initialize backtrace
bt = Backtrace(
    model=model,
    model_type='qwen3_moe',
    device='cuda'  # or 'cpu'
)

# Prepare input
prompt = "What is the capital of France?"
tokens = tokenizer(prompt, return_tensors="pt")
input_ids = tokens["input_ids"].cuda()
attention_mask = tokens["attention_mask"].cuda()

# Run generation with run_task() - ONE CALL!
results = bt.run_task(
    task="generation",
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
    tokenizer=tokenizer,
    max_new_tokens=10,
    return_relevance=True,
    return_scores=True,
    debug=False
)

# Access generated text and relevance
generated_text = tokenizer.decode(results['generated_ids'][0], skip_special_tokens=True)
print(f"Generated: {generated_text}")

# Access expert-level relevance
expert_relevance = bt.all_layer_expert_relevance
print(f"Expert routing across {len(expert_relevance)} layers")
```

**Key Features:**
- Grouped query attention
- Advanced expert routing
- Strong multilingual capabilities

### 4. GPT-OSS

**GPT-OSS** is an open-source MoEs implementation with configurable expert architectures.

```python
from dl_backtrace.moe_pytorch_backtrace import Backtrace
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load GPT-OSS model
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

# Initialize backtrace
bt = Backtrace(
    model=model,
    model_type='gpt_oss',
    device='cuda'  # or 'cpu'
)

# Prepare input
prompt = "What is the capital of France?"
tokens = tokenizer(prompt, return_tensors="pt")
input_ids = tokens["input_ids"].cuda()
attention_mask = tokens["attention_mask"].cuda()

# Run generation with run_task() - ONE CALL!
results = bt.run_task(
    task="generation",
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
    tokenizer=tokenizer,
    max_new_tokens=10,
    return_relevance=True,
    return_scores=True,
    debug=False
)

# Access generated text and relevance
generated_text = tokenizer.decode(results['generated_ids'][0], skip_special_tokens=True)
print(f"Generated: {generated_text}")

# Access expert-level relevance
expert_relevance = bt.all_layer_expert_relevance
print(f"Expert routing across {len(expert_relevance)} layers")
```

**Key Features:**
- Sliding window attention support
- Flexible expert configuration
- Full attention and feed-forward MoEs layers

---

## MoE-Specific Features

### Expert Relevance Tracking

DLBacktrace tracks relevance at the expert level, allowing you to understand which experts contribute to predictions.

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

MoEs models have specialized layer types:

- **MoEs Feed-Forward Layers**: Multiple expert networks with routing
- **MoEs Self-Attention Layers**: Expert-based attention mechanisms  
- **Router Layers**: Gate networks that select experts
- **Standard Transformer Layers**: Traditional attention and FFN

### Device Configuration

MoEs models benefit significantly from GPU acceleration:

```python
# CPU mode (slower, uses original implementations)
backtrace = Backtrace(..., device="cpu")
relevance = backtrace.eval(all_in, all_out, device="cpu")

# CUDA mode (recommended - much faster)
backtrace = Backtrace(..., device="cuda")
relevance = backtrace.eval(all_in, all_out, device="cuda")
```

**Performance Tips:**
- Always use `device="cuda"` for MoEs models when possible
- CUDA implementations provide 10-100x speedup for large MoEs models
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

### Complete MoEs Analysis Example

```python
import numpy as np
from dl_backtrace.moe_pytorch_backtrace import Backtrace
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("jetmoe/jetmoe-8b")
tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b")

# 2. Initialize backtrace
bt = Backtrace(
    model=model,
    model_type='jetmoe',  # or 'olmoe', 'qwen3_moe', 'gpt_oss'
    device="cuda"
)

# 3. Prepare input
prompt = "Analyze this important text for expert routing"
tokens = tokenizer(prompt, return_tensors="pt")
input_ids = tokens["input_ids"].cuda()
attention_mask = tokens["attention_mask"].cuda()

# 4. Run generation with run_task() - ONE CALL!
results = bt.run_task(
    task="generation",
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
    tokenizer=tokenizer,
    max_new_tokens=20,
    return_relevance=True,
    return_scores=True,
    debug=False
)

# 5. Access generated text
generated_text = tokenizer.decode(results['generated_ids'][0], skip_special_tokens=True)
print(f"\n✅ Generated text: {generated_text}")
print(f"📊 Relevance trace: {len(results['relevance_trace'])} generation steps")

# 6. Analyze expert routing across all generation steps
print("\n=== Expert Routing Analysis ===")
expert_relevance = bt.all_layer_expert_relevance

for layer_name, expert_rel in expert_relevance.items():
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

# 7. Analyze token relevance per generation step
print("\n=== Token Relevance Per Step ===")
for step_idx, step_data in enumerate(results['relevance_trace']):
    all_wt = step_data['all_wt']  # Token relevance
    expert_rel = step_data['expert_relevance']  # Expert relevance
    
    print(f"\nStep {step_idx}:")
    print(f"  Token relevance nodes: {len(all_wt)}")
    print(f"  Expert relevance layers: {len(expert_rel)}")
    
    # Show decoder embeddings relevance
    if 'decoder_embeddings' in all_wt:
        decoder_rel = all_wt['decoder_embeddings']
        if isinstance(decoder_rel, np.ndarray):
            token_scores = np.sum(decoder_rel, axis=-1).flatten()
            print(f"  Token relevance sum: {np.sum(token_scores):.4f}")

# 8. Visualize expert specialization
print("\n=== Expert Specialization Analysis ===")
# Aggregate expert relevance across all layers
all_expert_scores = []
for layer_name, expert_rel in expert_relevance.items():
    if "ff_expert" in layer_name or "attention_expert" in layer_name:
        # Sum across tokens/batch dimensions
        if len(expert_rel.shape) > 1:
            layer_expert_sum = expert_rel.sum(axis=tuple(range(len(expert_rel.shape)-1)))
            all_expert_scores.append(layer_expert_sum)

if all_expert_scores:
    # Average across layers
    avg_expert_scores = np.mean(all_expert_scores, axis=0)
    top_5_experts = avg_expert_scores.argsort()[-5:][::-1]
    
    print("\nTop 5 Most Active Experts (averaged across all layers):")
    for rank, expert_idx in enumerate(top_5_experts, 1):
        print(f"  {rank}. Expert {expert_idx}: {avg_expert_scores[expert_idx]:.4f}")
```

---

## Implementation Details

### MoEs Layer Processing

DLBacktrace processes MoEs layers with specialized implementations for each model type:

```python
# JetMoE Feed-Forward Layer
if node_class == "JetMoE_Feed_Forward":
    weights = all_wts[start_layer]
    ff_w = helper.rename_jetmoe_feed_forward_keys(weights)
    x = arr_from_key(child_nodes[0])
    temp_wt, ff_expert = UD2.launch_jetmoe_feed_forward(
        impl, all_wt[start_layer], x, ff_w, self.model
    )
    all_wt[child_nodes[0]] += to_np64(temp_wt)
    self.all_layer_expert_relevance[f"{start_layer}_ff_expert"] = ff_expert 
```

**Key Features:**

- **Device-Aware Processing**: Automatically handles CPU (`devide="cpu"`) or CUDA (`device="cuda"`) execution based on initialization
- **Dual Relevance Tracking**: Each MoEs layer returns:
  - `temp_wt`: Token-level relevance propagated to child nodes
  - `ff_expert` / `attn_expert`: Expert-level relevance stored separately
- **Model-Specific Implementations**: Optimized launchers for each architecture:
  - `launch_jetmoe_feed_forward()` / `launch_jetmoe_self_attention()`
  - `launch_olmoe_feed_forward()`
  - `launch_qwen3_moe_feed_forward()`
  - `launch_gpt_oss_feed_forward()` / `launch_gpt_oss_self_attention()`
- **JetMoE Unique Feature**: Only model with MoEs in both feed-forward AND self-attention layers
- **GPT-OSS Sliding Window**: Supports both full and sliding window attention patterns
- **Numerical Stability**: Token relevance accumulated using `to_np64()` (float64) to prevent precision loss
- **Hierarchical Storage**: 
  - Token relevance in `all_wt[node_name]`
  - Expert relevance in `self.all_layer_expert_relevance[layer_expert_key]`

### Expert Routing

The routing mechanism determines which experts process the token:

1. **Router Network**: Computes scores for each expert
2. **Top-K Selection**: Selects top-K experts based on scores
3. **Expert Execution**: Selected experts process the input
4. **Weighted Combination**: Expert outputs are weighted by router scores

DLBacktrace tracks relevance through this entire routing process.

---

## Performance Considerations

### Memory Usage

MoEs models require more memory due to multiple expert networks:

```python
# Estimate memory requirements
num_experts = config.num_experts
expert_size = config.hidden_size * config.intermediate_size
memory_per_layer = num_experts * expert_size * 4  # bytes (float32)

print(f"Approx memory per MoEs layer: {memory_per_layer / 1e9:.2f} GB")
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

1. **Always use CUDA** for MoEs models when possible
2. **Monitor memory usage** - MoEs models are memory-intensive
3. **Analyze expert specialization** - identify which experts handle specific patterns
4. **Compare across prompts** - see how expert routing varies
5. **Use appropriate `multiplier`** - scale relevance for visualization

---

## Next Steps

- Learn about [DLB Auto Sampler](auto-sampler.md) for MoEs text generation
- Explore [Temperature Scaling](temperature-scaling.md) for controlled generation
- Check [Pipeline](pipeline.md) for high-level MoEs workflows
- See [Examples](../../examples/colab-notebooks.md) for complete MoEs use cases

