# DLB Auto Sampler

The **DLB Auto Sampler** is a powerful text generation module that integrates DL-Backtrace's explainability capabilities directly into the generation process, enabling token-by-token relevance tracking while maintaining compatibility with HuggingFace's generation API.

---

## Overview

DLB Auto Sampler provides:

- **🎯 DLB-Native Generation**: All logits come from DL-Backtrace's execution engine
- **⚡ Multiple Sampling Strategies**: Greedy, temperature, top-k, top-p, and beam search
- **🔍 Token-Level Relevance**: Track relevance for each generated token
- **🤝 HuggingFace Compatible**: Drop-in replacement for standard generation
- **🎨 Flexible Control**: Full control over sampling parameters
- **⏱️ Efficient Execution**: Optimized for both CPU and CUDA

---

## Quick Start

### Basic Generation

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace import DLBacktrace
from dl_backtrace.pytorch_backtrace.dlbacktrace.core.dlb_auto_sampler import DLBAutoSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Initialize DL-Backtrace
sample_input = torch.randint(0, 1000, (1, 10))
dlb = DLBacktrace(
    model=model,
    input_for_graph=(sample_input,),
    device="cuda"
)

# Create auto sampler
sampler = DLBAutoSampler(dlb=dlb, tokenizer=tokenizer)

# Generate text
prompt = "The future of artificial intelligence"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

output = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9
)

# Decode output
generated_text = tokenizer.decode(output[0])
print(f"Generated: {generated_text}")
```

---

## Sampling Strategies

### 1. Greedy Decoding

Select the most probable token at each step (deterministic).

```python
output = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=None,  # Greedy when temperature is None
    top_k=None,
    top_p=None
)
```

**Use Cases:**
- Deterministic generation
- Factual completion
- Code generation

**Characteristics:**
- ✅ Deterministic and reproducible
- ✅ Fast execution
- ⚠️ May produce repetitive text
- ⚠️ Lower diversity

### 2. Temperature Sampling

Control randomness in generation with temperature parameter.

```python
# Low temperature (more focused, deterministic)
output_focused = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.5
)

# High temperature (more creative, random)
output_creative = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=1.5
)
```

**Temperature Effects:**
- `temperature < 1.0`: More focused, higher probability tokens preferred
- `temperature = 1.0`: No modification (standard distribution)
- `temperature > 1.0`: More random, flatter distribution

**Use Cases:**
- Creative writing: `temperature=1.2-1.5`
- Balanced generation: `temperature=0.8-1.0`
- Precise completion: `temperature=0.5-0.7`

### 3. Top-K Sampling

Sample from the k most probable tokens.

```python
output = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50  # Consider only top 50 tokens
)
```

**Top-K Values:**
- `top_k=1`: Equivalent to greedy
- `top_k=10-50`: Balanced quality and diversity
- `top_k=100-200`: Higher diversity

### 4. Top-P (Nucleus) Sampling

Sample from the smallest set of tokens whose cumulative probability exceeds p.

```python
output = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9  # Sample from top 90% probability mass
)
```

**Top-P Values:**
- `top_p=0.9`: Standard, good balance
- `top_p=0.95`: Higher diversity
- `top_p=0.8`: More focused

### 5. Beam Search

Explore multiple generation paths simultaneously.

```python
output = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    num_beams=4,  # Explore 4 parallel paths
    early_stopping=True,
    num_return_sequences=1
)
```

**Beam Parameters:**
- `num_beams=1`: Disable beam search (sampling mode)
- `num_beams=3-5`: Good balance
- `num_beams=10+`: Exhaustive search (slower)

**Use Cases:**
- Translation
- Summarization
- High-quality generation

---

## Advanced Usage

### Combined Sampling

Combine multiple sampling strategies for fine-grained control.

```python
output = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    temperature=0.8,      # Moderate randomness
    top_k=50,             # Limit to top 50 tokens
    top_p=0.9,            # Nucleus sampling
    repetition_penalty=1.2  # Penalize repetition
)
```

### Generation with Scores

Track generation scores for each token.

```python
output, scores = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.8,
    return_scores=True  # Return per-token scores
)

# Analyze scores
print(f"Generated {len(scores)} tokens")
print(f"Average score: {sum(scores) / len(scores):.4f}")
print(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]")
```

### Batch Generation

Generate multiple sequences in parallel (when supported).

```python
# Prepare multiple prompts
prompts = [
    "The capital of France is",
    "Machine learning is",
    "In the year 2050"
]

# Tokenize all prompts
input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.cuda()

# Generate for all prompts
outputs = []
for i in range(len(prompts)):
    output = sampler.generate(
        input_ids=input_ids[i:i+1],
        max_new_tokens=30,
        temperature=0.8
    )
    outputs.append(output)

# Decode all outputs
for i, output in enumerate(outputs):
    text = tokenizer.decode(output[0])
    print(f"Prompt {i+1}: {text}")
```

---

## Generation Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_ids` | Tensor | **Required** | Input token IDs |
| `max_new_tokens` | int | `50` | Maximum tokens to generate |
| `max_length` | int | `None` | Maximum total sequence length |
| `min_length` | int | `0` | Minimum sequence length |

### Sampling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | `1.0` | Sampling temperature |
| `top_k` | int | `50` | Top-k sampling parameter |
| `top_p` | float | `0.9` | Nucleus sampling threshold |
| `repetition_penalty` | float | `1.0` | Penalize token repetition |

### Beam Search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_beams` | int | `1` | Number of beams |
| `early_stopping` | bool | `True` | Stop when all beams complete |
| `num_return_sequences` | int | `1` | Number of sequences to return |
| `length_penalty` | float | `1.0` | Length penalty for beam search |

### Stopping Criteria

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eos_token_id` | int | `tokenizer.eos_token_id` | End-of-sequence token |
| `pad_token_id` | int | `tokenizer.pad_token_id` | Padding token |
| `max_time` | float | `None` | Maximum generation time (seconds) |

### Output Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `return_scores` | bool | `False` | Return per-token scores |
| `output_attentions` | bool | `False` | Return attention weights |
| `output_hidden_states` | bool | `False` | Return hidden states |

---

## Integration with DL-Backtrace

### Token-Level Relevance Tracking

The Auto Sampler allows relevance computation for generated sequences.

```python
# Generate with DLB
output = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.8
)

# Compute relevance for generated sequence
generated_ids = output[0]
attention_mask = torch.ones_like(generated_ids)

# Get node outputs
node_io = dlb.predict(generated_ids, attention_mask, debug=False)

# Compute relevance
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="generation"
)

print(f"Relevance computed for {len(relevance)} layers")
```

### Temperature Scaling in Generation

Temperature is automatically applied during generation:

```python
# Temperature is applied to logits before sampling
output = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.7  # Scaled logits used for sampling
)
```

This is different from applying temperature to final outputs - it affects the generation process itself.

---

## Performance Optimization

### CUDA Acceleration

For faster generation, use CUDA device:

```python
dlb = DLBacktrace(
    model=model,
    input_for_graph=(sample_input,),
    device="cuda"  # Enable CUDA acceleration
)

sampler = DLBAutoSampler(dlb=dlb, tokenizer=tokenizer)

# All tensors should be on CUDA
input_ids = input_ids.cuda()
```

**Performance Impact:**
- CUDA provides 10-50x speedup depending on model size
- Especially important for beam search
- Essential for large models (>1B parameters)

### Memory Management

For long sequences or large models:

```python
import torch

# Clear cache periodically
torch.cuda.empty_cache()

# Use smaller batches
max_new_tokens = 50  # Instead of 200

# Disable unnecessary outputs
output = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=max_new_tokens,
    output_attentions=False,
    output_hidden_states=False
)
```

### Deterministic Generation

For reproducible results:

```python
import torch
import numpy as np
import random

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Use greedy or low temperature
output = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.1,  # Low temperature for determinism
    do_sample=True
)
```

---

## Comparison with HuggingFace Generate

### Similarities

- ✅ Compatible parameter names
- ✅ Similar behavior for all sampling strategies
- ✅ Works with HuggingFace tokenizers
- ✅ Supports standard stopping criteria

### Differences

| Feature | HuggingFace | DLB Auto Sampler |
|---------|-------------|------------------|
| Logits Source | Model directly | DL-Backtrace engine |
| Relevance Tracking | ❌ No | ✅ Yes |
| Batch Generation | ✅ Full support | ⚠️ Single sequence |
| Model Caching | ✅ Yes | ❌ No (by design) |
| Custom Logits Processing | Limited | Full control |

### When to Use Each

**Use HuggingFace `generate()`:**
- Production inference at scale
- Batch processing
- When relevance not needed
- Maximum speed critical

**Use DLB Auto Sampler:**
- Need token-level explainability
- Analyzing generation process
- Research and debugging
- Custom generation workflows

---

## Best Practices

### 1. Temperature Selection

```python
# Creative writing
temperature = 1.2

# Balanced generation
temperature = 0.8

# Focused/factual
temperature = 0.5
```

### 2. Sampling Strategy Selection

```python
# For coherent, high-quality text
output = sampler.generate(
    input_ids=input_ids,
    temperature=0.8,
    top_p=0.9,
    top_k=50
)

# For diverse generation
output = sampler.generate(
    input_ids=input_ids,
    temperature=1.2,
    top_p=0.95
)

# For deterministic output
output = sampler.generate(
    input_ids=input_ids,
    num_beams=5,
    early_stopping=True
)
```

### 3. Memory Management

```python
# For very long sequences
max_new_tokens = min(desired_length, 100)  # Generate in chunks if needed

# Monitor memory
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory**
```python
# Solution: Reduce sequence length
output = sampler.generate(..., max_new_tokens=30)  # Instead of 100
```

**2. Slow Generation**
```python
# Solution: Use CUDA
dlb = DLBacktrace(..., device="cuda")
```

**3. Repetitive Output**
```python
# Solution: Increase temperature or use repetition penalty
output = sampler.generate(
    ...,
    temperature=1.0,
    repetition_penalty=1.2
)
```

**4. Low Quality Output**
```python
# Solution: Use beam search or lower temperature
output = sampler.generate(
    ...,
    num_beams=5,
    temperature=0.7
)
```

---

## Next Steps

- Learn about [Temperature Scaling](temperature-scaling.md) in detail
- Explore [MoE Models](moe-models.md) with Auto Sampler
- Check [Pipeline](pipeline.md) for integrated generation workflows
- See [Examples](../../examples/colab-notebooks.md) for complete use cases

