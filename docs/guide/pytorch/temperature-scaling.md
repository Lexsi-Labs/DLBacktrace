# Temperature Scaling

**Temperature scaling** is a powerful technique for controlling the confidence and diversity of model predictions by modulating the softmax distribution of logits. DL-Backtrace provides seamless temperature scaling integration for both inference and generation tasks.

---

## Overview

Temperature scaling provides:

- **🎛️ Confidence Control**: Adjust model confidence without retraining
- **🎨 Generation Diversity**: Control randomness in text generation
- **⚖️ Calibration**: Improve probability calibration for better uncertainty estimates
- **🔍 Analysis**: Study model behavior at different temperature settings
- **⚡ Zero Overhead**: Applied efficiently during inference

---

## Understanding Temperature

### What is Temperature?

Temperature (\(T\)) scales the logits before applying softmax:

\[
P(y_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
\]

Where:
- \(z_i\) = logit for class \(i\)
- \(T\) = temperature parameter
- \(P(y_i)\) = probability for class \(i\)

### Temperature Effects

```python
# Original logits: [2.0, 1.0, 0.5]

# T = 0.5 (Lower temperature - more confident)
# Probabilities: [0.76, 0.19, 0.05]  ← Sharp distribution

# T = 1.0 (No scaling - standard)
# Probabilities: [0.62, 0.23, 0.15]  ← Normal distribution

# T = 2.0 (Higher temperature - less confident)
# Probabilities: [0.48, 0.30, 0.22]  ← Flat distribution
```

### Temperature Ranges

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| \(T < 0.5\) | Very confident, sharp | Deterministic tasks |
| \(T = 0.5 - 0.8\) | Focused, reduced noise | Factual generation |
| \(T = 1.0\) | Standard (no scaling) | Default behavior |
| \(T = 1.2 - 1.5\) | Increased diversity | Creative generation |
| \(T > 2.0\) | Uniform, very random | Exploration |

---

## Usage in DL-Backtrace

### Basic Temperature Scaling

#### For Classification

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace import DLBacktrace
import torch

# Initialize DL-Backtrace
model = ...  # Your classification model
sample_input = torch.randn(1, 3, 224, 224)

dlb = DLBacktrace(
    model=model,
    input_for_graph=(sample_input,),
    device="cuda"
)

# Run prediction with temperature scaling
input_data = torch.randn(1, 3, 224, 224).cuda()

node_io = dlb.predict(
    input_data,
    temperature=0.8  # Apply temperature scaling
)

# Access scaled logits
logits = node_io["output"]["output_values"]
print(f"Scaled logits shape: {logits.shape}")
```

#### For Text Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Initialize DL-Backtrace
sample_input = torch.randint(0, 1000, (1, 10))
dlb = DLBacktrace(
    model=model,
    input_for_graph=(sample_input,),
    device="cuda"
)

# Generate with temperature
text = "The future of AI"
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
attention_mask = torch.ones_like(input_ids)

# Temperature affects the prediction
node_io = dlb.predict(
    input_ids,
    attention_mask,
    temperature=1.2  # Higher temperature for creative generation
)

logits = node_io["output"]["output_values"]
```

---

## Temperature in Different Contexts

### 1. Classification Tasks

Control confidence in predictions:

```python
# High confidence (low temperature)
node_io_confident = dlb.predict(input_data, temperature=0.5)
probs_confident = torch.softmax(node_io_confident["output"]["output_values"], dim=-1)

# Standard confidence
node_io_standard = dlb.predict(input_data, temperature=1.0)
probs_standard = torch.softmax(node_io_standard["output"]["output_values"], dim=-1)

# Low confidence (high temperature)
node_io_uncertain = dlb.predict(input_data, temperature=2.0)
probs_uncertain = torch.softmax(node_io_uncertain["output"]["output_values"], dim=-1)

print(f"Low T distribution: {probs_confident[0].tolist()}")
print(f"Standard distribution: {probs_standard[0].tolist()}")
print(f"High T distribution: {probs_uncertain[0].tolist()}")
```

### 2. Text Generation

Control generation diversity:

```python
from dl_backtrace.pytorch_backtrace.dlbacktrace.core.dlb_auto_sampler import DLBAutoSampler

# Create sampler
sampler = DLBAutoSampler(dlb=dlb, tokenizer=tokenizer)

# Generate with different temperatures
prompt = "Once upon a time"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

# Conservative generation (T=0.5)
output_conservative = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.5
)
text_conservative = tokenizer.decode(output_conservative[0])

# Balanced generation (T=0.8)
output_balanced = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.8
)
text_balanced = tokenizer.decode(output_balanced[0])

# Creative generation (T=1.2)
output_creative = sampler.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=1.2
)
text_creative = tokenizer.decode(output_creative[0])

print(f"Conservative (T=0.5): {text_conservative}")
print(f"Balanced (T=0.8): {text_balanced}")
print(f"Creative (T=1.2): {text_creative}")
```

### 3. Model Calibration

Improve probability calibration for uncertainty estimation:

```python
import numpy as np

# Find optimal temperature for calibration
def evaluate_calibration(dlb, validation_data, temperatures):
    results = {}
    
    for temp in temperatures:
        predictions = []
        confidences = []
        
        for data, label in validation_data:
            node_io = dlb.predict(data, temperature=temp)
            logits = node_io["output"]["output_values"]
            probs = torch.softmax(logits, dim=-1)
            
            pred = torch.argmax(probs, dim=-1)
            conf = torch.max(probs, dim=-1).values
            
            predictions.append(pred == label)
            confidences.append(conf.item())
        
        # Calculate Expected Calibration Error (ECE)
        accuracy = np.mean(predictions)
        avg_confidence = np.mean(confidences)
        ece = abs(accuracy - avg_confidence)
        
        results[temp] = {
            'accuracy': accuracy,
            'confidence': avg_confidence,
            'ece': ece
        }
    
    return results

# Test different temperatures
temperatures = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
calibration_results = evaluate_calibration(dlb, validation_data, temperatures)

# Find best temperature
best_temp = min(calibration_results.keys(), 
                key=lambda t: calibration_results[t]['ece'])
print(f"Best calibration temperature: {best_temp}")
```

---

## Advanced Usage

### Dynamic Temperature Selection

Adapt temperature based on input characteristics:

```python
def adaptive_temperature(logits, base_temp=1.0, adapt_factor=0.5):
    """
    Adjust temperature based on logit entropy.
    High entropy → increase temperature
    Low entropy → decrease temperature
    """
    # Calculate entropy of raw logits
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    
    # Normalize entropy (max entropy = log(num_classes))
    num_classes = logits.shape[-1]
    max_entropy = np.log(num_classes)
    normalized_entropy = entropy / max_entropy
    
    # Adjust temperature
    adjusted_temp = base_temp * (1 + adapt_factor * normalized_entropy)
    
    return adjusted_temp

# Use adaptive temperature
node_io = dlb.predict(input_data)
logits = node_io["output"]["output_values"]

# Calculate adaptive temperature
temp = adaptive_temperature(logits, base_temp=1.0, adapt_factor=0.5)

# Re-run with adjusted temperature
node_io = dlb.predict(input_data, temperature=temp.item())
```

### Temperature Annealing

Gradually change temperature during generation:

```python
def generate_with_annealing(sampler, input_ids, max_new_tokens,
                           start_temp=1.5, end_temp=0.5, steps=50):
    """
    Generate text with temperature annealing.
    Start with high temperature (exploration) and gradually reduce (exploitation).
    """
    generated = input_ids.clone()
    
    for step in range(max_new_tokens):
        # Calculate current temperature
        progress = step / max_new_tokens
        current_temp = start_temp + (end_temp - start_temp) * progress
        
        # Generate next token
        output = sampler.generate(
            input_ids=generated,
            max_new_tokens=1,
            temperature=current_temp
        )
        
        # Append new token
        generated = output
    
    return generated

# Use temperature annealing
output = generate_with_annealing(
    sampler=sampler,
    input_ids=input_ids,
    max_new_tokens=50,
    start_temp=1.5,  # Start creative
    end_temp=0.5     # End focused
)

text = tokenizer.decode(output[0])
print(f"Generated with annealing: {text}")
```

### Multi-Temperature Ensemble

Generate multiple outputs with different temperatures:

```python
def multi_temperature_ensemble(dlb, input_data, temperatures, weights=None):
    """
    Combine predictions from multiple temperatures.
    """
    if weights is None:
        weights = [1.0 / len(temperatures)] * len(temperatures)
    
    ensemble_probs = None
    
    for temp, weight in zip(temperatures, weights):
        node_io = dlb.predict(input_data, temperature=temp)
        logits = node_io["output"]["output_values"]
        probs = torch.softmax(logits, dim=-1)
        
        if ensemble_probs is None:
            ensemble_probs = weight * probs
        else:
            ensemble_probs += weight * probs
    
    return ensemble_probs

# Use ensemble
temperatures = [0.5, 1.0, 1.5]
weights = [0.2, 0.5, 0.3]  # Favor standard temperature

ensemble_probs = multi_temperature_ensemble(
    dlb=dlb,
    input_data=input_data,
    temperatures=temperatures,
    weights=weights
)

prediction = torch.argmax(ensemble_probs, dim=-1)
print(f"Ensemble prediction: {prediction}")
```

---

## Best Practices

### 1. Task-Specific Temperature Selection

```python
# Classification tasks
temperature_classification = 1.0  # Standard, well-calibrated

# Factual text generation
temperature_factual = 0.7  # Focused, less random

# Creative text generation
temperature_creative = 1.2  # More diverse

# Code generation
temperature_code = 0.2  # Very focused, deterministic

# Brainstorming
temperature_brainstorm = 1.5  # High diversity
```

### 2. Temperature Calibration Workflow

```python
# 1. Start with standard temperature
base_temp = 1.0

# 2. Evaluate on validation set
results = evaluate_calibration(dlb, validation_data, [base_temp])

# 3. If overconfident (ECE > threshold), increase temperature
if results[base_temp]['ece'] > 0.1:
    if results[base_temp]['confidence'] > results[base_temp]['accuracy']:
        # Model is overconfident
        calibrated_temp = base_temp * 1.5
    else:
        # Model is underconfident
        calibrated_temp = base_temp * 0.7
else:
    calibrated_temp = base_temp

# 4. Use calibrated temperature
node_io = dlb.predict(input_data, temperature=calibrated_temp)
```

### 3. Monitoring Temperature Effects

```python
def analyze_temperature_effects(dlb, input_data, temperature_range):
    """
    Analyze how temperature affects predictions.
    """
    results = {}
    
    for temp in temperature_range:
        node_io = dlb.predict(input_data, temperature=temp)
        logits = node_io["output"]["output_values"]
        probs = torch.softmax(logits, dim=-1)
        
        # Compute metrics
        max_prob = torch.max(probs).item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        prediction = torch.argmax(probs).item()
        
        results[temp] = {
            'max_prob': max_prob,
            'entropy': entropy,
            'prediction': prediction,
            'top3_probs': torch.topk(probs, 3).values.tolist()
        }
    
    return results

# Analyze
temps = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
analysis = analyze_temperature_effects(dlb, input_data, temps)

# Print results
for temp, metrics in analysis.items():
    print(f"\nTemperature {temp}:")
    print(f"  Max Probability: {metrics['max_prob']:.4f}")
    print(f"  Entropy: {metrics['entropy']:.4f}")
    print(f"  Prediction: {metrics['prediction']}")
    print(f"  Top 3 Probs: {metrics['top3_probs']}")
```

---

## Common Use Cases

### 1. Uncertainty Estimation

```python
# High temperature reveals model uncertainty
node_io_uncertain = dlb.predict(input_data, temperature=2.0)
probs = torch.softmax(node_io_uncertain["output"]["output_values"], dim=-1)

# Calculate prediction entropy
entropy = -torch.sum(probs * torch.log(probs + 1e-10))

if entropy > threshold:
    print("High uncertainty - consider manual review")
else:
    print("Low uncertainty - confident prediction")
```

### 2. A/B Testing Generation Styles

```python
# Test different generation styles
styles = {
    'conservative': 0.5,
    'balanced': 0.8,
    'creative': 1.2,
    'experimental': 1.5
}

outputs = {}
for style_name, temp in styles.items():
    output = sampler.generate(
        input_ids=input_ids,
        max_new_tokens=50,
        temperature=temp
    )
    outputs[style_name] = tokenizer.decode(output[0])

# Present different styles to users
for style, text in outputs.items():
    print(f"\n{style.upper()} STYLE:")
    print(text)
```

### 3. Confidence Thresholding

```python
# Only accept predictions above confidence threshold
min_confidence = 0.9
temperature = 1.0

node_io = dlb.predict(input_data, temperature=temperature)
probs = torch.softmax(node_io["output"]["output_values"], dim=-1)
max_prob = torch.max(probs).item()

if max_prob < min_confidence:
    # Try lower temperature for higher confidence
    node_io = dlb.predict(input_data, temperature=0.5)
    probs = torch.softmax(node_io["output"]["output_values"], dim=-1)
    max_prob = torch.max(probs).item()
    
    if max_prob < min_confidence:
        print("Prediction rejected - insufficient confidence")
    else:
        prediction = torch.argmax(probs)
        print(f"Prediction (low T): {prediction} (confidence: {max_prob:.4f})")
else:
    prediction = torch.argmax(probs)
    print(f"Prediction: {prediction} (confidence: {max_prob:.4f})")
```

---

## Performance Considerations

Temperature scaling has **minimal performance overhead**:

- Applied efficiently using in-place division
- No additional model forward passes required
- Same computational cost for any temperature value

```python
import time

# Measure overhead
start = time.time()
node_io_no_temp = dlb.predict(input_data)
time_no_temp = time.time() - start

start = time.time()
node_io_temp = dlb.predict(input_data, temperature=0.8)
time_temp = time.time() - start

print(f"Without temperature: {time_no_temp:.4f}s")
print(f"With temperature: {time_temp:.4f}s")
print(f"Overhead: {(time_temp - time_no_temp) * 1000:.2f}ms")
# Typically <1ms overhead
```

---

## Next Steps

- Learn about [DLB Auto Sampler](auto-sampler.md) for advanced generation
- Explore [Pipeline](pipeline.md) for high-level workflows
- Check [MoE Models](moe-models.md) for expert-based generation
- See [Examples](../../examples/colab-notebooks.md) for complete use cases

