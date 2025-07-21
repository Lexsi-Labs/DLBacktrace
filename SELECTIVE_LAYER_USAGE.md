# 🎯 Selective Layer Implementation - Quick Usage Guide

## ✨ What's New?

You can now choose **different implementations for specific layer types** while keeping others as original. Perfect for testing and gradual optimization!

## 🚀 Basic Usage

### Global Implementation (Before)
```python
# Old way - applies to ALL layers
dlb = DLBacktraceFX(model, input, layer_implementation="cuda")
```

### Selective Implementation (New!)
```python
# New way - choose per layer type
layer_config = {
    "linear": "cuda",           # Only linear layers use CUDA
    "attention": "cuda",        # Only attention layers use CUDA  
    "default": "original"       # Everything else uses original
}

dlb = DLBacktraceFX(model, input, layer_implementation=layer_config)
```

## 📋 Available Layer Types

| Layer Type | Description | Common Layers |
|-----------|-------------|---------------|
| `linear` | Linear/Dense layers | `nn.Linear`, MLP layers |
| `conv2d` | Convolutional layers | `nn.Conv2d` |
| `attention` | Attention mechanisms | Self-attention, multi-head attention |
| `embedding` | Embedding layers | `nn.Embedding` |
| `multiply` | Element-wise multiplication | `torch.mul`, `*` |
| `add` | Addition operations | `torch.add`, `+` |
| `default` | Fallback for all others | **Required in dict config** |

## 🎯 Common Use Cases

### 1. Test Linear Layer Upgrades Only
```python
# Perfect for testing - upgrade critical layers, keep others stable
config = {
    "linear": "cuda",         # Test CUDA linear performance
    "default": "original"     # Keep everything else original
}
```

### 2. Optimize Transformer Models
```python
# Target transformer bottlenecks
config = {
    "linear": "cuda",         # Critical for transformer performance
    "attention": "cuda",      # Optimize attention mechanisms
    "default": "original"     # Stable fallback
}
```

### 3. Gradual Migration Strategy
```python
# Phase 1: Test linear layers
phase1 = {"linear": "cuda", "default": "original"}

# Phase 2: Add attention
phase2 = {"linear": "cuda", "attention": "cuda", "default": "original"}

# Phase 3: Full acceleration
phase3 = "cuda"  # Or {"default": "cuda"}
```

### 4. Debug Specific Implementations
```python
# Test only multiplication operations
debug_config = {
    "multiply": "pytorch",    # Test PyTorch multiply implementation
    "default": "original"     # Keep everything else stable
}
```

## 📊 Performance Testing Example

```python
# Compare configurations
configs = [
    "original",                                    # Baseline
    {"linear": "cuda", "default": "original"},     # Linear only
    {"linear": "cuda", "attention": "cuda", "default": "original"},  # Critical layers
    "cuda"                                         # Full acceleration
]

for i, config in enumerate(configs):
    dlb = DLBacktraceFX(model, input, layer_implementation=config)
    # ... run tests and compare performance
```

## 🛡️ Safety Features

- **Auto-fallback**: If CUDA requested but unavailable, falls back to original
- **Validation**: Checks for valid layer types and implementations
- **Default required**: Must specify `"default"` in dictionary configs
- **Error handling**: Graceful handling of missing implementations

## ⚠️ Important Notes

1. **Always specify `"default"`** when using dictionary configuration
2. **CUDA availability** is checked automatically
3. **Mixed implementations** maintain numerical consistency
4. **Fallback behavior** ensures your code always works

## 🎉 Ready to Use!

```python
# Your perfect testing configuration
test_config = {
    "linear": "cuda",        # 🚀 Accelerate the bottleneck
    "default": "original"    # 🛡️ Keep everything else stable
}

dlb = DLBacktraceFX(model, input, layer_implementation=test_config)
io_data = dlb.predict(input)
dlb.evaluation(task="classification")
```

**Perfect for testing upgraded linear layers while keeping the rest of your system stable!** 🎯 