# Architecture

Understanding DL-Backtrace's architecture.

---

## System Overview

```
┌─────────────────┐
│  User's Model   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DLBacktrace   │  ← Main Entry Point
└────────┬────────┘
         │
         ├─────────────────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌──────────────────┐
│ Graph Builder   │   │ Trace Utils      │
└────────┬────────┘   └──────────────────┘
         │
         ▼
┌─────────────────┐
│ Execution Engine│
└────────┬────────┘
         │
         ├─────────────────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌──────────────────┐
│ Relevance Prop  │   │ Visualization    │
└─────────────────┘   └──────────────────┘
```

---

## Core Components

### 1. Graph Builder
Uses PyTorch Train Export to construct the computational graphs with fundamental torch aten node operations = layers + inline operation alongside weights and hyperparameter extraction for comprehensive node-wise model output tracing.

### 2. Execution Engine
High-performance forward pass execution using the computation graph to compute the output of each node while ensuring consistent precision and correct hyper-params and weight being used with settings to ensure deterministic results. With warning logs for extremely high values.

### 3. Relevance Propagation
Layer-specific algorithms (Linear, Convolutional, Attention) that distribute relevance based on each layer's mathematical properties and activation patterns
It distributes relevance scores across layers, providing insights into feature importance, information flow, and bias, enabling better model interpretation and validation without external dependencies.

---

## Data Flow

1. **Input**: Model + dummy input
2. **Trace**: Extract computational graph
3. **Execute**: Run forward pass
4. **Evaluate**: Calculate relevance
5. **Visualize**: Generate outputs

---



