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
│ DLBacktraceFX   │  ← Main Entry Point
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
- Traces model using `torch.export`
- Extracts nodes and parameters
- Builds NetworkX graph
- Performs topological sort

### 2. Execution Engine
- Executes operations
- Tracks activations
- Manages memory
- Handles devices

### 3. Relevance Propagation
- Calculates relevance scores
- Propagates backward
- Handles different layer types

### 4. Visualization
- Generates graph images
- Creates heatmaps
- Produces reports

---

## Data Flow

1. **Input**: Model + dummy input
2. **Trace**: Extract computational graph
3. **Execute**: Run forward pass
4. **Evaluate**: Calculate relevance
5. **Visualize**: Generate outputs

---

See [Developer Guide](../dev_notes/DEVELOPER_GUIDE.md) for implementation details.



