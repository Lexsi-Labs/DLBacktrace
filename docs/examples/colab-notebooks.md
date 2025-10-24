# Google Colab Notebooks

Interactive Google Colab notebooks to learn DL-Backtrace hands-on.

---

## PyTorch Examples

### Vision Models

#### ResNet Image Classification
Explain ResNet predictions on ImageNet.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mpo--AD8vNqm6Y05rb46Yzjx6VhLwXZh?usp=sharing)

**What you'll learn:**
- Loading pre-trained ResNet
- Preparing image inputs
- Calculating relevance
- Visualizing saliency maps

---

#### VGG Image Classification
Apply DL-Backtrace to VGG networks.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iJJZ0ApWHltTjnbGRKhJDrHlKTlm1koD?usp=sharing)

**What you'll learn:**
- VGG architecture tracing
- Multi-layer relevance analysis
- Comparing different VGG variants

---

#### Vision Transformer (ViT)
Explain ViT model decisions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BhzIw7Pf9-g1tqndaijwZ5FLaDUpjBaR?usp=sharing)

**What you'll learn:**
- Transformer architecture tracing
- Attention mechanism analysis
- Patch-level importance

---

#### DenseNet Classification
Analyze DenseNet predictions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CE2XBBGd5VSQuipJTyyRcb7mu5RSG6K5?usp=sharing)

**What you'll learn:**
- Dense connection tracing
- Feature reuse analysis
- Comparing DenseNet variants

---

#### EfficientNet Classification
Explain EfficientNet decisions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O-MyvIKWoADG2RrF43p2k8mUYpF9N_8m?usp=sharing)

**What you'll learn:**
- Mobile-friendly model explanation
- Compound scaling analysis
- Efficiency vs accuracy trade-offs

---

#### MobileNet Classification
Lightweight model explanations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BzsID9U3HndLrh67nPWRw_bm7UWLwLOH?usp=sharing)

**What you'll learn:**
- Depthwise separable convolutions
- Mobile deployment considerations
- Relevance in lightweight models

---

### NLP Models

#### BERT Sentiment Analysis
Explain BERT classifications.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ANZPjaAxl2oF2WHj23f87AR9-ZDIMBm9?usp=sharing)

**What you'll learn:**
- BERT model tracing
- Token-level attribution
- Attention pattern analysis

---

#### ALBERT Sentiment Classification
Lightweight BERT variant.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RuAW0FgtWqKkdVbc97VXf9z4oDXmA1ms?usp=sharing)

**What you'll learn:**
- Parameter sharing analysis
- ALBERT vs BERT comparisons
- Memory-efficient explanations

---

#### RoBERTa Classification
Robust BERT training.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Nw6lTSQKJvGU9JBZeXnA7EXboU7mE282?usp=sharing)

**What you'll learn:**
- RoBERTa architecture
- Bidirectional attention
- Token importance analysis

---

#### DistilBERT Sentiment
Distilled BERT model.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13_hqUC2vaJWfF-UheggHJU5RmWS5A2u3?usp=sharing)

**What you'll learn:**
- Knowledge distillation effects
- Speed vs accuracy trade-offs
- Compact model explanations

---

#### ELECTRA Classification
Efficient transformer training.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sht3uLej8g-4hMtaHm7VwwUuGAmAqpH_?usp=sharing)

**What you'll learn:**
- ELECTRA discriminative training
- Replaced token detection
- Efficient transformer explanations

---

#### XLNet Classification
Permutation language modeling.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZmVusCPgeXLGnbt7NzM-3SJuiGRgTzBa?usp=sharing)

**What you'll learn:**
- XLNet architecture
- Permutation-based attention
- Two-stream self-attention

---

### Generative Models

#### LLaMA-3.2-1B Text Generation
Explain LLaMA predictions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1i_CKoCfKdY4fcWyFdzuc_0e868jux12h?usp=sharing)

**What you'll learn:**
- Large language model tracing
- Causal attention analysis
- Token generation explanations

---

#### LLaMA-3.2-3B Text Generation
Larger LLaMA variant.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ki8kcc4ez8-kdvdlhtoq7Sed9v5hiaNs?usp=sharing)

**What you'll learn:**
- Scaling law effects
- Multi-head attention analysis
- Generation quality vs explanation

---

### Tabular Models

#### Custom Tabular Binary Classification
Explain tabular models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TqgeeBqQ1G9UGWfHV0MUloCccalpsRCh?usp=sharing)

**What you'll learn:**
- Feature importance for tabular data
- Binary classification explanations
- Custom model tracing

---

!!! note "TensorFlow Examples Deprecated"
    TensorFlow/Keras support is being deprecated. Please use the PyTorch backend for new projects. Existing TensorFlow notebooks are available for reference but may not receive updates.

---

## How to Use These Notebooks

### 1. Open in Colab

Click any badge above to open the notebook in Google Colab.

### 2. Copy to Your Drive

```
File → Save a copy in Drive
```

This creates your own editable copy.

### 3. Enable GPU (Optional)

```
Runtime → Change runtime type → Hardware accelerator → GPU
```

Speeds up execution significantly.

### 4. Run Cells

Execute cells in order:
```
Runtime → Run all
```

Or run individually with `Shift+Enter`.

### 5. Modify & Experiment

Try:
- Different models
- Different inputs
- Different parameters
- Your own data

---

## Tips for Success

!!! tip "Start with Simple Examples"
    Begin with classification notebooks before advanced architectures.

!!! tip "Read the Comments"
    Notebooks include detailed explanations inline.

!!! tip "Experiment"
    Modify code to understand how changes affect results.

!!! tip "Save Your Work"
    Copy notebooks to your Drive before making changes.

---

## Next Steps

- [Use Cases](use-cases.md) - Real-world applications
- [User Guide](../guide/introduction.md) - Comprehensive documentation
- [Best Practices](../guide/best-practices.md) - Tips for effective use
- [Developer Guide](../developer/contributing.md) - Contributing to the project



