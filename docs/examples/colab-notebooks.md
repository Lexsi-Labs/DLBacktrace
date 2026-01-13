# Google Colab Notebooks

Interactive Google Colab notebooks to learn DLBacktrace hands-on.

---

## PyTorch Examples

### Vision Models

#### ResNet Image Classification
Explain ResNet predictions on ImageNet.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O8Is0X-IrKxXzgJeR1Xxy21OOR-UGfm7?usp=sharing)

**What you'll learn:**
- Loading pre-trained ResNet
- Preparing image inputs
- Calculating relevance
- Visualizing saliency maps

---

#### VGG Image Classification
Apply DLBacktrace to VGG networks.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LxSazvy1Y2i0Ho1Qw7K905aNmvh66GzR?usp=sharing)

**What you'll learn:**
- VGG architecture tracing
- Multi-layer relevance analysis
- Comparing different VGG variants

---

#### Vision Transformer (ViT)
Explain ViT model decisions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1B1xN5w51-tRnHyycbHV6lxi0tKxONnoS?usp=sharing)

**What you'll learn:**
- Transformer architecture tracing
- Attention mechanism analysis
- Patch-level importance

---

#### DenseNet Classification
Analyze DenseNet predictions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TspNRXd-qf81iZepUpU_TrUokZynynbu?usp=sharing)

**What you'll learn:**
- Dense connection tracing
- Feature reuse analysis
- Comparing DenseNet variants

---

#### EfficientNet Classification
Explain EfficientNet decisions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xv9ghxp0OXcfYOseoq37KiJbz-UkLvWq?usp=sharing)

**What you'll learn:**
- Mobile-friendly model explanation
- Compound scaling analysis
- Efficiency vs accuracy trade-offs

---

#### MobileNet Classification
Lightweight model explanations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12D6NgYTf4Gud3BXLRfnUK1m5-RLZY6dv?usp=sharing)

**What you'll learn:**
- Depthwise separable convolutions
- Mobile deployment considerations
- Relevance in lightweight models

---

### NLP Models

#### BERT Sentiment Analysis
Explain BERT classifications.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eBAQ8TToJUs6EN4-md08cjfZUmAArX0s?usp=drive_link)

**What you'll learn:**
- BERT model tracing
- Token-level attribution
- Attention pattern analysis

---

### Generative Models

#### LLaMA-3.2-1B Text Generation
Explain LLaMA predictions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eZUK-PwI6iZeKnfcQHxIXO7xo8wYDEek?usp=sharing)

**What you'll learn:**
- Large language model tracing
- Causal attention analysis
- Token generation explanations

---

#### Qwen-3-0.6B Text Generation
Explain Qwen predictions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nJsxk7HPhiZ1DUIXn0eOnu0yjye-WOQT?usp=drive_link)

**What you'll learn:**
- Scaling law effects
- Multi-head attention analysis
- Generation quality vs explanation

---

### Mixture-of-Experts (MoEs) Models

#### JetMoE 
Explain JetMoE expert routing decisions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MKxR60Mf1F_TNMuE31T3GRWHFPTm5IhP?usp=drive_link)

**What you'll learn:**
- MoE model tracing
- Expert routing mechanisms
- Sparse activation patterns
- Load balancing in expert selection

---

#### OLMoE 
Explain OLMoE predictions and expert selection.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1e13QAFw4A8jKhZVb1jh-C_16A2u1vZIH?usp=drive_link)

**What you'll learn:**
- MoE model tracing
- Open language model architecture
- Expert specialization analysis
- Scaling efficiency in MoE systems

---

#### Qwen3-MoE 
Explain Qwen3-MoE multi-lingual predictions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ja26JGNt22rOMT0HCyjOMASXxfnMpkm8?usp=sharing)

**What you'll learn:**
- MoE model tracing
- Multi-lingual expert routing
- Cross-lingual knowledge transfer
- Advanced MoE architectures

---

#### GPT-Oss 
Explain GPT-Oss open-source MoE predictions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zZXTOV4NOrauNDXjjYB8rausbBGoLYtX?usp=sharing)

**What you'll learn:**
- MoE model tracing
- Open-source GPT architecture
- Expert gating mechanisms
- Performance optimization in MoE

---

### Tabular Models

#### Custom Tabular Binary Classification
Explain tabular models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13N2sfAxA_7GJsZ7VAKS5WIOg6GusQghI?usp=sharing)

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



