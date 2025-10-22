# Task Types

Different tasks require different relevance evaluation approaches.

---

## Classification Tasks

### Binary Classification

```python
relevance = dlb.evaluation(
    task="binary-classification"
)
```

For two-class problems (yes/no, positive/negative).

### Multi-class Classification

```python
relevance = dlb.evaluation(
    task="multi-class classification"
)
```

For problems with more than two classes.

---

## Regression Tasks

### Bounding Box Regression

```python
relevance = dlb.evaluation(
    task="bbox-regression"
)
```

For object detection models predicting bounding boxes.

---

## Segmentation Tasks

### Binary Segmentation

```python
relevance = dlb.evaluation(
    task="binary-segmentation",
    thresholding=0.5
)
```

For pixel-level binary classification (e.g., foreground/background).

---

## Encoder-Decoder Tasks

For sequence-to-sequence models:

```python
relevance = dlb.evaluation(
    model_type="Encoder_Decoder"
)
```

Used for translation, summarization, etc.

---

See [Parameters](parameters.md) for more configuration options.



