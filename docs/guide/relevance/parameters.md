# Evaluation Parameters

Complete reference for relevance evaluation parameters.

---

## Core Parameters

### mode
- **Type**: `str`
- **Default**: `"default"`
- **Options**: `"default"`, `"contrastive"`
- **Description**: Evaluation mode for relevance propagation

### multiplier
- **Type**: `float`
- **Default**: `100.0`
- **Description**: Starting relevance value at output layer

### task
- **Type**: `str`
- **Required**: Yes
- **Options**: 
  - `"binary-classification"`
  - `"multi-class classification"`
- **Description**: Type of task the model performs

---

## Optional Parameters

### thresholding
- **Type**: `float`
- **Default**: `0.5`
- **Description**: Threshold for segmentation tasks

### scaler
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Additional scaling factor for relevance

---

## Example Usage

```python
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="multi-class classification",
)
```

---



