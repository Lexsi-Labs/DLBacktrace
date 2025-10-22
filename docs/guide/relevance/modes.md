# Evaluation Modes

DL-Backtrace supports different evaluation modes for relevance propagation.

---

## Default Mode

Standard relevance propagation from output to input.

```python
relevance = dlb.evaluation(mode="default")
```

**Use when**: You want to understand which features contributed to the prediction.

---

## Contrastive Mode

Compare relevance between different classes or outputs.

```python
relevance = dlb.evaluation(mode="contrastive")
```

**Use when**: You want to understand what distinguishes one class from another.

---

## Choosing a Mode

- **Default**: Most common use case, general explanations
- **Contrastive**: When comparing predictions or classes

See [Relevance Overview](overview.md) for more details.



