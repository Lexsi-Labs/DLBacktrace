# PyTorch Examples

Code examples for using DL-Backtrace with PyTorch models.

---

## Vision Models

### ResNet Classification

```python
import torch
import torchvision.models as models
from dl_backtrace.pytorch_backtrace import DLBacktraceFX

# Load model
model = models.resnet18(pretrained=True)
model.eval()

# Initialize
dummy_input = torch.randn(1, 3, 224, 224)
dlb = DLBacktraceFX(model, input_for_graph=(dummy_input,))

# Analyze
node_io = dlb.predict(dummy_input)
relevance = dlb.evaluation(
    mode="default",
    multiplier=100.0,
    task="multi-class classification"
)
```

### Custom CNN

```python
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        return self.fc(x.flatten(1))

model = CustomCNN()
# Use with DLBacktraceFX as shown above
```

---

## NLP Models

### BERT Sentiment

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "This is amazing!"
inputs = tokenizer(text, return_tensors="pt")

dlb = DLBacktraceFX(
    model,
    input_for_graph=(inputs['input_ids'], inputs['attention_mask'])
)

node_io = dlb.predict(inputs['input_ids'], inputs['attention_mask'])
```

---

See [Colab Notebooks](colab-notebooks.md) for interactive examples.



