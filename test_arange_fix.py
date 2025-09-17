#!/usr/bin/env python
# Test script to verify arange fix

import os
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.export import Dim
from transformers import AutoTokenizer, AutoModelForCausalLM
from dl_backtrace.pytorch_backtrace import DLBacktraceFX

print("Starting arange fix test...")

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

# Create model and input
model = SimpleModel()
x = torch.randn(1, 10)

print("Creating DLBacktraceFX...")
try:
    dlb = DLBacktraceFX(
        model=model,
        input_for_graph=(x,),
        layer_implementation="pytorch",
        verbose=True
    )
    print("DLBacktraceFX created successfully")
    
    # Run prediction
    print("Running prediction...")
    result = dlb.predict(x)
    print("Prediction completed successfully")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
