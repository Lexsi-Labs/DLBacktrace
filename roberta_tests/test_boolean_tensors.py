#!/usr/bin/env python3
"""
Test boolean tensor handling (fix for the abs_cpu error)
"""

import torch
import torch.nn as nn
import sys
import os

# Add DL-Backtrace to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dl_backtrace', 'pytorch_backtrace'))
from dlbacktrace import DLBacktraceFX

def test_boolean_tensors():
    """Test model with operations that produce boolean tensors"""
    print('🔍 Testing boolean tensor handling...')

    class BooleanModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
        
        def forward(self, x):
            # Linear operation
            logits = self.linear(x)
            
            # Create boolean tensor (this was causing the error)
            mask = logits > 0.0  # This produces a boolean tensor
            
            # Use the boolean mask in a meaningful way
            return torch.where(mask, logits, torch.zeros_like(logits))

    # Test input
    x = torch.randn(2, 10)

    model = BooleanModel()

    print(f'Input shape: {x.shape}')

    # Direct execution
    with torch.no_grad():
        direct_output = model(x)
    print(f'Direct output: {direct_output.shape}')

    # Traced execution
    try:
        dlb = DLBacktraceFX(model, (x,), verbose=False)
        result = dlb.predict(x, debug=False)
        
        output_node = list(result.keys())[-1]
        traced_output = result[output_node]['output_values']
        print(f'Traced output: {traced_output.shape}')
        
        diff = torch.max(torch.abs(direct_output - traced_output)).item()
        print(f'Max difference: {diff:.6f}')
        
        if diff < 1e-5:
            print('✅ Boolean tensor handling working correctly!')
            return True
        else:
            print('❌ Boolean tensor handling has issues!')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_boolean_tensors()
    sys.exit(0 if success else 1)
