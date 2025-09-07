#!/usr/bin/env python3
"""
Test add operation (residual connections in transformers)
"""

import torch
import torch.nn as nn
import sys
import os

# Add DL-Backtrace to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dl_backtrace', 'pytorch_backtrace'))
from dlbacktrace import DLBacktraceFX

def test_add():
    """Test add operation (residual connections)"""
    print('🔍 Testing add operation (residual connections)...')

    class TestAdd(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 768)
            
        def forward(self, x):
            # Typical transformer residual connection
            residual = x
            output = self.linear(x)
            return output + residual  # This is the add operation

    # Test with RoBERTa-like dimensions
    x = torch.randn(2, 10, 768)

    model = TestAdd()

    print(f'Input shape: {x.shape}')

    # Direct execution
    with torch.no_grad():
        direct_output = model(x)
    print(f'Direct output: {direct_output.shape}, sample: {direct_output[0, 0, :3]}')

    # Traced execution
    try:
        dlb = DLBacktraceFX(model, (x,), verbose=False)
        result = dlb.predict(x, debug=False)
        
        output_node = list(result.keys())[-1]
        traced_output = result[output_node]['output_values']
        print(f'Traced output: {traced_output.shape}, sample: {traced_output[0, 0, :3]}')
        
        diff = torch.max(torch.abs(direct_output - traced_output)).item()
        print(f'Max difference: {diff:.6f}')
        
        if diff < 1e-5:
            print('✅ Add working correctly!')
            return True
        else:
            print('❌ Add has issues!')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == "__main__":
    success = test_add()
    sys.exit(0 if success else 1)
