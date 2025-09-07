#!/usr/bin/env python3
"""
Test cumsum operation (used for position embeddings in RoBERTa)
"""

import torch
import torch.nn as nn
import sys
import os

# Add DL-Backtrace to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dl_backtrace', 'pytorch_backtrace'))
from dlbacktrace import DLBacktraceFX

def test_cumsum():
    """Test cumsum operation precision"""
    print('🔍 Testing cumsum operation...')

    class TestCumsum(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            # RoBERTa uses cumsum for position generation
            return torch.cumsum(x, dim=-1)

    # Test with attention mask-like input (this is what RoBERTa does)
    x = torch.ones(2, 10, dtype=torch.long)  # Attention mask
    print(f'Input: {x[0]}')

    model = TestCumsum()

    # Direct execution
    with torch.no_grad():
        direct_output = model(x)
    print(f'Direct cumsum: {direct_output[0]}')

    # Traced execution
    try:
        dlb = DLBacktraceFX(model, (x,), verbose=False)
        result = dlb.predict(x, debug=False)
        
        output_node = list(result.keys())[-1]
        traced_output = result[output_node]['output_values']
        print(f'Traced cumsum: {traced_output[0]}')
        
        diff = torch.max(torch.abs(direct_output - traced_output)).item()
        print(f'Max difference: {diff:.6f}')
        
        if diff < 1e-5:
            print('✅ Cumsum working correctly!')
            return True
        else:
            print('❌ Cumsum has issues!')
            print(f'Direct: {direct_output.dtype}, Traced: {traced_output.dtype}')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == "__main__":
    success = test_cumsum()
    sys.exit(0 if success else 1)
