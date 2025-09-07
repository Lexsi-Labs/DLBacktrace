#!/usr/bin/env python3
"""
Test LayerNorm operation in isolation
"""

import torch
import torch.nn as nn
import sys
import os

# Add DL-Backtrace to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dl_backtrace', 'pytorch_backtrace'))
from dlbacktrace import DLBacktraceFX

def test_layernorm():
    """Test LayerNorm operation precision"""
    print('🔍 Testing LayerNorm operation...')

    class TestLayerNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm = nn.LayerNorm(768)
            
        def forward(self, x):
            return self.layer_norm(x)

    # Test with RoBERTa-like dimensions
    batch_size, seq_len, hidden_size = 2, 10, 768
    x = torch.randn(batch_size, seq_len, hidden_size)

    model = TestLayerNorm()

    print(f'Input shape: {x.shape}')

    # Direct execution
    with torch.no_grad():
        direct_output = model(x)
    print(f'Direct output: {direct_output.shape}, sample: {direct_output[0, 0, :5]}')

    # Traced execution
    try:
        dlb = DLBacktraceFX(model, (x,), verbose=False)
        result = dlb.predict(x, debug=False)
        
        output_node = list(result.keys())[-1]
        traced_output = result[output_node]['output_values']
        print(f'Traced output: {traced_output.shape}, sample: {traced_output[0, 0, :5]}')
        
        diff = torch.max(torch.abs(direct_output - traced_output)).item()
        print(f'Max difference: {diff:.6f}')
        
        if diff < 1e-5:
            print('✅ LayerNorm working correctly!')
            return True
        else:
            print('❌ LayerNorm has issues!')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == "__main__":
    success = test_layernorm()
    sys.exit(0 if success else 1)
