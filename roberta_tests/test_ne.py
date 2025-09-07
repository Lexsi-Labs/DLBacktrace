#!/usr/bin/env python3
"""
Test ne (not equal) operation used for attention mask processing
"""

import torch
import torch.nn as nn
import sys
import os

# Add DL-Backtrace to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dl_backtrace', 'pytorch_backtrace'))
from dlbacktrace import DLBacktraceFX

def test_ne():
    """Test ne (not equal) operation"""
    print('🔍 Testing ne (not equal) operation...')

    class TestNe(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, attention_mask):
            # RoBERTa uses ne for attention mask processing
            # attention_mask.ne(0) creates boolean mask
            return attention_mask.ne(0)

    # Test with attention mask
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])  # Typical attention mask

    model = TestNe()

    print(f'Input: {attention_mask}')

    # Direct execution
    with torch.no_grad():
        direct_output = model(attention_mask)
    print(f'Direct ne: {direct_output}')

    # Traced execution
    try:
        dlb = DLBacktraceFX(model, (attention_mask,), verbose=False)
        result = dlb.predict(attention_mask, debug=False)
        
        output_node = list(result.keys())[-1]
        traced_output = result[output_node]['output_values']
        print(f'Traced ne: {traced_output}')
        
        # For boolean tensors, check exact equality
        match = torch.equal(direct_output, traced_output)
        print(f'Exact match: {match}')
        
        if match:
            print('✅ Ne working correctly!')
            return True
        else:
            print('❌ Ne has issues!')
            print(f'Direct dtype: {direct_output.dtype}, Traced dtype: {traced_output.dtype}')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == "__main__":
    success = test_ne()
    sys.exit(0 if success else 1)
