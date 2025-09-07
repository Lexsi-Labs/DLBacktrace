#!/usr/bin/env python3
"""
Test slice operations on buffer tensors
"""

import torch
import torch.nn as nn
import sys
import os

# Add DL-Backtrace to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dl_backtrace', 'pytorch_backtrace'))
from dlbacktrace import DLBacktraceFX

def test_slice():
    """Test slice operation on buffer tensors"""
    print('🔍 Testing slice operation...')

    class TestSlice(nn.Module):
        def __init__(self):
            super().__init__()
            # Create a buffer tensor like RoBERTa's token_type_ids
            self.register_buffer('token_type_ids', torch.zeros(1, 514, dtype=torch.long))
            
        def forward(self, input_ids):
            seq_len = input_ids.shape[1]
            # This is what RoBERTa does - slice the buffer
            token_type_ids = self.token_type_ids[:, :seq_len]
            return token_type_ids

    # Test input
    input_ids = torch.tensor([[101, 102, 103]])  # 3 tokens
    seq_len = input_ids.shape[1]

    model = TestSlice()

    print(f'Input shape: {input_ids.shape}')
    print(f'Buffer shape: {model.token_type_ids.shape}')

    # Direct execution
    with torch.no_grad():
        direct_output = model(input_ids)
    print(f'Direct slice: {direct_output.shape}, values: {direct_output}')

    # Traced execution
    try:
        dlb = DLBacktraceFX(model, (input_ids,), verbose=False)
        result = dlb.predict(input_ids, debug=False)
        
        output_node = list(result.keys())[-1]
        traced_output = result[output_node]['output_values']
        print(f'Traced slice: {traced_output.shape}, values: {traced_output}')
        
        diff = torch.max(torch.abs(direct_output.float() - traced_output.float())).item()
        print(f'Max difference: {diff:.6f}')
        
        if diff < 1e-5:
            print('✅ Slice working correctly!')
            return True
        else:
            print('❌ Slice has issues!')
            print(f'Direct dtype: {direct_output.dtype}, Traced dtype: {traced_output.dtype}')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == "__main__":
    success = test_slice()
    sys.exit(0 if success else 1)
