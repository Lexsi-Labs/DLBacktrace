#!/usr/bin/env python3
"""
Test embedding operation in isolation
"""

import torch
import torch.nn as nn
import sys
import os

# Add DL-Backtrace to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dl_backtrace', 'pytorch_backtrace'))
from dlbacktrace import DLBacktraceFX

def test_embedding():
    """Test embedding operation precision"""
    print('🔍 Testing Embedding operation...')

    class TestEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            # RoBERTa-like embedding
            self.embedding = nn.Embedding(50264, 768)
            
        def forward(self, input_ids):
            return self.embedding(input_ids)

    # Simple test input
    input_ids = torch.tensor([[101, 102, 103]])

    model = TestEmbedding()

    print(f'Input: {input_ids}')

    # Direct execution
    with torch.no_grad():
        direct_output = model(input_ids)
    print(f'Direct output: {direct_output.shape}, sample: {direct_output[0, 0, :3]}')

    # Traced execution
    try:
        dlb = DLBacktraceFX(model, (input_ids,), verbose=False)
        result = dlb.predict(input_ids, debug=False)
        
        output_node = list(result.keys())[-1]
        traced_output = result[output_node]['output_values']
        print(f'Traced output: {traced_output.shape}, sample: {traced_output[0, 0, :3]}')
        
        diff = torch.max(torch.abs(direct_output - traced_output)).item()
        print(f'Max difference: {diff:.6f}')
        
        if diff < 1e-5:
            print('✅ Embedding working correctly!')
            return True
        else:
            print('❌ Embedding has issues!')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == "__main__":
    success = test_embedding()
    sys.exit(0 if success else 1)
