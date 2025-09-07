#!/usr/bin/env python3
"""
Test scaled_dot_product_attention operation in isolation
"""

import torch
import torch.nn as nn
import sys
import os

# Add DL-Backtrace to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dl_backtrace', 'pytorch_backtrace'))
from dlbacktrace import DLBacktraceFX

def test_attention():
    """Test attention mechanism precision"""
    print('🔍 Testing scaled_dot_product_attention operation...')

    class TestSDPA(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, q, k, v):
            # RoBERTa uses bidirectional attention (is_causal=False)
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )
            return output

    # Test with RoBERTa-like dimensions
    batch_size, num_heads, seq_len, head_dim = 2, 12, 10, 64
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    model = TestSDPA()

    print(f'Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}')

    # Direct execution
    with torch.no_grad():
        direct_output = model(q, k, v)
    print(f'Direct output: {direct_output.shape}, sample: {direct_output[0, 0, 0, :3]}')

    # Traced execution
    try:
        dlb = DLBacktraceFX(model, (q, k, v), verbose=False)
        result = dlb.predict(q, k, v, debug=False)
        
        output_node = list(result.keys())[-1]
        traced_output = result[output_node]['output_values']
        print(f'Traced output: {traced_output.shape}, sample: {traced_output[0, 0, 0, :3]}')
        
        diff = torch.max(torch.abs(direct_output - traced_output)).item()
        print(f'Max difference: {diff:.6f}')
        
        if diff < 1e-5:
            print('✅ SDPA working correctly!')
            return True
        else:
            print('❌ SDPA has issues!')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == "__main__":
    success = test_attention()
    sys.exit(0 if success else 1)
