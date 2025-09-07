#!/usr/bin/env python3
"""
Analyze tensor shapes throughout RoBERTa execution
"""

import torch
from transformers import RobertaForSequenceClassification
import sys
import os

# Add DL-Backtrace to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dl_backtrace', 'pytorch_backtrace'))
from dlbacktrace import DLBacktraceFX

def test_tensor_shapes():
    """Analyze RoBERTa tensor shapes safely"""
    print('🔍 Analyzing tensor shapes throughout execution...')

    class SimpleRoBERTa(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = RobertaForSequenceClassification.from_pretrained(
                'cardiffnlp/twitter-roberta-base-sentiment-latest',
                torch_dtype=torch.float32,
                use_cache=False
            ).eval()

        def forward(self, input_ids, attention_mask):
            return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

    model = SimpleRoBERTa()
    input_ids = torch.tensor([[101, 102, 103]])
    attention_mask = torch.ones_like(input_ids)

    try:
        # Get traced execution
        dlb = DLBacktraceFX(model, (input_ids, attention_mask), verbose=False)
        result = dlb.predict(input_ids, attention_mask, debug=False)

        print(f'Total nodes in execution: {len(result)}')
        
        # Analyze tensor dimensions safely
        print('\\nTensor dimension analysis:')
        dimension_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        
        for node_name, node_data in result.items():
            if 'output_values' in node_data:
                output = node_data['output_values']
                if isinstance(output, torch.Tensor):
                    dims = len(output.shape)
                    if dims in dimension_counts:
                        dimension_counts[dims] += 1
                    else:
                        dimension_counts[dims] = 1

        for dims, count in dimension_counts.items():
            print(f'  {dims}D tensors: {count}')

        # Find actual embedding-like tensors (3D with correct dimensions)
        print('\\nLooking for embedding-like tensors:')
        embedding_candidates = []
        
        for node_name, node_data in result.items():
            if 'output_values' in node_data:
                output = node_data['output_values']
                if isinstance(output, torch.Tensor) and len(output.shape) == 3:
                    batch, seq, hidden = output.shape
                    if batch == 1 and seq == 3 and hidden == 768:
                        embedding_candidates.append((node_name, output))

        if embedding_candidates:
            print(f'\\nFound {len(embedding_candidates)} embedding-like tensors!')
            for i, (node_name, tensor) in enumerate(embedding_candidates[:3]):
                print(f'{i+1}. {node_name}:')
                print(f'   Shape: {tensor.shape}')
                print(f'   Mean: {tensor.mean().item():.6f}')
                print(f'   Std: {tensor.std().item():.6f}')
                print(f'   First token sample: {tensor[0, 0, :5]}')
            return True
        else:
            print('❌ No properly shaped embedding tensors found!')
            print('This confirms the tensor shape issue in the execution engine.')
            return False

        # Look for the problematic node that should be 3D but is 1D
        print('\\nChecking for shape inconsistencies:')
        for node_name in ['embedding', 'embedding_1', 'add_38', 'layer_norm']:
            if node_name in result:
                output = result[node_name]['output_values']
                if isinstance(output, torch.Tensor):
                    print(f'{node_name}: shape={output.shape} (expected: [1, 3, 768])')
                    if len(output.shape) != 3:
                        print(f'  ⚠️  Shape mismatch detected!')
                        return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tensor_shapes()
    sys.exit(0 if success else 1)
