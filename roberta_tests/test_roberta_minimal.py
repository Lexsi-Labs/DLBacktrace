#!/usr/bin/env python3
"""
Minimal RoBERTa test with simple inputs to isolate issues
"""

import torch
from transformers import RobertaForSequenceClassification
import sys
import os

# Add DL-Backtrace to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dl_backtrace', 'pytorch_backtrace'))
from dlbacktrace import DLBacktraceFX

def test_roberta_minimal():
    """Test RoBERTa with minimal input to isolate issues"""
    print('🔍 Testing RoBERTa with minimal input...')

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
    # Use minimal 3-token input to reduce complexity
    input_ids = torch.tensor([[101, 102, 103]])
    attention_mask = torch.ones_like(input_ids)

    print(f'Input shape: {input_ids.shape}')
    print(f'Input IDs: {input_ids}')

    try:
        # Direct execution
        with torch.no_grad():
            direct_logits = model(input_ids, attention_mask)
        
        print(f'Direct logits: {direct_logits}')
        direct_pred = torch.argmax(direct_logits, dim=-1)
        print(f'Direct predictions: {direct_pred.tolist()}')

        # Traced execution
        dlb = DLBacktraceFX(model, (input_ids, attention_mask), verbose=False)
        result = dlb.predict(input_ids, attention_mask, debug=False)
        
        # Find output
        output_nodes = [k for k in result.keys() if 'output' in k.lower()]
        if output_nodes:
            traced_logits = result[output_nodes[-1]]['output_values']
        else:
            # Look for final nodes
            final_nodes = list(result.keys())[-3:]
            for node in final_nodes:
                output = result[node]['output_values']
                if isinstance(output, torch.Tensor) and output.shape == direct_logits.shape:
                    traced_logits = output
                    break
        
        print(f'Traced logits: {traced_logits}')
        traced_pred = torch.argmax(traced_logits, dim=-1)
        print(f'Traced predictions: {traced_pred.tolist()}')
        
        # Compare results
        logit_diff = torch.max(torch.abs(direct_logits - traced_logits)).item()
        pred_match = torch.equal(direct_pred, traced_pred)
        
        print(f'\\nResults:')
        print(f'  Max logit difference: {logit_diff:.6f}')
        print(f'  Predictions match: {pred_match}')
        
        if logit_diff < 1e-5 and pred_match:
            print('✅ Minimal RoBERTa test PASSED!')
            return True
        elif pred_match:
            print('⚠️  Minimal RoBERTa test: Predictions match but logits differ')
            return True
        else:
            print('❌ Minimal RoBERTa test FAILED - Different predictions!')
            return False
            
    except Exception as e:
        print(f'❌ Error during minimal RoBERTa test: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_roberta_minimal()
    sys.exit(0 if success else 1)
