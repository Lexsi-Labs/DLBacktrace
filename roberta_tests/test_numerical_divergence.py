#!/usr/bin/env python3
"""
Find where numerical divergence occurs between direct and traced execution
"""

import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification
import sys
import os

# Suppress TensorFlow warnings to reduce noise
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add DL-Backtrace to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dl_backtrace', 'pytorch_backtrace'))
from dlbacktrace import DLBacktraceFX

def test_numerical_divergence():
    """Find where numerical divergence starts"""
    print('🔍 Finding numerical divergence point...')

    class SimpleRoBERTa(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Use a smaller, faster model for testing
            self.bert = RobertaForSequenceClassification.from_pretrained(
                'cardiffnlp/twitter-roberta-base-sentiment-latest',
                torch_dtype=torch.float32,
                use_cache=False,
                local_files_only=False  # Allow download but with progress
            ).eval()

        def forward(self, input_ids, attention_mask):
            return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

    print("📥 Loading RoBERTa model...")
    model = SimpleRoBERTa()
    print("✅ Model loaded successfully")
    
    input_ids = torch.tensor([[101, 102, 103]])
    attention_mask = torch.ones_like(input_ids)

    # Get direct execution for comparison
    print("Getting direct model outputs...")
    with torch.no_grad():
        direct_logits = model(input_ids, attention_mask)
    print("✅ Direct execution completed")
    
    print(f"Direct logits: {direct_logits}")

    # Get traced execution
    print("Getting traced execution...")
    try:
        print("  📊 Initializing DLBacktraceFX...")
        dlb = DLBacktraceFX(model, (input_ids, attention_mask), verbose=False)
        print("  🔄 Running prediction...")
        result = dlb.predict(input_ids, attention_mask, debug=False)
        print("✅ Traced execution completed")

        # Find output node
        print("  🔍 Finding output node...")
        output_nodes = [k for k in result.keys() if 'output' in k.lower()]
        if not output_nodes:
            # Look for final nodes that might be the output
            final_nodes = list(result.keys())[-5:]
            print(f"  No obvious output nodes, checking final nodes: {final_nodes}")
            for node in final_nodes:
                output = result[node]['output_values']
                if isinstance(output, torch.Tensor) and output.shape == direct_logits.shape:
                    print(f"  Found matching output shape in node: {node}")
                    traced_logits = output
                    break
        else:
            traced_logits = result[output_nodes[-1]]['output_values']
            print(f"  Found output node: {output_nodes[-1]}")

        print(f"Traced logits: {traced_logits}")
        
        # Compare
        diff = torch.max(torch.abs(direct_logits - traced_logits)).item()
        print(f"Max difference: {diff:.6f}")

        if diff > 1e-5:
            print("❌ SIGNIFICANT DIVERGENCE DETECTED!")
            
            # Check early operations for divergence
            print("\\nChecking key early operations:")
            
            # Check word embeddings
            word_emb_nodes = [k for k in result.keys() if 'embedding' in k and 'weight' not in k][:5]
            for node in word_emb_nodes:
                output = result[node]['output_values']
                if isinstance(output, torch.Tensor) and len(output.shape) == 3:
                    print(f"{node}: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
            
            # Check first few attention outputs
            attn_nodes = [k for k in result.keys() if 'scaled_dot_product_attention' in k][:3]
            for node in attn_nodes:
                output = result[node]['output_values']
                if isinstance(output, torch.Tensor):
                    print(f"{node}: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
                    
            # Check layer norms
            ln_nodes = [k for k in result.keys() if 'layer_norm' in k][:3]
            for node in ln_nodes:
                output = result[node]['output_values']
                if isinstance(output, torch.Tensor):
                    print(f"{node}: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
            
            # Check if the issue is in different random states
            print("\\nChecking for randomness issues...")
            
            # Run direct execution again
            with torch.no_grad():
                direct_logits2 = model(input_ids, attention_mask)
            
            direct_diff = torch.max(torch.abs(direct_logits - direct_logits2)).item()
            print(f"Direct execution consistency: {direct_diff:.6f}")
            
            if direct_diff > 1e-6:
                print("⚠️  Direct execution is not deterministic!")
                return False
            else:
                print("✅ Direct execution is deterministic")
                return False  # Still failed due to divergence
        else:
            print("✅ NO SIGNIFICANT DIVERGENCE!")
            return True
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_numerical_divergence()
    sys.exit(0 if success else 1)
