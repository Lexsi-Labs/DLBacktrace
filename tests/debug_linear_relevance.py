#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from dl_backtrace.pytorch_backtrace.backtrace.backtrace import Backtrace

def debug_linear_layer_relevance():
    """Debug Linear layer relevance calculation to understand why CUDA gives zeros."""
    
    print("=== Linear Layer Relevance Debug ===")
    
    # Create simple test model with just Linear layers
    class SimpleLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.identity = nn.Identity()
            self.fc1 = nn.Linear(4, 3)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(3, 2)
            
        def forward(self, x):
            x = self.identity(x)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Initialize model and set deterministic weights
    model = SimpleLinearModel()
    model.eval()
    
    # Set fixed weights for reproducibility
    with torch.no_grad():
        model.fc1.weight.fill_(0.5)
        model.fc1.bias.fill_(0.1)
        model.fc2.weight.fill_(0.3)
        model.fc2.bias.fill_(0.05)
    
    # Create test input
    test_input = torch.tensor([[1.0, 2.0, 0.5, -0.5]], dtype=torch.float32)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Input values: {test_input.numpy()}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(test_input)
        print(f"Model output: {output.numpy()}")
    
    # Initialize backtrace
    backtrace = Backtrace(model)
    
    # Get layer outputs
    all_out = backtrace.predict_every(test_input)
    print(f"Number of layers captured: {len(all_out)}")
    
    for layer_name, output in all_out.items():
        if isinstance(output, dict) and 0 in output:
            data = output[0]
        else:
            data = output
        # Handle both tensor and numpy array
        if hasattr(data, 'cpu'):
            data_np = data.cpu().numpy()
        else:
            data_np = data
        print(f"Layer {layer_name}: shape={data_np.shape}, values={data_np.flatten()[:5]}")
    
    print("\n--- Original Implementation ---")
    try:
        original_results = backtrace.eval(all_out, mode="default", multiplier=1.0)
        
        print("Original relevance results:")
        for layer_name, relevance in original_results.items():
            if hasattr(relevance, 'shape'):
                avg_relevance = np.mean(np.abs(relevance))
                max_relevance = np.max(np.abs(relevance))
                min_relevance = np.min(relevance)
                print(f"  {layer_name}: avg={avg_relevance:.6f}, max={max_relevance:.6f}, min={min_relevance:.6f}")
            else:
                print(f"  {layer_name}: {relevance}")
                
    except Exception as e:
        print(f"Original implementation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- CUDA Implementation ---")
    try:
        cuda_results = backtrace.eval(all_out, mode="cuda_eval", multiplier=1.0)
        
        print("CUDA relevance results:")
        for layer_name, relevance in cuda_results.items():
            if hasattr(relevance, 'shape'):
                avg_relevance = np.mean(np.abs(relevance))
                max_relevance = np.max(np.abs(relevance))
                min_relevance = np.min(relevance)
                non_zero_count = np.count_nonzero(relevance)
                total_count = relevance.size
                print(f"  {layer_name}: avg={avg_relevance:.6f}, max={max_relevance:.6f}, min={min_relevance:.6f}, non_zero={non_zero_count}/{total_count}")
            else:
                print(f"  {layer_name}: {relevance}")
                
    except Exception as e:
        print(f"CUDA implementation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_linear_layer_relevance() 
 