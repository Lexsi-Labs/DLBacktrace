#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from dl_backtrace.pytorch_backtrace.backtrace.backtrace import Backtrace

def debug_relevance_propagation():
    """Debug exactly how relevance propagates through layers in both implementations."""
    
    print("=== Relevance Propagation Debug ===")
    
    # Create a very simple model with just 2 linear layers
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 3)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(3, 2)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x) 
            x = self.fc2(x)
            return x
    
    # Initialize model and set deterministic weights
    model = SimpleModel()
    model.eval()
    
    # Set fixed weights for reproducibility
    with torch.no_grad():
        model.fc1.weight.fill_(0.5)
        model.fc1.bias.fill_(0.1)
        model.fc2.weight.fill_(0.3)
        model.fc2.bias.fill_(0.05)
    
    # Create test input
    test_input = torch.tensor([[1.0, 2.0, 0.5, -0.5]], dtype=torch.float32)
    
    print(f"Input: {test_input.numpy()}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(test_input)
        print(f"Model output: {output.numpy()}")
    
    # Initialize backtrace
    backtrace = Backtrace(model)
    
    # Get layer outputs
    all_out = backtrace.predict_every(test_input)
    print(f"\nLayer outputs:")
    for layer_name, output in all_out.items():
        if isinstance(output, dict) and 0 in output:
            data = output[0]
        else:
            data = output
        if hasattr(data, 'cpu'):
            data_np = data.cpu().numpy()
        else:
            data_np = data
        print(f"  {layer_name}: {data_np.flatten()}")
    
    print("\n=== Original Implementation Debug ===")
    try:
        # Create a custom backtrace to monitor step by step
        original_results = backtrace.eval(all_out, mode="default", multiplier=1.0)
        
        print("Original relevance results:")
        for layer_name, relevance in original_results.items():
            rel_sum = np.sum(relevance) if hasattr(relevance, 'sum') else relevance
            rel_mean = np.mean(np.abs(relevance)) if hasattr(relevance, 'mean') else abs(relevance)
            print(f"  {layer_name}: sum={rel_sum:.6f}, mean_abs={rel_mean:.6f}")
                
    except Exception as e:
        print(f"Original implementation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== CUDA Implementation Debug ===")
    try:
        cuda_results = backtrace.eval(all_out, mode="cuda_eval", multiplier=1.0)
        
        print("CUDA relevance results:")
        for layer_name, relevance in cuda_results.items():
            rel_sum = np.sum(relevance) if hasattr(relevance, 'sum') else relevance
            rel_mean = np.mean(np.abs(relevance)) if hasattr(relevance, 'mean') else abs(relevance)
            non_zero = np.count_nonzero(relevance) if hasattr(relevance, 'sum') else (1 if relevance != 0 else 0)
            total = relevance.size if hasattr(relevance, 'size') else 1
            print(f"  {layer_name}: sum={rel_sum:.6f}, mean_abs={rel_mean:.6f}, non_zero={non_zero}/{total}")
                
    except Exception as e:
        print(f"CUDA implementation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Starting Weight Comparison ===")
    # Check what starting weights are calculated
    from dl_backtrace.pytorch_backtrace.backtrace.utils import prop as UP
    
    # Get the output layer
    layer_stack = backtrace.layer_stack
    out_layer = backtrace.model_resource[2][0]
    print(f"Output layer: {out_layer}")
    
    # Check starting weight calculation 
    out_data = all_out[out_layer]
    if isinstance(out_data, dict):
        out_data = out_data[0] if 0 in out_data else list(out_data.values())[0]
    if hasattr(out_data, 'cpu'):
        out_data = out_data.cpu().numpy()
    
    print(f"Output layer data: {out_data}")
    
    start_wt = UP.calculate_start_wt(out_data, scaler=0, thresholding=0.5, task="binary-classification")
    print(f"Calculated starting weight: {start_wt}")
    print(f"Starting weight sum: {np.sum(start_wt)}")
    
    print("\n=== Layer Stack ===")
    print("Processing order (layer stack):")
    for i, layer in enumerate(layer_stack):
        layer_class = backtrace.model_resource[1][layer]["class"]
        print(f"  {i}: {layer} ({layer_class})")

if __name__ == "__main__":
    debug_relevance_propagation() 
