#!/usr/bin/env python3
"""
Debug script to identify numerical issues in CUDA vs Original implementation comparison.
This script helps pinpoint where NaN values and large differences originate.
"""

import torch
import numpy as np
import torch.nn as nn
from dl_backtrace.pytorch_backtrace.backtrace.backtrace import Backtrace as B

def debug_layer_by_layer(model, test_input):
    """
    Debug each layer individually to identify where numerical issues occur.
    """
    print("=== Layer-by-Layer Debugging ===")
    
    # Initialize both implementations
    backtrace_orig = B(model=model)
    backtrace_cuda = B(model=model)
    
    # Get layer outputs
    layer_outputs = backtrace_orig.predict_every(test_input)
    print(f"Captured {len(layer_outputs)} layer outputs")
    
    # Process each layer individually
    problematic_layers = []
    
    for layer_name, output in layer_outputs.items():
        print(f"\n--- Debugging Layer: {layer_name} ---")
        
        # Check for NaN or inf in input data
        if isinstance(output, np.ndarray):
            has_nan = np.isnan(output).any()
            has_inf = np.isinf(output).any()
            output_range = (np.min(output), np.max(output))
        else:
            has_nan = torch.isnan(output).any()
            has_inf = torch.isinf(output).any()
            output_range = (torch.min(output).item(), torch.max(output).item())
        
        print(f"  Input data - NaN: {has_nan}, Inf: {has_inf}, Range: {output_range}")
        
        # Try processing this layer with both implementations
        try:
            # Create minimal layer outputs for testing
            test_outputs = {layer_name: output}
            
            # Process with original implementation
            orig_result = backtrace_orig.eval(test_outputs, mode='default', scaler=1)
            
            # Process with CUDA implementation  
            cuda_result = backtrace_cuda.eval(test_outputs, mode='cuda_eval', scaler=1)
            
            # Compare results
            if layer_name in orig_result and layer_name in cuda_result:
                orig_vals = orig_result[layer_name]
                cuda_vals = cuda_result[layer_name]
                
                # Check for issues
                orig_has_nan = np.isnan(orig_vals).any()
                cuda_has_nan = np.isnan(cuda_vals).any()
                
                if orig_has_nan or cuda_has_nan:
                    print(f"  ❌ NaN detected - Original: {orig_has_nan}, CUDA: {cuda_has_nan}")
                    problematic_layers.append((layer_name, "NaN", orig_has_nan, cuda_has_nan))
                else:
                    max_diff = np.max(np.abs(orig_vals - cuda_vals))
                    print(f"  ✅ Max difference: {max_diff:.2e}")
                    
                    if max_diff > 1e-3:  # Threshold for significant difference
                        problematic_layers.append((layer_name, "Large_diff", max_diff, None))
                        
        except Exception as e:
            print(f"  ❌ Error processing {layer_name}: {e}")
            problematic_layers.append((layer_name, "Error", str(e), None))
    
    return problematic_layers

def debug_activation_functions():
    """
    Test activation function implementations for numerical stability.
    """
    print("\n=== Activation Function Debugging ===")
    
    # Test edge cases that might cause NaN
    test_values = [
        0.0, 1.0, -1.0, 1e-10, -1e-10, 
        1e10, -1e10, np.inf, -np.inf, 
        1e-308, -1e-308  # Near machine epsilon
    ]
    
    for val in test_values:
        print(f"\nTesting value: {val}")
        
        # Test different activation scenarios
        try:
            # Simulate division by zero scenarios common in relevance calculations
            if val != 0:
                reciprocal = 1.0 / val
                print(f"  1/x = {reciprocal}")
            else:
                print(f"  1/x = undefined (division by zero)")
                
            # Test common relevance calculation patterns
            if val > 0:
                p_sum = val
                n_sum = 0
            else:
                p_sum = 0  
                n_sum = -val
                
            total = p_sum + n_sum
            if total > 0:
                p_ratio = p_sum / total
                n_ratio = n_sum / total
                print(f"  Ratios - Positive: {p_ratio}, Negative: {n_ratio}")
            else:
                print(f"  Ratios - undefined (total sum is zero)")
                
        except Exception as e:
            print(f"  Error with {val}: {e}")

def debug_data_types():
    """
    Check for data type inconsistencies between implementations.
    """
    print("\n=== Data Type Debugging ===")
    
    # Create sample data with different types
    test_data = {
        'float32_np': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'float64_np': np.array([1.0, 2.0, 3.0], dtype=np.float64), 
        'float32_torch': torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        'float64_torch': torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
    }
    
    for name, data in test_data.items():
        print(f"  {name}: {type(data)}, dtype: {data.dtype}")
        
        # Test conversion manually (avoid creating Backtrace with None model)
        try:
            if hasattr(data, 'detach'):
                converted = data.detach().numpy()
            elif hasattr(data, 'numpy'):
                converted = data.numpy()
            else:
                converted = data
            print(f"    Converted: {type(converted)}, dtype: {converted.dtype}")
        except Exception as e:
            print(f"    Conversion error: {e}")

def main():
    """
    Main debugging function - run all debug tests.
    """
    print("=== CUDA Numerical Issues Debugging Script ===\n")
    
    # Create a simple test model and data
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    test_input = torch.randn(1, 10)
    
    print("Running debugging tests...")
    
    # Debug different aspects
    debug_activation_functions()
    debug_data_types()
    
    # Debug layer by layer (commented out for now since we need the actual VGG model)
    # problematic_layers = debug_layer_by_layer(model, test_input)
    
    print("\n=== Debugging Complete ===")
    print("\nRecommended next steps:")
    print("1. Check for division by zero in relevance calculations")
    print("2. Verify activation function implementations match between orig/CUDA")
    print("3. Ensure consistent data types (float32 vs float64)")
    print("4. Add numerical stability checks (e.g., epsilon for division)")
    print("5. Compare intermediate results step-by-step in problematic layers")

if __name__ == "__main__":
    main() 
 