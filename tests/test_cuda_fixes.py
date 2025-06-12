#!/usr/bin/env python3
"""
Simple test script to validate CUDA fixes for numerical stability.
Tests individual layer implementations for edge cases.
"""

import torch
import numpy as np
import sys
import os

def test_linear_cuda_stability():
    """Test Linear CUDA implementation for numerical stability."""
    print("Testing Linear CUDA numerical stability...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        # Import the CUDA Linear function
        from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.Linear.cuda_version.wt_fc_ops import calculate_wt_fc_interface as linear_cuda
        
        # Test case 1: All zeros (previously caused NaN)
        print("  Test 1: All zeros...")
        row_weights = torch.zeros(3, dtype=torch.float32).cuda()
        input_acts = torch.zeros(3, dtype=torch.float32).cuda()
        weights_matrix = torch.zeros(3, 5, dtype=torch.float32).cuda()
        bias_vector = torch.zeros(3, dtype=torch.float32).cuda()
        
        result = linear_cuda(row_weights, input_acts, weights_matrix, bias_vector,
                           False, 0.0, False, 0.0, False, 0)
        
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        print(f"    All zeros: NaN={has_nan}, Inf={has_inf} ✅" if not (has_nan or has_inf) else f"    All zeros: NaN={has_nan}, Inf={has_inf} ❌")
        
        # Test case 2: Extreme values
        print("  Test 2: Extreme values...")
        row_weights = torch.tensor([1e10, -1e10, 1e-10], dtype=torch.float32).cuda()
        input_acts = torch.tensor([1e-10, 1e10, -1e10], dtype=torch.float32).cuda()
        weights_matrix = torch.randn(3, 5, dtype=torch.float32).cuda() * 1e5
        bias_vector = torch.tensor([1e10, -1e10, 0], dtype=torch.float32).cuda()
        
        result = linear_cuda(row_weights, input_acts, weights_matrix, bias_vector,
                           False, 0.0, False, 0.0, False, 0)
        
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        print(f"    Extreme values: NaN={has_nan}, Inf={has_inf} ✅" if not (has_nan or has_inf) else f"    Extreme values: NaN={has_nan}, Inf={has_inf} ❌")
        
        # Test case 3: With activation bounds
        print("  Test 3: Activation bounds...")
        row_weights = torch.randn(3, dtype=torch.float32).cuda()
        input_acts = torch.randn(3, dtype=torch.float32).cuda()
        weights_matrix = torch.randn(3, 5, dtype=torch.float32).cuda()
        bias_vector = torch.randn(3, dtype=torch.float32).cuda()
        
        result = linear_cuda(row_weights, input_acts, weights_matrix, bias_vector,
                           True, -1.0, True, 1.0, True, 1)  # ReLU with bounds
        
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        print(f"    Activation bounds: NaN={has_nan}, Inf={has_inf} ✅" if not (has_nan or has_inf) else f"    Activation bounds: NaN={has_nan}, Inf={has_inf} ❌")
        
        return not (has_nan or has_inf)
        
    except Exception as e:
        print(f"❌ Linear CUDA test error: {e}")
        return False

def test_conv2d_cuda_stability():
    """Test Conv2D CUDA implementation for numerical stability."""
    print("Testing Conv2D CUDA numerical stability...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
        
    try:
        # Import the CUDA Conv2D function
        from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.Conv2D.build_cuda_version.wt_conv_ops import calculate_wt_conv2d_interface as conv_cuda
        
        # Test case 1: All zeros
        print("  Test 1: All zeros...")
        patches = torch.zeros(2, 3, 3, 4, dtype=torch.float32).cuda()  # (L, K_h, K_w, C)
        kernel_weights = torch.zeros(1, 3, 3, 4, 6, dtype=torch.float32).cuda()  # (1, K_h, K_w, C, F)
        bias = torch.zeros(6, dtype=torch.float32).cuda()
        grad_scales = torch.zeros(2, 6, dtype=torch.float32).cuda()
        
        result = conv_cuda(patches, kernel_weights, bias, grad_scales,
                          2, 3, 3, 4, 6,  # L, K_h, K_w, C, F
                          0, 0, 0.0, 0.0, False, False)
        
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        print(f"    All zeros: NaN={has_nan}, Inf={has_inf} ✅" if not (has_nan or has_inf) else f"    All zeros: NaN={has_nan}, Inf={has_inf} ❌")
        
        # Test case 2: Extreme values
        print("  Test 2: Extreme values...")
        patches = torch.randn(2, 3, 3, 4, dtype=torch.float32).cuda() * 1e5
        kernel_weights = torch.randn(1, 3, 3, 4, 6, dtype=torch.float32).cuda() * 1e-5
        bias = torch.tensor([1e10, -1e10, 0, 1e-10, -1e-10, 1e5], dtype=torch.float32).cuda()
        grad_scales = torch.randn(2, 6, dtype=torch.float32).cuda() * 1e3
        
        result = conv_cuda(patches, kernel_weights, bias, grad_scales,
                          2, 3, 3, 4, 6,  # L, K_h, K_w, C, F  
                          0, 0, 0.0, 0.0, False, False)
        
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        print(f"    Extreme values: NaN={has_nan}, Inf={has_inf} ✅" if not (has_nan or has_inf) else f"    Extreme values: NaN={has_nan}, Inf={has_inf} ❌")
        
        return not (has_nan or has_inf)
        
    except Exception as e:
        print(f"❌ Conv2D CUDA test error: {e}")
        return False

def test_maxpool_cuda_stability():
    """Test MaxPool CUDA implementation for numerical stability."""
    print("Testing MaxPool CUDA numerical stability...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
        
    try:
        # Import the CUDA MaxPool function
        from dl_backtrace.pytorch_backtrace.backtrace.refactored_utils.layers.WtMaxunit2D.build_cuda_version.wt_maxunit2d_ops import calculate_wt_maxunit2d_interface as maxpool_cuda
        
        # Test case 1: All equal values (division by zero risk)
        print("  Test 1: All equal values...")
        patch = torch.ones(3, 3, 4, dtype=torch.float32).cuda() * 5.0
        weights = torch.ones(4, dtype=torch.float32).cuda()
        
        result = maxpool_cuda(patch, weights, 3)
        
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        print(f"    Equal values: NaN={has_nan}, Inf={has_inf} ✅" if not (has_nan or has_inf) else f"    Equal values: NaN={has_nan}, Inf={has_inf} ❌")
        
        # Test case 2: With -inf padding
        print("  Test 2: -inf padding...")
        patch_inf = torch.full((3, 3, 4), -float('inf'), dtype=torch.float32).cuda()
        patch_inf[1, 1, :] = 10.0  # One non-inf value per channel
        weights = torch.ones(4, dtype=torch.float32).cuda()
        
        result = maxpool_cuda(patch_inf, weights, 3)
        
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()  
        print(f"    -inf padding: NaN={has_nan}, Inf={has_inf} ✅" if not has_nan else f"    -inf padding: NaN={has_nan}, Inf={has_inf} ❌")
        
        return not has_nan  # Allow -inf in output, but not NaN
        
    except Exception as e:
        print(f"❌ MaxPool CUDA test error: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("CUDA Numerical Stability Test Suite")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot run tests.")
        return
    
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    
    # Run all tests
    results = []
    results.append(("Linear", test_linear_cuda_stability()))
    results.append(("Conv2D", test_conv2d_cuda_stability()))
    results.append(("MaxPool", test_maxpool_cuda_stability()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:15}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests PASSED! CUDA fixes are working.")
        print("You can now run the VGG benchmark with confidence:")
        print("python tests/VGG_benchmark_cuda_eval.py")
    else:
        print("⚠️  Some tests FAILED. Check the CUDA implementations.")
        print("Review the error messages above for debugging.")
    print("=" * 60)

if __name__ == "__main__":
    main() 
