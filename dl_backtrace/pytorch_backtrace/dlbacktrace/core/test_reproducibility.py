#!/usr/bin/env python3
"""
Test script for exact reproducibility in DL-Backtrace core.
"""

import torch
import torch.nn as nn
import numpy as np
from reproducibility import (
    setup_exact_reproducibility, 
    verify_exact_reproducibility,
    print_reproducibility_info
)

def create_test_model():
    """Create a simple test model."""
    return nn.Sequential(
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

def test_basic_reproducibility():
    """Test basic PyTorch reproducibility."""
    print("🧪 Testing basic PyTorch reproducibility...")
    
    def test_function():
        model = create_test_model()
        x = torch.randn(2, 10)
        with torch.no_grad():
            return model(x).numpy()
    
    return verify_exact_reproducibility(test_function, num_runs=3, seed=42)

def test_traced_model_reproducibility():
    """Test traced model reproducibility."""
    print("🧪 Testing traced model reproducibility...")
    
    def test_function():
        from torch.export import export
        model = create_test_model()
        x = torch.randn(2, 10)
        
        # Export model
        exported_program = export(model, (x,))
        
        # Run inference using the module
        with torch.no_grad():
            return exported_program.module()(x).numpy()
    
    return verify_exact_reproducibility(test_function, num_runs=3, seed=42)

def main():
    """Run reproducibility tests."""
    print("🎯 DL-Backtrace Core Reproducibility Tests")
    print("=" * 50)
    
    # Set up reproducibility
    setup_exact_reproducibility(seed=42)
    
    # Print current configuration
    print_reproducibility_info()
    
    # Run tests
    tests = [
        ("Basic PyTorch", test_basic_reproducibility),
        ("Traced Model", test_traced_model_reproducibility),
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results[name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:15}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All reproducibility tests PASSED!")
        print("✅ Exact reproducibility is working correctly.")
    else:
        print("⚠️  Some reproducibility tests FAILED!")
        print("❌ Check your configuration.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
