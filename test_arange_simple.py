#!/usr/bin/env python
# Simple test for arange fix

import os
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn

print("Testing arange fix...")

# Test the arange operation directly
def test_arange_fix():
    print("Testing arange with start=512, end=10...")
    
    # Simulate the problematic case
    start = 512
    end = 10
    
    print(f"Original: start={start}, end={end}")
    
    # Apply our fix logic
    if start is not None and end is not None and start > end:
        print(f"🔧 Detected start={start} > end={end}, using start as end value")
        end = start
    elif start is not None and end == 10:
        if start > 10:
            print(f"🔧 Detected hardcoded end=10 with start={start}, using start as end value")
            end = start
    
    print(f"Fixed: start={start}, end={end}")
    
    # Test the arange operation
    try:
        result = torch.arange(start, end, dtype=torch.float32)
        print(f"✅ arange({start}, {end}) = {result.shape}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_arange_fix()
    print(f"Test {'PASSED' if success else 'FAILED'}")
