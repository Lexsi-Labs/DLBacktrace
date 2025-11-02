#!/usr/bin/env python3
"""
Test script for the updated DL-Backtrace Pipeline

This script tests the pipeline with the new DLBacktraceFX API changes.
"""

import sys
import os

# Add the pipeline directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import DLBacktracePipeline
from config import PipelineConfig
from model_registry import ModelRegistry


def test_simple_pipeline():
    """Test the simple pipeline creation and basic functionality"""
    print("="*60)
    print("TEST 1: Simple Pipeline Creation")
    print("="*60)
    
    try:
        # Test simple pipeline creation
        pipeline = DLBacktracePipeline.create_simple(
            model_name="gpt2",
            device="cpu",
            verbose=True,
            save_results=False  # Don't save for testing
        )
        
        print("✅ Simple pipeline created successfully")
        print(f"   Model name: {pipeline.config.model_name}")
        print(f"   Device: {pipeline.config.device}")
        print(f"   Model info: {pipeline.model_info.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple pipeline creation failed: {e}")
        return False


def test_config_validation():
    """Test configuration validation"""
    print("\n" + "="*60)
    print("TEST 2: Configuration Validation")
    print("="*60)
    
    try:
        # Test valid config
        config = PipelineConfig(
            model_name="gpt2",
            device="cpu",
            verbose=False
        )
        print("✅ Valid config created successfully")
        
        # Test invalid device
        try:
            invalid_config = PipelineConfig(
                model_name="gpt2",
                device="invalid_device"
            )
            print("❌ Invalid device validation failed")
            return False
        except ValueError as e:
            print(f"✅ Invalid device correctly rejected: {e}")
        
        # Test missing model_name
        try:
            invalid_config = PipelineConfig(model_name="")
            print("❌ Empty model_name validation failed")
            return False
        except ValueError as e:
            print(f"✅ Empty model_name correctly rejected: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def test_model_registry():
    """Test model registry functionality"""
    print("\n" + "="*60)
    print("TEST 3: Model Registry")
    print("="*60)
    
    try:
        # List available models
        models = ModelRegistry.list_models()
        print(f"✅ Found {len(models)} registered models:")
        for name, desc in list(models.items())[:3]:  # Show first 3
            print(f"   - {name}: {desc}")
        
        # Test getting model info
        model_info = ModelRegistry.get_model_info("gpt2")
        print(f"✅ GPT-2 model info retrieved:")
        print(f"   - Type: {model_info.model_type}")
        print(f"   - Path: {model_info.base_model_path}")
        print(f"   - Max length: {model_info.default_max_length}")
        
        # Test invalid model
        try:
            invalid_info = ModelRegistry.get_model_info("nonexistent-model")
            print("❌ Invalid model validation failed")
            return False
        except ValueError as e:
            print(f"✅ Invalid model correctly rejected: {str(e)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Model registry test failed: {e}")
        return False


def test_dlbacktrace_initialization():
    """Test DL-Backtrace initialization with new API"""
    print("\n" + "="*60)
    print("TEST 4: DL-Backtrace Initialization")
    print("="*60)
    
    try:
        # Create pipeline
        pipeline = DLBacktracePipeline.create_simple(
            model_name="gpt2",
            device="cpu",
            verbose=False,
            save_results=False
        )
        
        print("🔄 Loading model...")
        # Load model (this might take a while)
        model, tokenizer = pipeline.load_model()
        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")
        
        print("🔄 Preparing sample inputs...")
        # Prepare sample inputs
        sample_inputs = pipeline.prepare_inputs(["Hello world"])
        print(f"✅ Sample inputs prepared: {list(sample_inputs.keys())}")
        
        print("🔄 Initializing DL-Backtrace...")
        # Initialize DL-Backtrace (this tests the new API)
        pipeline.initialize_dlbt(sample_inputs)
        print("✅ DL-Backtrace initialized successfully")
        
        # Verify the DL-Backtrace instance has the expected attributes
        if hasattr(pipeline.dlbt, 'layer_implementation'):
            print(f"✅ Layer implementation config: {type(pipeline.dlbt.layer_implementation)}")
        else:
            print("⚠️  No layer_implementation attribute (expected with new API)")
        
        if hasattr(pipeline.dlbt, 'get_layer_implementation'):
            print("✅ get_layer_implementation method available")
        else:
            print("❌ get_layer_implementation method missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ DL-Backtrace initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device_configuration():
    """Test device configuration and layer implementation selection"""
    print("\n" + "="*60)
    print("TEST 5: Device Configuration")
    print("="*60)
    
    try:
        # Test CPU configuration
        cpu_config = PipelineConfig(
            model_name="gpt2",
            device="cpu",
            verbose=False
        )
        print(f"✅ CPU config created: device={cpu_config.device}")
        
        # Test CUDA configuration (even if CUDA not available)
        cuda_config = PipelineConfig(
            model_name="gpt2",
            device="cuda",
            verbose=False
        )
        print(f"✅ CUDA config created: device={cuda_config.device}")
        
        # Note: We don't test actual CUDA initialization since it may not be available
        print("✅ Device configuration tests passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Device configuration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("🚀 Running DL-Backtrace Pipeline Tests")
    print("🔧 Testing updated API compatibility...")
    
    tests = [
        ("Simple Pipeline Creation", test_simple_pipeline),
        ("Configuration Validation", test_config_validation),
        ("Model Registry", test_model_registry),
        ("Device Configuration", test_device_configuration),
        ("DL-Backtrace Initialization", test_dlbacktrace_initialization),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is compatible with new DLBacktraceFX API")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)