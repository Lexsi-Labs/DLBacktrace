"""
Exact reproducibility utilities for PyTorch backtrace.
This module ensures exact reproducibility for DL-Backtrace PyTorch operations.
"""

import os
import random
import numpy as np
import warnings
from typing import Optional, Union, Dict, Any, Tuple
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def setup_exact_reproducibility(seed: int = 42, 
                               disable_optimizations: bool = True,
                               verbose: bool = True) -> None:
    """
    Set up exact reproducibility for PyTorch backtrace operations.
    
    Args:
        seed: Random seed to use for all random number generators
        disable_optimizations: Whether to disable optimizations for exact reproducibility
    """
    if verbose:
        print(f"🔧 Setting up exact PyTorch reproducibility with seed={seed}")
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # PyTorch reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if disable_optimizations:
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Set cuDNN to deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Disable TensorFloat-32 (TF32) for exact reproducibility
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        
        # Set default dtype to float32 for consistency
        torch.set_default_dtype(torch.float32)
        
        # Force deterministic SDPA (Scaled Dot Product Attention) path
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except ImportError:
            pass
        
        if verbose:
            print("🔧 PyTorch optimizations disabled for exact reproducibility")
    else:
        if verbose:
            print("⚠️  PyTorch optimizations enabled - results may not be exactly reproducible")
    
    # Traced model specific settings
    if disable_optimizations:
        # Disable torch.export optimizations that can cause non-determinism
        os.environ['TORCH_EXPORT_DISABLE_OPTIMIZATIONS'] = '1'
        os.environ['TORCH_EXPORT_DETERMINISTIC'] = '1'
        
        # Disable FX graph optimizations
        os.environ['TORCH_FX_DISABLE_OPTIMIZATIONS'] = '1'
        
        if verbose:
            print("🔧 Torch.export optimizations disabled for exact reproducibility")
    
    # Set additional environment variables for traced models
    os.environ['TORCH_LOGS'] = '+dynamic'  # Enable dynamic shape logging
    os.environ['TORCH_DYNAMIC_SHAPES_DETERMINISTIC'] = '1'
    
    if verbose:
        print("✅ Exact PyTorch reproducibility configured")

def create_deterministic_dynamic_shapes(input_shapes: Dict[str, Tuple[int, ...]], 
                                      seed: int = 42) -> Dict[str, Dict[int, 'Dim']]:
    """
    Create deterministic dynamic shapes for torch.export.
    
    Args:
        input_shapes: Dictionary mapping input names to their shapes
        seed: Seed for deterministic dimension creation
        
    Returns:
        Dictionary of dynamic shapes for torch.export
    """
    from torch.export import Dim
    
    # Set seed for deterministic dimension creation
    random.seed(seed)
    np.random.seed(seed)
    
    dynamic_shapes = {}
    
    for input_name, shape in input_shapes.items():
        dynamic_shapes[input_name] = {}
        
        for dim_idx, dim_size in enumerate(shape):
            if dim_size > 1:  # Only make dimensions > 1 dynamic
                # Create deterministic dimension names
                dim_name = f"{input_name}_dim_{dim_idx}"
                
                # Use deterministic min/max values
                min_val = max(1, dim_size // 2)
                max_val = dim_size * 2
                
                dynamic_shapes[input_name][dim_idx] = Dim(
                    dim_name, 
                    min=min_val, 
                    max=max_val
                )
    
    return dynamic_shapes

def export_model_deterministically(model: torch.nn.Module, 
                                 sample_inputs: Tuple[torch.Tensor, ...],
                                 dynamic_shapes: Optional[Dict] = None,
                                 seed: int = 42) -> 'torch.export.ExportedProgram':
    """
    Export a model deterministically for reproducible tracing.
    
    Args:
        model: PyTorch model to export
        sample_inputs: Sample inputs for tracing
        dynamic_shapes: Optional dynamic shape constraints
        seed: Seed for deterministic export
        
    Returns:
        Exported program with deterministic behavior
    """
    from torch.export import export, export_for_training
    
    # Set up reproducibility
    setup_exact_reproducibility(seed)
    
    # Ensure model is in eval mode and deterministic
    model.eval()
    model.requires_grad_(False)
    
    # Set model to deterministic mode
    for module in model.modules():
        if hasattr(module, 'training'):
            module.training = False
        if hasattr(module, 'eval'):
            module.eval()
    
    # Export with deterministic settings
    try:
        if dynamic_shapes:
            exported_program = export_for_training(
                model,
                sample_inputs,
                dynamic_shapes=dynamic_shapes
            )
        else:
            exported_program = export(model, sample_inputs)
        
        # Run decompositions deterministically
        exported_program = exported_program.run_decompositions(decomp_table={})
        
        print("✅ Model exported deterministically")
        return exported_program
        
    except Exception as e:
        print(f"❌ Error during deterministic export: {e}")
        raise

def verify_exact_reproducibility(test_function, 
                                num_runs: int = 3, 
                                tolerance: float = 1e-6,
                                seed: int = 42) -> bool:
    """
    Verify that a function produces exactly reproducible results.
    
    Args:
        test_function: Function to test for reproducibility
        num_runs: Number of runs to test
        tolerance: Tolerance for considering results identical
        seed: Seed to use for testing
        
    Returns:
        True if results are exactly reproducible, False otherwise
    """
    results = []
    
    for run in range(num_runs):
        # Reset reproducibility for each run
        setup_exact_reproducibility(seed)
        
        # Run the test function
        result = test_function()
        results.append(result)
    
    # Check if all results are identical (within tolerance)
    if len(results) < 2:
        return True
    
    # Convert results to numpy arrays for comparison
    try:
        results = [np.array(r) if not isinstance(r, np.ndarray) else r for r in results]
        
        # Check if all results are close to each other
        for i in range(1, len(results)):
            if not np.allclose(results[0], results[i], atol=tolerance, rtol=tolerance):
                print(f"❌ Exact reproducibility test failed: results differ between runs")
                print(f"   Run 0: {results[0]}")
                print(f"   Run {i}: {results[i]}")
                return False
        
        print(f"✅ Exact reproducibility test passed: {num_runs} runs produced identical results")
        return True
        
    except Exception as e:
        print(f"❌ Error during exact reproducibility test: {e}")
        return False

def get_reproducibility_info() -> Dict[str, Any]:
    """
    Get information about current reproducibility settings.
    
    Returns:
        Dictionary with reproducibility configuration info
    """
    info = {
        'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'torch_export_disable_optimizations': os.environ.get('TORCH_EXPORT_DISABLE_OPTIMIZATIONS'),
        'torch_export_deterministic': os.environ.get('TORCH_EXPORT_DETERMINISTIC'),
        'torch_fx_disable_optimizations': os.environ.get('TORCH_FX_DISABLE_OPTIMIZATIONS'),
        'torch_logs': os.environ.get('TORCH_LOGS'),
        'torch_dynamic_shapes_deterministic': os.environ.get('TORCH_DYNAMIC_SHAPES_DETERMINISTIC'),
    }
    
    # PyTorch info
    try:
        info['pytorch_version'] = torch.__version__
        info['pytorch_deterministic'] = torch.backends.cudnn.deterministic
        info['pytorch_benchmark'] = torch.backends.cudnn.benchmark
        info['pytorch_allow_tf32'] = torch.backends.cudnn.allow_tf32
    except ImportError:
        info['pytorch_available'] = False
    
    return info

def print_reproducibility_info() -> None:
    """Print current reproducibility configuration."""
    info = get_reproducibility_info()
    
    print("🔍 Current Exact Reproducibility Configuration:")
    print("=" * 60)
    
    for key, value in info.items():
        if value is not None:
            print(f"{key}: {value}")
    
    print("=" * 60)

# Auto-setup when imported (can be disabled by setting environment variable)
if os.environ.get('DL_BACKTRACE_AUTO_REPRODUCIBILITY', 'true').lower() == 'true':
    setup_exact_reproducibility(verbose=False)

# Export main functions
__all__ = [
    'setup_exact_reproducibility',
    'create_deterministic_dynamic_shapes',
    'export_model_deterministically',
    'verify_exact_reproducibility',
    'get_reproducibility_info',
    'print_reproducibility_info'
]
