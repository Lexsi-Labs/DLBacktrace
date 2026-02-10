import os
import torch
import numpy as np

# ─── Import precompiled CUDA extension ───
_precompiled = False

try:
    import wt_fc_v3_ops as custom_linear_layer_cuda_ops
    _precompiled = True
except ImportError:
    # Try importing from the cuda_version_v3 directory (in-tree build)
    _cuda_version_dir = os.path.join(os.path.dirname(__file__), "cuda_version_v3")
    import importlib.util
    _so_files = [f for f in os.listdir(_cuda_version_dir) if f.endswith('.so')] if os.path.isdir(_cuda_version_dir) else []
    if _so_files:
        _spec = importlib.util.spec_from_file_location("wt_fc_v3_ops", os.path.join(_cuda_version_dir, _so_files[0]))
        custom_linear_layer_cuda_ops = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(custom_linear_layer_cuda_ops)
        _precompiled = True
    else:
        raise ImportError(
            "Precompiled CUDA extension 'wt_fc_v3_ops' not found. "
            "Please run 'bash compile_layers.sh' or 'python setup.py develop' "
            "in the cuda_version_v3 directory to build it."
        )


def calculate_wt_fc_cuda(relevance_y, input_array, w, b, act):
    """
    CUDA-accelerated version that maintains the original algorithm structure.
    Handles batch processing and multi-dimensional inputs in Python,
    delegates core computation to CUDA kernel.
    
    Args:
        relevance_y: relevance at the output (same shape as linear output)
        input_array: input to the linear layer (can be any shape: [B, D], [B, T, D], etc.)
        w: weight matrix of the linear layer (shape: [out_dim, in_dim])
        b: bias vector (shape: [out_dim]) or None
        act: dict containing activation info with keys: "type", "range", "func"

    Returns:
        relevance_x: relevance at the input, same shape as input_array
    """
    # Validate inputs
    if relevance_y is None or input_array is None or w is None:
        print(f"[CUDA ERROR] One or more inputs is None")
        return None
        
    # Flatten input except for last dim (same as original)
    original_shape = input_array.shape
    batch_dims = original_shape[:-1]
    feature_dim = original_shape[-1]

    input_flat = input_array.reshape(-1, feature_dim)
    relevance_flat = relevance_y.reshape(-1, relevance_y.shape[-1])

    # Process each batch element individually (maintains original logic)
    relevance_x_flat = []
    cuda_device = torch.device("cuda")
    
    # Convert weights to CUDA once (they're the same for all batch elements)
    w_torch = torch.tensor(w, dtype=torch.float32, device=cuda_device)
    b_torch = torch.tensor(b, dtype=torch.float32, device=cuda_device) if b is not None else torch.empty(0, device=cuda_device)
    
    # Parse activation parameters once
    act_type = 0 if act["type"] == "mono" else 1
    act_lower = -float('inf') if act["range"]["l"] is None else float(act["range"]["l"])
    act_upper = float('inf') if act["range"]["u"] is None else float(act["range"]["u"])
    
    # Convert activation function string to int
    act_func_int = 0  # default: identity
    if act["func"] is not None:
        act_func_str = act["func"]
        if act_func_str == "sigmoid": act_func_int = 1
        elif act_func_str == "swish": act_func_int = 2
        elif act_func_str == "wave": act_func_int = 3
        elif act_func_str == "pulse": act_func_int = 4
        elif act_func_str == "absolute": act_func_int = 5
        elif act_func_str == "hard_sigmoid": act_func_int = 6
        elif act_func_str == "tanh": act_func_int = 7
    
    for i in range(input_flat.shape[0]):
        inp = input_flat[i]            # shape: (input_dim,)
        wts = relevance_flat[i]        # shape: (output_dim,)
        
        # Convert to CUDA tensors for single batch element
        inp_torch = torch.tensor(inp, dtype=torch.float32, device=cuda_device)
        wts_torch = torch.tensor(wts, dtype=torch.float32, device=cuda_device)
        
        cuda_function = custom_linear_layer_cuda_ops.launch_calculate_wt_fc_kernel
        
        # Call CUDA kernel for single batch element with simplified parameters
        try:
            result = cuda_function(
                wts_torch, inp_torch, w_torch, b_torch,
                act_type, act_lower, act_upper, act_func_int
            )

            torch.cuda.synchronize()
            
            if result is None:
                print(f"[CUDA ERROR] Kernel returned None for batch element {i}")
                return None
                
            relevance_x_flat.append(result.cpu().numpy())
        except Exception as e:
            print(f"[CUDA ERROR] Kernel failed for batch element {i}: {e}")
            return None
    
    # Reshape back to original dimensions
    relevance_x_flat = np.array(relevance_x_flat)
    relevance_x = relevance_x_flat.reshape(*batch_dims, feature_dim)
    return relevance_x
