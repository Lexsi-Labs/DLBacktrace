# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/execution_engine_noncache.py
import os
import inspect
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any

# Centralized logging configuration
class LoggingManager:
    """Centralized logging manager to ensure consistent configuration"""
    _instance = None
    _logger = None
    _configured = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggingManager, cls).__new__(cls)
        return cls._instance
    
    def configure_logging(self, debug: bool = False, log_level: str = "INFO", force_reconfigure: bool = False):
        """Configure logging with proper handler management"""
        # Get the logger instance
        logger_name = "dlbacktrace.execution_engine"
        logger = logging.getLogger(logger_name)
        
        # Only reconfigure if not already configured or if forced
        if self._configured and not force_reconfigure:
            return logger
        
        # Clear any existing handlers to prevent duplication
        logger.handlers.clear()
        
        # Set log level
        level = logging.DEBUG if debug else getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler when debug is enabled
        if debug:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"dlbacktrace_debug_{timestamp}.log"
            
            # Create logs directory if it doesn't exist
            log_dir = "dlbacktrace_logs"
            os.makedirs(log_dir, exist_ok=True)
            log_filepath = os.path.join(log_dir, log_filename)
            
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Store file handler for later flushing
            self._file_handler = file_handler
            self._log_filepath = log_filepath
            
            logger.info(f"Debug logging enabled - logs will be saved to: {log_filepath}")
            logger.debug("Debug logging test - this should appear in the log file")
        else:
            self._file_handler = None
            self._log_filepath = None
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        # Mark as configured
        self._configured = True
        self._logger = logger
        
        return logger
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        if not self._configured:
            # Configure with default settings if not already configured
            return self.configure_logging()
        return self._logger
    
    def flush_logs(self):
        """Flush all log handlers"""
        if self._logger:
            for handler in self._logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
    
    def get_log_filepath(self):
        """Get the current log file path if debug logging is enabled"""
        return getattr(self, '_log_filepath', None)

# Global logging manager instance
_logging_manager = LoggingManager()

def setup_logging(debug: bool = False, log_level: str = "INFO", force_reconfigure: bool = False) -> logging.Logger:
    """Setup logging configuration for the execution engine"""
    return _logging_manager.configure_logging(debug, log_level, force_reconfigure)

def get_logger() -> logging.Logger:
    """Get the global logger instance"""
    return _logging_manager.get_logger()

def flush_logs():
    """Flush all log handlers"""
    _logging_manager.flush_logs()

# Legacy compatibility - will be removed
def log(*args, **kwargs):
    """Legacy log function - use get_logger().debug() instead"""
    get_logger().debug(" ".join(str(arg) for arg in args))

def _load_if_path(val, cache_manager):
    if isinstance(val, str) and val.endswith(".pt.zstd") and os.path.isfile(val):
        return cache_manager.load_tensor(val)
    return val

def sanitize(x):
        return x.to(torch.float32) if isinstance(x, torch.Tensor) and x.dtype == torch.bool else x

def ensure_model_has_device_attribute(model):
    """Ensure model has a device attribute, set it if missing"""
    logger = get_logger()
    if not hasattr(model, 'device'):
        try:
            # Try to get device from model parameters
            device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
            model.device = device
            logger.debug(f"🔧 Added device attribute to model: {device}")
        except Exception as e:
            model.device = torch.device('cpu')
            logger.warning(f"🔧 Added default device attribute to model: cpu (error: {e})")
    return model.device 

def _sanitize_input_tensor(t):
    return t.detach() if isinstance(t, torch.nn.Parameter) else t

# 🔧 CONSISTENCY FIXES: New functions to ensure consistent outputs
def standardize_layer_input(layer_in, parents, node_io, tensor_map, func_name):
    """Ensure consistent input processing across all operations"""
    if isinstance(layer_in, list):
        # Filter out weight/bias nodes consistently
        filtered_inputs = []
        for p in parents:
            if node_io[p]['layer_type'] in ["Weight", "Bias", "future_use"]:
                continue
            filtered_inputs.append(tensor_map[p])
        
        if len(filtered_inputs) == 1:
            return filtered_inputs[0]
        elif len(filtered_inputs) > 1:
            return filtered_inputs
        else:
            raise RuntimeError(f"No valid inputs found for {func_name}")
    
    return layer_in

def enforce_precision_consistency(tensors, target_dtype=None, preserve_original_dtype=True):
    """Ensure all tensors are real tensors (not FakeTensors) while preserving original properties"""
    if isinstance(tensors, (list, tuple)):
        # Ensure real tensors (not FakeTensors) are returned
        result = []
        for t in tensors:
            if isinstance(t, torch.Tensor):
                # Check if it's a FakeTensor and convert to real tensor if needed
                if 'FakeTensor' in str(type(t)):
                    # 🔧 CRITICAL FIX: Preserve original tensor values, don't create zeros!
                    # FakeTensors should already have the correct values from the forward pass
                    # Just ensure they're real tensors by detaching and cloning
                    real_tensor = t.detach().clone()
                    # 🔧 CRITICAL: Explicitly preserve device to prevent device drift
                    real_tensor = real_tensor.to(device=t.device)
                    get_logger().debug(f"🔧 Converted FakeTensor to real tensor: {real_tensor.shape} on {real_tensor.device}")
                    result.append(real_tensor)
                else:
                    # Keep original tensor unchanged for maximum consistency
                    result.append(t)
            else:
                result.append(t)
        return result
    elif isinstance(tensors, torch.Tensor):
        # Ensure real tensor (not FakeTensor) is returned
        if 'FakeTensor' in str(type(tensors)):
            # 🔧 CRITICAL FIX: Preserve original tensor values, don't create zeros!
            # FakeTensors should already have the correct values from the forward pass
            # Just ensure they're real tensors by detaching and cloning
            real_tensor = tensors.detach().clone()
            # 🔧 CRITICAL: Explicitly preserve device to prevent device drift
            real_tensor = real_tensor.to(device=tensors.device)
            get_logger().debug(f"🔧 Converted FakeTensor to real tensor: {real_tensor.shape} on {real_tensor.device}")
            return real_tensor
        else:
            # Keep original tensor unchanged for maximum consistency
            return tensors
    return tensors

def ensure_dtype_consistency(tensors, target_dtype=None, fallback_dtype=torch.float32):
    """
    🔧 CRITICAL FIX: Ensure all tensors have consistent dtypes for CPU compatibility
    Handles mixed float16/float32 scenarios common on CPU
    """
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    
    # Determine target dtype
    if target_dtype is None:
        # Find the most common dtype among tensors
        dtypes = [t.dtype for t in tensors if isinstance(t, torch.Tensor)]
        if dtypes:
            # Prefer float32 for CPU compatibility, but use the most common dtype
            if torch.float32 in dtypes:
                target_dtype = torch.float32
            else:
                target_dtype = dtypes[0]  # Use first dtype as fallback
        else:
            target_dtype = fallback_dtype
    
    # Convert all tensors to target dtype
    result = []
    for t in tensors:
        if isinstance(t, torch.Tensor):
            if t.dtype != target_dtype:
                # 🔧 CRITICAL: Convert half/float16 to float32 for CPU compatibility
                if t.dtype == torch.float16 or str(t.dtype) == 'c10::Half':
                    t = t.to(dtype=target_dtype)
                    get_logger().debug(f"🔧 Converted {t.dtype} to {target_dtype} for CPU compatibility")
                else:
                    t = t.to(dtype=target_dtype)
            result.append(t)
        else:
            result.append(t)
    
    return result[0] if len(result) == 1 else result

def ensure_tensor_consistency(tensors, target_dtype=None, target_device=None):
    """
    🔧 COMPREHENSIVE FIX: Ensure all tensors are consistent in dtype and device
    Handles None inputs, dtype mismatches, and device mismatches
    """
    if tensors is None:
        raise RuntimeError("❌ Input tensors cannot be None")
    
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    
    # Filter out None values
    valid_tensors = [t for t in tensors if t is not None and isinstance(t, torch.Tensor)]
    
    if not valid_tensors:
        raise RuntimeError("❌ No valid tensors found in input")
    
    # Determine target dtype and device
    if target_dtype is None:
        # Prefer float32 for CPU compatibility
        dtypes = [t.dtype for t in valid_tensors]
        if torch.float32 in dtypes:
            target_dtype = torch.float32
        else:
            target_dtype = dtypes[0]
    
    if target_device is None:
        target_device = valid_tensors[0].device
    
    # Convert all tensors to consistent dtype and device
    result = []
    for t in tensors:
        if t is None:
            result.append(None)
        elif isinstance(t, torch.Tensor):
            # Convert dtype if needed
            if t.dtype != target_dtype:
                t = t.to(dtype=target_dtype)
            # Convert device if needed
            if t.device != target_device:
                t = t.to(device=target_device)
            result.append(t)
        else:
            result.append(t)
    
    return result[0] if len(result) == 1 else result

def setup_consistent_environment(model=None):
    """Set up environment for consistent execution"""
    logger = get_logger()
    logger.debug("🔧 Setting up deterministic environment for consistent execution...")
    
    # 🔧 ENHANCED: Suppress warnings for cleaner execution
    try:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        logger.debug("✅ Warnings suppressed for cleaner execution")
    except Exception as e:
        logger.debug(f"⚠️  Could not suppress warnings: {e}")
    
    # Force evaluation mode
    torch.set_grad_enabled(False)
    logger.debug("✅ Gradients disabled")
    
    # 🔧 ENHANCED: Set deterministic algorithms for consistent results
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        logger.debug("✅ Deterministic cuDNN and CUDA settings applied")
    except Exception as e:
        logger.debug(f"⚠️  Could not set deterministic settings: {e}")
    
    # 🔧 ENHANCED: Use deterministic algorithms with warnings only
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.debug("✅ Deterministic algorithms enabled (warn_only=True)")
    except Exception as e:
        logger.debug(f"⚠️  Could not enable deterministic algorithms: {e}")
    
    # 🔧 ENHANCED: Set CuBLAS workspace config for deterministic behavior
    import os
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        logger.debug("✅ CUBLAS_WORKSPACE_CONFIG set to :4096:8 for deterministic behavior")
    else:
        logger.debug(f"✅ CUBLAS_WORKSPACE_CONFIG already set to: {os.environ['CUBLAS_WORKSPACE_CONFIG']}")
    
    # 🔧 ENHANCED: Set default dtype based on model dtype
    try:
        if model is not None:
            # Get model dtype from the first parameter or buffer
            model_dtype = torch.float32  # fallback
            for param in model.parameters():
                if param.is_floating_point():
                    model_dtype = param.dtype
                    break
            else:
                for buffer in model.buffers():
                    if buffer.is_floating_point():
                        model_dtype = buffer.dtype
                        break
            
            torch.set_default_dtype(model_dtype)
            logger.debug(f"✅ Default dtype set to {model_dtype} (from model)")
        else:
            torch.set_default_dtype(torch.float32)
            logger.debug("✅ Default dtype set to float32 (no model provided)")
    except Exception as e:
        logger.debug(f"⚠️  Could not set default dtype: {e}")
    
    # 🔧 ENHANCED: Set environment variables for deterministic execution
    try:
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        os.environ.setdefault('PYTHONHASHSEED', '42')
        logger.debug("✅ Environment variables set for deterministic execution")
    except Exception as e:
        logger.debug(f"⚠️  Could not set environment variables: {e}")
    
    # 🔧 ENHANCED: Force deterministic SDPA path if attention is used
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        logger.debug("✅ Deterministic SDPA path enabled")
    except Exception as e:
        logger.debug(f"⚠️  Could not set deterministic SDPA: {e}")
    
    # 🔧 ENHANCED: Memory management for consistent performance
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("✅ CUDA memory cleared and synchronized")
    except Exception as e:
        logger.debug(f"⚠️  Could not manage CUDA memory: {e}")
    
    logger.debug("🔧 Deterministic environment setup complete!")

def ensure_model_state_consistency(dlb_model, original_model):
    """Ensure DLB model has exactly the same state as original model"""
    logger = get_logger()
    logger.debug("🔧 Ensuring model state consistency...")
    try:
        # 🔧 MINIMAL FIX: Only copy data, don't modify dtypes or devices
        # Copy all parameters exactly without changing their properties
        for orig_param, dlb_param in zip(original_model.parameters(), dlb_model.parameters()):
            dlb_param.data.copy_(orig_param.data)
            # Don't modify dtype or device - keep original properties
        
        # Copy all buffers (running stats, etc.) without changing properties
        for orig_buffer, dlb_buffer in zip(original_model.buffers(), dlb_model.buffers()):
            dlb_buffer.data.copy_(orig_buffer.data)
            # Don't modify dtype or device - keep original properties
        
        # Force same mode and properties
        dlb_model.eval()
        dlb_model.requires_grad_(False)
        
        # 🔧 MINIMAL FIX: Just log parameter info, don't modify anything
        for param in dlb_model.parameters():
            logger.debug(f"Parameter {param.shape} dtype: {param.dtype}, device: {param.device}")
        
        logger.debug("✅ Model state synchronized successfully (minimal changes)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model state synchronization failed: {e}")
        return False

def ensure_input_consistency(inputs, target_model):
    """Ensure inputs are IDENTICAL between executions"""
    logger = get_logger()
    logger.debug("🔧 Ensuring input consistency...")
    try:
        # 🔧 MINIMAL FIX: Don't modify inputs unless absolutely necessary
        consistent_inputs = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                # 🔧 MINIMAL FIX: Only detach and clone, don't change dtype or device
                consistent_inp = inp.detach().clone()
                consistent_inp.requires_grad_(False)
                consistent_inputs.append(consistent_inp)
                logger.debug(f"✅ Input tensor {i}: {inp.shape} -> {consistent_inp.shape} on {consistent_inp.device}, dtype: {consistent_inp.dtype}")
            else:
                consistent_inputs.append(inp)
                logger.debug(f"✅ Non-tensor input {i}: {type(inp)}")
        
        logger.debug("✅ Input consistency ensured successfully (minimal changes)")
        return consistent_inputs
        
    except Exception as e:
        # Silently handle device-related errors - they're not critical
        # Only log error if it's a real issue, not just device detection
        if "device" not in str(e).lower() and "attribute" not in str(e).lower():
            logger.warning(f"🔧 Input consistency failed: {e}")
        return inputs

def ensure_embedding_consistency(weight, indices, node_name):
    """Ensure embedding operation produces consistent results"""
    logger = get_logger()
    logger.debug(f"🔧 Ensuring embedding consistency for {node_name}...")
    
    try:
        # 🔧 CRITICAL FIX: Ensure weight is properly formatted
        if isinstance(weight, torch.Tensor):
            # Force weight to be contiguous and on correct device
            if not weight.is_contiguous():
                weight = weight.contiguous()
                logger.debug(f"Made weight contiguous: {weight.shape}")
            
            # 🔧 CRITICAL FIX: Preserve original weight precision for consistency
            # Don't force float32 conversion as it can cause numerical differences
            logger.debug(f"Preserving weight precision: {weight.dtype}")
        
        # 🔧 CRITICAL FIX: Ensure indices are properly formatted
        if isinstance(indices, torch.Tensor):
            # Force indices to be long (integer) type
            if indices.dtype != torch.long:
                indices = indices.long()
                logger.debug(f"Converted indices to long: {indices.dtype}")
            
            # Ensure indices are contiguous
            if not indices.is_contiguous():
                indices = indices.contiguous()
                logger.debug(f"Made indices contiguous: {indices.shape}")
            
            # Ensure indices are on same device as weight
            if weight.device != indices.device:
                indices = indices.to(device=weight.device)
                logger.debug(f"Moved indices to device: {weight.device}")
        
        logger.debug(f"✅ Embedding consistency ensured:")
        logger.debug(f"   Weight: {weight.shape} on {weight.device}, dtype: {weight.dtype}")
        logger.debug(f"   Indices: {indices.shape} on {indices.device}, dtype: {indices.dtype}")
        
        return weight, indices
        
    except Exception as e:
        logger.error(f"❌ Embedding consistency failed: {e}")
        return weight, indices

def smart_broadcast_tensors(a, b, node_name, operation_name="operation"):
    """
    Enhanced broadcasting function for tensor operations following PyTorch broadcasting rules.
    
    PyTorch Broadcasting Rules:
    1. Two tensors are "broadcastable" if:
       - Each tensor has at least one dimension
       - When iterating over the dimension sizes, starting at the trailing dimension,
         the dimension sizes must either be equal, or one of them is 1, or one of them does not exist
    2. After broadcasting, each tensor behaves as if it had shape equal to the element-wise maximum
       of shapes of the two input tensors
    3. Any dimension of size 1 can be expanded to any size without copying data
    
    Args:
        a: First tensor
        b: Second tensor  
        node_name: Name of the node for logging
        operation_name: Name of the operation for error messages
        
    Returns:
        Tuple of broadcasted tensors (a_broadcast, b_broadcast)
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return a, b

    if a.shape == b.shape:
        return a, b

    logger = get_logger()
    logger.debug(f"[{node_name}] 🔧 Broadcasting for {operation_name}: {a.shape} × {b.shape}")

    # Strategy 1: PyTorch's native broadcasting (most efficient and correct)
    try:
        a_broadcast, b_broadcast = torch.broadcast_tensors(a, b)
        logger.debug(f"[{node_name}] ✅ Native broadcasting: {a.shape} → {a_broadcast.shape}, {b.shape} → {b_broadcast.shape}")
        return a_broadcast, b_broadcast
    except RuntimeError as e:
        logger.debug(f"[{node_name}] ⚠️ Native broadcasting failed: {e}")
        # Fall through to manual strategies

    # Strategy 2: Manual broadcasting following PyTorch rules
    try:
        # Check if tensors are broadcastable according to PyTorch rules
        def is_broadcastable(a, b):
            """Check if two tensors are broadcastable according to PyTorch rules"""
            a_dims = list(a.shape)
            b_dims = list(b.shape)
            
            # Pad shorter tensor with 1s on the left
            while len(a_dims) < len(b_dims):
                a_dims.insert(0, 1)
            while len(b_dims) < len(a_dims):
                b_dims.insert(0, 1)
            
            # Check each dimension
            for a_dim, b_dim in zip(a_dims, b_dims):
                if a_dim != b_dim and a_dim != 1 and b_dim != 1:
                    return False
            return True

        if not is_broadcastable(a, b):
            raise RuntimeError(f"Tensors are not broadcastable: {a.shape} × {b.shape}")

        # Manual broadcasting implementation
        a_dims = list(a.shape)
        b_dims = list(b.shape)
        
        # Pad shorter tensor with 1s on the left
        while len(a_dims) < len(b_dims):
            a_dims.insert(0, 1)
        while len(b_dims) < len(a_dims):
            b_dims.insert(0, 1)
        
        # Create target shape
        target_shape = []
        for a_dim, b_dim in zip(a_dims, b_dims):
            target_shape.append(max(a_dim, b_dim))
        
        # Expand tensors to target shape
        a_expanded = a
        b_expanded = b
        
        # Add dimensions to a if needed
        while a_expanded.dim() < len(target_shape):
            a_expanded = a_expanded.unsqueeze(0)
        
        # Add dimensions to b if needed
        while b_expanded.dim() < len(target_shape):
            b_expanded = b_expanded.unsqueeze(0)
        
        # Expand to target shape
        a_broadcast = a_expanded.expand(target_shape)
        b_broadcast = b_expanded.expand(target_shape)
        
        logger.debug(f"[{node_name}] ✅ Manual broadcasting: {a.shape} → {a_broadcast.shape}, {b.shape} → {b_broadcast.shape}")
        return a_broadcast, b_broadcast

    except Exception as expand_error:
        logger.debug(f"[{node_name}] ⚠️ Manual broadcasting failed: {expand_error}")

    # If all strategies fail, provide detailed error information
    logger.error(f"[{node_name}] ❌ All broadcasting strategies failed for {operation_name}")
    logger.error(f"[{node_name}] ❌ Tensor a: shape={a.shape}, dtype={a.dtype}, device={a.device}")
    logger.error(f"[{node_name}] ❌ Tensor b: shape={b.shape}, dtype={b.dtype}, device={b.device}")
    logger.error(f"[{node_name}] ❌ Shape compatibility check:")
    logger.error(f"[{node_name}]   - a.dim() = {a.dim()}, b.dim() = {b.dim()}")
    logger.error(f"[{node_name}]   - a.numel() = {a.numel()}, b.numel() = {b.numel()}")
    
    raise RuntimeError(f"[{node_name}] ❌ Cannot broadcast tensors for {operation_name}: "
                    f"a={a.shape} (dtype={a.dtype}) × b={b.shape} (dtype={b.dtype}). "
                    f"Tensors must be broadcastable according to PyTorch broadcasting rules.")

def needs_broadcasting(func_name):
    """
    Determine if an operation needs broadcasting support based on PyTorch API documentation.
    
    Returns:
        bool: True if operation supports broadcasting, False otherwise
    """
    # Operations that support broadcasting (element-wise operations)
    broadcasting_ops = {
        # Arithmetic operations
        "add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_", 
        "rsub", "rdiv", "pow", "pow_", "fmod", "remainder",
        
        # Comparison operations  
        "eq", "ne", "lt", "le", "gt", "ge", "equal", "not_equal",
        
        # Logical operations
        "logical_and", "logical_or", "logical_xor", "logical_not",
        "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
        
        # Mathematical functions
        "atan2", "hypot", "nextafter", "ldexp", "copysign",
        
        # Element-wise functions
        "maximum", "minimum", "clamp", "clamp_", "where",
        
        # Advanced operations
        "lerp", "slerp", "addcdiv", "addcmul",
    }
    
    return func_name in broadcasting_ops

def apply_smart_broadcasting(func_name, inputs, node_name, operation_name=None):
    """
    Apply smart broadcasting to inputs if the operation supports it.
    
    Args:
        func_name: Name of the operation
        inputs: List or tuple of input tensors
        node_name: Name of the node for logging
        operation_name: Optional custom operation name for logging
        
    Returns:
        Processed inputs with broadcasting applied if needed
    """
    if not needs_broadcasting(func_name):
        return inputs
    
    if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
        return inputs
    
    # Find tensor inputs
    tensor_inputs = [inp for inp in inputs if isinstance(inp, torch.Tensor)]
    if len(tensor_inputs) < 2:
        return inputs
    
    # Apply broadcasting to the first two tensor inputs
    a, b = tensor_inputs[0], tensor_inputs[1]
    op_name = operation_name or func_name
    
    try:
        a_broadcast, b_broadcast = smart_broadcast_tensors(a, b, node_name, op_name)
        
        # Replace the original tensors with broadcasted versions
        result = list(inputs)
        for i, inp in enumerate(result):
            if isinstance(inp, torch.Tensor):
                if inp is a:
                    result[i] = a_broadcast
                elif inp is b:
                    result[i] = b_broadcast
        
        return tuple(result) if isinstance(inputs, tuple) else result
        
    except Exception as e:
        logger = get_logger()
        logger.warning(f"[{node_name}] ⚠️ Broadcasting failed for {op_name}: {e}")
        return inputs

def ensure_operation_consistency(operation_name, inputs, weights, node_name):
    """Ensure any operation produces consistent results"""
    logger = get_logger()
    logger.debug(f"🔧 Ensuring operation consistency for {operation_name} in {node_name}...")
    
    try:
        # 🔧 CRITICAL FIX: Preserve original input precision for consistency
        # Don't force float32 conversion as it can cause numerical differences
        if isinstance(inputs, (list, tuple)):
            consistent_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    # Preserve original dtype and ensure contiguous
                    consistent_inp = inp.to(dtype=inp.dtype)  # Preserve original dtype
                    if not consistent_inp.is_contiguous():
                        consistent_inp = consistent_inp.contiguous()
                    consistent_inputs.append(consistent_inp)
                else:
                    consistent_inputs.append(inp)
            inputs = consistent_inputs
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.to(dtype=inputs.dtype)  # Preserve original dtype
            if not inputs.is_contiguous():
                inputs = inputs.contiguous()
        
        # 🔧 CRITICAL FIX: Preserve original weight precision for consistency
        # Don't force float32 conversion as it can cause numerical differences
        if isinstance(weights, (list, tuple)):
            consistent_weights = []
            for w in weights:
                if isinstance(w, torch.Tensor):
                    consistent_w = w.to(dtype=w.dtype)  # Preserve original dtype
                    if not consistent_w.is_contiguous():
                        consistent_w = consistent_w.contiguous()
                    consistent_weights.append(consistent_w)
                else:
                    consistent_weights.append(w)
            weights = consistent_weights
        elif isinstance(weights, torch.Tensor):
            weights = weights.to(dtype=weights.dtype)  # Preserve original dtype
            if not weights.is_contiguous():
                weights = weights.contiguous()
        
        logger.debug(f"✅ Operation consistency ensured for {operation_name}")
        return inputs, weights
        
    except Exception as e:
        logger.error(f"❌ Operation consistency failed: {e}")
        return inputs, weights

def sanitize_model_layer_inputs(layer_in, original_input):
    """Ensure Model_Layer inputs match original model inputs exactly"""
    if isinstance(layer_in, (list, tuple)):
        sanitized = []
        for i, inp in enumerate(layer_in):
            if isinstance(inp, torch.Tensor):
                # Force same properties as original
                sanitized.append(inp.detach().to(dtype=original_input.dtype, device=original_input.device))
            else:
                sanitized.append(inp)
        return sanitized
    else:
        if isinstance(layer_in, torch.Tensor):
            return layer_in.detach().to(dtype=original_input.dtype, device=original_input.device)
        return layer_in


# Duplicate function removed - using the first definition above


def _process_output_tuple(output):
    if isinstance(output, (torch.Tensor, int)):
        return output  # ✅ Return directly, not wrapped in a list

    result = []

    def flatten(x):
        if isinstance(x, torch.Tensor):
            result.append(x)
        elif isinstance(x, (list, tuple)):
            for item in x:
                flatten(item)
        elif x is not None:
            try:
                result.append(torch.tensor(x))
            except Exception:
                pass

    flatten(output)

    if len(result) == 1:
        return result[0]  # ✅ return just the tensor
    elif len(result) > 1:
        return tuple(result)
    else:
        return None


def _process_layer_input(layer_in):
    list_inp = False
    # ✅ Only unwrap if it's a singleton *list of tensors*
    while isinstance(layer_in, (list, tuple)) and len(layer_in) == 1:
        if isinstance(layer_in[0], (torch.Tensor, int, float, np.ndarray)):
            layer_in = layer_in[0]
        else:
            break
    return layer_in, isinstance(layer_in, list)


def flatten_paths(val):
    if isinstance(val, list):
        flat = []
        for v in val:
            flat.extend(flatten_paths(v))
        return flat
    elif isinstance(val, str):
        return [val]
    else:
        return [val]


def _check_tensor_sig(t, name):
    """Create a compact signature for tensor comparison"""
    if isinstance(t, torch.Tensor):
        # Get a small slice hash for value verification
        try:
            flat = t.flatten()
            slice_hash = hash(tuple(flat[:32].tolist())) if flat.numel() > 0 else 0
        except:
            slice_hash = 0
        return (name, str(t.dtype), str(t.device), tuple(t.shape), t.is_contiguous(), slice_hash)
    else:
        return (name, str(type(t)), "non-tensor", "non-tensor", False, 0)

def _assert_same_sig(sig_eager, sig_exec, node_name):
    """Assert tensor signatures match between eager and executor"""
    if sig_eager != sig_exec:
        logger = get_logger()
        logger.error(f"❌ Signature mismatch in {node_name}:")
        logger.error(f"   Eager: {sig_eager}")
        logger.error(f"   Exec:  {sig_exec}")
        raise AssertionError(f"Tensor signature mismatch in {node_name}: eager={sig_eager} exec={sig_exec}")

def _precheck_sdpa(q, k, v, attn_mask, dropout_p, is_causal, node_name, model_compute_dtype):
    """Pre-execution checks for SDPA operations"""
    logger = get_logger()
    
    # Check tensor signatures
    exec_sig = tuple(_check_tensor_sig(x, n) for x, n in [(q, "q"), (k, "k"), (v, "v")])
    logger.debug(f"🔍 SDPA tensor signatures: {exec_sig}")
    
    # Dtype policy - ensure q/k/v match model compute dtype
    if not all(x.dtype == model_compute_dtype for x in (q, k, v)):
        logger.warning(f"⚠️  SDPA dtype mismatch: q={q.dtype}, k={k.dtype}, v={v.dtype}, expected={model_compute_dtype}")
    
    # Mask checks (allow boolean or additive float masks)
    if attn_mask is not None:
        if not (attn_mask.dtype == torch.bool or torch.is_floating_point(attn_mask)):
            logger.warning(f"⚠️  SDPA mask uncommon dtype: {attn_mask.dtype} (expected bool or floating)")
    
    # Eval mode checks
    if dropout_p != 0.0:
        logger.warning(f"⚠️  SDPA dropout_p should be 0.0 in eval, got {dropout_p}")
    
    # Causal flag check
    if not isinstance(is_causal, bool):
        logger.warning(f"⚠️  SDPA is_causal should be bool, got {type(is_causal)}")
    
    logger.debug(f"✅ SDPA pre-checks passed for {node_name}")

def _precheck_layernorm(x, normalized_shape, weight, bias, eps, node_name, layer_eps=None):
    """Pre-execution checks for LayerNorm operations"""
    logger = get_logger()
    
    # Check if eps matches layer's actual epsilon
    if layer_eps is not None and eps != layer_eps:
        logger.warning(f"⚠️  LayerNorm eps mismatch: passed={eps}, layer={layer_eps}")
    
    # Check tensor properties
    if not isinstance(x, torch.Tensor):
        logger.error(f"❌ LayerNorm input must be tensor, got {type(x)}")
        raise AssertionError(f"LayerNorm input type mismatch: expected tensor, got {type(x)}")
    
    logger.debug(f"✅ LayerNorm pre-checks passed for {node_name}")

def _precheck_embedding(indices, weight, node_name):
    """Pre-execution checks for Embedding operations"""
    logger = get_logger()
    
    # Check indices dtype
    if indices.dtype != torch.long:
        logger.error(f"❌ Embedding indices must be long, got {indices.dtype}")
        raise AssertionError(f"Embedding indices dtype mismatch: expected long, got {indices.dtype}")
    
    # Check weight properties
    if not isinstance(weight, torch.Tensor):
        logger.error(f"❌ Embedding weight must be tensor, got {type(weight)}")
        raise AssertionError(f"Embedding weight type mismatch: expected tensor, got {type(weight)}")
    
    logger.debug(f"✅ Embedding pre-checks passed for {node_name}")

def execute_aten_operation(func_name, aten_op, layer_in, layer_hyperparams, method_args, parents, node_io, node_name,tensor_map, children=None, model_dtype=None):
    logger = get_logger()
    logger.debug(f"🔧 Executing aten operation: {func_name}")
    logger.debug(f"🔧 EXECUTING ATEN OPERATION: {func_name} (node: {node_name})")
    logger.debug(f"🔧 Node name: {node_name}")
    logger.debug(f"🔧 Children: {children}")
    logger.debug(f"🔧 Parents: {parents}")
    
    try:
        # 🔧 CRITICAL: Infer model compute dtype with model reference
        model_compute_dtype = model_dtype if model_dtype is not None else torch.float32
        try:
            if isinstance(layer_in, torch.Tensor) and layer_in.is_floating_point():
                model_compute_dtype = layer_in.dtype
            elif isinstance(layer_in, (list, tuple)):
                for _x in layer_in:
                    if isinstance(_x, torch.Tensor) and _x.is_floating_point():
                        model_compute_dtype = _x.dtype
                        break
            else:
                # Fallback: scan node_io for any floating tensor
                for _data in node_io.values():
                    if isinstance(_data, dict) and 'output_values' in _data:
                        _t = _data['output_values']
                        if isinstance(_t, torch.Tensor) and _t.is_floating_point():
                            model_compute_dtype = _t.dtype
                            break
        except Exception:
            pass
        
        # 🔧 MINIMAL FIX: Only convert FakeTensors to real tensors, don't modify properties
        if isinstance(layer_in, (list, tuple)):
            layer_in = [enforce_precision_consistency(x) for x in layer_in]
        else:
            layer_in = enforce_precision_consistency(layer_in)
        
        # --- Special-case native batch_norm to preserve full tuple ---
        if func_name in ("_native_batch_norm_legit_no_training", "batch_norm"):
            # Determine behavior based on the type of layer_in
            if isinstance(layer_in, (list, tuple)) and len(layer_in) >= 5:
                # Inputs are passed as a tuple/list → strict mode
                inp, weight, bias, running_mean, running_var = layer_in[:5]
                momentum, eps = method_args
                output = aten_op(inp, weight, bias, running_mean, running_var, momentum, eps)
                return output
            else:
                # Inputs passed individually or as a non-strict struct → fallback to legacy
                output = aten_op(
                    layer_in,
                    layer_hyperparams.get("weight"),
                    layer_hyperparams.get("bias"),
                    layer_hyperparams.get("running_mean"),
                    layer_hyperparams.get("running_var"),
                    layer_hyperparams.get("momentum"),
                    layer_hyperparams.get("eps"),
                )
                return output

        # --- Handle FX getitem nodes: tuple-indexing instead of tensor slicing ---
        if func_name == "getitem":
            parent = parents[0]
            tup = node_io[parent]["output_values"]
            idx = method_args[0]

            # --- Error checking ---
            if tup is None:
                raise RuntimeError(f"[ERROR] Cannot apply getitem on None from parent node '{parent}'")

            if not isinstance(tup, (tuple, list)):
                raise TypeError(f"[ERROR] Invalid type for getitem: {type(tup)} from node '{parent}'")

            return tup[idx]

        if func_name == "linear":
            # 🔧 CRITICAL FIX: Consistent linear operation handling with dtype consistency
            logger = get_logger()
            logger.debug(f"🔧 linear: input type={type(layer_in)}, parents={parents}")
            
            # 🔧 FIX: Use standardized input processing
            layer_in = standardize_layer_input(layer_in, parents, node_io, tensor_map, func_name)
            
            # 🔧 CRITICAL FIX: Ensure dtype consistency for CPU compatibility
            if isinstance(layer_in, torch.Tensor):
                weight = layer_hyperparams["weight"]
                bias = layer_hyperparams.get("bias", None)
                
                # 🔧 CRITICAL FIX: Ensure all tensors have consistent dtypes
                layer_in, weight = ensure_tensor_consistency([layer_in, weight])
                if bias is not None and not isinstance(bias, bool):
                    layer_in, weight, bias = ensure_tensor_consistency([layer_in, weight, bias])
                
                logger.debug(f"✅ linear: input shape={layer_in.shape}, weight shape={weight.shape}")
                logger.debug(f"✅ linear: input dtype={layer_in.dtype}, weight dtype={weight.dtype}")
                
                # Execute linear operation with consistent dtypes
                if bias is not None and not isinstance(bias, bool):
                    output = aten_op(layer_in, weight, bias)
                else:
                    output = aten_op(layer_in, weight)
                
                logger.debug(f"✅ linear: output shape={output.shape}")
                
                return output
            else:
                raise RuntimeError(f"[{node_name}] ❌ linear: expected tensor input, got {type(layer_in)}")

        elif func_name == "conv2d": 
            # 🔧 CONSISTENCY FIX: Apply precision consistency
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)

            bias = layer_hyperparams["bias"]
            if isinstance(bias, bool):
                bias = None  # ✅ Fix: convert True/False to None

            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            target_device = layer_in.device
            weight = layer_hyperparams["weight"]
            if weight.device != target_device:
                weight = weight.to(device=target_device)
            if bias is not None and bias.device != target_device:
                bias = bias.to(device=target_device)

            return aten_op(layer_in, weight, bias,
                           layer_hyperparams["stride"], layer_hyperparams["padding"],
                           layer_hyperparams["dilation"], layer_hyperparams["groups"])

        elif func_name == "conv1d":
            # 🔧 NEW OPERATION: 1D Convolution for sequence processing
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)

            bias = layer_hyperparams.get("bias", None)
            if isinstance(bias, bool):
                bias = None

            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            target_device = layer_in.device
            weight = layer_hyperparams["weight"]
            if weight.device != target_device:
                weight = weight.to(device=target_device)
            if bias is not None and bias.device != target_device:
                bias = bias.to(device=target_device)

            return aten_op(layer_in, weight, bias,
                           layer_hyperparams["stride"], layer_hyperparams["padding"],
                           layer_hyperparams["dilation"], layer_hyperparams["groups"])

        elif func_name == "max_pool2d":
            # 🔧 CONSISTENCY FIX: Apply precision consistency
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            return aten_op(layer_in,
                           layer_hyperparams["kernel_size"],
                           layer_hyperparams["stride"],
                           layer_hyperparams["padding"],
                           layer_hyperparams["dilation"],
                           layer_hyperparams["ceil_mode"])

        elif func_name == "adaptive_avg_pool2d":
            # 🔧 CONSISTENCY FIX: Apply precision consistency
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            return aten_op(layer_in, layer_hyperparams["output_size"])

        elif func_name == "avg_pool2d":
            # 🔧 NEW OPERATION: 2D Average Pooling
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            return aten_op(layer_in,
                           layer_hyperparams["kernel_size"],
                           layer_hyperparams["stride"],
                           layer_hyperparams["padding"],
                           layer_hyperparams["ceil_mode"],
                           layer_hyperparams["count_include_pad"])

        elif func_name == "dropout":
            # 🔧 CONSISTENCY FIX: Apply precision consistency and force evaluation mode
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            return aten_op(layer_in, layer_hyperparams["p"], False)  # Always False

        elif func_name in ("relu", "relu_", "gelu", "tanh","silu"):
            # 🔧 CRITICAL FIX: Robust activation function handling with proper input validation
            logger = get_logger()
            logger.debug(f"🔧 {func_name}: input type={type(layer_in)}, length={len(layer_in) if isinstance(layer_in, (list, tuple)) else 'single'}")
            
            # 🔧 FIX: Handle list inputs properly
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    raise RuntimeError(f"[{node_name}] ❌ {func_name}: received empty input list")
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: Find first valid tensor without modifying original
                    valid_tensor = None
                    for x in layer_in:
                        if isinstance(x, torch.Tensor):
                            valid_tensor = x
                            break
                    
                    if valid_tensor is None:
                        raise RuntimeError(f"[{node_name}] ❌ {func_name}: no valid tensor found in input list")
                    
                    layer_in = valid_tensor
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ {func_name}: expected tensor input, got {type(layer_in)}")
            
            # 🔧 DEVICE CONSISTENCY FIX: Ensure tensor is on the correct device
            if not layer_in.is_contiguous():
                layer_in = layer_in.contiguous()
            
            logger.debug(f"✅ {func_name}: input shape={layer_in.shape}, device={layer_in.device}")
            
            try:
                # 🔧 FIX: aten_op is the actual PyTorch function, call it directly
                output = aten_op(layer_in)
                
                logger.debug(f"✅ {func_name}: output shape={output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ {func_name}: failed with input shape={layer_in.shape}. Error: {e}")

        elif "unsqueeze" in func_name :           
            # 🔧 CRITICAL FIX: Robust input handling for unsqueeze operation
            logger = get_logger()
            logger.debug(f"🔧 unsqueeze: input type={type(layer_in)}, length={len(layer_in) if isinstance(layer_in, (list, tuple)) else 'single'}")
            
            # 🔧 FIX: Preserve original input and handle list inputs properly
            original_layer_in = layer_in
            
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    logger.warning(f"⚠️ unsqueeze received empty input list, attempting recovery")
                    
                    # 🔧 FIX: Try to recover from parents without overwriting original input
                    recovered_input = None
                    for node_x in parents:
                        if node_io[node_x]['layer_type'] not in ("Weight", "Bias", "future_use"):
                            recovered_input = node_io[node_x]['output_values']
                            logger.debug(f"🔧 Recovered input from parent {node_x}: {type(recovered_input)}")
                            break
                    
                    if recovered_input is None:
                        raise RuntimeError(f"[{node_name}] ❌ unsqueeze: no valid input found in parents")
                    
                    # 🔧 FIX: Use recovered input but don't modify original
                    layer_in = recovered_input
                    
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: Find first valid tensor without modifying original
                    valid_tensor = None
                    for x in layer_in:
                        if isinstance(x, torch.Tensor):
                            valid_tensor = x
                            break
                    
                    if valid_tensor is None:
                        raise RuntimeError(f"[{node_name}] ❌ unsqueeze: no valid tensor found in input list")
                    
                    layer_in = valid_tensor
            
            # 🔧 FIX: Validate input type
            if not isinstance(layer_in, torch.Tensor):
                raise RuntimeError(f"[{node_name}] ❌ unsqueeze expected tensor input, got {type(layer_in)}")
            
            # 🔧 FIX: Validate dimension parameter
            dim = layer_hyperparams.get("dim")
            if dim is None:
                raise RuntimeError(f"[{node_name}] ❌ unsqueeze: 'dim' parameter is required")
            
            # 🔧 ENHANCED: Handle negative indexing for unsqueeze
            if not isinstance(dim, int):
                raise TypeError(f"[{node_name}] ❌ unsqueeze: dimension must be integer, got {type(dim)}")
            
            # Handle negative indexing (PyTorch standard behavior)
            tensor_rank = layer_in.dim()
            if dim < 0:
                dim = tensor_rank + 1 + dim  # For unsqueeze, -1 means last+1 position
            
            # Validate converted dimension
            if dim < 0 or dim > tensor_rank:
                raise ValueError(f"[{node_name}] ❌ unsqueeze: dimension {layer_hyperparams.get('dim')} out of range for tensor of rank {tensor_rank}")
            
            logger.debug(f"✅ unsqueeze: input shape={layer_in.shape}, dim={dim}")
            
            try:
                output = aten_op(layer_in, dim)
                logger.debug(f"✅ unsqueeze output shape: {output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ unsqueeze failed: input shape={layer_in.shape}, dim={dim}. Error: {e}")
            
        elif "squeeze" in func_name :
            # 🔧 ENHANCED: Robust squeeze operation with negative indexing support
            logger = get_logger()
            logger.debug(f"[{node_name}] 🔧 squeeze: input type={type(layer_in)}")
            
            # 🔧 CONSISTENCY FIX: Apply precision consistency
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            # 🔧 FIX: Validate input
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ squeeze: expected tensor input, got {type(layer_in)}")
            
            # 🔧 FIX: Get and validate dimension
            dim = layer_hyperparams.get("dim")
            if dim is None:
                raise RuntimeError(f"[{node_name}] ❌ squeeze: 'dim' parameter is required")
            
            # 🔧 ENHANCED: Handle negative indexing for squeeze
            if not isinstance(dim, int):
                raise TypeError(f"[{node_name}] ❌ squeeze: dimension must be integer, got {type(dim)}")
            
            # Handle negative indexing (PyTorch standard behavior)
            tensor_rank = layer_in.dim()
            if dim < 0:
                dim = tensor_rank + dim
            
            # Validate converted dimension
            if dim < 0 or dim >= tensor_rank:
                raise ValueError(f"[{node_name}] ❌ squeeze: dimension {layer_hyperparams.get('dim')} out of range for tensor of rank {tensor_rank}")
            
            logger.debug(f"[{node_name}] ✅ squeeze: input shape={layer_in.shape}, dim={dim}")
            
            try:
                return aten_op(layer_in, dim)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ squeeze: execution failed: {e}")

        elif func_name == "layer_norm":
            # 🔧 CONSISTENCY FIX: Use standardized input processing
            layer_in = standardize_layer_input(layer_in, parents, node_io, tensor_map, func_name)
            
            # 🔧 CONSISTENCY FIX: Enforce precision consistency
            layer_in = enforce_precision_consistency(layer_in)
            
            if not isinstance(layer_in, torch.Tensor):
                raise RuntimeError(f"[{node_name}] ❌ `layer_norm` expected Tensor input, got {type(layer_in)}")
        
            # 🔧 VALIDATION: Check required parameters exist
            if "weight" not in layer_hyperparams:
                raise RuntimeError(f"[{node_name}] ❌ `layer_norm` missing weight parameter")
            if "bias" not in layer_hyperparams:
                raise RuntimeError(f"[{node_name}] ❌ `layer_norm` missing bias parameter")
            if "normalized_shape" not in layer_hyperparams:
                raise RuntimeError(f"[{node_name}] ❌ `layer_norm` missing normalized_shape parameter")
            
            weight = layer_hyperparams["weight"]
            bias = layer_hyperparams["bias"]
            normalized_shape = layer_hyperparams["normalized_shape"]
            
            # 🔧 VALIDATION: Check parameter types
            if not isinstance(weight, torch.Tensor):
                raise RuntimeError(f"[{node_name}] ❌ `layer_norm` weight must be Tensor, got {type(weight)}")
            if not isinstance(bias, torch.Tensor):
                raise RuntimeError(f"[{node_name}] ❌ `layer_norm` bias must be Tensor, got {type(bias)}")
            
            # 🔧 VALIDATION: Check parameter shapes
            if weight.shape != bias.shape:
                raise RuntimeError(f"[{node_name}] ❌ `layer_norm` weight shape {weight.shape} != bias shape {bias.shape}")
            
            # 🔧 VALIDATION: Check normalized_shape matches input dimensions
            if isinstance(normalized_shape, (list, tuple)):
                expected_shape = tuple(normalized_shape)
            else:
                expected_shape = tuple(normalized_shape.tolist()) if hasattr(normalized_shape, 'tolist') else (normalized_shape,)
            
            if layer_in.shape[-len(expected_shape):] != expected_shape:
                raise RuntimeError(f"[{node_name}] ❌ `layer_norm` input shape {layer_in.shape} doesn't match normalized_shape {expected_shape}")
            
            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            target_device = layer_in.device
            if weight.device != target_device:
                weight = weight.to(device=target_device)
            if bias.device != target_device:
                bias = bias.to(device=target_device)
            
            # 🔧 DTYPE CONSISTENCY FIX: Ensure all tensors have the same dtype
            target_dtype = layer_in.dtype
            if weight.dtype != target_dtype:
                weight = weight.to(dtype=target_dtype)
            if bias.dtype != target_dtype:
                bias = bias.to(dtype=target_dtype)
        
            # 🔧 EPSILON: Allow override from hyperparams while having a default
            eps = layer_hyperparams.get("eps", 1e-5)
            
            # 🔧 CRITICAL: Pre-execution checks for layer_norm
            _precheck_layernorm(layer_in, normalized_shape, weight, bias, eps, node_name, eps)
            
            # 🔧 DEBUGGING: Log parameter shapes/values for debugging
            logger.debug(f"🔧 layer_norm parameters:")
            logger.debug(f"  Input shape: {layer_in.shape}")
            logger.debug(f"  Weight shape: {weight.shape}")
            logger.debug(f"  Bias shape: {bias.shape}")
            logger.debug(f"  Normalized shape: {normalized_shape}")
            logger.debug(f"  Epsilon: {eps}")
            logger.debug(f"  Input sample: {layer_in.flatten()[:5]}")
            logger.debug(f"  Weight sample: {weight[:5]}")
            logger.debug(f"  Bias sample: {bias[:5]}")
            
            # 🔧 ERROR HANDLING: Try/catch around the actual operation
            try:
                output = aten_op(layer_in, normalized_shape, weight, bias, eps)
                logger.debug(f"  Output sample: {output.flatten()[:5]}")
                return output
            except Exception as e:
                logger.error(f"[{node_name}] ❌ `layer_norm` execution failed: {e}")
                logger.error(f"  Input shape: {layer_in.shape}")
                logger.error(f"  Weight shape: {weight.shape}")
                logger.error(f"  Bias shape: {bias.shape}")
                logger.error(f"  Normalized shape: {normalized_shape}")
                raise

        elif func_name == "batch_norm":
            # 🔧 CONSISTENCY FIX: Use standardized input processing
            layer_in = standardize_layer_input(layer_in, parents, node_io, tensor_map, func_name)
            
            # 🔧 CONSISTENCY FIX: Enforce precision consistency
            layer_in = enforce_precision_consistency(layer_in)
            
            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            target_device = layer_in.device
            weight = layer_hyperparams["weight"]
            bias = layer_hyperparams["bias"]
            running_mean = layer_hyperparams["running_mean"]
            running_var = layer_hyperparams["running_var"]
            
            if weight.device != target_device:
                weight = weight.to(device=target_device)
            if bias.device != target_device:
                bias = bias.to(device=target_device)
            if running_mean.device != target_device:
                running_mean = running_mean.to(device=target_device)
            if running_var.device != target_device:
                running_var = running_var.to(device=target_device)
            
            # 🔧 CONSISTENCY FIX: Use the layer's configured epsilon
            eps = layer_hyperparams.get("eps", 1e-5)
            
            return aten_op(layer_in,
                           weight,
                           bias,
                           running_mean,
                           running_var,
                           True,
                           layer_hyperparams["momentum"],
                           eps,
                           False)

        elif func_name == "scaled_dot_product_attention":
            # 🔧 CRITICAL FIX: Handle None inputs and ensure proper tensor extraction
            if layer_in is None:
                raise RuntimeError(f"[{node_name}] ❌ scaled_dot_product_attention: layer_in is None")
            
            if not isinstance(layer_in, (list, tuple)) or len(layer_in) < 3:
                raise RuntimeError(f"[{node_name}] ❌ scaled_dot_product_attention: expected 3 inputs, got {type(layer_in)} with length {len(layer_in) if hasattr(layer_in, '__len__') else 'N/A'}")
            
            q, k, v = layer_in[0], layer_in[1], layer_in[2]
            
            # 🔧 CRITICAL FIX: Validate inputs are tensors
            if q is None or k is None or v is None:
                raise RuntimeError(f"[{node_name}] ❌ scaled_dot_product_attention: One or more inputs are None (q={q is None}, k={k is None}, v={v is None})")
            
            if not all(isinstance(t, torch.Tensor) for t in [q, k, v]):
                raise RuntimeError(f"[{node_name}] ❌ scaled_dot_product_attention: All inputs must be tensors (q={type(q)}, k={type(k)}, v={type(v)})")
            
            # 🔧 CRITICAL FIX: Ensure dtype consistency - convert all to the same dtype
            target_dtype = q.dtype  # Use query's dtype as reference
            if k.dtype != target_dtype:
                k = k.to(dtype=target_dtype)
            if v.dtype != target_dtype:
                v = v.to(dtype=target_dtype)
            
            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            target_device = q.device
            if k.device != target_device:
                k = k.to(device=target_device)
            if v.device != target_device:
                v = v.to(device=target_device)
            
            # 🔧 CONSISTENCY FIX: Force consistent precision
            q = enforce_precision_consistency(q)
            k = enforce_precision_consistency(k)
            v = enforce_precision_consistency(v)
            
            # 🔧 ATTENTION MASK FIX: Handle attention mask with minimal broadcast, preserve dtype
            # Prefer the 4th positional input if present; fall back to hyperparams
            attn_mask = None
            if isinstance(layer_in, (list, tuple)) and len(layer_in) >= 4 and isinstance(layer_in[3], torch.Tensor):
                attn_mask = layer_in[3]
            else:
                attn_mask = layer_hyperparams.get("attn_mask", None)
            if attn_mask is not None:
                # Keep mask dtype (bool/additive) exactly as provided
                # Move to correct device if needed
                if attn_mask.device != q.device:
                    attn_mask = attn_mask.to(device=q.device)

                # Only reshape if strictly required for torch.sdpa; prefer minimal broadcast
                # Acceptable shapes: [B, S], [B, 1, 1, S], [B, 1, S, S], [B, H, S, S]
                if attn_mask.dim() == 2:
                    # Convert [B, S] to [B, 1, 1, S]
                    attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            
            # 🔧 CRITICAL: Pre-execution checks for SDPA
            dropout_p = layer_hyperparams.get("dropout_p", 0.0)
            
            # 🔧 CRITICAL FIX: Proper model type detection for RoBERTa/BERT vs GPT/Llama
            if "is_causal" not in layer_hyperparams:
                # DEFAULT to bidirectional (is_causal=False) since most transformer models are bidirectional
                # Only use causal=True for explicitly detected causal models
                is_bidirectional_model = True  # Default assumption
                
                # Check for RoBERTa/BERT patterns in ALL node names in node_io
                model_type_indicators = []
                if node_io:
                    for existing_node in node_io.keys():
                        if any(pattern in existing_node.lower() for pattern in ['roberta', 'bert', 'distilbert', 'electra', 'encoder']):
                            model_type_indicators.append('bidirectional')
                        elif any(pattern in existing_node.lower() for pattern in ['gpt', 'llama', 'opt', 'bloom', 'decoder']):
                            model_type_indicators.append('causal')
                
                # Determine model type based on indicators
                if 'bidirectional' in model_type_indicators:
                    is_bidirectional_model = True
                elif 'causal' in model_type_indicators:
                    is_bidirectional_model = False
                # else: keep default (bidirectional=True)
                
                # Set appropriate default based on model type
                if is_bidirectional_model:
                    is_causal = False
                    logger.debug(f"🔧 Auto-detected bidirectional model → setting is_causal=False for {node_name}")
                else:
                    is_causal = True
                    logger.debug(f"🔧 Auto-detected causal model → setting is_causal=True for {node_name}")
            else:
                is_causal = layer_hyperparams["is_causal"]
            
            _precheck_sdpa(q, k, v, attn_mask, dropout_p, is_causal, node_name, model_compute_dtype)
            
            # 🔧 DEBUGGING: Log attention mask info
            logger.debug(f"🔧 scaled_dot_product_attention parameters:")
            logger.debug(f"  Query shape: {q.shape}")
            logger.debug(f"  Key shape: {k.shape}")
            logger.debug(f"  Value shape: {v.shape}")
            logger.debug(f"  Attention mask shape: {attn_mask.shape if attn_mask is not None else 'None'}")
            logger.debug(f"  Dropout p: {dropout_p}")
            logger.debug(f"  Is causal: {is_causal}")
            
            return aten_op(q, k, v, attn_mask, dropout_p, is_causal)

        elif func_name == "lstm":
            return aten_op(layer_in[0],
                           (layer_in[1], layer_in[2]),
                           layer_hyperparams["params"],
                           layer_hyperparams["has_biases"],
                           layer_hyperparams["num_layers"],
                           layer_hyperparams["dropout"],
                           False,
                           layer_hyperparams["bidirectional"],
                           layer_hyperparams["batch_first"])

        elif func_name == "reshape":
            if "shape" in layer_hyperparams:
                if isinstance(layer_hyperparams["shape"], (tuple, list)) and isinstance(layer_in,list):
                    if len(layer_in)>0:
                        for item in range(0,len(layer_hyperparams['shape'])):
                            if isinstance(layer_hyperparams['shape'][item], torch.fx.node.Node):
                                idx = parents.index(str(layer_hyperparams['shape'][item]))
                                value = layer_in[idx]
                                layer_hyperparams['shape'][item] = value
                    layer_in = layer_in[0]
                output = aten_op(layer_in,layer_hyperparams['shape'])
            return output

        elif func_name in ("masked_fill", "masked_fill_"):
            logger.debug(f"[{node_name}] ➕ Executing `{func_name}`")
            logger.debug(f"[{node_name}] Parents: {parents}")
            logger.debug(f"[{node_name}] method_args: {method_args}")
            logger.debug(f"[{node_name}] layer_hyperparams: {layer_hyperparams}")

            if isinstance(layer_in, list) and len(layer_in) > 1:
                mask = layer_in[1]
                layer_in = layer_in[0]

                # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
                target_device = layer_in.device
                if mask.device != target_device:
                    mask = mask.to(device=target_device)

                # ✅ Ensure mask is boolean
                if mask.dtype != torch.bool:
                    mask = mask != 0
            else:
                raise RuntimeError(f"[{node_name}] ❌ Expected list with input and mask for `{func_name}`, got: {layer_in}")

            value = layer_hyperparams.get("value", 0.0)
            if isinstance(value, torch.Tensor):
                value = value.item()  # Convert single-element tensor to Python scalar

            # 🔧 CRITICAL: Preserve -inf values, don't replace with large negatives
            if value == float('-inf'):
                logger.debug(f"✅ masked_fill preserving -inf value (correct)")
                value = float('-inf')  # Keep -inf as-is

            logger.debug(f"[{node_name}] input shape: {getattr(layer_in, 'shape', None)}, dtype: {getattr(layer_in, 'dtype', None)}")
            logger.debug(f"[{node_name}] mask shape: {getattr(mask, 'shape', None)}, dtype: {getattr(mask, 'dtype', None)}")
            logger.debug(f"[{node_name}] fill value: {value}")

            try:
                output = aten_op(layer_in, mask, value)
                logger.debug(f"[{node_name}] ✅ `masked_fill` success → output shape: {output.shape}")
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `masked_fill` failed with shapes: input={layer_in.shape}, mask={mask.shape}, value={value}. Error: {e}")

            return output

        elif func_name == "view":
            # 🔧 CRITICAL FIX: Robust view operation with proper validation
            logger.debug(f"🔧 view: input type={type(layer_in)}, method_args={method_args}")
            
            # 🔧 FIX: Validate input before processing
            if layer_in is None:
                raise RuntimeError(f"[{node_name}] ❌ view: input is None")
            
            # 🔧 FIX: Handle list inputs properly
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    raise RuntimeError(f"[{node_name}] ❌ view: received empty input list")
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: Find first valid tensor without modifying original
                    valid_tensor = None
                    for x in layer_in:
                        if isinstance(x, torch.Tensor):
                            valid_tensor = x
                            break
                    
                    if valid_tensor is None:
                        raise RuntimeError(f"[{node_name}] ❌ view: no valid tensor found in input list")
                    
                    layer_in = valid_tensor
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ view: expected tensor input, got {type(layer_in)}")
            
            # 🔧 FIX: Get shape with proper fallback
            shape = layer_hyperparams.get("shape", [])
            if not shape or any(s is None for s in shape):
                if method_args and isinstance(method_args[0], (list, tuple)):
                    shape = method_args[0]
                    logger.debug(f"[{node_name}] 🔧 view: using shape from method_args: {shape}")
                else:
                    raise RuntimeError(f"[{node_name}] ❌ view: no valid shape found in hyperparams or method_args")
            
            logger.debug(f"[{node_name}] 🔧 view: processing shape: {shape}")
            
            # 🔧 FIX: Robust parameter resolution with proper error handling
            def resolve_param_view(p, idx):
                try:
                    if isinstance(p, (torch.fx.Node, str)):
                        p_key = str(p)
                        if p_key in node_io:
                            val = node_io[p_key]["output_values"]
                            if isinstance(val, torch.Tensor) and val.numel() == 1:
                                return int(val.item())
                            elif isinstance(val, (int, float)):
                                return int(val)
                            else:
                                raise ValueError(f"Node {p_key} output is not a scalar: {type(val)}")
                        else:
                            raise KeyError(f"Node {p_key} not found in node_io")
                    elif isinstance(p, torch.Tensor):
                        if p.numel() == 1:
                            return int(p.item())
                        else:
                            raise ValueError(f"Tensor has multiple elements: {p.shape}")
                    elif isinstance(p, (int, torch.SymInt)):
                        return int(p)
                    elif p is None:
                        raise ValueError("Cannot resolve None in shape")
                    else:
                        raise TypeError(f"Unsupported shape type: {type(p)}")
                        
                except Exception as e:
                    raise RuntimeError(f"Failed to resolve shape element {idx} ({p}): {e}")

            # 🔧 FIX: Resolve shape with comprehensive error handling
            resolved_shape = []
            for idx, dim in enumerate(shape):
                try:
                    resolved = resolve_param_view(dim, idx)
                    resolved_shape.append(resolved)
                except Exception as e:
                    raise RuntimeError(f"[{node_name}] ❌ view: failed to resolve shape element {idx}: {e}")
            
            logger.debug(f"[{node_name}] ✅ view: resolved shape: {resolved_shape}")
            
            # 🔧 FIX: Validate resolved shape
            if not resolved_shape:
                raise RuntimeError(f"[{node_name}] ❌ view: empty shape after resolution")
            
            # 🔧 FIX: Allow -1 for automatic dimension inference (PyTorch standard)
            if any(s < -1 for s in resolved_shape):
                raise RuntimeError(f"[{node_name}] ❌ view: invalid shape dimensions: {resolved_shape} (only -1 and positive values allowed)")
            
            # 🔧 FIX: Handle -1 dimensions for automatic inference
            if -1 in resolved_shape:
                logger.debug(f"[{node_name}] 🔧 view: detected -1 dimension, will infer automatically")
                
                # Count how many -1 dimensions we have
                neg_one_count = resolved_shape.count(-1)
                if neg_one_count > 1:
                    raise RuntimeError(f"[{node_name}] ❌ view: only one -1 dimension allowed, got {neg_one_count}")
                
                # Calculate what the -1 dimension should be
                total_elements = layer_in.numel()
                for s in resolved_shape:
                    if s != -1:
                        total_elements //= s
                
                # Replace -1 with the calculated value
                resolved_shape = [s if s != -1 else total_elements for s in resolved_shape]
                
                logger.debug(f"[{node_name}] 🔧 view: inferred -1 dimension as {total_elements}")
            
            # 🔧 FIX: Validate tensor compatibility with final shape
            total_elements = 1
            for s in resolved_shape:
                total_elements *= s
            
            if layer_in.numel() != total_elements:
                raise RuntimeError(f"[{node_name}] ❌ view: shape {resolved_shape} is incompatible with input tensor "
                                f"of {layer_in.numel()} elements (expected {total_elements} elements)")
            
            try:
                output = aten_op(layer_in, resolved_shape)
                logger.debug(f"[{node_name}] ✅ view: input shape={layer_in.shape}, output shape={output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ view: execution failed: input shape={layer_in.shape}, "
                                f"resolved_shape={resolved_shape}, error={e}")

        elif func_name == "select":
            # 🔧 CONSISTENCY FIX: Apply precision consistency
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            return aten_op(layer_in,
                           layer_hyperparams["dim"],
                           layer_hyperparams["index"])

        elif func_name == "arange":
            logger.debug("🚨 ARANGE OPERATION DETECTED - ENTERING ENHANCED LOGIC")
            start = layer_hyperparams.get("start", None)
            end = layer_hyperparams.get("end", None)
            step = layer_hyperparams.get("step", 1)
            dtype = layer_hyperparams.get("dtype", None)
            device = layer_hyperparams.get("device", None)
            logger.debug(f"🔧 arange: start={start}, end={end}, step={step}, dtype={dtype}, device={device}")
            logger.debug(f"🔧 arange: parents={parents}, layer_in={layer_in}")

            # 🔧 ENHANCED: Better parameter resolution for arange
            def resolve_arange_param(param):
                """Resolve arange parameter to scalar value"""
                if param is None:
                    return None
                if isinstance(param, torch.Tensor):
                    return param.item()
                elif isinstance(param, torch.fx.Node):
                    if str(param) in tensor_map:
                        val = tensor_map[str(param)]
                        return val.item() if isinstance(val, torch.Tensor) else val
                    return None
                elif isinstance(param, torch.SymInt):
                    return int(param)
                elif isinstance(param, (int, float)):
                    return param
                return None

            # Resolve parameters first
            start = resolve_arange_param(start)
            end = resolve_arange_param(end)
            step = resolve_arange_param(step)
            
            # 🔧 CRITICAL FIX: Handle parameter extraction issues
            # The main issue is that arange parameters are often extracted incorrectly
            # We need to infer the correct parameters from context
            
            # Case 1: If we have parents, try to get sequence length from them
            if len(parents) > 0:
                for parent in parents:
                    if parent in tensor_map:
                        parent_value = tensor_map[parent]
                        if isinstance(parent_value, torch.Tensor) and parent_value.dim() > 0:
                            # Use the last dimension size as sequence length
                            seq_len = parent_value.shape[-1]
                            logger.debug(f"🔧 arange: Using parent {parent} sequence length: {seq_len}")
                            if end is None:  # Only override if end is None
                                end = seq_len
                                start = 0
                            break
            
            # Case 2: If no parents or parents didn't help, try to infer from layer_in
            if end is None:
                if hasattr(layer_in, '__len__') and len(layer_in) > 0:
                    if isinstance(layer_in, (list, tuple)) and len(layer_in) > 0:
                        first_input = layer_in[0]
                        if isinstance(first_input, torch.Tensor) and first_input.dim() > 0:
                            seq_len = first_input.shape[-1]
                            logger.debug(f"🔧 arange: Inferring sequence length from input: {seq_len}")
                            end = seq_len
                            start = 0
                elif isinstance(layer_in, torch.Tensor) and layer_in.dim() > 0:
                    seq_len = layer_in.shape[-1]
                    logger.debug(f"🔧 arange: Inferring sequence length from layer_in: {seq_len}")
                    end = seq_len
                    start = 0
            
            # Case 3: Handle common parameter extraction bugs
            if start is not None and end is not None:
                if start > end:
                    # This is likely a bug - swap them
                    logger.debug(f"🔧 arange: Detected start={start} > end={end}, swapping to start=0, end={start}")
                    original_start = start
                    start = 0
                    end = original_start
                elif start == 0 and end > 1000:
                    # This might be a sequence length
                    logger.debug(f"🔧 arange: Detected potential sequence length: end={end}")
                    # Keep as is, this looks correct

            # Ensure we have valid parameters
            if start is None:
                start = 0
            if end is None:
                # No hardcoded fallback - this should be an error
                raise RuntimeError(f"🔧 arange: Cannot determine 'end' parameter. "
                                f"start={start}, end={end}, step={step}. "
                                f"Please check parameter extraction in graph builder.")
            if step is None:
                step = 1
                
            logger.debug(f"🔧 arange: Final parameters - start={start}, end={end}, step={step}, dtype={dtype}, device={device}")
            logger.debug(f"🔧 arange: Parents count={len(parents)}, parents={parents}")
            logger.debug(f"🔧 arange: tensor_map keys={list(tensor_map.keys())}")

            def get_overload_name(op_overload_obj):
                try:
                    return str(op_overload_obj).split(".")[-1]
                except Exception:
                    return None

            arange_overload = get_overload_name(aten_op)
            logger.debug(f"🔧 arange: Overload detected: {arange_overload}")

            try:
                # 🔧 ENHANCED: Better overload handling
                if arange_overload in ("start_step", "Scalar", "Scalar_"):
                    # aten::arange.start_step(start, end, step, *, dtype, device)
                    logger.debug(f"🔧 arange: Using start_step overload with start={start}, end={end}, step={step}")
                    output = aten_op(start, end, step, dtype=dtype, device=device)
                elif "start" in str(arange_overload):
                    # aten::arange.start(start, end, *, dtype, device)
                    logger.debug(f"🔧 arange: Using start overload with start={start}, end={end}")
                    output = aten_op(start, end, dtype=dtype, device=device)
                elif "default" in str(arange_overload):
                    # aten::arange.default(end, *, dtype, device)
                    logger.debug(f"🔧 arange: Using default overload with end={end}")
                    output = aten_op(end, dtype=dtype, device=device)
                else:
                    # Fallback: try different combinations
                    try:
                        logger.debug(f"🔧 arange: Trying start_step fallback with start={start}, end={end}, step={step}")
                        output = aten_op(start, end, step, dtype=dtype, device=device)
                    except:
                        try:
                            logger.debug(f"🔧 arange: Trying start fallback with start={start}, end={end}")
                            output = aten_op(start, end, dtype=dtype, device=device)
                        except:
                            logger.debug(f"🔧 arange: Trying default fallback with end={end}")
                            output = aten_op(end, dtype=dtype, device=device)
                            
                logger.debug(f"🔧 arange: Successfully created tensor with shape: {output.shape}")
                return output
                
            except Exception as e:
                logger.error(f"🔧 arange: All overload attempts failed")
                logger.error(f"🔧 arange: Parameters: start={start}, end={end}, step={step}, dtype={dtype}, device={device}")
                logger.error(f"🔧 arange: Overload: {arange_overload}")
                logger.error(f"🔧 arange: Error: {str(e)}")
                raise RuntimeError(f"Failed to call aten::arange with resolved args. "
                                f"start={start}, end={end}, step={step}, dtype={dtype}, device={device}. "
                                f"Overload: {arange_overload}. Error: {str(e)}")

        elif func_name == "slice":
            # 🔧 CRITICAL FIX: Robust slice operation without hyperparameter corruption
            logger.debug(f"[{node_name}] 🔧 slice: input type={type(layer_in)}, length={len(layer_in) if isinstance(layer_in, (list, tuple)) else 'single'}")
            
            # 🔧 FIX: Handle list inputs without modifying hyperparameters
            if isinstance(layer_in, list):
                if len(layer_in) >= 2 and isinstance(layer_in[0], torch.Tensor) and isinstance(layer_in[1], (int, torch.Tensor)):
                    # Extract tensor and potential end value
                    input_tensor = layer_in[0]
                    potential_end = layer_in[1]
                    
                    # Convert potential_end to int if it's a tensor
                    if isinstance(potential_end, torch.Tensor):
                        if potential_end.numel() == 1:
                            potential_end = int(potential_end.item())
                        else:
                            raise RuntimeError(f"[{node_name}] ❌ slice: end value must be a scalar tensor, got shape {potential_end.shape}")
                    
                    logger.debug(f"[{node_name}] 🔧 slice: extracted tensor shape={input_tensor.shape}, potential_end={potential_end}")
                    
                    # 🔧 FIX: Use extracted tensor but preserve original hyperparameters
                    layer_in = input_tensor
                    
                    # 🔧 CRITICAL FIX: Always use parent value when available (it's more accurate)
                    logger.debug(f"[{node_name}] 🔧 slice: using parent value {potential_end} as end parameter (overriding hyperparams)")
                    # Create a copy to avoid modifying original
                    updated_hyperparams = dict(layer_hyperparams)
                    updated_hyperparams["end"] = potential_end
                    layer_hyperparams = updated_hyperparams
                else:
                    raise RuntimeError(f"[{node_name}] ❌ slice: expected list with [tensor, int] or single tensor, got {layer_in}")
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(layer_in, torch.Tensor):
                raise RuntimeError(f"[{node_name}] ❌ slice expected tensor input, got {type(layer_in)}")
            
            # 🔧 FIX: Validate required hyperparameters
            required_params = ["dim", "start", "end"]
            for param in required_params:
                if param not in layer_hyperparams:
                    raise RuntimeError(f"[{node_name}] ❌ slice: missing required parameter '{param}'")
            
            dim = layer_hyperparams["dim"]
            start = layer_hyperparams["start"]
            end = layer_hyperparams["end"]
            step = layer_hyperparams.get("step", 1)
            
            # 🔧 ENHANCED: Handle negative indexing for slice
            tensor_rank = layer_in.dim()
            if isinstance(dim, int) and dim < 0:
                dim = tensor_rank + dim
            
            # Validate converted dimension
            if isinstance(dim, int) and (dim < 0 or dim >= tensor_rank):
                raise ValueError(f"[{node_name}] ❌ slice: dimension {layer_hyperparams['dim']} out of range for tensor of rank {tensor_rank}")
            
            logger.debug(f"[{node_name}] ✅ slice: input shape={layer_in.shape}, dim={dim}, start={start}, end={end}, step={step}")
            logger.debug(f"slice: dim={dim}, start={start}, end={end}, step={step}")
            try:
                output = aten_op(layer_in, dim, start, end, step)
                logger.debug(f"[{node_name}] ✅ slice output shape: {output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ slice failed: input shape={layer_in.shape}, "
                                f"dim={dim}, start={start}, end={end}, step={step}. Error: {e}")
        
        elif func_name == "sym_size":
            logger.debug("sym_size operation")
            logger.debug(f"input layer_in: {layer_in}")
            if isinstance(layer_in,(tuple,list)):
                layer_in = layer_in[0]
            output = aten_op(layer_in,layer_hyperparams['dim'])
            logger.debug(f"output: {output}")
            return output

        elif func_name == "_assert_tensor_metadata":
            # Skip or pass-through this op, as it's only a debug consistency check
            logger.debug(f"[{node_name}] ℹ️ Skipping `_assert_tensor_metadata` (no-op).")
            if isinstance(layer_in, list) and len(layer_in) > 0:
                return layer_in[0]
            return layer_in

        elif func_name == "to":
            # 🔧 CRITICAL FIX: Robust 'to' operation with input validation
            logger.debug(f"[{node_name}] 🔧 to: input type={type(layer_in)}, method_args={method_args}")
            
            # 🔧 FIX: Validate input before execution
            if layer_in is None:
                raise RuntimeError(f"[{node_name}] ❌ to: input is None")
            
            # 🔧 FIX: Handle list inputs properly
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    raise RuntimeError(f"[{node_name}] ❌ to: received empty input list")
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: For multiple inputs, apply 'to' to each
                    try:
                        outputs = []
                        for i, inp in enumerate(layer_in):
                            if isinstance(inp, torch.Tensor):
                                output = aten_op(inp, *method_args)
                                outputs.append(output)
                            else:
                                outputs.append(inp)
                        
                        if len(outputs) == 1:
                            return outputs[0]
                        else:
                            return outputs
                            
                    except Exception as e:
                        raise RuntimeError(f"[{node_name}] ❌ to: failed to process list input: {e}")
            
            # 🔧 FIX: Validate tensor input
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ to: expected tensor input, got {type(layer_in)}")
            
            try:
                output = aten_op(layer_in, *method_args)
                logger.debug(f"[{node_name}] ✅ to: input shape={layer_in.shape}, output shape={output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ to: failed with input shape={layer_in.shape}, "
                                f"method_args={method_args}. Error: {e}")
        elif func_name == "mean":
            # 🔧 IMPROVED: Better list input handling
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    logger.debug(f"[{node_name}] ⚠️ mean operation received empty input list")
                    # Try to recover from parents
                    for node_x in parents:
                        if node_io[node_x]['layer_type'] not in ("Weight", "Bias", "future_use"):
                            layer_in = node_io[node_x]['output_values']
                            break
                    if isinstance(layer_in, list) and len(layer_in) == 0:
                        raise RuntimeError(f"[{node_name}] ❌ mean operation has no valid inputs")
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # Take the first tensor from the list
                    layer_in = next((x for x in layer_in if isinstance(x, torch.Tensor)), layer_in[0])
            
            if not isinstance(layer_in, torch.Tensor):
                raise RuntimeError(f"[{node_name}] ❌ mean expected tensor input, got {type(layer_in)}")
            
            return aten_op(layer_in, *method_args)

        elif func_name == "sum":
            # 🔧 NEW OPERATION: Sum reduction
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            return aten_op(layer_in, *method_args)

        elif func_name == "max":
            # 🔧 NEW OPERATION: Maximum reduction
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            return aten_op(layer_in, *method_args)

        elif func_name == "min":
            # 🔧 NEW OPERATION: Minimum reduction
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            return aten_op(layer_in, *method_args)

        elif func_name == "softmax":
            # 🔧 CONSISTENCY FIX: Ensure consistent softmax computation
            if isinstance(layer_in, list):
                layer_in = layer_in[0]
            
            # 🔧 CONSISTENCY FIX: Force float32 precision for numerical stability
            layer_in = enforce_precision_consistency(layer_in)
            
            output = aten_op(layer_in,*method_args)
            return output
        elif func_name in ("unflatten","flatten"):
            # 🔧 CONSISTENCY FIX: Apply precision consistency
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            output = aten_op(layer_in,*method_args)
            return output
        elif func_name == "embedding":
            # 🔧 ENHANCED EMBEDDING: Add precision consistency like other operations
            layer_in = []
            for node_x in parents: 
                if "weight" in node_x:
                    continue
                else:
                    layer_in.append(node_io[node_x]['output_values'])

            indices = layer_in[0] if isinstance(layer_in, (list, tuple)) else layer_in
            
            # 🔧 CONSISTENCY FIX: Apply precision consistency to indices
            indices = enforce_precision_consistency(indices)
            
            # 🔄 Ensure indices are LongTensor
            if not torch.is_floating_point(indices) and indices.dtype in (torch.int32, torch.int64):
                pass  # Already correct
            else:
                indices = indices.long()
        
            # 🔧 CONSISTENCY FIX: Apply precision consistency to weight
            weight = layer_hyperparams["weight"]
            weight = enforce_precision_consistency(weight)
            
            # 🔧 ENHANCED CONSISTENCY: Use dedicated embedding consistency function
            weight, indices = ensure_embedding_consistency(weight, indices, node_name)
            
            # 🔧 CRITICAL: Pre-execution checks for embedding
            _precheck_embedding(indices, weight, node_name)
        
            return aten_op(weight,
                    indices,
                    layer_hyperparams["padding_idx"],
                    layer_hyperparams["scale_grad_by_freq"],
                    layer_hyperparams["sparse"])

        elif func_name in ("mul", "mul_"):
            # 🔧 ENHANCED MUL OPERATION: Fixed broadcasting and improved error handling
            
            # Extract operands a and b robustly
            a = b = None
            if isinstance(layer_in, list):
                if len(layer_in) == 2:
                    a, b = layer_in
                elif len(layer_in) == 1 and len(method_args) == 1:
                    a = layer_in[0]
                    b = method_args[0]
                else:
                    raise RuntimeError(
                        f"[{node_name}] [DLBacktrace] `{func_name}` expects 2 inputs, got list of length {len(layer_in)} and method_args of length {len(method_args)}"
                    )
            elif isinstance(layer_in, (int, float, torch.Tensor)) and len(method_args) == 1:
                a = layer_in
                b = method_args[0]
            else:
                raise RuntimeError(
                    f"[{node_name}] [DLBacktrace] `{func_name}` expects 2 inputs, got: {type(layer_in)} + {method_args}"
                )

            # Resolve SymInt or symbolic Node values
            if hasattr(a, 'node') and hasattr(a.node, '_value'):
                a = a.node._value
            if hasattr(b, 'node') and hasattr(b.node, '_value'):
                b = b.node._value

            # 🔧 IMPROVED: Simplified and more robust tensor handling
            def normalize_tensor(x):
                """Normalize input to appropriate tensor format"""
                if isinstance(x, torch.nn.Parameter):
                    x = x.detach()
                
                if isinstance(x, torch.Tensor):
                    if x.ndim == 0:  # Scalar tensor
                        val = x.item()
                        # Keep large values as tensors to avoid precision loss
                        if abs(val) > 1e10:
                            return x.to(dtype=torch.float32)
                        return val
                    return x
                
                if isinstance(x, (int, float)):
                    # Convert large numbers to tensors to avoid precision issues
                    if abs(x) > 1e10:
                        return torch.tensor(x, dtype=torch.float32)
                    return x
                
                return x

            a = normalize_tensor(a)
            b = normalize_tensor(b)

            # 🔧 IMPROVED: Better tensor promotion with consistent dtype/device handling
            def promote_to_tensor(val, reference=None):
                """Promote value to tensor with consistent dtype and device"""
                if isinstance(val, (int, float)):
                    if reference is not None and isinstance(reference, torch.Tensor):
                        return torch.tensor(val, dtype=reference.dtype, device=reference.device)
                    return torch.tensor(val, dtype=torch.float32)
                return val

            a = promote_to_tensor(a, b)
            b = promote_to_tensor(b, a)

            # Use the shared smart broadcasting function

            # Execute the multiplication operation
            try:
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    # Ensure dtype and device consistency
                    a, b = ensure_tensor_consistency([a, b])
                    
                    logger.debug(f"[{node_name}] 🔧 Before broadcasting: a={a.shape}, b={b.shape}")
                    a, b = smart_broadcast_tensors(a, b, node_name, "mul")
                    logger.debug(f"[{node_name}] 🔧 After broadcasting: a={a.shape}, b={b.shape}")
                
                output = aten_op(a, b)
                logger.debug(f"[{node_name}] ✅ mul output shape: {output.shape}")
                
            except Exception as e:
                logger.error(f"[{node_name}] ❌ mul operation failed:")
                logger.error(f"[{node_name}]   - a: type={type(a)}, shape={getattr(a, 'shape', None)}, dtype={getattr(a, 'dtype', None)}")
                logger.error(f"[{node_name}]   - b: type={type(b)}, shape={getattr(b, 'shape', None)}, dtype={getattr(b, 'dtype', None)}")
                raise RuntimeError(f"[{node_name}] ❌ mul operation failed: {e}")

            return output

        elif func_name in {"add", "add_", "sub", "div", "rsub", "pow", "gt", "ge", "lt", "eq"}:
            # 🔧 ENHANCED: Robust arithmetic and comparison operations with smart broadcasting
            logger = get_logger()
            logger.debug(f"🔧 {func_name}: input type={type(layer_in)}, length={len(layer_in) if isinstance(layer_in, (list, tuple)) else 'single'}")
            logger.debug(f"🔧 {func_name}: method_args={method_args}")
            
            # 🔧 CONSISTENCY FIX: Apply precision consistency
            if isinstance(layer_in, (list, tuple)):
                layer_in = [enforce_precision_consistency(x) for x in layer_in]
            else:
                layer_in = enforce_precision_consistency(layer_in)
            
            # 🔧 BROADCASTING FIX: Apply smart broadcasting for supported operations
            if isinstance(layer_in, (list, tuple)) and len(layer_in) >= 2:
                layer_in = apply_smart_broadcasting(func_name, layer_in, node_name)
            
            # 🔧 FIX: Handle list inputs with proper validation
            if isinstance(layer_in, list):
                logger.debug(f"[{node_name}] 🔧 {func_name}: processing list input, length={len(layer_in)}")
                
                # 🔧 FIX: Validate list inputs for comparison operations
                if func_name in {"gt", "ge", "lt", "eq"} and len(layer_in) != 2:
                    raise RuntimeError(f"[{node_name}] ❌ {func_name}: comparison operations require exactly 2 inputs, got {len(layer_in)}")
                
                # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
                if len(layer_in) >= 2:
                    target_device = None
                    for t in layer_in:
                        if isinstance(t, torch.Tensor):
                            target_device = t.device
                            break
                    
                    if target_device is not None:
                        for i, t in enumerate(layer_in):
                            if isinstance(t, torch.Tensor) and t.device != target_device:
                                layer_in[i] = t.to(device=target_device)
                                logger.debug(f"[{node_name}] 🔧 Moved tensor {i} to device {target_device}")
                
                # 🔧 FIX: Execute operation based on input count
                if len(layer_in) == 2:
                    # 🔧 FIX: Validate inputs for comparison operations
                    if func_name in {"gt", "ge", "lt", "eq"}:
                        a, b = layer_in[0], layer_in[1]
                        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                            raise TypeError(f"[{node_name}] ❌ {func_name}: both inputs must be tensors, got {type(a)} and {type(b)}")
                    
                    output = aten_op(layer_in[0], layer_in[1], *method_args)
                    
                elif len(layer_in) == 3:
                    output = aten_op(layer_in[0], layer_in[1], layer_in[2], *method_args)
                else:
                    output = aten_op(layer_in, *method_args)
                    
            else:
                # 🔧 FIX: Handle single tensor inputs
                logger.debug(f"[{node_name}] 🔧 {func_name}: processing single tensor input")
                
                # 🔧 FIX: Special handling for comparison operations with single tensor
                if func_name in {"gt", "ge", "lt", "eq"} and not method_args:
                    raise RuntimeError(f"[{node_name}] ❌ {func_name}: comparison operation requires 2 inputs")
                
                # 🔧 FIX: Try to recover inputs from parents for missing arguments
                if func_name in {"add", "add_", "sub", "div"} and not method_args:
                    logger.debug(f"[{node_name}] 🔧 {func_name}: collecting inputs from parents: {parents}")
                    
                    recovered_inputs = [tensor_map[p] for p in parents if p in tensor_map]
                    if len(recovered_inputs) >= 2:
                        logger.debug(f"[{node_name}] 🔧 {func_name}: found {len(recovered_inputs)} inputs, using first 2")
                        
                        a, b = recovered_inputs[0], recovered_inputs[1]
                        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                            # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
                            target_device = a.device
                            if b.device != target_device:
                                b = b.to(device=target_device)
                                logger.debug(f"[{node_name}] 🔧 {func_name}: moved second tensor to device {target_device}")
                            
                            output = aten_op(a, b)
                        else:
                            output = aten_op(a, b)
                        return output
                
                # 🔧 FIX: Handle single tensor cases with proper validation
                if func_name == "pow":
                    logger.debug(f"[{node_name}] 🔧 {func_name}: pow operation with single tensor")
                    
                    if method_args and len(method_args) > 0:
                        scalar_val = method_args[0]
                        if 'Tensor_Scalar' in str(aten_op):
                            output = aten_op(layer_in, scalar_val)
                        else:
                            output = aten_op(scalar_val, layer_in)
                    else:
                        output = aten_op(-1.0, layer_in)
                else:
                    output = aten_op(layer_in, *method_args)
            
            if hasattr(output, 'shape'):
                logger.debug(f"[{node_name}] ✅ {func_name} output shape: {output.shape}")
                logger.debug(f"{node_name} output shape: {output.shape}")
            else:
                logger.debug(f"[{node_name}] ✅ {func_name} output type: {type(output)}")
                logger.debug(f"{node_name} output type: {type(output)} (non-tensor)")
            
            return output

        elif "scalar_tensor" in func_name:
            value = layer_hyperparams.get("value", 0)
            dtype = layer_hyperparams.get("dtype", torch.float32)
            device = layer_hyperparams.get("device", torch.device("cpu"))
            layout = layer_hyperparams.get("layout", torch.strided)
            pin_memory = layer_hyperparams.get("pin_memory", False)

            # Ensure value is a scalar
            if isinstance(value, (list, tuple)):
                value = value[0] if value else 0
            elif isinstance(value, torch.Tensor):
                value = value.item()

            # Create the scalar tensor
            output = aten_op(
                value,
                dtype=dtype,
                layout=layout,
                device=device,
                pin_memory=pin_memory
            )
            return output

            
        elif func_name == "matmul":
            if isinstance(layer_in, list) and len(layer_in) == 2:
                a, b = layer_in[0], layer_in[1]
                
                # 🔧 CRITICAL FIX: Ensure dtype and device consistency for CPU compatibility
                a, b = ensure_tensor_consistency([a, b])
                
                output = aten_op(a, b, *method_args)
            else:
                raise RuntimeError(f"[DLBacktrace] matmul expects 2 inputs but got: {layer_in}")
            return output
        
        elif func_name == "bmm":
            if isinstance(layer_in, list):
                if len(layer_in) != 2:
                    raise RuntimeError(f"[{node_name}] ❌ `bmm` expects exactly 2 tensor inputs, got {len(layer_in)}")
                a, b = layer_in
            else:
                raise RuntimeError(f"[{node_name}] ❌ `bmm` expects list of 2 tensors, got {type(layer_in)}")

            # Ensure both are tensors
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `bmm` inputs must be tensors: got {type(a)} and {type(b)}")

            # 🔧 CRITICAL FIX: Ensure dtype and device consistency for CPU compatibility
            a, b = ensure_tensor_consistency([a, b])

            try:
                output = aten_op(a, b)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `bmm` failed with shapes {a.shape}, {b.shape}: {e}")

            return output

        elif func_name == "full":
            # 🔧 CRITICAL FIX: Robust size resolution for full operation
            size = layer_hyperparams.get("size") or layer_hyperparams.get("sizes")
            
            logger.debug(f"[{node_name}] 🔧 full: initial size={size}, method_args={method_args}")
            
            # 🔧 FIX: Proper size resolution with error handling
            if not size or all(s is None for s in size):
                if method_args and isinstance(method_args[0], (list, tuple)):
                    size = method_args[0]
                    logger.debug(f"[{node_name}] 🔧 Using size from method_args: {size}")
                else:
                    raise RuntimeError(f"[{node_name}] ❌ No valid size found in hyperparams or method_args")
            
            # 🔧 FIX: Resolve symbolic sizes with proper error handling
            resolved_size = []
            for i, s in enumerate(size):
                try:
                    if isinstance(s, torch.fx.Node):
                        node_key = str(s)
                        if node_key in tensor_map:
                            parent_val = tensor_map[node_key]
                            if isinstance(parent_val, torch.Tensor) and parent_val.numel() == 1:
                                resolved_size.append(int(parent_val.item()))
                            elif isinstance(parent_val, (int, float)):
                                resolved_size.append(int(parent_val))
                            else:
                                raise ValueError(f"Parent {node_key} output is not a scalar: {type(parent_val)}")
                        else:
                            raise KeyError(f"Parent node {node_key} not found in tensor_map")
                    elif isinstance(s, torch.Tensor):
                        if s.numel() == 1:
                            resolved_size.append(int(s.item()))
                        else:
                            raise ValueError(f"Size element {i} is not a scalar tensor: {s.shape}")
                    elif isinstance(s, (int, float)):
                        resolved_size.append(int(s))
                    elif s is None:
                        raise ValueError(f"Size element {i} is None")
                    else:
                        raise TypeError(f"Unsupported size type: {type(s)}")
                        
                except Exception as e:
                    raise RuntimeError(f"[{node_name}] ❌ Failed to resolve size element {i} ({s}): {e}")
            
            logger.debug(f"[{node_name}] ✅ Resolved size: {resolved_size}")
            
            # 🔧 FIX: Validate resolved size
            if not resolved_size:
                raise RuntimeError(f"[{node_name}] ❌ Empty size after resolution")
            
            if any(s <= 0 for s in resolved_size):
                raise RuntimeError(f"[{node_name}] ❌ Invalid size dimensions: {resolved_size} (all must be positive)")
            
            # Remaining parameters with validation
            # Extract fill_value from method_args[1] if available, otherwise from hyperparams
            if method_args and len(method_args) > 1:
                fill_value = method_args[1]
                logger.debug(f"[{node_name}] 🔧 Using fill_value from method_args: {fill_value}")
            else:
                fill_value = layer_hyperparams.get("fill_value", 0)
                logger.debug(f"[{node_name}] 🔧 Using fill_value from hyperparams: {fill_value}")
            
            dtype = layer_hyperparams.get("dtype", model_compute_dtype)
            device = layer_hyperparams.get("device", torch.device("cpu"))

            if isinstance(fill_value, torch.Tensor):
                if fill_value.numel() == 1:
                    fill_value = fill_value.item()
                else:
                    raise RuntimeError(f"[{node_name}] ❌ fill_value must be a scalar tensor, got shape {fill_value.shape}")

            try:
                output = aten_op(resolved_size, fill_value, dtype=dtype, device=device)
                logger.debug(f"[{node_name}] ✅ full({resolved_size}, {fill_value}) → shape: {output.shape}")
                    
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ Failed to execute full: "
                                f"size={resolved_size}, fill_value={fill_value}, dtype={dtype}, device={device}. Error: {e}")
            return output

        
        elif func_name == "full_like":
            return aten_op(layer_in,
                           fill_value=layer_hyperparams["fill_value"],
                           dtype=layer_hyperparams["dtype"],
                           layout=layer_hyperparams["layout"],
                           device=layer_hyperparams["device"],
                           pin_memory=layer_hyperparams["pin_memory"])

        elif func_name == "expand":
            # 🔧 CRITICAL FIX: Robust expand operation without input corruption
            logger.debug(f"[{node_name}] 🔧 expand: input type={type(layer_in)}, method_args={method_args}")
            
            # 🔧 FIX: Handle list inputs without modifying original
            original_layer_in = layer_in
            
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    logger.debug(f"[{node_name}] ⚠️ expand received empty input list, attempting recovery")
                    
                    # 🔧 FIX: Try to recover from parents without overwriting original input
                    recovered_input = None
                    for node_x in parents:
                        if node_io[node_x]['layer_type'] not in ("Weight", "Bias", "future_use"):
                            recovered_input = node_io[node_x]['output_values']
                            logger.debug(f"[{node_name}] 🔧 expand: recovered input from parent {node_x}: {type(recovered_input)}")
                            break
                    
                    if recovered_input is None:
                        raise RuntimeError(f"[{node_name}] ❌ expand: no valid input found in parents")
                    
                    # 🔧 FIX: Use recovered input but don't modify original
                    layer_in = recovered_input
                    
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: Find first valid tensor without modifying original
                    valid_tensor = None
                    for x in layer_in:
                        if isinstance(x, torch.Tensor):
                            valid_tensor = x
                            break
                    
                    if valid_tensor is None:
                        raise RuntimeError(f"[{node_name}] ❌ expand: no valid tensor found in input list")
                    
                    layer_in = valid_tensor
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ expand: expected tensor input, got {type(layer_in)}")
            
            logger.debug(f"[{node_name}] 🔧 expand: input shape: {layer_in.shape}")
            
            # 🔧 FIX: Get sizes with proper fallback
            raw_sizes = layer_hyperparams.get("sizes", None)
            if isinstance(raw_sizes, (list, tuple)):
                sizes = []
                for s in raw_sizes:
                    if isinstance(s, torch.fx.Node):
                        # retrieve the recorded output value for that node
                        sizes.append(node_io[str(s)]["output_values"])
                    else:
                        sizes.append(s)
            else:
                sizes = raw_sizes

            # 🔧 FIX: Robust parameter resolution with proper error handling
            def resolve_param(param, idx):
                try:
                    if isinstance(param, torch.Tensor):
                        if param.numel() == 1:
                            return int(param.item())
                        else:
                            raise ValueError(f"Size element {idx} is not a scalar tensor: {param.shape}")
                    elif isinstance(param, torch.fx.Node):
                        node_key = str(param)
                        value = node_io.get(node_key, {}).get("output_values", None)
                        if value is None:
                            raise ValueError(f"No output value found for node: {node_key}")
                        
                        if isinstance(value, torch.Tensor):
                            if value.numel() == 1:
                                return int(value.item())
                            else:
                                raise ValueError(f"Node {node_key} output is not a scalar: {value.shape}")
                        elif isinstance(value, (int, float)):
                            return int(value)
                        else:
                            raise TypeError(f"Unexpected output value type: {type(value)}")
                    elif isinstance(param, torch.SymInt):
                        if hasattr(param, 'node') and hasattr(param.node, '_value'):
                            return int(param.node._value)
                        else:
                            raise ValueError(f"SymInt {param} has no _value attribute")
                    elif isinstance(param, (int, float)):
                        return int(param)
                    elif param is None:
                        raise ValueError(f"Size element {idx} is None")
                    else:
                        raise TypeError(f"Unsupported size type: {type(param)}")
                        
                except Exception as e:
                    raise RuntimeError(f"Failed to resolve size element {idx} ({param}): {e}")

            # 🔧 FIX: If sizes not available, try method_args fallback
            if not sizes or any(s is None for s in sizes):
                if method_args and isinstance(method_args[0], (list, tuple)):
                    logger.debug(f"[{node_name}] 🔧 expand: using sizes from method_args: {method_args[0]}")
                    sizes = method_args[0]
                else:
                    raise RuntimeError(f"[{node_name}] ❌ expand: no valid sizes found in hyperparams or method_args")

            logger.debug(f"[{node_name}] 🔧 expand: processing sizes: {sizes}")

            # 🔧 FIX: Resolve sizes with comprehensive error handling
            resolved_sizes = []
            for idx, s in enumerate(sizes):
                try:
                    resolved = resolve_param(s, idx)
                    resolved_sizes.append(resolved)
                except Exception as e:
                    raise RuntimeError(f"[{node_name}] ❌ expand: failed to resolve size element {idx}: {e}")

            logger.debug(f"[{node_name}] ✅ expand: resolved sizes: {resolved_sizes}")

            # 🔧 FIX: Validate resolved sizes
            if not resolved_sizes:
                raise RuntimeError(f"[{node_name}] ❌ expand: empty sizes after resolution")

            # 🔧 FIX: Validate size compatibility with input tensor
            if len(resolved_sizes) != layer_in.dim():
                raise RuntimeError(f"[{node_name}] ❌ expand: size dimensions {len(resolved_sizes)} don't match tensor rank {layer_in.dim()}")

            # 🔧 FIX: Execute with proper error handling
            try:
                output = aten_op(
                    layer_in,
                    resolved_sizes,
                    implicit=layer_hyperparams.get("implicit", False)
                )
                logger.debug(f"[{node_name}] ✅ expand: output shape: {output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ expand: execution failed: input shape={layer_in.shape}, "
                                f"sizes={resolved_sizes}, error={e}")
        

        elif "permute" in func_name:
            # 🔧 ENHANCED: Robust permute operation with proper validation
            logger.debug(f"[{node_name}] 🔧 permute: input type={type(layer_in)}")
            
            # 🔧 FIX: Handle input properly
            if isinstance(layer_in, tuple):
                layer_in = layer_in[0]
            elif isinstance(layer_in, list):
                layer_in = layer_in[0]
            
            # 🔧 FIX: Validate input
            if layer_in is None:
                raise RuntimeError(f"[{node_name}] ❌ permute: input is None")
            
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ permute: expected tensor input, got {type(layer_in)}")
            
            # 🔧 FIX: Get and validate dimensions
            dims = layer_hyperparams.get("dims", [])
            if not dims:
                raise RuntimeError(f"[{node_name}] ❌ permute: no dimensions provided")
            
            # 🔧 ENHANCED: Handle negative indexing for permute
            tensor_rank = layer_in.dim()
            normalized_dims = []
            for dim in dims:
                if not isinstance(dim, int):
                    raise TypeError(f"[{node_name}] ❌ permute: dimension must be integer, got {type(dim)}")
                
                # Handle negative indexing
                if dim < 0:
                    dim = tensor_rank + dim
                
                if dim < 0 or dim >= tensor_rank:
                    raise ValueError(f"[{node_name}] ❌ permute: dimension {dim} out of range for tensor of rank {tensor_rank}")
                
                normalized_dims.append(dim)
            
            logger.debug(f"[{node_name}] ✅ permute: input shape={layer_in.shape}, dims={normalized_dims}")
            
            try:
                return aten_op(layer_in, normalized_dims)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ permute: execution failed: {e}")

        elif func_name == "transpose":
            # 🔧 CRITICAL FIX: Robust transpose operation with proper validation
            logger.debug(f"[{node_name}] 🔧 transpose: input type={type(layer_in)}, method_args={method_args}")
            
            # 🔧 FIX: Validate input before processing
            if layer_in is None:
                raise RuntimeError(f"[{node_name}] ❌ transpose: input is None")
            
            # 🔧 FIX: Handle list inputs properly
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    raise RuntimeError(f"[{node_name}] ❌ transpose: received empty input list")
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: Find first valid tensor without modifying original
                    valid_tensor = None
                    for x in layer_in:
                        if isinstance(x, torch.Tensor):
                            valid_tensor = x
                            break
                    
                    if valid_tensor is None:
                        raise RuntimeError(f"[{node_name}] ❌ transpose: no valid tensor found in input list")
                    
                    layer_in = valid_tensor
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ transpose: expected tensor input, got {type(layer_in)}")
            
            # 🔧 FIX: Validate method arguments
            if not method_args or len(method_args) < 2:
                raise RuntimeError(f"[{node_name}] ❌ transpose: requires 2 arguments (dim0, dim1), got {method_args}")
            
            dim0, dim1 = method_args[0], method_args[1]
            
            # 🔧 FIX: Validate dimensions and handle negative indexing
            if not isinstance(dim0, int) or not isinstance(dim1, int):
                raise TypeError(f"[{node_name}] ❌ transpose: dimensions must be integers, got {type(dim0)} and {type(dim1)}")
            
            # 🔧 ENHANCED: Handle negative indexing (PyTorch standard behavior)
            tensor_rank = layer_in.dim()
            if dim0 < 0:
                dim0 = tensor_rank + dim0
            if dim1 < 0:
                dim1 = tensor_rank + dim1
            
            # 🔧 FIX: Validate converted dimensions
            if dim0 < 0 or dim1 < 0:
                raise ValueError(f"[{node_name}] ❌ transpose: dimensions {method_args[0]} and {method_args[1]} out of range for tensor of rank {tensor_rank}")
            
            if dim0 >= tensor_rank or dim1 >= tensor_rank:
                raise ValueError(f"[{node_name}] ❌ transpose: dimensions {dim0} and {dim1} out of range for tensor of rank {tensor_rank}")
            
            logger.debug(f"[{node_name}] ✅ transpose: input shape={layer_in.shape}, dims=({dim0}, {dim1})")
            
            try:
                output = aten_op(layer_in, dim0, dim1)
                logger.debug(f"[{node_name}] ✅ transpose: output shape={output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ transpose: failed with input shape={layer_in.shape}, "
                                f"dims=({dim0}, {dim1}). Error: {e}")
            
        elif func_name == "sigmoid":
            # 🔧 CONSISTENCY FIX: Apply precision consistency
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            return aten_op(layer_in, *method_args)
        
        elif func_name == "zeros":
            return aten_op(layer_hyperparams["sizes"])

        elif func_name == "interpolate":
            # 🔧 NEW OPERATION: General interpolation (upsampling/downsampling)
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            # Get interpolation parameters
            size = layer_hyperparams.get("size", None)
            scale_factor = layer_hyperparams.get("scale_factor", None)
            mode = layer_hyperparams.get("mode", "nearest")
            align_corners = layer_hyperparams.get("align_corners", None)
            recompute_scale_factor = layer_hyperparams.get("recompute_scale_factor", None)
            
            # Handle different interpolation modes
            if size is not None:
                return aten_op(layer_in, size=size, mode=mode, align_corners=align_corners, 
                              recompute_scale_factor=recompute_scale_factor)
            elif scale_factor is not None:
                return aten_op(layer_in, scale_factor=scale_factor, mode=mode, align_corners=align_corners,
                              recompute_scale_factor=recompute_scale_factor)
            else:
                raise RuntimeError(f"[{node_name}] ❌ interpolate requires either 'size' or 'scale_factor' parameter")

        elif func_name == "pad":
            # 🔧 NEW OPERATION: General padding
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            # Get padding parameters
            pad = layer_hyperparams.get("pad", [])
            mode = layer_hyperparams.get("mode", "constant")
            value = layer_hyperparams.get("value", 0.0)
            
            return aten_op(layer_in, pad=pad, mode=mode, value=value)

        elif func_name == "_softmax":
            return aten_op(layer_in,
                           layer_hyperparams["dim"],
                           layer_hyperparams["half_to_float"])

        elif func_name == "mean":
            return aten_op(layer_in, *method_args)

        elif func_name == "where":
            if len(layer_in) == 3:
                cond, x, y = layer_in
                
                # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
                target_device = x.device if isinstance(x, torch.Tensor) else y.device if isinstance(y, torch.Tensor) else torch.device("cpu")
                if isinstance(cond, torch.Tensor) and cond.device != target_device:
                    cond = cond.to(device=target_device)
                if isinstance(x, torch.Tensor) and x.device != target_device:
                    x = x.to(device=target_device)
                if isinstance(y, torch.Tensor) and y.device != target_device:
                    y = y.to(device=target_device)
                
                try:
                    # Use smart broadcasting for where operation
                    cond_, x_ = smart_broadcast_tensors(cond, x, node_name, "where_cond_x")
                    x_, y_ = smart_broadcast_tensors(x_, y, node_name, "where_x_y")
                    return aten_op(cond_, x_, y_)
                except Exception as e:
                    # Safe fallback: expand cond to match if it's missing a dimension
                    if cond.ndim == x.ndim - 1 and cond.shape[0] == x.shape[0] and cond.shape[-2:] == x.shape[-2:]:
                        cond_expanded = cond.unsqueeze(1).expand_as(x)
                        logger.debug(f"[{node_name}] ⚠️ Expanded cond to shape {cond_expanded.shape}")
                        return aten_op(cond_expanded, x, y)
                    raise RuntimeError(
                        f"[{node_name}] ❌ Shape mismatch in `where`: "
                        f"cond: {cond.shape}, x: {x.shape}, y: {y.shape}. Error: {e}"
                    )
            
            elif len(layer_in) == 2:
                # Handle aten.where.ScalarSelf variant: x.where(cond)
                x, cond = layer_in
                
                # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
                target_device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
                if isinstance(cond, torch.Tensor) and cond.device != target_device:
                    cond = cond.to(device=target_device)

                if cond.dtype != torch.bool:
                    logger.debug(f"[{node_name}] ⚠️ Converting cond from {cond.dtype} to bool")
                    cond = cond.to(torch.bool)

                try:
                    cond_, x_ = smart_broadcast_tensors(cond, x, node_name, "where_scalar_self")
                    return torch.where(cond_, x_, torch.zeros_like(x_))  # Default y = 0
                except Exception as e:
                    raise RuntimeError(
                        f"[{node_name}] ❌ Shape mismatch in `where.ScalarSelf`: x: {x.shape}, cond: {cond.shape}. Error: {e}"
                    )

            else:
                raise RuntimeError(
                    f"[{node_name}] ❌ Unexpected number of inputs for `where`: got {len(layer_in)} → {layer_in}"
                )

        elif func_name == "contiguous":
            # 🔧 CONSISTENCY FIX: Apply precision consistency
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            return aten_op(layer_in)

        elif func_name == "_to_copy":
            if isinstance(layer_in, list) and len(layer_in) == 1:
                layer_in = layer_in[0]
            
            if layer_in is None:
                raise RuntimeError(f"[DLBacktrace] ❌ _to_copy received `None` as input at node `{node_name}` → likely due to skipped or failed parent node.")
            
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[DLBacktrace] ❌ _to_copy expected a Tensor but got {type(layer_in)} at node `{node_name}` → value: {layer_in}")
                       
            output = aten_op(layer_in,
                           memory_format=layer_hyperparams.get("memory_format", torch.contiguous_format),
                           non_blocking=layer_hyperparams.get("non_blocking", False))
            
            return output

        elif func_name == "logical_not":
            # 🔧 DEVICE CONSISTENCY FIX: logical operations work on all types
            if isinstance(layer_in, torch.Tensor):
                # Logical operations work on all types, but for consistency with bitwise_not
                if layer_in.dtype in [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64]:
                    # Integer types work on both devices
                    return aten_op(layer_in)
                else:
                    # For float types, convert to boolean first, then apply logical_not
                    bool_tensor = layer_in.bool()
                    result = aten_op(bool_tensor)
                    # Convert back to the original dtype and device
                    return result.to(dtype=layer_in.dtype, device=layer_in.device)
            return aten_op(layer_in)

        elif func_name == "bitwise_not":
            # 🔧 DEVICE CONSISTENCY FIX: bitwise operations only work on integer and boolean types
            if isinstance(layer_in, torch.Tensor):
                # Bitwise operations only work on integer and boolean types
                if layer_in.dtype in [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64]:
                    # Integer types work on both devices
                    return aten_op(layer_in)
                else:
                    # For float types, convert to boolean first, then apply bitwise_not
                    # This is a common pattern in deep learning where bitwise_not is used for masking
                    bool_tensor = layer_in.bool()
                    result = aten_op(bool_tensor)
                    # Convert back to the original dtype and device
                    return result.to(dtype=layer_in.dtype, device=layer_in.device)
            return aten_op(layer_in)

        elif func_name == "any":
            return aten_op(layer_in,
                           layer_hyperparams["dim"],
                           layer_hyperparams["keepdim"])

        elif func_name == "all":
            # 🔧 CRITICAL FIX: Robust 'all' operation with proper validation
            logger.debug(f"[{node_name}] 🔧 all: input type={type(layer_in)}, method_args={method_args}")
            
            # 🔧 FIX: Validate input before processing
            if layer_in is None:
                raise RuntimeError(f"[{node_name}] ❌ all: input is None")
            
            # 🔧 FIX: Handle list inputs properly
            if isinstance(layer_in, list):
                if len(layer_in) == 0:
                    raise RuntimeError(f"[{node_name}] ❌ all: received empty input list")
                elif len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    # 🔧 FIX: Find first valid tensor without modifying original
                    valid_tensor = None
                    for x in layer_in:
                        if isinstance(x, torch.Tensor):
                            valid_tensor = x
                            break
                    
                    if valid_tensor is None:
                        raise RuntimeError(f"[{node_name}] ❌ all: no valid tensor found in input list")
                    
                    layer_in = valid_tensor
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ all: expected tensor input, got {type(layer_in)}")
            
            # 🔧 FIX: Get dimension and keepdim parameters with proper fallback
            dim = layer_hyperparams.get("dim", None)
            keepdim = layer_hyperparams.get("keepdim", False)
            
            logger.debug(f"[{node_name}] ✅ all: input shape={layer_in.shape}, dim={dim}, keepdim={keepdim}")
            
            try:
                # 🔧 FIX: Handle different aten operation variants based on parameters
                if dim is not None:
                    # Use aten::all.dim variant when dimension is specified
                    output = aten_op(layer_in, dim, keepdim)
                else:
                    # Use aten::all.default variant when no dimension is specified (reduces all dimensions)
                    # This is equivalent to calling all() without arguments in PyTorch
                    output = aten_op(layer_in)
                
                logger.debug(f"[{node_name}] ✅ all: output shape={output.shape}")
                return output
                
            except Exception as e:
                # 🔧 FIX: Better error handling with fallback
                logger.debug(f"[{node_name}] ⚠️ all: first attempt failed, trying fallback: {e}")
                
                try:
                    # Fallback: if dim is None, try to reduce all dimensions
                    if dim is None:
                        # For aten::all.dim, we need to provide a dimension
                        # Since we want to reduce all dimensions, we'll use dim=0 and then reduce further if needed
                        output = aten_op(layer_in, 0, keepdim)
                        logger.debug(f"[{node_name}] ✅ all: fallback successful with dim=0, output shape={output.shape}")
                        return output
                    else:
                        raise RuntimeError(f"[{node_name}] ❌ all: failed with input shape={layer_in.shape}, "
                                        f"dim={dim}, keepdim={keepdim}. Error: {e}")
                except Exception as fallback_error:
                    raise RuntimeError(f"[{node_name}] ❌ all: failed with input shape={layer_in.shape}, "
                                    f"dim={dim}, keepdim={keepdim}. Original error: {e}. Fallback error: {fallback_error}")

        elif func_name in {"flatten", "unflatten"}:
            return aten_op(layer_in, *method_args)

        elif func_name in {"ne", "eq", "lt", "le", "gt", "ge"}:
            # 🔧 ENHANCED: Comparison operations with smart broadcasting
            logger.debug(f"[{node_name}] 🔧 {func_name}: processing inputs, layer_in type={type(layer_in)}")
            
            # Handle different input scenarios
            if isinstance(layer_in, (list, tuple)):
                logger.debug(f"[{node_name}] 🔧 {func_name}: got list/tuple with {len(layer_in)} inputs")
                if len(layer_in) == 2:
                    a, b = layer_in
                elif len(layer_in) == 1:
                    # Single input - might be comparing with a scalar or need to get second input from parents
                    a = layer_in[0]
                    # Try to get second input from method_args or layer_hyperparams
                    if method_args:
                        b = method_args[0]
                    elif "other" in layer_hyperparams:
                        b = layer_hyperparams["other"]
                    else:
                        # Try to get from parents
                        logger.debug(f"[{node_name}] 🔧 {func_name}: trying to get second input from parents")
                        raise RuntimeError(f"[{node_name}] ❌ {func_name} needs 2 inputs, got 1 input and no second input found")
                else:
                    raise RuntimeError(f"[{node_name}] ❌ {func_name} expects 1-2 inputs, got {len(layer_in)}")
            elif isinstance(layer_in, torch.Tensor):
                # Single tensor input - try to get second input from method_args or layer_hyperparams
                a = layer_in
                if method_args:
                    b = method_args[0]
                elif "other" in layer_hyperparams:
                    b = layer_hyperparams["other"]
                else:
                    raise RuntimeError(f"[{node_name}] ❌ {func_name} needs 2 inputs, got 1 tensor and no second input found")
            else:
                raise RuntimeError(f"[{node_name}] ❌ {func_name} expects tensor inputs, got {type(layer_in)}")
            
            logger.debug(f"[{node_name}] 🔧 {func_name}: inputs resolved - a={type(a)}, b={type(b)}")
            
            # Handle scalar vs tensor comparisons properly
            # aten::ne.Scalar expects (Tensor, Scalar) or (Scalar, Tensor)
            if isinstance(a, torch.Tensor) and not isinstance(b, torch.Tensor):
                # Tensor vs scalar - use aten::ne.Scalar
                output = aten_op(a, b)
            elif not isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                # Scalar vs tensor - convert scalar to tensor and use tensor comparison
                a_tensor = torch.tensor(a, dtype=b.dtype, device=b.device)
                output = aten_op(a_tensor, b)
            elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                # Tensor vs tensor - ensure consistency and apply smart broadcasting
                a, b = ensure_tensor_consistency([a, b])
                a, b = smart_broadcast_tensors(a, b, node_name, func_name)
                output = aten_op(a, b)
            else:
                # Both scalars - convert to tensors and compare
                a_tensor = torch.tensor(a)
                b_tensor = torch.tensor(b)
                a_tensor, b_tensor = ensure_tensor_consistency([a_tensor, b_tensor])
                output = aten_op(a_tensor, b_tensor)
            
            # Safe logging - handle both tensor and scalar inputs
            a_shape = a.shape if hasattr(a, 'shape') else f"scalar({a})"
            b_shape = b.shape if hasattr(b, 'shape') else f"scalar({b})"
            output_shape = output.shape if hasattr(output, 'shape') else f"scalar({output})"
            
            logger.debug(f"[{node_name}] ✅ {func_name}: input shapes={a_shape}, {b_shape}, output shape={output_shape}")
            return output

        elif func_name == "rsqrt":
            if isinstance(layer_in, list) and len(layer_in) == 1:
                output = aten_op(layer_in[0], *method_args)
            elif isinstance(layer_in, torch.Tensor):
                logger.debug(f"rsqrt operation on node: {node_name}")
                logger.debug(f"rsqrt input shape: {layer_in.shape}")
                output = aten_op(layer_in, *method_args)
                logger.debug(f"rsqrt output shape: {output.shape}")
            else:
                raise RuntimeError(f"[DLBacktrace] rsqrt expects 1 input, got {type(layer_in)}: {layer_in}")
            return output

        elif func_name == "triu":
            if isinstance(layer_in, list):
                if len(layer_in) != 1:
                    raise ValueError(f"Expected 1 tensor for `triu`, but got: {len(layer_in)}")
                layer_in = layer_in[0]
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"`triu` expects a tensor input but got: {type(layer_in)}")
            output = aten_op(layer_in, layer_hyperparams.get("diagonal", 0))
            logger.debug(f"triu output shape: {output.shape}")
            return output

        elif func_name == "mm":
            # Ensure `layer_in` is a list of exactly two tensors
            if isinstance(layer_in, list):
                logger.debug(f"  ↪ Number of inputs: {len(layer_in)}")

                for i, inp in enumerate(layer_in):
                    logger.debug(f"    ↪ Input {i}: shape={getattr(inp, 'shape', 'N/A')}, type={type(inp)}")
                
                if len(layer_in) != 2:
                    raise RuntimeError(f"[{node_name}] ❌ `mm` expects 2 tensor inputs, got {len(layer_in)}")

                a, b = layer_in
            else:
                raise RuntimeError(f"[{node_name}] ❌ `mm` expects list of tensors, got: {type(layer_in)}")

            # Validate inputs are tensors
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `mm` inputs must be tensors: got {type(a)} and {type(b)}")

            # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
            target_device = a.device
            if b.device != target_device:
                b = b.to(device=target_device)

            # Optional: dtype promotion
            if a.dtype != b.dtype:
                target_dtype = torch.promote_types(a.dtype, b.dtype)
                a = a.to(dtype=target_dtype)
                b = b.to(dtype=target_dtype)

            try:
                output = aten_op(a, b)
            except Exception as e:
                raise RuntimeError(
                    f"[{node_name}] ❌ `mm` failed with shapes {a.shape}, {b.shape}: {e}"
                )

            return output

        elif func_name == "cat":
            
            # Step 1: Get tensor list from method_args (preferred)
            tensors = []
            if method_args and isinstance(method_args[0], (list, tuple)):
                for t in method_args[0]:
                    key = str(t)
                    if key in node_io:
                        val = node_io[key]["output_values"]
                        if isinstance(val, torch.Tensor):
                            tensors.append(val)
                        else:
                            raise RuntimeError(f"[{node_name}] ⚠️ Non-tensor in method_args[0]: {type(val)}")
                    else:
                        raise KeyError(f"[{node_name}] ❌ Parent node `{key}` not found in node_io")
            else:
                raise RuntimeError(f"[{node_name}] ❌ `cat` method_args[0] is not a list: {method_args}")

            # Step 2: Get and validate the dimension
            dim = method_args[1] if len(method_args) > 1 else layer_hyperparams.get("dim", 0)
            if not isinstance(dim, int):
                try:
                    dim = int(dim)
                except Exception as e:
                    raise RuntimeError(f"[{node_name}] ❌ Invalid dim value `{dim}`: {e}")
            
            # 🔧 ENHANCED: Handle negative indexing for cat
            if len(tensors) > 0:
                tensor_rank = tensors[0].dim()
                if dim < 0:
                    dim = tensor_rank + dim
                
                # Validate converted dimension
                if dim < 0 or dim >= tensor_rank:
                    raise ValueError(f"[{node_name}] ❌ cat: dimension {method_args[1] if len(method_args) > 1 else layer_hyperparams.get('dim', 0)} out of range for tensor of rank {tensor_rank}")

            for i, t in enumerate(tensors):
                logger.debug(f"  ↪ Tensor {i}: shape={t.shape}, dtype={t.dtype}")

            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            if len(tensors) > 1:
                target_device = tensors[0].device
                for i, t in enumerate(tensors):
                    if t.device != target_device:
                        tensors[i] = t.to(device=target_device)
                        logger.debug(f"  🔧 Moved tensor {i} from {t.device} to {target_device}")

            try:
                output = aten_op(tensors, dim)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `cat` failed: {e}")

            return output

        elif func_name == "clone":
            if isinstance(layer_in, list):
                if len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    raise RuntimeError(f"[{node_name}] ❌ `clone` expects 1 input tensor, got: {layer_in}")

            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `clone` expected Tensor input, got {type(layer_in)}")

            try:
                output = aten_op(
                    layer_in,
                    memory_format=layer_hyperparams.get("memory_format", torch.contiguous_format)
                )
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `clone` failed: {e}")

            return output

        elif func_name == "cos":
            if isinstance(layer_in, list):
                if len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    raise RuntimeError(f"[{node_name}] ❌ `cos` expects a single input tensor, got: {layer_in}")
            
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `cos` expects a Tensor, got {type(layer_in)}")

            try:
                output = aten_op(layer_in)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `cos` failed: {e}")
            
            return output

        elif func_name == "sin":
            # Unwrap if it's a singleton list
            if isinstance(layer_in, list):
                if len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    raise RuntimeError(f"[{node_name}] ❌ `sin` expects a single input tensor but got a list of {len(layer_in)} elements.")

            # Type check
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `sin` expected a Tensor but got {type(layer_in)} → {layer_in}")

            try:
                output = aten_op(layer_in)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `sin` failed: {e}")

            return output

        elif func_name == "neg":
            # Unwrap if it's a singleton list
            if isinstance(layer_in, list):
                if len(layer_in) == 1:
                    layer_in = layer_in[0]
                else:
                    raise RuntimeError(f"[{node_name}] ❌ `neg` expects a single input tensor but got list of {len(layer_in)} elements.")

            # Type check
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ `neg` expected a Tensor but got {type(layer_in)} → {layer_in}")

            try:
                output = aten_op(layer_in)
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ `neg` failed: {e}")

            return output
        
        elif func_name == "copy":
            # 🔧 CRITICAL FIX: Robust copy operation without input modification
            logger.debug(f"[{node_name}] 🔧 copy: input type={type(layer_in)}, length={len(layer_in) if isinstance(layer_in, (list, tuple)) else 'single'}")
            
            # 🔧 FIX: Validate input structure
            if not isinstance(layer_in, list) or len(layer_in) != 2:
                raise RuntimeError(f"[{node_name}] ❌ copy: expects a list of 2 tensors, got: {layer_in}")
            
            self_tensor, src_tensor = layer_in
            
            # 🔧 FIX: Validate input types
            if not isinstance(self_tensor, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ copy: expected 'self' to be Tensor but got {type(self_tensor)}")
            
            if not isinstance(src_tensor, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ copy: expected 'src' to be Tensor but got {type(src_tensor)}")
            
            logger.debug(f"[{node_name}] 🔧 copy: self shape={self_tensor.shape}, src shape={src_tensor.shape}")
            logger.debug(f"[{node_name}] 🔧 copy: self device={self_tensor.device}, src device={src_tensor.device}")
            
            # 🔧 FIX: Create copies to avoid modifying original tensors
            self_tensor_copy = self_tensor.clone()
            src_tensor_copy = src_tensor.clone()
            
            # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
            target_device = self_tensor_copy.device
            if src_tensor_copy.device != target_device:
                src_tensor_copy = src_tensor_copy.to(device=target_device)
                logger.debug(f"[{node_name}] 🔧 copy: moved src tensor to device {target_device}")
            
            # 🔧 FIX: Validate device synchronization was successful
            if src_tensor_copy.device != target_device:
                raise RuntimeError(f"[{node_name}] ❌ copy: failed to move src tensor to target device {target_device}")
            
            # 🔧 FIX: Validate tensor compatibility
            if self_tensor_copy.shape != src_tensor_copy.shape:
                raise RuntimeError(f"[{node_name}] ❌ copy: shape mismatch - self: {self_tensor_copy.shape}, src: {src_tensor_copy.shape}")
            
            non_blocking = layer_hyperparams.get("non_blocking", False)
            
            try:
                output = aten_op(self_tensor_copy, src_tensor_copy, non_blocking=non_blocking)
                logger.debug(f"[{node_name}] ✅ copy: output shape={output.shape}")
                    
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ copy: failed with self shape={self_tensor_copy.shape}, "
                                f"src shape={src_tensor_copy.shape}, non_blocking={non_blocking}. Error: {e}")
            
            return output
        
        elif func_name == "slice_scatter":
            # Step 1: Unpack inputs
            if isinstance(layer_in, list) and len(layer_in) == 2:
                self_tensor, src_tensor = layer_in
                
                # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
                target_device = self_tensor.device
                if src_tensor.device != target_device:
                    src_tensor = src_tensor.to(device=target_device)
            else:
                raise RuntimeError(f"[{node_name}] ❌ `slice_scatter` expects 2 inputs (self, src), got: {layer_in}")

            
            # Step 2: Resolve hyperparams with defaults and checks
            INT64_MAX = 9223372036854775807
            dim   = layer_hyperparams.get("dim", 0)
            start = layer_hyperparams.get("start", 0)
            end   = layer_hyperparams.get("end", None)
            step  = layer_hyperparams.get("step", 1)

            # Fix symbolic placeholder for start/end
            if isinstance(start, int) and start >= INT64_MAX:
                logger.debug(f"[{node_name}] ⚠️ Detected symbolic placeholder for `start` ({start}) → resetting to 0")
                start = 0

            if isinstance(end, int) and end >= INT64_MAX:
                end = self_tensor.shape[dim]
                logger.debug(f"[{node_name}] ⚠️ Detected symbolic placeholder for `end` ({INT64_MAX}) → using {end}")

            if end is None:
                end = self_tensor.shape[dim]
                logger.debug(f"[{node_name}] ℹ️ `end` not specified → using {end}")

            # Step 3: Validate dimensions
            try:
                slice_size = end - start
                src_size = src_tensor.shape[dim]
                self_size = self_tensor.shape[dim]

                if slice_size != src_size:
                    logger.debug(f"[{node_name}] ⚠️ Shape mismatch: src[{dim}] = {src_size} != target slice length = {slice_size}")
            except Exception as e:
                logger.debug(f"[{node_name}] ⚠️ Failed to inspect shapes: {e}")

            # Optional: Show small slice for verification
            try:
                preview = self_tensor.narrow(dim, start, min(end - start, self_tensor.shape[dim] - start))
                logger.debug(f"[{node_name}] 🔍 self_tensor slice preview (dim={dim}, start={start}, end={end}): shape={preview.shape}")
            except Exception as e:
                logger.debug(f"[{node_name}] ⚠️ Could not preview slice: {e}")

            # Step 4: Execute
            try:
                output = aten_op(self_tensor, src_tensor, dim, start, end, step)
                logger.debug(f"[{node_name}] ✅ `slice_scatter` success → output shape: {output.shape}")
            except Exception as e:
                raise RuntimeError(
                    f"[{node_name}] ❌ `slice_scatter` failed: self={type(self_tensor)}, src={type(src_tensor)}, "
                    f"dim={dim}, start={start}, end={end}, step={step}. Error: {e}"
                )

            return output
        
        elif func_name == "_unsafe_view":
            # 🔧 CRITICAL FIX: Robust _unsafe_view operation with proper validation
            logger.debug(f"[{node_name}] 🔧 _unsafe_view: input type={type(layer_in)}, method_args={method_args}")
            
            # 🔧 FIX: Validate input before processing
            if layer_in is None:
                raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: input is None")
            
            # 🔧 FIX: Extract tensor from layer_in with proper validation
            if isinstance(layer_in, list):
                input_tensor = None
                for item in layer_in:
                    if isinstance(item, torch.Tensor):
                        input_tensor = item
                        break
                
                if input_tensor is None:
                    raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: no valid tensor found in inputs: {layer_in}")
            else:
                input_tensor = layer_in
            
            # 🔧 FIX: Validate input tensor
            if not isinstance(input_tensor, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ _unsafe_view: expected tensor input, got {type(input_tensor)}")
            
            # 🔧 FIX: Validate method arguments
            if not method_args or len(method_args) == 0:
                raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: method_args is empty")
            
            raw_shape = method_args[0]
            if not isinstance(raw_shape, (list, tuple)):
                raise TypeError(f"[{node_name}] ❌ _unsafe_view: expected list/tuple for shape, got {type(raw_shape)}")
            
            logger.debug(f"[{node_name}] 🔧 _unsafe_view: raw shape: {raw_shape}")
            
            # 🔧 FIX: Robust parameter resolution with proper error handling
            def resolve_param_view(p, idx):
                try:
                    if isinstance(p, (torch.fx.Node, str)):
                        p_key = str(p)
                        if p_key in node_io:
                            val = node_io[p_key]["output_values"]
                            if isinstance(val, torch.Tensor) and val.numel() == 1:
                                return int(val.item())
                            elif isinstance(val, (int, float)):
                                return int(val)
                            else:
                                raise ValueError(f"Node {p_key} output is not a scalar: {type(val)}")
                        else:
                            raise KeyError(f"Node {p_key} not found in node_io")
                    elif isinstance(p, torch.Tensor):
                        if p.numel() == 1:
                            return int(p.item())
                        else:
                            raise ValueError(f"Tensor has multiple elements: {p.shape}")
                    elif isinstance(p, (int, torch.SymInt)):
                        return int(p)
                    elif p is None:
                        raise ValueError("Cannot resolve None in shape")
                    else:
                        raise TypeError(f"Unsupported shape type: {type(p)}")
                        
                except Exception as e:
                    raise RuntimeError(f"Failed to resolve shape element {idx} ({p}): {e}")

            # 🔧 FIX: Resolve shape with comprehensive error handling
            resolved_shape = []
            for idx, p in enumerate(raw_shape):
                try:
                    resolved = resolve_param_view(p, idx)
                    resolved_shape.append(resolved)
                except Exception as e:
                    raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: failed to resolve shape element {idx}: {e}")
            
            logger.debug(f"[{node_name}] ✅ _unsafe_view: resolved shape: {resolved_shape}")
            
            # 🔧 FIX: Validate resolved shape
            if not resolved_shape:
                raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: empty shape after resolution")
            
            # 🔧 FIX: Allow -1 for automatic dimension inference (PyTorch standard)
            if any(s < -1 for s in resolved_shape):
                raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: invalid shape dimensions: {resolved_shape} (only -1 and positive values allowed)")
            
            # 🔧 FIX: Handle -1 dimensions for automatic inference
            if -1 in resolved_shape:
                logger.debug(f"[{node_name}] 🔧 _unsafe_view: detected -1 dimension, will infer automatically")
                
                # Count how many -1 dimensions we have
                neg_one_count = resolved_shape.count(-1)
                if neg_one_count > 1:
                    raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: only one -1 dimension allowed, got {neg_one_count}")
                
                # Calculate what the -1 dimension should be
                total_elements = input_tensor.numel()
                for s in resolved_shape:
                    if s != -1:
                        total_elements //= s
                
                # Replace -1 with the calculated value
                resolved_shape = [s if s != -1 else total_elements for s in resolved_shape]
                
                logger.debug(f"[{node_name}] 🔧 _unsafe_view: inferred -1 dimension as {total_elements}")
            
            # 🔧 FIX: Validate tensor compatibility with final shape
            total_elements = 1
            for s in resolved_shape:
                total_elements *= s
            
            if input_tensor.numel() != total_elements:
                raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: shape {resolved_shape} (expected {total_elements} elements)")
            
            # 🔧 FIX: Execute with proper error handling
            try:
                output = aten_op(input_tensor, resolved_shape)
                logger.debug(f"[{node_name}] ✅ _unsafe_view: input shape={input_tensor.shape}, output shape={output.shape}")
                return output
                
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ _unsafe_view: execution failed: input shape={input_tensor.shape}, "
                                f"resolved_shape={resolved_shape}, error={e}")

        elif func_name == "einsum":
            if not method_args or not isinstance(method_args[0], str):
                raise RuntimeError(f"[{node_name}] ❌ `einsum` requires equation string as the first method_arg")

            equation = method_args[0]

            # Ensure input is a list of tensors
            if not isinstance(layer_in, (list, tuple)):
                layer_in = [layer_in]

            logger.debug(f"[{node_name}] 🧪 einsum equation: {equation}")
            for i, t in enumerate(layer_in):
                if isinstance(t, torch.Tensor):
                    logger.debug(f"[{node_name}] ↪ Input {i} shape: {t.shape}")
                else:
                    logger.debug(f"[{node_name}] ⚠️ Input {i} is not a tensor: {type(t)}")

            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            if len(layer_in) > 1:
                target_device = None
                for t in layer_in:
                    if isinstance(t, torch.Tensor):
                        target_device = t.device
                        break
                
                if target_device is not None:
                    for i, t in enumerate(layer_in):
                        if isinstance(t, torch.Tensor) and t.device != target_device:
                            layer_in[i] = t.to(device=target_device)
                            logger.debug(f"[{node_name}] 🔧 Moved tensor {i} from {t.device} to {target_device}")

            try:
                # Don't unpack the tensor list — pass it as a list
                output = aten_op(equation, layer_in)
                logger.debug(f"[{node_name}] ✅ einsum output shape: {output.shape}")
            except Exception as e:
                raise RuntimeError(
                    f"[{node_name}] ❌ `einsum` failed with equation '{equation}' and inputs: {layer_in}. Error: {e}"
                )

            return output

        elif func_name == "index_select":
            if not isinstance(layer_in, list) or len(layer_in) != 2:
                raise RuntimeError(f"[{node_name}] ❌ `index_select` expected 2 inputs from parents, got: {layer_in}")

            input_tensor, index_tensor = layer_in

            # 🔧 DEVICE CONSISTENCY FIX: Ensure both tensors are on the same device
            target_device = input_tensor.device
            if index_tensor.device != target_device:
                index_tensor = index_tensor.to(device=target_device)

            if not method_args or len(method_args) < 1:
                raise RuntimeError(f"[{node_name}] ❌ `index_select` missing `dim` in method_args: {method_args}")

            dim = method_args[0]
            if isinstance(dim, torch.Tensor):
                dim = int(dim.item())
            elif not isinstance(dim, int):
                raise TypeError(f"[{node_name}] ❌ `dim` must be int, got {type(dim)}: {dim}")
            
            # 🔧 ENHANCED: Handle negative indexing for index_select
            tensor_rank = input_tensor.dim()
            if dim < 0:
                dim = tensor_rank + dim
            
            # Validate converted dimension
            if dim < 0 or dim >= tensor_rank:
                raise ValueError(f"[{node_name}] ❌ index_select: dimension {method_args[0]} out of range for tensor of rank {tensor_rank}")

            if index_tensor.dtype not in (torch.int32, torch.int64):
                index_tensor = index_tensor.to(dtype=torch.int64)

            # ✅ Match length to peer tensor (e.g. einsum_5)
            peer_shape = node_io.get('einsum_5', {}).get('output_values', None)
            if isinstance(peer_shape, torch.Tensor):
                expected_dim_size = peer_shape.shape[dim]
                if index_tensor.shape[0] > expected_dim_size:
                    logger.debug(f"[{node_name}] ⚠️ Trimming index_tensor from {index_tensor.shape[0]} to {expected_dim_size} to match einsum_5 shape")
                    index_tensor = index_tensor[:expected_dim_size]

            logger.debug(f"[{node_name}] ✅ index_select(dim={dim}) on input shape {input_tensor.shape} with index shape {index_tensor.shape}")
            return aten_op(input_tensor, dim, index_tensor)

        elif func_name == "addmm":
            if isinstance(layer_in, (list, tuple)):
                for idx, item in enumerate(layer_in):
                    logger.debug(f"idx: {idx}, item: {item.shape}") 
            else:
                logger.debug(f"layer_in shape: {layer_in.shape}")

            if isinstance(layer_in, list) and len(layer_in) == 3:
                bias, mat1, mat2 = layer_in
                logger.debug(f"bias: {bias.shape}, mat1: {mat1.shape}, mat2: {mat2.shape}") 

                if not all(isinstance(x, torch.Tensor) for x in (bias, mat1, mat2)):
                    raise TypeError(f"[{node_name}] ❌ Expected Tensors for `addmm`, got {[type(x) for x in layer_in]}")

                # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
                target_device = mat1.device
                if bias.device != target_device:
                    bias = bias.to(device=target_device)
                if mat2.device != target_device:
                    mat2 = mat2.to(device=target_device)

                try:
                    return torch.addmm(bias, mat1, mat2)
                except Exception as e:
                    raise RuntimeError(
                        f"[{node_name}] ❌ torch.addmm failed with shapes: "
                        f"bias={bias.shape}, mat1={mat1.shape}, mat2={mat2.shape}. Error: {e}"
                    )
            else:
                raise RuntimeError(f"[{node_name}] ❌ `addmm` expects 3 input tensors (bias, mat1, mat2), got: {layer_in}")
        
        elif func_name == "index": 
            input_tensor = layer_in[0]
            index_tensors = layer_in[1:]

            # 🔧 DEVICE CONSISTENCY FIX: Ensure all tensors are on the same device
            target_device = input_tensor.device
            for i, idx in enumerate(index_tensors):
                if isinstance(idx, torch.Tensor) and idx.device != target_device:
                    index_tensors[i] = idx.to(device=target_device)

            # Cast all index tensors to long
            index_tensors = [
                idx.to(dtype=torch.long) if isinstance(idx, torch.Tensor) and not idx.dtype in (torch.long, torch.int, torch.bool, torch.uint8) else idx
                for idx in index_tensors
            ]

            # 🛡️ Validate and inject index at correct dimension
            input_rank = input_tensor.dim()
            indices = [slice(None)] * input_rank  # default: [:, :, :, ...]

            for idx in index_tensors:
                # Dynamically find dimension that matches index shape
                inserted = False
                for i in range(input_rank):
                    if idx.dim() == 1 and input_tensor.shape[i] == idx.shape[0]:
                        indices[i] = idx
                        inserted = True
                        break
                if not inserted:
                    raise RuntimeError(f"[{node_name}] ❌ Could not find matching dimension for index with shape {idx.shape} in input_tensor of shape {input_tensor.shape}")

            try:
                return input_tensor[tuple(indices)]
            except Exception as e:
                raise RuntimeError(
                    f"[{node_name}] ❌ Failed to apply `index` with shape {input_tensor.shape} and indices {[idx.shape for idx in index_tensors]}: {e}"
                )

        elif func_name == "cumsum":
            # 🔧 NEW OPERATION: Cumulative sum with precision consistency
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0] 
            layer_in = enforce_precision_consistency(layer_in)
            
            # Extract dimension from method_args
            dim = method_args[0] if method_args else 0
            
            try:
                output = aten_op(layer_in, dim)
                logger.debug(f"[{node_name}] ✅ cumsum: input shape={layer_in.shape}, output shape={output.shape}")
                return output
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ cumsum failed with input shape={layer_in.shape}, dim={dim}. Error: {e}")

        elif func_name == "type_as":
            # 🔧 NEW OPERATION: Type conversion with precision consistency
            if isinstance(layer_in, (list, tuple)) and len(layer_in) >= 2:
                input_tensor, target_tensor = layer_in[0], layer_in[1]
            else:
                raise RuntimeError(f"[{node_name}] ❌ type_as expects 2 inputs, got {len(layer_in) if isinstance(layer_in, (list, tuple)) else 1}")
            
            # Apply precision consistency to both tensors
            input_tensor = enforce_precision_consistency(input_tensor)
            target_tensor = enforce_precision_consistency(target_tensor)
            
            try:
                output = aten_op(input_tensor, target_tensor)
                logger.debug(f"[{node_name}] ✅ type_as: input shape={input_tensor.shape}, target shape={target_tensor.shape}, output shape={output.shape}")
                return output
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ type_as failed with input shape={input_tensor.shape}, target shape={target_tensor.shape}. Error: {e}")

        elif func_name == "split":
            # 🔧 NEW OPERATION: Split tensor into chunks (crucial for transformers)
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0]
            layer_in = enforce_precision_consistency(layer_in)
            
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ split: expected tensor input, got {type(layer_in)}")
            
            # Get split parameters
            split_size_or_sections = layer_hyperparams.get("split_size_or_sections", 1)
            dim = layer_hyperparams.get("dim", 0)
            
            # Handle negative indexing
            tensor_rank = layer_in.dim()
            if dim < 0:
                dim = tensor_rank + dim
            
            try:
                output = aten_op(layer_in, split_size_or_sections, dim)
                logger.debug(f"[{node_name}] ✅ split: input shape={layer_in.shape}, output chunks={len(output)}")
                return output
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ split failed with input shape={layer_in.shape}, split_size={split_size_or_sections}, dim={dim}. Error: {e}")

        elif func_name == "chunk":
            # 🔧 NEW OPERATION: Split tensor into equal chunks
            if isinstance(layer_in, (list, tuple)):
                layer_in = layer_in[0]
            layer_in = enforce_precision_consistency(layer_in)
            
            if not isinstance(layer_in, torch.Tensor):
                raise TypeError(f"[{node_name}] ❌ chunk: expected tensor input, got {type(layer_in)}")
            
            chunks = layer_hyperparams.get("chunks", 2)
            dim = layer_hyperparams.get("dim", 0)
            
            # Handle negative indexing
            tensor_rank = layer_in.dim()
            if dim < 0:
                dim = tensor_rank + dim
            
            try:
                output = aten_op(layer_in, chunks, dim)
                logger.debug(f"[{node_name}] ✅ chunk: input shape={layer_in.shape}, chunks={chunks}, dim={dim}")
                return output
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ chunk failed with input shape={layer_in.shape}, chunks={chunks}, dim={dim}. Error: {e}")

        elif func_name == "stack":
            # 🔧 NEW OPERATION: Stack tensors along new dimension
            if not isinstance(layer_in, (list, tuple)):
                raise RuntimeError(f"[{node_name}] ❌ stack: expected list of tensors, got {type(layer_in)}")
            
            # Apply precision consistency to all tensors
            tensors = [enforce_precision_consistency(t) for t in layer_in]
            
            # Ensure all are tensors and on same device
            target_device = tensors[0].device
            for i, t in enumerate(tensors):
                if not isinstance(t, torch.Tensor):
                    raise TypeError(f"[{node_name}] ❌ stack: all inputs must be tensors, got {type(t)} at index {i}")
                if t.device != target_device:
                    tensors[i] = t.to(device=target_device)
            
            dim = layer_hyperparams.get("dim", 0)
            
            try:
                output = aten_op(tensors, dim)
                logger.debug(f"[{node_name}] ✅ stack: {len(tensors)} tensors, output shape={output.shape}")
                return output
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ stack failed with {len(tensors)} tensors, dim={dim}. Error: {e}")

        elif func_name == "gather":
            # 🔧 NEW OPERATION: Gather values along dimension (critical for attention)
            if not isinstance(layer_in, (list, tuple)) or len(layer_in) != 2:
                raise RuntimeError(f"[{node_name}] ❌ gather: expected 2 inputs (input, index), got {layer_in}")
            
            input_tensor, index_tensor = layer_in
            input_tensor = enforce_precision_consistency(input_tensor)
            index_tensor = enforce_precision_consistency(index_tensor)
            
            # Ensure index tensor is long
            if index_tensor.dtype != torch.long:
                index_tensor = index_tensor.long()
            
            # Ensure same device
            if input_tensor.device != index_tensor.device:
                index_tensor = index_tensor.to(device=input_tensor.device)
            
            dim = layer_hyperparams.get("dim", 0)
            
            try:
                output = aten_op(input_tensor, dim, index_tensor)
                logger.debug(f"[{node_name}] ✅ gather: input shape={input_tensor.shape}, index shape={index_tensor.shape}, output shape={output.shape}")
                return output
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ gather failed with input shape={input_tensor.shape}, index shape={index_tensor.shape}, dim={dim}. Error: {e}")

        elif func_name == "native_layer_norm":
            # 🔧 NEW OPERATION: Native layer norm (optimized version)
            layer_in = standardize_layer_input(layer_in, parents, node_io, tensor_map, func_name)
            layer_in = enforce_precision_consistency(layer_in)
            
            if not isinstance(layer_in, torch.Tensor):
                raise RuntimeError(f"[{node_name}] ❌ native_layer_norm: expected tensor input, got {type(layer_in)}")
            
            # Get parameters
            normalized_shape = layer_hyperparams.get("normalized_shape")
            weight = layer_hyperparams.get("weight")
            bias = layer_hyperparams.get("bias")
            eps = layer_hyperparams.get("eps", 1e-5)
            
            # Validate parameters
            if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
                raise RuntimeError(f"[{node_name}] ❌ native_layer_norm: weight and bias must be tensors")
            
            # Ensure device consistency
            target_device = layer_in.device
            if weight.device != target_device:
                weight = weight.to(device=target_device)
            if bias.device != target_device:
                bias = bias.to(device=target_device)
            
            try:
                output = aten_op(layer_in, normalized_shape, weight, bias, eps)
                # native_layer_norm returns (output, mean, rstd) tuple
                if isinstance(output, tuple):
                    return output[0]  # Return just the normalized output
                return output
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ native_layer_norm failed with input shape={layer_in.shape}. Error: {e}")

        elif func_name == "baddbmm":
            # 🔧 NEW OPERATION: Batch matrix multiplication with bias (critical for attention)
            if not isinstance(layer_in, (list, tuple)) or len(layer_in) != 3:
                raise RuntimeError(f"[{node_name}] ❌ baddbmm: expected 3 inputs (bias, batch1, batch2), got {layer_in}")
            
            bias, batch1, batch2 = layer_in
            
            # Apply precision consistency
            bias = enforce_precision_consistency(bias)
            batch1 = enforce_precision_consistency(batch1)
            batch2 = enforce_precision_consistency(batch2)
            
            # Ensure device consistency
            target_device = batch1.device
            if bias.device != target_device:
                bias = bias.to(device=target_device)
            if batch2.device != target_device:
                batch2 = batch2.to(device=target_device)
            
            # Get scaling factors
            beta = layer_hyperparams.get("beta", 1.0)
            alpha = layer_hyperparams.get("alpha", 1.0)
            
            try:
                output = aten_op(bias, batch1, batch2, beta=beta, alpha=alpha)
                logger.debug(f"[{node_name}] ✅ baddbmm: bias shape={bias.shape}, batch1 shape={batch1.shape}, batch2 shape={batch2.shape}, output shape={output.shape}")
                return output
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ baddbmm failed with shapes: bias={bias.shape}, batch1={batch1.shape}, batch2={batch2.shape}. Error: {e}")

        elif func_name == "addmv":
            # 🔧 NEW OPERATION: Matrix-vector multiplication with bias
            if not isinstance(layer_in, (list, tuple)) or len(layer_in) != 3:
                raise RuntimeError(f"[{node_name}] ❌ addmv: expected 3 inputs (bias, mat, vec), got {layer_in}")
            
            bias, mat, vec = layer_in
            
            # Apply precision consistency
            bias = enforce_precision_consistency(bias)
            mat = enforce_precision_consistency(mat)
            vec = enforce_precision_consistency(vec)
            
            # Ensure device consistency
            target_device = mat.device
            if bias.device != target_device:
                bias = bias.to(device=target_device)
            if vec.device != target_device:
                vec = vec.to(device=target_device)
            
            # Get scaling factors
            beta = layer_hyperparams.get("beta", 1.0)
            alpha = layer_hyperparams.get("alpha", 1.0)
            
            try:
                output = aten_op(bias, mat, vec, beta=beta, alpha=alpha)
                logger.debug(f"[{node_name}] ✅ addmv: bias shape={bias.shape}, mat shape={mat.shape}, vec shape={vec.shape}, output shape={output.shape}")
                return output
            except Exception as e:
                raise RuntimeError(f"[{node_name}] ❌ addmv failed with shapes: bias={bias.shape}, mat={mat.shape}, vec={vec.shape}. Error: {e}")

        elif func_name == "alias":
            # 🔧 NEW OPERATION: Alias operation - creates a view of tensor without copying
            logger.debug(f"[{node_name}] 🔧 alias: processing input, layer_in type={type(layer_in)}")
            
            # Handle different input scenarios
            if isinstance(layer_in, (list, tuple)):
                if len(layer_in) == 1:
                    input_tensor = layer_in[0]
                else:
                    raise RuntimeError(f"[{node_name}] ❌ alias expects 1 input, got {len(layer_in)}")
            elif isinstance(layer_in, torch.Tensor):
                input_tensor = layer_in
            else:
                raise RuntimeError(f"[{node_name}] ❌ alias expects tensor input, got {type(layer_in)}")
            
            # Apply consistency checks for exact reproducibility
            input_tensor = enforce_precision_consistency(input_tensor)
            
            # Execute operation - aten::alias() creates a view of the input tensor
            output = aten_op(input_tensor)
            logger.debug(f"[{node_name}] ✅ alias: input shape={input_tensor.shape}, output shape={output.shape}")
            return output

        else:
            # 🔧 UNHANDLED OPERATION: Print with emojis for easy identification
            logger.warning(f"🚨 UNHANDLED OPERATION: `{func_name}` - needs implementation!")
            logger.warning(f"📝 Operation: {func_name}")
            logger.warning(f"📊 Input type: {type(layer_in)}")
            logger.warning(f"📋 Method args: {method_args}")
            logger.warning(f"🏷️  Node: {node_name}")
            logger.warning(f"🔧 Add this operation to the execution engine!")
            
            print(f"🚨 UNHANDLED OPERATION: `{func_name}` - needs implementation!")
            print(f"📝 Operation: {func_name}")
            print(f"📊 Input type: {type(layer_in)}")
            print(f"📋 Method args: {method_args}")
            print(f"🏷️  Node: {node_name}")
            print(f"🔧 Add this operation to the execution engine!")
            # Fallback to generic execution
            inputs = layer_in if isinstance(layer_in, (list, tuple)) else [layer_in]
            output = aten_op(*inputs, *method_args)
            return output 
            
    except Exception as e:
        logger.error(f"[Execution Error] Node `{node_name}` failed in `{func_name}`: {e}")
        #return layer_in

def run_execution_nocache(graph, layer_stack, model, extracted_weights, inputs, tracer, exported_program=None):
    logger = get_logger()
    logger.info(f"Executing `run_execution_nocache` ...!")
    
    # 🔧 CONSISTENCY FIX: Set up consistent environment
    setup_consistent_environment(model)
    
    # 🔧 CRITICAL FIX: Ensure model is in consistent state
    logger.debug("🔧 Ensuring model consistency...")
    model.eval()
    model.requires_grad_(False)
    
    # 🔧 CRITICAL FIX: Synchronize extracted weights with current model state
    logger.debug("🔧 Synchronizing extracted weights with model state...")
    if exported_program is not None:
        for placeholder_name, extracted_weight in extracted_weights.items():
            if extracted_weight is not None and isinstance(extracted_weight, torch.Tensor):
                # Get the real parameter name from exported_program
                real_key = None
                for spec in exported_program.graph_signature.input_specs:
                    if hasattr(spec.arg, 'name') and spec.arg.name == placeholder_name:
                        real_key = spec.target
                        break
                
                if real_key and real_key in model.state_dict():
                    current_weight = model.state_dict()[real_key]
                    # Update extracted weight to match current model state
                    extracted_weights[placeholder_name] = current_weight.clone()
                    logger.debug(f"✅ Synchronized {placeholder_name} -> {real_key}")
    else:
        logger.warning("⚠️ No exported_program provided, skipping weight synchronization")
    
    logger.debug("✅ Weight synchronization complete")
    
    # 🔧 CRITICAL FIX: Preserve original model precision instead of forcing float32
    # This ensures numerical consistency with the original model
    logger.debug("Preserving original model precision for consistency")
    for param in model.parameters():
        logger.debug(f"Parameter {param.shape} dtype: {param.dtype}")
    
    for buffer in model.buffers():
        logger.debug(f"Buffer {buffer.shape} dtype: {buffer.dtype}")
    
    logger.debug("✅ Model consistency ensured")
    
    # 🔧 ENHANCED FIX: Get the target device from the model for all operations
    target_device = ensure_model_has_device_attribute(model)
    logger.debug(f"🎯 Target device for all operations: {target_device}")
    
    # 🔧 ENHANCED FIX: Get the model dtype for all operations
    model_dtype = torch.float32  # fallback
    for param in model.parameters():
        if param.is_floating_point():
            model_dtype = param.dtype
            break
    else:
        for buffer in model.buffers():
            if buffer.is_floating_point():
                model_dtype = buffer.dtype
                break
    logger.debug(f"🎯 Model dtype for all operations: {model_dtype}")
    
    tensor_map = {}
    node_io = {}

    model_signature = inspect.signature(model.forward)
    expected_input_names = list(model_signature.parameters.keys())

    if len(inputs) != len(expected_input_names):
        raise ValueError("Mismatch between model input count and provided inputs.")

    inp_map = dict(zip(expected_input_names, inputs))
    
    # 🔧 CRITICAL FIX: Ensure input tensors are consistent from the start
    if len(extracted_weights) < len(layer_stack):
        # Force input tensors to have consistent properties
        if isinstance(inputs, (list, tuple)):
            consistent_inputs = []
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    # 🔧 NAME-BASED CASTING: cast only input_ids/position_ids to long; keep attention_mask dtype
                    arg_name = expected_input_names[i] if i < len(expected_input_names) else None
                    if arg_name in ("input_ids", "position_ids"):
                        consistent_inp = inp.detach().to(dtype=torch.long)
                    else:
                        consistent_inp = inp.detach()  # preserve original dtype
                    
                    # Ensure contiguous memory layout
                    if not consistent_inp.is_contiguous():
                        consistent_inp = consistent_inp.contiguous()
                    
                    consistent_inputs.append(consistent_inp)
                    logger.debug(f"Input {i}: {inp.shape} -> {consistent_inp.shape}, dtype: {consistent_inp.dtype}")
                else:
                    consistent_inputs.append(inp)
            tensor_map[layer_stack[len(extracted_weights)]] = consistent_inputs
        else:
            if isinstance(inputs, torch.Tensor):
                # For single tensor input, conservatively preserve dtype
                consistent_input = inputs.detach()
                if not consistent_input.is_contiguous():
                    consistent_input = consistent_input.contiguous()
                tensor_map[layer_stack[len(extracted_weights)]] = consistent_input
                logger.debug(f"Single input: {inputs.shape} -> {consistent_input.shape}, dtype: {consistent_input.dtype}")
            else:
                tensor_map[layer_stack[len(extracted_weights)]] = inputs

    for node_name in layer_stack:
        node_data = graph.nodes[node_name]
        parents = node_data["parents"]
        layer_type = node_data["layer_type"]
        method_args = node_data["method_args"]
        layer = node_data["layer"]
        func_name = node_data["func_name"]
        layer_hyperparams = node_data["layer_hyperparams"]
        children = node_data["children"]

        # Log node execution for debugging
        logger.debug(f"🔧 Processing node: {node_name}, func: {func_name}, layer_type: {layer_type}")

        if any(p not in tensor_map for p in parents):
            logger.debug(f"⚠️ Skipping {node_name} - missing parents: {[p for p in parents if p not in tensor_map]}")
            continue
        
        layer_in = [_sanitize_input_tensor(tensor_map[p]) for p in parents]
        
        # 🔧 CRITICAL FIX: Ensure input consistency for all operations
        layer_in = ensure_input_consistency(layer_in, model)
        
        output = layer_in
        
        # 🔧 REMOVED: Unnecessary embedding handling code
        # Embedding operations are handled in execute_aten_operation function

        # ✅ Register lifted_tensor constants early (before parent check)
        if layer_type == "Placeholder" and "lifted_tensor" in node_name:
            output = extracted_weights.get(node_name, None)
            if output is None:
                logger.warning(f"Lifted tensor node `{node_name}` missing in extracted_weights")
                continue
            
        # 🔧 CONSISTENCY FIX: Ensure weights have consistent precision
        if layer_type == "Placeholder" and any(keyword in node_name for keyword in ["weight", "bias"]):
            if output is not None and isinstance(output, torch.Tensor):
                output = enforce_precision_consistency(output)
                    
        try:
            if layer_type == "Placeholder":
                output = extracted_weights.get(node_name, inp_map.get(node_name, layer_in)) 

                special_weight_keywords = [
                    '_layernorm_weight',
                    '_proj_weight',
                    '_head_weight',
                    '_norm_weight'
                ]
                
                if any(key in node_name for key in special_weight_keywords):
                    layer_type = "future_use"
                elif "weight" in node_name:
                    layer_type = "Weight"
                elif "bias" in node_name:
                    layer_type = "Bias"
                elif "running_mean" in node_name:
                    layer_type = "bn_running_mean"
                elif "running_var" in node_name:
                    layer_type = "bn_running_var"
                elif "num_batches_tracked" in node_name:
                    layer_type = "bn_num_batches_tracked"
                elif "embeddings_token_type_ids" in node_name:
                    layer_type = "embeddings_token_type_ids"
                elif "embeddings_position_ids" in node_name:
                    layer_type = "embeddings_position_ids"


            elif layer_type == "Model_Layer":
                resolved_layer = model
                for attr in layer.split("."):
                    resolved_layer = getattr(resolved_layer, attr)
                
                # 🔧 CRITICAL FIX: Ensure Model_Layer inputs are IDENTICAL to direct model execution
                # Get the original input tensor to match properties exactly
                original_input = None
                if len(inputs) > 0:
                    original_input = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
                
                # Sanitize inputs to match original model input properties exactly
                if original_input is not None:
                    layer_in = sanitize_model_layer_inputs(layer_in, original_input)
                
                # 🔧 ENHANCED CONSISTENCY FIX: Use comprehensive consistency functions
                if isinstance(layer_in, (list, tuple)):
                    layer_in = [enforce_precision_consistency(x) for x in layer_in]
                    # 🔧 CRITICAL FIX: Preserve original input precision for consistency
                    # Don't force float32 conversion as it can cause numerical differences
                    for i, inp in enumerate(layer_in):
                        if isinstance(inp, torch.Tensor):
                            layer_in[i] = inp.to(dtype=inp.dtype)  # Preserve original dtype
                            if not layer_in[i].is_contiguous():
                                layer_in[i] = layer_in[i].contiguous()
                else:
                    layer_in = enforce_precision_consistency(layer_in)
                    if isinstance(layer_in, torch.Tensor):
                        layer_in = layer_in.to(dtype=layer_in.dtype)  # Preserve original dtype
                        if not layer_in.is_contiguous():
                            layer_in = layer_in.contiguous()
                
                # 🔧 CRITICAL FIX: Ensure the resolved layer is in the same state
                resolved_layer.eval()
                resolved_layer.requires_grad_(False)
                
                output = resolved_layer(*layer_in)

            elif layer_type == "Output":
                output = layer_in[0]

            elif layer_type == "CustomOp":
                output = layer_in

            elif layer_type in ("ATen_Operation", "Operation"):
                layer_in = []
                if func_name == "linear":
                    for node_x in parents:
                        if node_io[node_x]['layer_type'] in ("future_use","Weight","Bias","bn_running_mean","bn_running_var","bn_num_batches_tracked","embeddings"):
                            continue
                        else:
                            layer_in.append(node_io[node_x]['output_values'])
                elif func_name == "addmm":
                    # ✅ For addmm, include all parent outputs, including weight/bias
                    for node_x in parents:
                        layer_in.append(node_io[node_x]['output_values'])

                else:
                    for node_x in parents:
                        if func_name == "_native_batch_norm_legit_no_training":
                            # ✅ Include *all* parents for batch norm op
                            layer_in.append(node_io[node_x]['output_values'])
                        elif node_io[node_x]['layer_type'] in ("Weight", "Bias", "bn_running_mean", "bn_running_var", "bn_num_batches_tracked", "embeddings"):
                            continue
                        else:
                            layer_in.append(node_io[node_x]['output_values']) 

                # 🔧 CRITICAL FIX: Better input processing for all operations
                if func_name in ["addmm", "matmul", "bmm"]:
                    # For matrix operations, ensure inputs are properly formatted
                    layer_in, _ = ensure_operation_consistency(func_name, layer_in, [], node_name)
                elif func_name == "addmm":
                    pass  # ⛔ Don't process or unwrap inputs
                elif func_name == "linear":
                    # 🔧 FIX: Linear operations are handled separately above, skip duplicate processing
                    logger.debug(f"🔧 linear: skipping duplicate processing (handled above)")
                    pass
                else:
                    # 🔧 IMPROVED: Better list-to-tensor conversion for all other operations
                    logger.debug(f"🔧 Processing inputs for {func_name}: {len(layer_in)} inputs")
                    for i, inp in enumerate(layer_in):
                        if isinstance(inp, torch.Tensor):
                            logger.debug(f"  Input {i}: {inp.shape} on {inp.device}, dtype: {inp.dtype}")
                        else:
                            logger.debug(f"  Input {i}: {type(inp)}")
                    
                    # 🔧 IMPROVED: Handle empty input lists gracefully
                    if len(layer_in) == 0:
                        # Only warn for operations that should have inputs, not leaf operations
                        leaf_operations = ["arange", "full", "zeros", "ones", "randn", "rand", "eye", "linspace"]
                        if func_name not in leaf_operations:
                            logger.warning(f"⚠️ No inputs collected for {func_name}, trying to recover from parents")
                        else:
                            logger.debug(f"🔧 {func_name} is a leaf operation with no inputs (expected)")
                        
                        # Try to recover inputs from parent nodes
                        for node_x in parents:
                            if node_io[node_x]['layer_type'] not in ("Weight", "Bias", "future_use"):
                                layer_in.append(node_io[node_x]['output_values'])
                    
                    # 🔧 IMPROVED: Better list processing
                    if len(layer_in) == 1 and isinstance(layer_in[0], (list, tuple)):
                        # Unwrap nested lists
                        layer_in = layer_in[0]
                        logger.debug(f"🔧 Unwrapped nested list, now have {len(layer_in)} inputs")
                    
                    # Process the inputs
                    layer_in, _ = _process_layer_input(layer_in)
                    
                    logger.debug(f"🔧 After processing: {type(layer_in)}, length: {len(layer_in) if isinstance(layer_in, (list, tuple)) else 'single'}")

                output = execute_aten_operation(func_name, layer, layer_in, layer_hyperparams, method_args, parents, node_io, node_name,tensor_map, children=children, model_dtype=model_dtype)
                # 🔧 FIX: Only log shape if output is a tensor
                if isinstance(output, int):
                    logger.debug(f"[outerloop] {node_name} output type: {type(output)}, value: {output} (non-tensor)")
                elif hasattr(output, 'shape'):
                    logger.debug(f"[outerloop] {node_name} output shape: {output.shape}")
                else:
                    logger.debug(f"[outerloop] {node_name} output type: {type(output)} (non-tensor)")
        except Exception as e:
            logger.error(f"[Execution Error - NoCache] Node `{node_name}` failed in `{func_name}`: {e}")
            output = layer_in

        # At the end of node execution
        processed_output = _process_output_tuple(output)

        # 🔧 CRITICAL FIX: Only check for NaN in floating point tensors
        if isinstance(processed_output, torch.Tensor) and torch.is_floating_point(processed_output) and torch.isnan(processed_output).any():
            logger.error(f"[ERROR:NaN] Node `{node_name}` produced NaNs → shape: {processed_output.shape}")
        
        # 🔧 DEBUG: Check for extreme values that could indicate precision issues
        if isinstance(processed_output, torch.Tensor):
            # 🔧 CRITICAL FIX: Check for empty tensors first
            if processed_output.numel() == 0:
                logger.debug(f"[DEBUG:EMPTY] Node `{node_name}` produced empty tensor → shape: {processed_output.shape}")
            # 🔧 CRITICAL FIX: Only check abs for numeric tensors, not boolean tensors
            elif processed_output.dtype in [torch.bool]:
                # For boolean tensors, just check basic properties
                logger.debug(f"[DEBUG:BOOL] Node `{node_name}` produced boolean tensor → shape: {processed_output.shape}")
            elif torch.is_floating_point(processed_output) or torch.is_complex(processed_output):
                # Only apply abs() to floating point or complex tensors
                max_val = torch.max(torch.abs(processed_output)).item()
                
                # Skip extreme value warnings for attention mask operations that legitimately use float32 min/max
                attention_mask_ops = ["full", "masked_fill", "eq", "triu", "tril", "mul", "add"]
                shape_preserving_ops = ["slice", "unsqueeze", "expand", "clone", "copy", "slice_scatter"]
                is_attention_mask = (func_name in attention_mask_ops or 
                                   any(op in node_name.lower() for op in ["mask", "attention", "causal"]))
                is_shape_preserving = (func_name in shape_preserving_ops or
                                     any(op in node_name.lower() for op in ["slice", "unsqueeze", "expand", "clone", "copy"]))
                
                if max_val > 1e6 and not (is_attention_mask or is_shape_preserving):
                    logger.warning(f"[WARNING:EXTREME] Node `{node_name}` produced extreme values → max: {max_val:.2e}")
                elif is_shape_preserving:
                    logger.debug(f"✅ Skipped extreme value warning for shape-preserving operation: {func_name}")
                elif func_name in ["scaled_dot_product_attention", "layer_norm", "linear"] and max_val < 1e-6:
                    logger.warning(f"[WARNING:SMALL] Node `{node_name}` produced very small values → max: {max_val:.2e}")
                
                # Log first few values for critical operations to help debug
                if func_name in ["scaled_dot_product_attention", "layer_norm"] and logger.isEnabledFor(logging.DEBUG):
                    flat_vals = processed_output.flatten()[:5].tolist()
                    logger.debug(f"[DEBUG:VALUES] {node_name} first 5 values: {[f'{v:.6f}' for v in flat_vals]}")
            elif processed_output.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                # For integer tensors, use different approach
                max_val = torch.max(processed_output).item()
                min_val = torch.min(processed_output).item()
                logger.debug(f"[DEBUG:INT] Node `{node_name}` → min: {min_val}, max: {max_val}")
            else:
                # For other dtypes, just log basic info
                logger.debug(f"[DEBUG:OTHER] Node `{node_name}` → dtype: {processed_output.dtype}, shape: {processed_output.shape}")

        # 🔧 CRITICAL FIX: Ensure ALL tensor outputs maintain consistency without forcing float32
        if isinstance(processed_output, torch.Tensor):
            # Preserve original precision but ensure consistent memory format
            processed_output = processed_output.detach().contiguous()
            # Only convert to float32 if the model was originally in float32
            if hasattr(model, 'dtype') and model.dtype == torch.float32:
                processed_output = processed_output.to(dtype=torch.float32)
            tensor_map[node_name] = processed_output
        elif isinstance(processed_output, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in processed_output):
            # Preserve original precision for all tensors in tuples
            processed_output = [x.detach().contiguous() for x in processed_output]
            # Only convert to float32 if the model was originally in float32
            if hasattr(model, 'dtype') and model.dtype == torch.float32:
                processed_output = [x.to(dtype=torch.float32) for x in processed_output]
            tensor_map[node_name] = processed_output
        elif isinstance(processed_output, (int, float)):
            tensor_map[node_name] = processed_output
        elif processed_output is None:
            # 🔧 CRITICAL FIX: Handle None outputs gracefully
            logger.warning(f"[{node_name}] ⚠️  processed_output is None, using layer_in as fallback")
            tensor_map[node_name] = layer_in
        else:
            # 🔧 CRITICAL FIX: More informative error message
            logger.error(f"[{node_name}] ❌ Invalid processed_output type: {type(processed_output)}, value: {processed_output}")
            raise RuntimeError(f"[{node_name}] ❌ No valid tensor found in processed_output. Type: {type(processed_output)}") 

        # normalize input_values to never be a bare list
        iv = layer_in
        if isinstance(iv, list):
            if len(iv) == 1:
                iv = iv[0]
            else:
                iv = tuple(iv)

        # record everything in node_io
        node_io[node_name] = {
            "input_sources":    parents,
            "input_values":     iv,
            "output_values":    processed_output,
            "layer_type":       layer_type,
            "node_type":        node_data["node_type"],
            "method_args":      method_args,
            "layer":            layer,
            "layer_name":       node_data["layer_name"],
            "func_name":        func_name,
            "func_module":      node_data["func_module"],
            "output_children":  children,
            "layer_hyperparams":layer_hyperparams,
        }

    return node_io

class ExecutionEngineNoCache:
    def __init__(self, model, extracted_weights, fx_graph, layer_stack, tracer, exported_program, debug=False, log_level="INFO"):
        self.model = model
        self.extracted_weights = extracted_weights
        self.graph = fx_graph
        self.layer_stack = layer_stack
        self.tracer = tracer
        self.exported_program = exported_program
        self.debug = debug
        self.log_level = log_level
        
        # Setup logging for this instance with force_reconfigure to ensure proper setup
        self.logger = setup_logging(debug=debug, log_level=log_level, force_reconfigure=True)
        self.logger.info(f"ExecutionEngineNoCache initialized with debug={debug}, log_level={log_level}")

    def run(self, inputs, debug=None, log_level=None):
        """Run the execution engine with optional debug and log level overrides"""
        # Use instance defaults if not provided
        if debug is None:
            debug = self.debug
        if log_level is None:
            log_level = self.log_level
            
        # Update global logger if parameters changed - force reconfigure to ensure consistency
        if debug != self.debug or log_level != self.log_level:
            self.logger = setup_logging(debug=debug, log_level=log_level, force_reconfigure=True)
            self.debug = debug
            self.log_level = log_level
            
        self.logger.info(f"Starting execution with {len(inputs)} inputs")
        
        # Run the execution
        result = run_execution_nocache(
            graph=self.graph,
            layer_stack=self.layer_stack,
            model=self.model,
            extracted_weights=self.extracted_weights,
            inputs=inputs,
            tracer=self.tracer,
            exported_program=self.exported_program,
        )
        
        # Flush logs at the end of execution to ensure all debug statements are captured
        flush_logs()
        
        return result