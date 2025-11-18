"""
Model utilities for handling wrapped and direct HuggingFace models.
"""

def unwrap_model(model, max_depth=8):
    """
    Unwrap a model to find the underlying HuggingFace transformer model.
    
    Walks the `.model` chain until it finds a model with the expected
    HuggingFace structure (has .layers, .embed_tokens, etc.).
    
    Args:
        model: The model to unwrap (wrapper or direct HuggingFace model)
        max_depth: Maximum depth to search (default: 8)
    
    Returns:
        tuple: (core_model, lm_head)
            - core_model: The transformer model with .layers, .embed_tokens, etc.
            - lm_head: The language model head (or None if not found)
    
    Raises:
        AttributeError: If no valid HuggingFace model structure is found
    
    Examples:
        # Direct HuggingFace model
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        core, lm_head = unwrap_model(model)
        # core = model.model, lm_head = model.lm_head
        
        # Wrapped model
        class MyWrapper(nn.Module):
            def __init__(self):
                self.model = AutoModelForCausalLM.from_pretrained(...)
            def forward(self, ...): ...
        
        wrapped = MyWrapper()
        core, lm_head = unwrap_model(wrapped)
        # core = wrapped.model.model, lm_head = wrapped.model.lm_head
    """
    obj = model
    seen = set()
    
    for depth in range(max_depth):
        # Check if this object has the HuggingFace transformer structure
        if _is_transformer_core(obj):
            # Found the core transformer model
            # Now find the lm_head by walking back up
            lm_head = _find_lm_head(model, obj)
            return obj, lm_head
        
        # Prevent infinite loops
        obj_id = id(obj)
        if obj_id in seen:
            break
        seen.add(obj_id)
        
        # Try to unwrap by accessing .model attribute
        if hasattr(obj, "model"):
            obj = obj.model
        else:
            break
    
    raise AttributeError(
        f"Could not find HuggingFace transformer structure in {type(model).__name__}. "
        f"Expected attributes: .layers, .embed_tokens, .norm. "
        f"Searched {depth + 1} levels deep."
    )


def _is_transformer_core(obj):
    """
    Check if an object has the HuggingFace transformer core structure.
    
    A valid transformer core should have:
    - .layers (list of transformer layers)
    - .embed_tokens (embedding layer)
    - .norm (final layer norm)
    """
    return (
        hasattr(obj, "layers") and 
        hasattr(obj, "embed_tokens") and
        hasattr(obj, "norm")
    )


def _find_lm_head(root_model, core_model):
    """
    Find the lm_head by searching from root_model down to core_model.
    
    The lm_head is typically at the same level as the core model,
    or one level up from it.
    
    Args:
        root_model: The original model (possibly wrapped)
        core_model: The unwrapped core transformer model
    
    Returns:
        The lm_head module, or None if not found
    """
    # Strategy 1: Check if root_model has lm_head
    if hasattr(root_model, "lm_head"):
        return root_model.lm_head
    
    # Strategy 2: Walk the .model chain and check each level
    obj = root_model
    seen = set()
    for _ in range(8):
        if id(obj) in seen:
            break
        seen.add(id(obj))
        
        if hasattr(obj, "lm_head"):
            return obj.lm_head
        
        # Stop if we've reached the core model
        if obj is core_model:
            break
        
        if hasattr(obj, "model"):
            obj = obj.model
        else:
            break
    
    # Strategy 3: Check for alternative names
    for attr_name in ["output", "lm_head", "head", "classifier"]:
        if hasattr(root_model, attr_name):
            return getattr(root_model, attr_name)
    
    return None


def get_model_structure_info(model):
    """
    Get information about the model structure for debugging.
    
    Returns:
        dict: Information about the model structure
    """
    try:
        core, lm_head = unwrap_model(model)
        return {
            "success": True,
            "core_type": type(core).__name__,
            "lm_head_type": type(lm_head).__name__ if lm_head else None,
            "num_layers": len(core.layers) if hasattr(core, "layers") else None,
            "has_embed_tokens": hasattr(core, "embed_tokens"),
            "has_norm": hasattr(core, "norm"),
        }
    except AttributeError as e:
        return {
            "success": False,
            "error": str(e),
            "model_type": type(model).__name__,
            "has_model_attr": hasattr(model, "model"),
        }