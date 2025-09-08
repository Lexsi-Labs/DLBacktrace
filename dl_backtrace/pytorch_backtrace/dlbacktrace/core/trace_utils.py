# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/trace_utils.py

import logging
from typing import Dict, List, Optional, Any

# Set up logger for this module
logger = logging.getLogger(__name__)

def extract_placeholders(exported_program) -> List[str]:
    """
    Extract placeholder names from exported program graph signature.
    
    Args:
        exported_program: The exported PyTorch program
        
    Returns:
        List of placeholder names starting with 'p_', 'b_', or 'c_'
        
    Raises:
        ValueError: If exported_program is None or missing required attributes
    """
    if exported_program is None:
        raise ValueError("exported_program cannot be None")
    
    if not hasattr(exported_program, 'graph_signature'):
        raise ValueError("exported_program missing graph_signature attribute")
    
    if not hasattr(exported_program.graph_signature, 'input_specs'):
        raise ValueError("exported_program.graph_signature missing input_specs attribute")
    
    fx_placeholders = [
        spec.arg.name for spec in exported_program.graph_signature.input_specs
        if hasattr(spec.arg, 'name') and (
            spec.arg.name.startswith("p_") or 
            spec.arg.name.startswith("b_") or
            spec.arg.name.startswith("c_")
        )
    ]
    
    logger.debug(f"Extracted {len(fx_placeholders)} placeholders: {fx_placeholders}")
    return fx_placeholders  


def map_placeholders_to_state_dict(exported_program, model) -> Dict[str, str]:
    """
    Map placeholder names to their corresponding state dict keys.
    
    Args:
        exported_program: The exported PyTorch program
        model: The PyTorch model
        
    Returns:
        Dictionary mapping placeholder names to state dict keys
        
    Raises:
        ValueError: If exported_program or model is None
    """
    if exported_program is None:
        raise ValueError("exported_program cannot be None")
    
    if model is None:
        raise ValueError("model cannot be None")
    
    mapping = {}
    state_dict_keys = set(model.state_dict().keys())
    buffer_keys = set(dict(model.named_buffers()).keys())

    for spec in exported_program.graph_signature.input_specs:
        if hasattr(spec.arg, 'name'):  # Ensure `spec.arg` has a valid name
            placeholder_name = spec.arg.name
            target_name = spec.target  # Direct mapping from `InputSpec`
            # 🔧 FIX: Include 'c_' prefix to match extract_placeholders()
            if (placeholder_name.startswith("p_") or 
                placeholder_name.startswith("b_") or 
                placeholder_name.startswith("c_")):
                if target_name in state_dict_keys or target_name in buffer_keys:
                    mapping[placeholder_name] = target_name  # Direct match
                    logger.debug(f"Mapped placeholder {placeholder_name} -> {target_name}")
    
    logger.debug(f"Created {len(mapping)} placeholder mappings")
    return mapping


def get_weight_from_placeholder(placeholder_name: str, exported_program, model, 
                               placeholder_to_real_name: Dict[str, str]) -> Optional[Any]:
    """
    Retrieve weight tensor from placeholder name.
    
    Args:
        placeholder_name: The placeholder name to look up
        exported_program: The exported PyTorch program
        model: The PyTorch model
        placeholder_to_real_name: Mapping from placeholder to real names
        
    Returns:
        The weight tensor if found, None otherwise
    """
    if placeholder_name is None:
        logger.warning("placeholder_name is None")
        return None
        
    real_key = placeholder_to_real_name.get(placeholder_name)

    if real_key is None:
        logger.warning(f"No mapping found for placeholder {placeholder_name}")
        return None

    # 🔧 OPTIMIZATION: Cache these expensive operations
    buffers = dict(model.named_buffers())
    exported_state_dict = exported_program.state_dict
    full_state_dict = model.state_dict()

    if real_key in exported_state_dict:
        logger.debug(f"Found {placeholder_name} in exported_state_dict")
        return exported_state_dict[real_key]
    elif real_key in full_state_dict:
        logger.debug(f"Found {placeholder_name} in full_state_dict")
        return full_state_dict[real_key]
    elif real_key in buffers:
        logger.debug(f"Found {placeholder_name} in buffers")
        return buffers[real_key]
    else:
        logger.error(f"{placeholder_name} (mapped to {real_key}) not found in any state_dict")
        return None
