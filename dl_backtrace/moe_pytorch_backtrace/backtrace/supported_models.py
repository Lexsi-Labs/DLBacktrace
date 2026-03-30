"""
Supported MoE model registry for DLBacktrace.

Single source of truth for all supported Mixture-of-Experts architectures.
Used for auto-detection, validation, and user documentation.
"""

SUPPORTED_MOE_MODELS = {
    "olmoe": {
        "class_names": ["OlmoeForCausalLM"],
        "example_ids": ["allenai/OLMoE-1B-7B-0125-Instruct"],
        "description": "OLMoE (Open Language Model with Mixture of Experts)",
        "attn_attr": "self_attn",
    },
    "jetmoe": {
        "class_names": ["JetMoEForCausalLM"],
        "example_ids": ["jetmoe/jetmoe-8b"],
        "description": "JetMoE",
        "attn_attr": "self_attention",
    },
    "gpt_oss": {
        "class_names": ["GptOssForCausalLM", "GPTOSSForCausalLM"],
        "example_ids": ["openai/gpt-oss-20b"],
        "description": "GPT-OSS (Open-Source MoE by OpenAI)",
        "attn_attr": "self_attn",
    },
    "qwen3_moe": {
        "class_names": ["Qwen3MoeForCausalLM", "Qwen2MoeForCausalLM"],
        "example_ids": ["Qwen/Qwen3-30B-A3B"],
        "description": "Qwen3-MoE / Qwen2-MoE",
        "attn_attr": "self_attn",
    },
}

# Build reverse lookup: class_name → model_type
_CLASS_TO_TYPE = {}
for _mtype, _info in SUPPORTED_MOE_MODELS.items():
    for _cls in _info["class_names"]:
        _CLASS_TO_TYPE[_cls] = _mtype


def detect_model_type(model) -> str:
    """Auto-detect MoE model type from a HuggingFace model instance.
    
    Inspects the model's class hierarchy (including wrapper classes) to find
    a matching architecture in the supported models registry.
    
    Args:
        model: A HuggingFace model instance (can be wrapped in nn.Module).
    
    Returns:
        The model type string (e.g., 'olmoe', 'jetmoe', 'gpt_oss', 'qwen3_moe').
    
    Raises:
        ValueError: If the model architecture is not recognized.
    """
    # Check the model and all nested .model attributes (handles wrappers)
    candidate = model
    checked_classes = []
    for _ in range(5):  # max depth to prevent infinite loops
        cls_name = type(candidate).__name__
        checked_classes.append(cls_name)
        if cls_name in _CLASS_TO_TYPE:
            return _CLASS_TO_TYPE[cls_name]
        # Try unwrapping common wrapper patterns
        if hasattr(candidate, 'model'):
            candidate = candidate.model
        elif hasattr(candidate, 'module'):  # DataParallel
            candidate = candidate.module
        else:
            break
    
    supported_list = "\n".join(
        f"  - {mtype}: {info['description']} ({', '.join(info['class_names'])})"
        for mtype, info in SUPPORTED_MOE_MODELS.items()
    )
    raise ValueError(
        f"Could not auto-detect MoE model type.\n"
        f"Checked classes: {checked_classes}\n\n"
        f"Supported architectures:\n{supported_list}\n\n"
        f"You can also pass model_type='...' explicitly."
    )


def list_supported_models() -> str:
    """Return a formatted string listing all supported MoE models."""
    lines = ["Supported MoE Models:", "=" * 50]
    for mtype, info in SUPPORTED_MOE_MODELS.items():
        lines.append(f"\n  {mtype}")
        lines.append(f"    Description : {info['description']}")
        lines.append(f"    Classes     : {', '.join(info['class_names'])}")
        lines.append(f"    Example IDs : {', '.join(info['example_ids'])}")
    return "\n".join(lines)
