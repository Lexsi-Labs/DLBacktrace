"""
Model Registry for DL-Backtrace Pipeline

Provides model-specific configurations and metadata.
"""

from typing import Dict, Any, Optional, Tuple, Type
from dataclasses import dataclass
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelInfo:
    """Information about a registered model"""
    
    model_name: str
    model_type: str  # "classification", "causal_lm", "masked_lm", "seq2seq"
    base_model_path: str
    supports_dynamic_shapes: bool = False
    default_batch_size: int = 1
    default_max_length: int = 512
    custom_model_class: Optional[Type] = None
    custom_forward_signature: Optional[str] = None
    description: str = ""


class ModelRegistry:
    """Registry of models supported by DL-Backtrace"""
    
    # Pre-defined model configurations
    MODELS: Dict[str, ModelInfo] = {
        # Classification models
        "bert-sst2": ModelInfo(
            model_name="bert-sst2",
            model_type="classification",
            base_model_path="textattack/bert-base-uncased-SST-2",
            default_batch_size=2,
            default_max_length=128,
            description="BERT model fine-tuned on SST-2 sentiment classification"
        ),
        
        "roberta-base": ModelInfo(
            model_name="roberta-base",
            model_type="classification",
            base_model_path="roberta-base",
            default_batch_size=1,
            default_max_length=512,
            description="RoBERTa base model"
        ),
        
        "distilbert": ModelInfo(
            model_name="distilbert",
            model_type="classification",
            base_model_path="distilbert-base-uncased",
            default_batch_size=1,
            default_max_length=512,
            description="DistilBERT model"
        ),
        
        # Causal LM models
        "llama-3.2-1b": ModelInfo(
            model_name="llama-3.2-1b",
            model_type="causal_lm",
            base_model_path="meta-llama/Llama-3.2-1B",
            supports_dynamic_shapes=True,
            default_batch_size=1,
            default_max_length=256,
            description="Meta Llama 3.2 1B model"
        ),
        
        "phi-4": ModelInfo(
            model_name="phi-4",
            model_type="causal_lm",
            base_model_path="microsoft/phi-4",
            supports_dynamic_shapes=True,
            default_batch_size=1,
            default_max_length=512,
            description="Microsoft Phi-4 model"
        ),
        
        "gpt2": ModelInfo(
            model_name="gpt2",
            model_type="causal_lm",
            base_model_path="gpt2",
            default_batch_size=1,
            default_max_length=256,
            description="GPT-2 base model"
        ),
    }
    
    @classmethod
    def register_model(cls, info: ModelInfo):
        """Register a new model"""
        cls.MODELS[info.model_name] = info
    
    @classmethod
    def get_model_info(cls, model_name: str) -> ModelInfo:
        """Get information for a specific model"""
        if model_name not in cls.MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in registry. "
                f"Available models: {list(cls.MODELS.keys())}"
            )
        return cls.MODELS[model_name]
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """List all registered models"""
        return {name: info.description for name, info in cls.MODELS.items()}
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Type:
        """Get the appropriate model class for a given type"""
        mapping = {
            "classification": AutoModelForSequenceClassification,
            "causal_lm": AutoModelForCausalLM,
            "masked_lm": AutoModelForSequenceClassification,  # Can use sequence classification
            "seq2seq": AutoModelForSequenceClassification,  # Use sequence classification as default
        }
        
        if model_type not in mapping:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return mapping[model_type]


def get_model_info(model_name: str) -> ModelInfo:
    """Convenience function to get model information"""
    return ModelRegistry.get_model_info(model_name)

