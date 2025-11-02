"""
Model Registry for DL-Backtrace Pipeline

Provides model-specific configurations and metadata.
"""

from typing import Dict, Any, Optional, Tuple, Type
from dataclasses import dataclass
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForImageClassification
)
import torchvision.models as tv_models


@dataclass
class ModelInfo:
    """Information about a registered model"""
    
    model_name: str
    model_type: str  # "text_classification", "image_classification", "causal_lm", "generation"
    base_model_path: str
    modality: str = "text"  # "text", "image", "multimodal"
    supports_dynamic_shapes: bool = False
    default_batch_size: int = 1
    default_max_length: int = 512
    default_image_size: Tuple[int, int] = (224, 224)
    num_classes: Optional[int] = None
    custom_model_class: Optional[Type] = None
    custom_forward_signature: Optional[str] = None
    description: str = ""


class ModelRegistry:
    """Registry of models supported by DL-Backtrace"""
    
    # Pre-defined model configurations
    MODELS: Dict[str, ModelInfo] = {
        # ===== IMAGE CLASSIFICATION MODELS =====
        "vgg": ModelInfo(
            model_name="vgg",
            model_type="image_classification",
            base_model_path="torchvision://vgg16",
            modality="image",
            default_batch_size=4,
            default_image_size=(224, 224),
            num_classes=1000,
            description="VGG-16 for image classification"
        ),
        
        "resnet": ModelInfo(
            model_name="resnet",
            model_type="image_classification",
            base_model_path="torchvision://resnet50",
            modality="image",
            default_batch_size=4,
            default_image_size=(224, 224),
            num_classes=1000,
            description="ResNet-50 for image classification"
        ),
        
        "vit": ModelInfo(
            model_name="vit",
            model_type="image_classification",
            base_model_path="google/vit-base-patch16-224",
            modality="image",
            default_batch_size=4,
            default_image_size=(224, 224),
            num_classes=1000,
            description="Vision Transformer (ViT) base model for image classification"
        ),
        
        "densenet": ModelInfo(
            model_name="densenet",
            model_type="image_classification",
            base_model_path="torchvision://densenet121",
            modality="image",
            default_batch_size=4,
            default_image_size=(224, 224),
            num_classes=1000,
            description="DenseNet-121 for image classification"
        ),
        
        "efficientnet": ModelInfo(
            model_name="efficientnet",
            model_type="image_classification",
            base_model_path="torchvision://efficientnet_b0",
            modality="image",
            default_batch_size=4,
            default_image_size=(224, 224),
            num_classes=1000,
            description="EfficientNet-B0 for image classification"
        ),
        
        "mobilenet": ModelInfo(
            model_name="mobilenet",
            model_type="image_classification",
            base_model_path="torchvision://mobilenet_v2",
            modality="image",
            default_batch_size=4,
            default_image_size=(224, 224),
            num_classes=1000,
            description="MobileNet-V2 for image classification"
        ),
        
        # ===== TEXT CLASSIFICATION MODELS =====
        "bert-base": ModelInfo(
            model_name="bert-base",
            model_type="text_classification",
            base_model_path="bert-base-uncased",
            modality="text",
            default_batch_size=2,
            default_max_length=512,
            num_classes=2,
            description="BERT base model for text classification"
        ),
        
        "albert": ModelInfo(
            model_name="albert",
            model_type="text_classification",
            base_model_path="albert-base-v2",
            modality="text",
            default_batch_size=2,
            default_max_length=512,
            num_classes=2,
            description="ALBERT base model for text classification"
        ),
        
        "roberta": ModelInfo(
            model_name="roberta",
            model_type="text_classification",
            base_model_path="roberta-base",
            modality="text",
            default_batch_size=2,
            default_max_length=512,
            num_classes=2,
            description="RoBERTa base model for text classification"
        ),
        
        "distilbert": ModelInfo(
            model_name="distilbert",
            model_type="text_classification",
            base_model_path="distilbert-base-uncased",
            modality="text",
            default_batch_size=2,
            default_max_length=512,
            num_classes=2,
            description="DistilBERT base model for text classification"
        ),
        
        "electra": ModelInfo(
            model_name="electra",
            model_type="text_classification",
            base_model_path="google/electra-base-discriminator",
            modality="text",
            default_batch_size=2,
            default_max_length=512,
            num_classes=2,
            description="ELECTRA base model for text classification"
        ),
        
        "xlnet": ModelInfo(
            model_name="xlnet",
            model_type="text_classification",
            base_model_path="xlnet-base-cased",
            modality="text",
            default_batch_size=2,
            default_max_length=512,
            num_classes=2,
            description="XLNet base model for text classification"
        ),
        
        # ===== GENERATION MODELS =====
        "llama3.2-1b": ModelInfo(
            model_name="llama3.2-1b",
            model_type="generation",
            base_model_path="meta-llama/Llama-3.2-1B",
            modality="text",
            supports_dynamic_shapes=True,
            default_batch_size=1,
            default_max_length=512,
            description="Meta Llama 3.2 1B model for text generation"
        ),
        
        "llama3.2-3b": ModelInfo(
            model_name="llama3.2-3b",
            model_type="generation",
            base_model_path="meta-llama/Llama-3.2-3B",
            modality="text",
            supports_dynamic_shapes=True,
            default_batch_size=1,
            default_max_length=512,
            description="Meta Llama 3.2 3B model for text generation"
        ),
        
        "qwen3-0.6b": ModelInfo(
            model_name="qwen3-0.6b",
            model_type="generation",
            base_model_path="Qwen/Qwen2.5-0.5B",
            modality="text",
            supports_dynamic_shapes=True,
            default_batch_size=1,
            default_max_length=512,
            description="Qwen3 0.6B model for text generation"
        ),
        
        "qwen3-1.7b": ModelInfo(
            model_name="qwen3-1.7b",
            model_type="generation",
            base_model_path="Qwen/Qwen2.5-1.5B",
            modality="text",
            supports_dynamic_shapes=True,
            default_batch_size=1,
            default_max_length=512,
            description="Qwen3 1.7B model for text generation"
        ),
        
        "qwen3-4b": ModelInfo(
            model_name="qwen3-4b",
            model_type="generation",
            base_model_path="Qwen/Qwen2.5-3B",
            modality="text",
            supports_dynamic_shapes=True,
            default_batch_size=1,
            default_max_length=512,
            description="Qwen3 4B model for text generation"
        ),
        
        "qwen3-8b": ModelInfo(
            model_name="qwen3-8b",
            model_type="generation",
            base_model_path="Qwen/Qwen2.5-7B",
            modality="text",
            supports_dynamic_shapes=True,
            default_batch_size=1,
            default_max_length=512,
            description="Qwen3 8B model for text generation"
        ),
        
        "qwen3-14b": ModelInfo(
            model_name="qwen3-14b",
            model_type="generation",
            base_model_path="Qwen/Qwen2.5-14B",
            modality="text",
            supports_dynamic_shapes=True,
            default_batch_size=1,
            default_max_length=512,
            description="Qwen3 14B model for text generation"
        ),
        
        "qwen3-32b": ModelInfo(
            model_name="qwen3-32b",
            model_type="generation",
            base_model_path="Qwen/Qwen2.5-32B",
            modality="text",
            supports_dynamic_shapes=True,
            default_batch_size=1,
            default_max_length=512,
            description="Qwen3 32B model for text generation"
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
            "text_classification": AutoModelForSequenceClassification,
            "image_classification": AutoModelForImageClassification,
            "generation": AutoModelForCausalLM,
            # Legacy support
            "classification": AutoModelForSequenceClassification,
            "causal_lm": AutoModelForCausalLM,
        }
        
        if model_type not in mapping:
            raise ValueError(f"Unknown model type: {model_type}. Supported types: {list(mapping.keys())}")
        
        return mapping[model_type]
    
    @classmethod
    def get_processor_class(cls, model_info: ModelInfo):
        """Get the appropriate processor/tokenizer class for a model"""
        if model_info.modality == "text":
            return AutoTokenizer
        elif model_info.modality == "image":
            return AutoImageProcessor
        else:
            raise ValueError(f"Unsupported modality: {model_info.modality}")
    
    @classmethod
    def is_torchvision_model(cls, base_model_path: str) -> bool:
        """Check if model is from torchvision"""
        return base_model_path.startswith("torchvision://")
    
    @classmethod
    def load_torchvision_model(cls, base_model_path: str, num_classes: int = 1000, **kwargs):
        """Load a torchvision model"""
        model_name = base_model_path.replace("torchvision://", "")
        
        # Map model names to torchvision functions
        model_mapping = {
            "densenet121": tv_models.densenet121,
            "resnet50": tv_models.resnet50,
            "vgg16": tv_models.vgg16,
            "efficientnet_b0": tv_models.efficientnet_b0,
            "mobilenet_v2": tv_models.mobilenet_v2,
        }
        
        if model_name not in model_mapping:
            raise ValueError(f"Unsupported torchvision model: {model_name}")
        
        model_fn = model_mapping[model_name]
        
        # Load with pretrained weights
        model = model_fn(pretrained=True, **kwargs)
        
        # Modify classifier for custom number of classes if needed
        if num_classes != 1000:
            if hasattr(model, 'classifier'):
                if hasattr(model.classifier, 'in_features'):
                    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                elif isinstance(model.classifier, nn.Sequential):
                    # For models like VGG
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            elif hasattr(model, 'fc'):
                # For ResNet models
                model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        return model


def get_model_info(model_name: str) -> ModelInfo:
    """Convenience function to get model information"""
    return ModelRegistry.get_model_info(model_name)


def register_custom_model(
    model_name: str,
    base_model_path: str,
    model_type: str = "causal_lm",
    **kwargs
) -> None:
    """
    Register a custom model in the registry.
    
    Args:
        model_name: Unique name for the model
        base_model_path: HuggingFace model path or local path
        model_type: Type of model ("causal_lm", "classification", etc.)
        **kwargs: Additional ModelInfo parameters
    """
    info = ModelInfo(
        model_name=model_name,
        model_type=model_type,
        base_model_path=base_model_path,
        **kwargs
    )
    ModelRegistry.register_model(info)
    print(f"✅ Registered custom model: {model_name} -> {base_model_path}")

