"""
Configuration for DL-Backtrace Pipeline
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class TaskType(Enum):
    """Supported task types"""
    BINARY_CLASSIFICATION = "binary-classification"
    MULTI_CLASS_CLASSIFICATION = "multi-class-classification"
    REGRESSION = "regression"
    GENERATION = "generation"


class RelevanceMode(Enum):
    """Relevance propagation modes"""
    DEFAULT = "default"
    CONSERVATIVE = "conservative"
    LIBERAL = "liberal"


@dataclass
class PipelineConfig:
    """Configuration for DL-Backtrace Pipeline"""
    
    # Required
    model_name: str
    
    # Device configuration
    device: str = "cpu"  # "cpu" or "cuda"
    
    # Model configuration
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # DL-Backtrace configuration
    dynamic_shapes: Optional[Dict] = None
    verbose: bool = False
    strict_cpu: bool = True
    
    # Input configuration
    batch_size: int = 1
    max_length: int = 512
    
    # Evaluation configuration
    mode: str = "default"
    start_wt: List = field(default_factory=list)
    multiplier: float = 100.0
    scaler: float = 1.0
    thresholding: float = 0.5
    task: str = "binary-classification"
    
    # Temperature scaling
    temperature: float = 1.0
    
    # Debug configuration
    debug: bool = False
    
    # Output configuration
    save_results: bool = True
    output_dir: str = "backtrace_results"
    save_timing: bool = True
    save_relevance: bool = True
    save_visualization: bool = False
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.model_name:
            raise ValueError("model_name is mandatory")
        
        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"device must be 'cpu' or 'cuda', got: {self.device}")
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")
        
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary"""
        model_name = config_dict.pop('model_name')
        return cls(model_name=model_name, **config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'model_kwargs': self.model_kwargs,
            'tokenizer_kwargs': self.tokenizer_kwargs,
            'dynamic_shapes': self.dynamic_shapes,
            'verbose': self.verbose,
            'strict_cpu': self.strict_cpu,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'mode': self.mode,
            'start_wt': self.start_wt,
            'multiplier': self.multiplier,
            'scaler': self.scaler,
            'thresholding': self.thresholding,
            'task': self.task,
            'temperature': self.temperature,
            'debug': self.debug,
            'save_results': self.save_results,
            'output_dir': self.output_dir,
            'save_timing': self.save_timing,
            'save_relevance': self.save_relevance,
            'save_visualization': self.save_visualization,
        }

