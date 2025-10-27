"""
DL-Backtrace Pipeline

High-level interface for running DL-Backtrace on PyTorch models.
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from dl_backtrace.pytorch_backtrace.dlbacktrace.dlbacktrace import DLBacktraceFX

from .config import PipelineConfig
from .model_registry import ModelRegistry, ModelInfo


class DLBacktracePipeline:
    """
    High-level pipeline for running DL-Backtrace on PyTorch models.
    
    This pipeline handles:
    - Model loading and initialization
    - Input preprocessing
    - DL-Backtrace execution
    - Result saving and visualization
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the DL-Backtrace pipeline.
        
        Args:
            config: Pipeline configuration (model_name is mandatory)
        """
        self.config = config
        
        # Get model information from registry
        self.model_info: ModelInfo = ModelRegistry.get_model_info(config.model_name)
        
        # Initialize internal state
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.dlbt: Optional[DLBacktraceFX] = None
        self.results: Dict[str, Any] = {}
        
        # Create output directory
        if self.config.save_results:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model(self) -> Tuple[nn.Module, Any]:
        """
        Load model and tokenizer based on configuration.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"🔄 Loading model: {self.model_info.base_model_path}")
        
        try:
            # Import transformers if needed
            from transformers import AutoTokenizer, AutoModel
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_info.base_model_path,
                **self.config.tokenizer_kwargs
            )
            
            # Set pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model using appropriate class
            model_class = ModelRegistry.get_model_class(self.model_info.model_type)
            
            # Build model kwargs
            model_kwargs = {
                "torch_dtype": torch.float32,
                "use_cache": False,
                **self.config.model_kwargs
            }
            
            print(f"📦 Model type: {self.model_info.model_type}")
            print(f"📦 Using class: {model_class.__name__}")
            
            self.model = model_class.from_pretrained(
                self.model_info.base_model_path,
                **model_kwargs
            )
            
            self.model.eval()
            self.model.requires_grad_(False)
            
            # Set up tokenizer padding
            if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
                if hasattr(self.tokenizer, 'eos_token'):
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Model and tokenizer loaded successfully")
            return self.model, self.tokenizer
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def prepare_inputs(self, text: List[str]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model.
        
        Args:
            text: List of input text strings
        
        Returns:
            Dictionary of input tensors
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        
        # Tokenize inputs
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        return tokens
    
    def initialize_dlbt(self, sample_inputs: Dict[str, torch.Tensor]):
        """
        Initialize DL-Backtrace FX.
        
        Args:
            sample_inputs: Sample inputs for graph tracing
        """
        print("🔄 Initializing DL-Backtrace FX...")
        
        # Prepare input tuple for DL-Backtrace
        input_args = tuple(sample_inputs.values())
        
        # Set up dynamic shapes if supported
        dynamic_shapes = self.config.dynamic_shapes
        if dynamic_shapes is None and self.model_info.supports_dynamic_shapes:
            # Auto-generate dynamic shapes
            dynamic_shapes = {}
            for key, value in sample_inputs.items():
                shape = list(value.shape)
                # Make batch and sequence dimensions dynamic
                dynamic_shape_dict = {}
                if len(shape) > 0:
                    dynamic_shape_dict[0] = None  # Dynamic batch size
                if len(shape) > 1:
                    dynamic_shape_dict[1] = None  # Dynamic sequence length
                if dynamic_shape_dict:
                    dynamic_shapes[key] = dynamic_shape_dict
        
        self.dlbt = DLBacktraceFX(
            model=self.model,
            input_for_graph=input_args,
            dynamic_shapes=dynamic_shapes,
            device=self.config.device,
            verbose=self.config.verbose,
            strict_cpu=self.config.strict_cpu
        )
        
        print("✅ DL-Backtrace FX initialized")
    
    def run(self, input_text: List[str]) -> Dict[str, Any]:
        """
        Run the full DL-Backtrace pipeline.
        
        Args:
            input_text: List of input text strings
        
        Returns:
            Dictionary containing results
        """
        start_time = time.time()
        
        # Step 1: Load model
        self.model, self.tokenizer = self.load_model()
        
        # Step 2: Prepare inputs
        print("🔄 Preparing inputs...")
        inputs = self.prepare_inputs(input_text)
        
        # Step 3: Initialize DL-Backtrace
        self.initialize_dlbt(inputs)
        
        # Step 4: Run prediction (forward pass)
        print("🔄 Running forward pass...")
        predict_start = time.time()
        node_io = self.dlbt.predict(*inputs.values(), temperature=self.config.temperature, debug=self.config.debug)
        predict_time = time.time() - predict_start
        
        print(f"✅ Forward pass completed in {predict_time:.2f}s")
        
        # Step 5: Run evaluation (relevance propagation)
        print("🔄 Running relevance propagation...")
        eval_start = time.time()
        all_wt = self.dlbt.evaluation(
            mode=self.config.mode,
            start_wt=self.config.start_wt,
            multiplier=self.config.multiplier,
            scaler=self.config.scaler,
            thresholding=self.config.thresholding,
            task=self.config.task,
            debug=self.config.debug
        )
        eval_time = time.time() - eval_start
        
        print(f"✅ Relevance propagation completed in {eval_time:.2f}s")
        
        # Step 6: Collect results
        total_time = time.time() - start_time
        
        self.results = {
            "model_name": self.config.model_name,
            "model_info": {
                "base_model_path": self.model_info.base_model_path,
                "model_type": self.model_info.model_type,
                "description": self.model_info.description,
            },
            "inputs": input_text,
            "input_shape": {k: list(v.shape) for k, v in inputs.items()},
            "timing": {
                "total_time": round(total_time, 2),
                "predict_time": round(predict_time, 2),
                "eval_time": round(eval_time, 2),
            },
            "node_io": self._serialize_node_io(node_io),
            "relevance": self._serialize_relevance(all_wt),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Step 7: Save results if configured
        if self.config.save_results:
            self._save_results()
        
        # Step 8: Visualize if configured
        if self.config.save_visualization:
            self._visualize()
        
        print(f"✅ Pipeline completed in {total_time:.2f}s")
        
        return self.results
    
    def _serialize_node_io(self, node_io: Dict) -> Dict:
        """Serialize node I/O data for saving"""
        serialized = {}
        for node_name, node_data in node_io.items():
            serialized[node_name] = {
                "layer_type": node_data.get("layer_type"),
                "output_shape": str(node_data.get("output_values", {}).shape) if hasattr(node_data.get("output_values"), "shape") else "N/A"
            }
        return serialized
    
    def _serialize_relevance(self, all_wt: Dict) -> Dict:
        """Serialize relevance values for saving"""
        serialized = {}
        for key, value in all_wt.items():
            if isinstance(value, (list, tuple)):
                serialized[key] = [f"shape: {v.shape}, sum: {np.sum(v):.4f}" for v in value if hasattr(v, "shape")]
            elif hasattr(value, "shape"):
                serialized[key] = f"shape: {value.shape}, sum: {np.sum(value):.4f}"
            else:
                serialized[key] = str(value)
        return serialized
    
    def _save_results(self):
        """Save results to disk"""
        # Save JSON results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.model_name}_{timestamp}.json"
        output_path = os.path.join(self.config.output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
    
    def _visualize(self):
        """Generate visualizations"""
        if self.dlbt is None:
            print("⚠️ Cannot visualize: DL-Backtrace not initialized")
            return
        
        try:
            # Save graph visualization
            graph_path = os.path.join(self.config.output_dir, f"{self.config.model_name}_graph.png")
            self.dlbt.visualize(save_path=graph_path)
            print(f"📊 Graph visualization saved to: {graph_path}")
            
            # Save relevance visualization
            if hasattr(self.dlbt, 'all_wt'):
                relevance_path = os.path.join(self.config.output_dir, f"{self.config.model_name}_relevance")
                self.dlbt.visualize_dlbacktrace(output_path=relevance_path)
                print(f"📊 Relevance visualization saved to: {relevance_path}")
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
    
    def print_relevance_summary(self):
        """Print summary of relevance values"""
        if self.dlbt is None:
            print("⚠️ DL-Backtrace not initialized")
            return
        
        print("\n" + "="*60)
        print("RELEVANCE SUMMARY")
        print("="*60)
        self.dlbt.print_all_relevance_info()
        print("="*60)
    
    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]) -> 'DLBacktracePipeline':
        """Create pipeline from configuration dictionary"""
        config = PipelineConfig.from_dict(config_dict)
        return cls(config)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DLBacktracePipeline':
        """Create pipeline from YAML configuration file"""
        import yaml
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_config_dict(config_dict)

