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
from typing import Dict, Any, List, Optional, Tuple, Union
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
        Load model and processor/tokenizer based on configuration.
        
        Returns:
            Tuple of (model, processor/tokenizer)
        """
        print(f"🔄 Loading model: {self.model_info.base_model_path}")
        print(f"📦 Model type: {self.model_info.model_type}")
        print(f"📦 Modality: {self.model_info.modality}")
        
        try:
            # Handle torchvision models
            if ModelRegistry.is_torchvision_model(self.model_info.base_model_path):
                return self._load_torchvision_model()
            
            # Handle HuggingFace models
            elif self.model_info.modality == "text":
                return self._load_text_model()
            
            elif self.model_info.modality == "image":
                return self._load_image_model()
            
            else:
                raise ValueError(f"Unsupported modality: {self.model_info.modality}")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"💡 Suggestion: Check if model '{self.model_info.base_model_path}' exists and is accessible")
            print(f"💡 Available models in registry: {list(ModelRegistry.MODELS.keys())}")
            raise
    
    def _load_torchvision_model(self) -> Tuple[nn.Module, None]:
        """Load a torchvision model"""
        print(f"📦 Loading torchvision model...")
        
        self.model = ModelRegistry.load_torchvision_model(
            self.model_info.base_model_path,
            num_classes=self.model_info.num_classes or 1000,
            **self.config.model_kwargs
        )
        
        # Move to correct device
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        
        self.model.eval()
        self.model.requires_grad_(False)
        
        # No tokenizer for image models
        self.tokenizer = None
        
        print("✅ Torchvision model loaded successfully")
        print(f"📊 Model device: {next(self.model.parameters()).device}")
        print(f"📊 Model dtype: {next(self.model.parameters()).dtype}")
        
        return self.model, None
    
    def _load_text_model(self) -> Tuple[nn.Module, Any]:
        """Load a text model with tokenizer"""
        print(f"📦 Loading text model and tokenizer...")
        
        # Load tokenizer
        processor_class = ModelRegistry.get_processor_class(self.model_info)
        try:
            self.tokenizer = processor_class.from_pretrained(
                self.model_info.base_model_path,
                **self.config.tokenizer_kwargs
            )
        except Exception as e:
            print(f"⚠️ Tokenizer loading failed: {e}")
            print("🔄 Trying with trust_remote_code=True...")
            self.tokenizer = processor_class.from_pretrained(
                self.model_info.base_model_path,
                trust_remote_code=True,
                **self.config.tokenizer_kwargs
            )
        
        # Set pad token if missing
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"📝 Set pad_token to eos_token: {self.tokenizer.eos_token}")
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print("📝 Added [PAD] as pad_token")
        
        # Load model
        model_class = ModelRegistry.get_model_class(self.model_info.model_type)
        model_kwargs = {
            "torch_dtype": torch.float32,
            "use_cache": False,
            **self.config.model_kwargs
        }
        
        # Handle device placement for large models
        if self.config.device == "cuda" and torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        
        try:
            self.model = model_class.from_pretrained(
                self.model_info.base_model_path,
                **model_kwargs
            )
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
            print("🔄 Trying with trust_remote_code=True...")
            model_kwargs["trust_remote_code"] = True
            self.model = model_class.from_pretrained(
                self.model_info.base_model_path,
                **model_kwargs
            )
        
        # Move to correct device if needed
        if self.config.device == "cuda" and torch.cuda.is_available():
            if not next(self.model.parameters()).is_cuda:
                self.model = self.model.cuda()
        elif self.config.device == "cpu":
            if next(self.model.parameters()).is_cuda:
                self.model = self.model.cpu()
        
        self.model.eval()
        self.model.requires_grad_(False)
        
        # Resize token embeddings if tokenizer was modified
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
            if len(self.tokenizer) != self.model.config.vocab_size:
                print(f"📝 Resizing token embeddings: {self.model.config.vocab_size} -> {len(self.tokenizer)}")
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        print("✅ Text model and tokenizer loaded successfully")
        print(f"📊 Model device: {next(self.model.parameters()).device}")
        print(f"📊 Model dtype: {next(self.model.parameters()).dtype}")
        print(f"📊 Vocab size: {len(self.tokenizer)}")
        
        return self.model, self.tokenizer
    
    def _load_image_model(self) -> Tuple[nn.Module, Any]:
        """Load an image model with processor"""
        print(f"📦 Loading image model and processor...")
        
        # Load processor
        processor_class = ModelRegistry.get_processor_class(self.model_info)
        try:
            self.tokenizer = processor_class.from_pretrained(
                self.model_info.base_model_path,
                **self.config.processor_kwargs
            )
        except Exception as e:
            print(f"⚠️ Processor loading failed: {e}")
            print("🔄 Trying with trust_remote_code=True...")
            self.tokenizer = processor_class.from_pretrained(
                self.model_info.base_model_path,
                trust_remote_code=True,
                **self.config.processor_kwargs
            )
        
        # Load model
        model_class = ModelRegistry.get_model_class(self.model_info.model_type)
        model_kwargs = {
            "torch_dtype": torch.float32,
            **self.config.model_kwargs
        }
        
        try:
            self.model = model_class.from_pretrained(
                self.model_info.base_model_path,
                **model_kwargs
            )
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
            print("🔄 Trying with trust_remote_code=True...")
            model_kwargs["trust_remote_code"] = True
            self.model = model_class.from_pretrained(
                self.model_info.base_model_path,
                **model_kwargs
            )
        
        # Move to correct device
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        
        self.model.eval()
        self.model.requires_grad_(False)
        
        print("✅ Image model and processor loaded successfully")
        print(f"📊 Model device: {next(self.model.parameters()).device}")
        print(f"📊 Model dtype: {next(self.model.parameters()).dtype}")
        
        return self.model, self.tokenizer
    
    def prepare_inputs(self, inputs: Union[List[str], List[Any], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model based on modality.
        
        Args:
            inputs: Input data (text strings, images, or tensors)
        
        Returns:
            Dictionary of input tensors
        """
        if self.model_info.modality == "text":
            return self._prepare_text_inputs(inputs)
        elif self.model_info.modality == "image":
            return self._prepare_image_inputs(inputs)
        else:
            raise ValueError(f"Unsupported modality: {self.model_info.modality}")
    
    def _prepare_text_inputs(self, text_inputs: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare text inputs for text models"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        
        # Tokenize inputs
        tokens = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # Move to correct device
        device = next(self.model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        return tokens
    
    def _prepare_image_inputs(self, image_inputs: Union[List[Any], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare image inputs for image models"""
        import torch
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Handle different input types
        if isinstance(image_inputs, torch.Tensor):
            # Already a tensor
            processed_images = image_inputs
        elif isinstance(image_inputs, list):
            # List of images (PIL Images, numpy arrays, or file paths)
            processed_images = []
            
            # Define transforms for torchvision models
            if ModelRegistry.is_torchvision_model(self.model_info.base_model_path):
                transform = transforms.Compose([
                    transforms.Resize(self.config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                for img in image_inputs:
                    if isinstance(img, str):
                        # File path
                        img = Image.open(img).convert('RGB')
                    elif isinstance(img, np.ndarray):
                        # Numpy array
                        img = Image.fromarray(img)
                    
                    if not isinstance(img, Image.Image):
                        raise ValueError(f"Unsupported image type: {type(img)}")
                    
                    processed_images.append(transform(img))
                
                processed_images = torch.stack(processed_images)
            
            else:
                # HuggingFace model - use processor
                if self.tokenizer is None:
                    raise ValueError("Image processor not loaded. Call load_model() first.")
                
                processed = self.tokenizer(
                    images=image_inputs,
                    return_tensors="pt",
                    **self.config.processor_kwargs
                )
                processed_images = processed['pixel_values']
        
        else:
            raise ValueError(f"Unsupported image input type: {type(image_inputs)}")
        
        # Move to correct device
        device = next(self.model.parameters()).device
        processed_images = processed_images.to(device)
        
        # Return in expected format
        if ModelRegistry.is_torchvision_model(self.model_info.base_model_path):
            return {"input": processed_images}
        else:
            return {"pixel_values": processed_images}
    
    def initialize_dlbt(self, sample_inputs: Dict[str, torch.Tensor]):
        """
        Initialize DL-Backtrace FX.
        
        Args:
            sample_inputs: Sample inputs for graph tracing
        """
        print("🔄 Initializing DL-Backtrace FX...")
        
        # Prepare input tuple for DL-Backtrace
        # Handle different input formats
        if ModelRegistry.is_torchvision_model(self.model_info.base_model_path):
            # For torchvision models, use the tensor directly
            input_args = (sample_inputs["input"],)
        else:
            # For HuggingFace models, use all input values
            input_args = tuple(sample_inputs.values())
        
        # Set up dynamic shapes if supported
        dynamic_shapes = self.config.dynamic_shapes
        if dynamic_shapes is None and self.model_info.supports_dynamic_shapes:
            # Auto-generate dynamic shapes for text models
            if self.model_info.modality == "text":
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
        
        # Initialize DL-Backtrace FX with device-based layer implementation
        # The device parameter now internally handles layer implementation selection
        self.dlbt = DLBacktraceFX(
            model=self.model,
            input_for_graph=input_args,
            dynamic_shapes=dynamic_shapes,
            device=self.config.device,
            verbose=self.config.verbose,
            strict_cpu=self.config.strict_cpu
        )
        
        print("✅ DL-Backtrace FX initialized")
    
    def run(self, inputs: Union[List[str], List[Any], torch.Tensor], labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the full DL-Backtrace pipeline.
        
        Args:
            inputs: Input data (text strings, images, or tensors)
            labels: Ground truth labels for classification tasks (optional)
        
        Returns:
            Dictionary containing results
        """
        start_time = time.time()
        
        # Step 1: Load model
        self.model, self.tokenizer = self.load_model()
        
        # Step 2: Prepare inputs
        print("🔄 Preparing inputs...")
        processed_inputs = self.prepare_inputs(inputs)
        
        # Step 3: Initialize DL-Backtrace
        self.initialize_dlbt(processed_inputs)
        
        # Step 4: Run task-specific pipeline
        if self.model_info.model_type in ["text_classification", "image_classification"]:
            return self._run_classification_pipeline(processed_inputs, inputs, labels, start_time)
        elif self.model_info.model_type == "generation":
            return self._run_generation_pipeline(processed_inputs, inputs, start_time)
        else:
            raise ValueError(f"Unsupported model type: {self.model_info.model_type}")
    
    def _run_classification_pipeline(self, processed_inputs: Dict[str, torch.Tensor], 
                                   raw_inputs: Union[List[str], List[Any]], 
                                   labels: Optional[List[str]], 
                                   start_time: float) -> Dict[str, Any]:
        """Run classification pipeline (text or image)"""
        
        # Step 4: Run prediction (forward pass)
        print("🔄 Running forward pass...")
        predict_start = time.time()
        
        try:
            node_io = self.dlbt.predict(*processed_inputs.values(), debug=self.config.debug)
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            print("🔄 Attempting with fallback strategies...")
            raise e
        
        predict_time = time.time() - predict_start
        print(f"✅ Forward pass completed in {predict_time:.2f}s")
        
        # Step 5: Run evaluation (relevance propagation)
        print("🔄 Running relevance propagation...")
        eval_start = time.time()
        
        # Use labels if provided for targeted relevance
        target_labels = labels or self.config.labels
        
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
                "modality": self.model_info.modality,
                "description": self.model_info.description,
            },
            "task_type": "classification",
            "inputs": self._serialize_inputs(raw_inputs),
            "labels": target_labels,
            "input_shape": {k: list(v.shape) for k, v in processed_inputs.items()},
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
        
        print(f"✅ Classification pipeline completed in {total_time:.2f}s")
        
        return self.results
    
    def _run_generation_pipeline(self, processed_inputs: Dict[str, torch.Tensor], 
                               raw_inputs: Union[List[str], List[Any]], 
                               start_time: float) -> Dict[str, Any]:
        """Run generation pipeline"""
        
        # Step 4: Run generation with DL-Backtrace
        print("🔄 Running text generation...")
        predict_start = time.time()
        
        # Extract input_ids for generation
        if "input_ids" in processed_inputs:
            input_ids = processed_inputs["input_ids"]
        else:
            raise ValueError("Generation requires input_ids")
        
        # Generate text using DL-Backtrace auto sampler
        try:
            generation_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
                "num_beams": self.config.num_beams,
                "num_return_sequences": self.config.num_return_sequences,
                "early_stopping": self.config.early_stopping,
                "return_scores": self.config.return_scores,
                "debug": self.config.debug
            }
            
            generated_ids = self.dlbt.sample_auto(
                tokenizer=self.tokenizer,
                input_ids=input_ids,
                **generation_kwargs
            )
            
            # Decode generated text
            if isinstance(generated_ids, tuple):
                # If return_scores=True, unpack
                generated_ids, scores = generated_ids
            else:
                scores = None
            
            generated_texts = []
            for seq in generated_ids:
                text = self.tokenizer.decode(seq, skip_special_tokens=True)
                generated_texts.append(text)
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            raise e
        
        predict_time = time.time() - predict_start
        print(f"✅ Generation completed in {predict_time:.2f}s")
        
        # Step 5: Run relevance propagation if requested
        eval_time = 0
        all_wt = None
        
        if self.config.return_relevance:
            print("🔄 Running relevance propagation...")
            eval_start = time.time()
            
            # Run prediction first to get node_io
            node_io = self.dlbt.predict(*processed_inputs.values(), debug=self.config.debug)
            
            all_wt = self.dlbt.evaluation(
                mode=self.config.mode,
                start_wt=self.config.start_wt,
                multiplier=self.config.multiplier,
                scaler=self.config.scaler,
                thresholding=self.config.thresholding,
                task="generation",
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
                "modality": self.model_info.modality,
                "description": self.model_info.description,
            },
            "task_type": "generation",
            "inputs": self._serialize_inputs(raw_inputs),
            "generated_texts": generated_texts,
            "generation_scores": scores.tolist() if scores is not None else None,
            "generation_config": generation_kwargs,
            "input_shape": {k: list(v.shape) for k, v in processed_inputs.items()},
            "timing": {
                "total_time": round(total_time, 2),
                "predict_time": round(predict_time, 2),
                "eval_time": round(eval_time, 2),
            },
            "relevance": self._serialize_relevance(all_wt) if all_wt else None,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Step 7: Save results if configured
        if self.config.save_results:
            self._save_results()
        
        # Step 8: Visualize if configured
        if self.config.save_visualization:
            self._visualize()
        
        print(f"✅ Generation pipeline completed in {total_time:.2f}s")
        
        return self.results
    
    def _serialize_inputs(self, inputs: Union[List[str], List[Any]]) -> List[str]:
        """Serialize inputs for saving"""
        if isinstance(inputs, list) and len(inputs) > 0:
            if isinstance(inputs[0], str):
                return inputs
            else:
                # For images or other non-string inputs
                return [f"<{type(inp).__name__}>" for inp in inputs]
        return ["<unknown>"]
    
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
    
    def diagnose_compatibility(self) -> Dict[str, Any]:
        """
        Diagnose model compatibility with DL-Backtrace.
        
        Returns:
            Dictionary containing compatibility analysis
        """
        if self.dlbt is None:
            print("⚠️ DL-Backtrace not initialized. Call load_model() and initialize_dlbt() first.")
            return {}
        
        return self.dlbt.diagnose_model_compatibility()
    
    def debug_execution_differences(self, input_text: List[str]) -> Dict[str, Any]:
        """
        Debug differences between direct model execution and DL-Backtrace execution.
        
        Args:
            input_text: List of input text strings
            
        Returns:
            Dictionary containing debug information
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded. Call load_model() first.")
        
        if self.dlbt is None:
            raise ValueError("DL-Backtrace not initialized. Call initialize_dlbt() first.")
        
        # Prepare inputs
        inputs = self.prepare_inputs(input_text)
        input_args = tuple(inputs.values())
        
        return self.dlbt.debug_execution_differences(*input_args)
    
    def generate_text(self, input_text: str, **generation_kwargs) -> str:
        """
        Generate text using DL-Backtrace auto sampling.
        
        Args:
            input_text: Input prompt text
            **generation_kwargs: Generation parameters (temp, top_k, top_p, max_new_tokens, etc.)
            
        Returns:
            Generated text string
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded. Call load_model() first.")
        
        if self.dlbt is None:
            raise ValueError("DL-Backtrace not initialized. Call initialize_dlbt() first.")
        
        # Check if model supports generation
        if self.model_info.model_type not in ["causal_lm"]:
            raise ValueError(f"Text generation not supported for model type: {self.model_info.model_type}")
        
        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        # Set default generation parameters
        default_kwargs = {
            "max_new_tokens": 50,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9,
            "debug": self.config.debug
        }
        default_kwargs.update(generation_kwargs)
        
        # Generate using DL-Backtrace
        try:
            generated_ids = self.dlbt.sample_auto(
                tokenizer=self.tokenizer,
                input_ids=input_ids,
                **default_kwargs
            )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Return only the newly generated part
            input_length = len(input_text)
            return generated_text[input_length:].strip()
            
        except Exception as e:
            print(f"❌ Text generation failed: {e}")
            raise
    
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
    
    @classmethod
    def create_simple(cls, model_name: str, device: str = "cpu", **kwargs) -> 'DLBacktracePipeline':
        """
        Create a simple pipeline with minimal configuration.
        
        Args:
            model_name: Name of the model (must be in registry)
            device: Device to use ("cpu" or "cuda")
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured DLBacktracePipeline instance
        """
        config = PipelineConfig(
            model_name=model_name,
            device=device,
            **kwargs
        )
        return cls(config)
    
    def run_text_classification(self, texts: List[str], labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run text classification analysis.
        
        Args:
            texts: List of input text strings
            labels: Optional ground truth labels
            
        Returns:
            Dictionary containing classification results
        """
        if self.model_info.modality != "text" or self.model_info.model_type != "text_classification":
            raise ValueError(f"Model {self.config.model_name} is not a text classification model")
        
        print(f"🚀 Running text classification on {len(texts)} samples")
        return self.run(texts, labels)
    
    def run_image_classification(self, images: List[Any], labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run image classification analysis.
        
        Args:
            images: List of images (PIL Images, numpy arrays, or file paths)
            labels: Optional ground truth labels
            
        Returns:
            Dictionary containing classification results
        """
        if self.model_info.modality != "image" or self.model_info.model_type != "image_classification":
            raise ValueError(f"Model {self.config.model_name} is not an image classification model")
        
        print(f"🚀 Running image classification on {len(images)} samples")
        return self.run(images, labels)
    
    def run_text_generation(self, prompts: List[str], **generation_kwargs) -> Dict[str, Any]:
        """
        Run text generation analysis.
        
        Args:
            prompts: List of input prompt strings
            **generation_kwargs: Generation parameters (overrides config)
            
        Returns:
            Dictionary containing generation results
        """
        if self.model_info.modality != "text" or self.model_info.model_type != "generation":
            raise ValueError(f"Model {self.config.model_name} is not a text generation model")
        
        # Update config with generation kwargs
        for key, value in generation_kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        print(f"🚀 Running text generation on {len(prompts)} prompts")
        return self.run(prompts)
    
    def run_simple_analysis(self, inputs: Union[str, Any], label: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a simple end-to-end analysis with a single input.
        
        Args:
            inputs: Single input (text string, image, etc.)
            label: Optional ground truth label
            
        Returns:
            Dictionary containing analysis results
        """
        # Convert single input to list
        if isinstance(inputs, str):
            input_list = [inputs]
            print(f"🚀 Running simple analysis on: '{inputs[:50]}{'...' if len(inputs) > 50 else ''}'")
        else:
            input_list = [inputs]
            print(f"🚀 Running simple analysis on: {type(inputs).__name__}")
        
        labels = [label] if label else None
        
        # Run the full pipeline
        results = self.run(input_list, labels)
        
        # Print summary
        print(f"\n📊 Analysis Summary:")
        print(f"   Model: {self.config.model_name}")
        print(f"   Model type: {self.model_info.model_type}")
        print(f"   Modality: {self.model_info.modality}")
        print(f"   Total time: {results['timing']['total_time']}s")
        print(f"   Prediction time: {results['timing']['predict_time']}s")
        print(f"   Evaluation time: {results['timing']['eval_time']}s")
        
        # Print task-specific results
        if results.get('task_type') == 'generation':
            print(f"   Generated texts: {len(results.get('generated_texts', []))}")
            if results.get('generated_texts'):
                print(f"   Sample output: '{results['generated_texts'][0][:100]}{'...' if len(results['generated_texts'][0]) > 100 else ''}'")
        
        # Print relevance summary if available
        if hasattr(self, 'dlbt') and hasattr(self.dlbt, 'all_wt'):
            print(f"\n🔍 Relevance Analysis:")
            self.print_relevance_summary()
        
        return results

