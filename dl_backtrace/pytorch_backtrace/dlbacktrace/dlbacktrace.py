# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/dlbacktrace.py

from .core.graph_builder import build_graph
from .core.execution_engine_noncache import ExecutionEngineNoCache
from .core.trace_utils import (
    extract_placeholders,
    map_placeholders_to_state_dict,
    get_weight_from_placeholder
)
from .core.config import activation_master
from .core.dlb_auto_sampler import DLBAutoSampler
from .core.relevance_propagation import RelevancePropagator
from .core.visualization import visualize_graph, visualize_relevance, visualize_relevance_auto 
from .core.token_relevance_visuals import (
    plot_tokenwise_relevance_map_swapped,
    plot_input_heatmap_for_token,
)

import numpy as np 
import torch
import inspect

class DLBacktrace:
    def __init__(self, model, input_for_graph, dynamic_shapes=None, device="cpu", verbose=False, strict_cpu=True):
        """
        Initialize DL-Backtrace FX for model tracing and explainability.
        
        Args:
            model: PyTorch model to trace
            input_for_graph: Sample input for graph tracing
            dynamic_shapes: Optional dynamic shape constraints
            device (str): Compute device for layer implementations. Options:
                - "cpu": Use optimized refactored implementations for all layers
                - "cuda": Use CUDA-accelerated implementations where available:
                  * Linear: CUDA kernel
                  * Embedding: CUDA kernel
                  * Attention: CUDA kernel
                  * Conv2D: CUDA kernel
                  * And many more...
            verbose (bool): Enable verbose initialization logs.
            strict_cpu (bool): When running on CPU, disable MKL-DNN and pin threads for stricter determinism.
        """
        # 🔧 CRITICAL: Set up deterministic environment for consistent tracing
        print("Setting up DL-Bactrace", flush=True)
        self.verbose = verbose
        self.strict_cpu = strict_cpu
        self._setup_deterministic_environment(seed=42, verbose=self.verbose, strict_cpu=self.strict_cpu)
        
        self.model = model
        if self.verbose:
            print("---------------------------v1------------------------------------------", flush=True)
            print("🔧 Initializing DL-Backtrace FX...", flush=True)
        self.input_for_graph = input_for_graph
        # Normalize sample inputs to a tuple for exporter compatibility
        if isinstance(self.input_for_graph, torch.Tensor):
            self.input_for_graph = (self.input_for_graph,)
        elif not isinstance(self.input_for_graph, (tuple, list)):
            self.input_for_graph = (self.input_for_graph,)
        self.dynamic_shapes = dynamic_shapes
        self.model.eval()
        self.model.requires_grad_(False)
        # Parse device configuration and create layer implementation mapping
        self.layer_implementation = self._parse_device_config(device)
        
        if self.verbose:
            print(f"🚀 Layer Implementation Configuration:", flush=True)
            if isinstance(self.layer_implementation, str):
                print(f"   Global: {self.layer_implementation.upper()}", flush=True)
            else:
                print(f"   Layer-specific configuration:", flush=True)
                for layer_type, impl in self.layer_implementation.items():
                    print(f"     {layer_type}: {impl.upper()}", flush=True)
        
        # Cache manager removed - using non-cache execution only
        if self.verbose:
            print("---------------------------v2------------------------------------------", flush=True)
            print("🔧 Exporting and tracing model...", flush=True)
        # Export and trace model
        self._trace_model()
        if self.verbose:
            print("---------------------------v3------------------------------------------", flush=True)
            print("🔧 Extracting placeholders...", flush=True)
        # Placeholder mapping
        self.fx_placeholders = extract_placeholders(self.exported_program)
        if self.verbose:
            print("---------------------------v4------------------------------------------", flush=True)
            print("🔧 Mapping placeholders to state dict...", flush=True)
        self.placeholder_to_real_name = map_placeholders_to_state_dict(
            self.exported_program, self.model
        )
        if self.verbose:
            print("---------------------------v5------------------------------------------", flush=True)
            print("🔧 Extracting weights...", flush=True)
        self.extracted_weights = {
            p: get_weight_from_placeholder(
                p, self.exported_program, self.model, self.placeholder_to_real_name
            ) for p in self.fx_placeholders
        }
        if self.verbose:
            print("---------------------------v6------------------------------------------", flush=True)
            print("🔧 Building computation graph...", flush=True)

        # Graph + metadata
        self.graph, self.layer_stack = build_graph(
            self.tracer, self.extracted_weights
        )
        if self.verbose:
            print("---------------------------v7------------------------------------------", flush=True)
            print("🔧 Initializing I/O and bookkeeping...", flush=True)
        # I/O and bookkeeping
        self.node_io = {}
        self.activation_dict = {}
        if self.verbose:
            print("---------------------------v8------------------------------------------", flush=True)
            print("✅ DL-Backtrace FX initialization complete!", flush=True)
    
    def _parse_device_config(self, device):
        """
        Parse device configuration and create layer-specific implementation mapping.
        
        Args:
            device (str): Either "cpu" or "cuda"
            
        Returns:
            dict: Layer-specific implementation mapping
        """
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"device must be 'cpu' or 'cuda', got: {device}")
        
        if device == "cpu":
            # CPU mode: use refactored implementations for all layers
            config = {
                "linear": "original",
                "conv2d": "refactored",
                "attention": "original",
                "embedding": "refactored",
                "wt_add_equal": "refactored",
                "wt_mul": "refactored",
                "default": "refactored"
            }
            
        else:  # device == "cuda"
            # Check CUDA availability
            if not torch.cuda.is_available():
                print("⚠️  CUDA device requested but CUDA not available. Falling back to CPU mode.")
                return self._parse_device_config("cpu")
            
            # CUDA mode: use CUDA implementations where available
            config = {
                "linear": "cuda",
                "embedding": "cuda",
                "attention": "cuda",
                "conv2d": "cuda",  
                "wt_add_equal": "refactored",  
                "wt_mul": "refactored",
                "default": "refactored"
            }
        
        return config
    
    def get_layer_implementation(self, layer_type):
        """Get the implementation to use for a specific layer type."""
        # Map common layer names to our configuration keys
        layer_mapping = {
            "MLP_Layer": "linear",
            "Linear": "linear", 
            "DL_Layer": "conv2d",
            "Conv2D": "conv2d",
            "Attention": "attention",
            "NLP_Embedding": "embedding",
            "Mathematical_Operation_mul": "wt_mul",
            "Mathematical_Operation_add": "wt_add_equal"
        }
        
        mapped_type = layer_mapping.get(layer_type, layer_type.lower())
        return self.layer_implementation.get(mapped_type, self.layer_implementation["default"])

    def _trace_model(self):
        """Export model deterministically using the reproducibility module."""
        from dl_backtrace.pytorch_backtrace.dlbacktrace.core.reproducibility import export_model_deterministically
        
        try:
            # Use the deterministic export function
            self.exported_program = export_model_deterministically(
                model=self.model,
                sample_inputs=self.input_for_graph,
                dynamic_shapes=self.dynamic_shapes,
                seed=42
            )
            self.tracer = self.exported_program.graph_module
        except Exception as e:
            # Fallback: Try with different export strategies for complex models
            print(f"⚠️ Primary export failed: {e}")
            print("🔄 Trying alternative export strategies...")
            
            self.exported_program = self._fallback_export()
            self.tracer = self.exported_program.graph_module

    def _fallback_export(self):
        """Fallback export strategies for complex models like Llama/RoBERTa"""
        import torch
        from torch.export import export
        
        # Strategy 1: Try with stricter constraints
        try:
            print("📋 Strategy 1: Strict mode with no dynamic shapes")
            return export(
                self.model,
                args=self.input_for_graph,
                strict=True,
                preserve_module_call_signature=(),
            )
        except Exception as e1:
            print(f"❌ Strategy 1 failed: {e1}")
        
        # Strategy 2: Try with relaxed constraints
        try:
            print("📋 Strategy 2: Non-strict mode")
            return export(
                self.model,
                args=self.input_for_graph,
                strict=False,
            )
        except Exception as e2:
            print(f"❌ Strategy 2 failed: {e2}")
        
        # Strategy 3: Try with explicit dynamic shapes for transformers
        try:
            print("📋 Strategy 3: Transformer-specific dynamic shapes")
            # Common dynamic shapes for transformer models
            if self.dynamic_shapes is None:
                # Infer from input shapes
                if isinstance(self.input_for_graph[0], torch.Tensor):
                    batch_size, seq_len = self.input_for_graph[0].shape[:2]
                    # Create dynamic shapes for typical transformer inputs
                    self.dynamic_shapes = {
                        "input_ids": {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq_len")},
                        "attention_mask": {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq_len")},
                    }
                    
            return export(
                self.model,
                args=self.input_for_graph,
                dynamic_shapes=self.dynamic_shapes,
                strict=True,
            )
        except Exception as e3:
            print(f"❌ Strategy 3 failed: {e3}")
        
        # Strategy 4: Last resort - FX tracing
        try:
            print("📋 Strategy 4: FX symbolic tracing (last resort)")
            import torch.fx as fx
            
            # Create a wrapper to handle complex forward signatures
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, *args):
                    return self.model(*args)
            
            wrapped_model = ModelWrapper(self.model)
            traced = fx.symbolic_trace(wrapped_model)
            
            # Create a mock ExportedProgram-like object
            class MockExportedProgram:
                def __init__(self, graph_module):
                    self.graph_module = graph_module
                    self.graph_signature = None  # Will be handled later
                    
            return MockExportedProgram(traced)
            
        except Exception as e4:
            print(f"❌ Strategy 4 failed: {e4}")
            raise RuntimeError(f"All export strategies failed. Last error: {e4}")

    def _infer_model_type(self):
        """Infer model type for better tracing strategies"""
        model_name = self.model.__class__.__name__.lower()
        
        if any(name in model_name for name in ['llama', 'llm', 'gpt', 'opt']):
            return 'causal_lm'
        elif any(name in model_name for name in ['bert', 'roberta', 'electra', 'distilbert']):
            return 'masked_lm'
        elif any(name in model_name for name in ['t5', 'bart', 'pegasus']):
            return 'seq2seq'
        else:
            return 'unknown'

    def apply_temperature_scaling(self, temperature: float = 1.0):
        """
        Apply temperature scaling to the output logits in-place.

        Args:
            temperature (float):
                > 1.0  -> softer / more uniform distribution
                < 1.0  -> sharper / more peaked distribution
                = 1.0  -> no change

        Returns:
            dict: Updated node_io with scaled logits.
        """
        print(f"Using Temperature Scaling: {temperature}")
        if not hasattr(self, "node_io") or "output" not in self.node_io:
            raise ValueError("No 'output' node found in self.node_io")

        if "output_values" not in self.node_io["output"]:
            raise ValueError("'output_values' not found in self.node_io['output']")

        logits = self.node_io["output"]["output_values"]
        if logits is None:
            raise ValueError("self.node_io['output']['output_values'] is None")

        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        if temperature == 1.0:
            return self.node_io  # no-op

        # Apply scaling in-place
        self.node_io["output"]["output_values"] = logits / temperature

        return self.node_io 

    def predict(self, *inputs, temperature: float = 1.0, debug=None):
        """
        Execute the model with the given inputs and return node I/O data.
        
        Args:
            *inputs: Input tensors for the model
            debug (bool | None): Enable debug logs. If None, defaults to self.verbose.
            
        Returns:
            dict: Node I/O data containing execution results
        """
        if debug is None:
            debug = bool(getattr(self, "verbose", False))
        
        # 🔧 ENHANCED: Better input preprocessing for complex models
        processed_inputs = self._preprocess_inputs(inputs, debug)
        
        if debug:
            print(f"🔧 DLB Predict: Debug mode enabled")
            print(f"   Original input count: {len(inputs)}")
            print(f"   Processed input count: {len(processed_inputs)}")
            for i, inp in enumerate(processed_inputs):
                if isinstance(inp, torch.Tensor):
                    print(f"   Input {i}: {inp.shape} on {inp.device}, dtype: {inp.dtype}")
                else:
                    print(f"   Input {i}: {type(inp)}")
            print(f"   Model type: {self._infer_model_type()}")
            print(f"   Execution engine: Non-cache (ExecutionEngineNoCache)")
            print(f"   Layer stack length: {len(self.layer_stack)}")
        
        # Always use non-cache execution engine
        executor = ExecutionEngineNoCache(
            model=self.model,
            extracted_weights=self.extracted_weights,
            fx_graph=self.graph,
            layer_stack=self.layer_stack,
            tracer=self.tracer,
            exported_program=self.exported_program,
            debug=debug,
            log_level="DEBUG" if debug else "INFO"
        )
        
        if debug:
            print(f"🔧 Starting execution with {type(executor).__name__}")
        
        try:
            self.node_io = executor.run(processed_inputs, debug=debug)
        except Exception as e:
            if debug:
                print(f"❌ Execution failed: {e}")
                print("🔄 Attempting recovery strategies...")
            
            # Try recovery strategies
            self.node_io = self._attempt_execution_recovery(executor, processed_inputs, debug, e)
        
        if debug:
            print(f"🔧 Execution completed successfully")
            print(f"   Output nodes: {len(self.node_io)}")

        # 🔧 Apply temperature scaling if requested
        if temperature != 1.0:
            try:
                print(f"Calling temperature_scaling...")
                self.apply_temperature_scaling(temperature)
                if debug:
                    print(f"✅ Applied temperature scaling with temperature={temperature}")
            except Exception as te:
                if debug:
                    print(f"⚠️ Temperature scaling skipped due to error: {te}")
            
        return self.node_io

    def _preprocess_inputs(self, inputs, debug=False):
        """Preprocess inputs for better compatibility with complex models"""
        processed = []
        
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                # Ensure tensor is contiguous and detached
                processed_inp = inp.detach().contiguous()
                
                # Handle common transformer input patterns
                if inp.dtype == torch.float16:
                    # Convert half precision to float32 for better compatibility
                    processed_inp = processed_inp.to(torch.float32)
                    if debug:
                        print(f"   🔧 Converted input {i} from float16 to float32")
                
                # Ensure reasonable tensor shapes
                if processed_inp.dim() == 1 and processed_inp.shape[0] > 1:
                    # Add batch dimension if missing
                    processed_inp = processed_inp.unsqueeze(0)
                    if debug:
                        print(f"   🔧 Added batch dimension to input {i}: {inp.shape} -> {processed_inp.shape}")
                
                processed.append(processed_inp)
            else:
                processed.append(inp)
        
        return processed

    def _attempt_execution_recovery(self, executor, inputs, debug, original_error):
        """Attempt to recover from execution failures"""
        if debug:
            print("🔄 Trying recovery strategies:")
        
        # Recovery 1: Try with simplified inputs
        try:
            if debug:
                print("   📋 Strategy 1: Simplified inputs")
            
            simplified_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    # Ensure basic tensor properties
                    simple_inp = inp.clone().detach().requires_grad_(False)
                    if simple_inp.device.type != 'cpu':
                        simple_inp = simple_inp.cpu()
                    simplified_inputs.append(simple_inp)
                else:
                    simplified_inputs.append(inp)
            
            return executor.run(simplified_inputs, debug=debug)
            
        except Exception as e1:
            if debug:
                print(f"   ❌ Strategy 1 failed: {e1}")
        
        # Recovery 2: Try with smaller batch size
        try:
            if debug:
                print("   📋 Strategy 2: Smaller batch size")
            
            reduced_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor) and inp.shape[0] > 1:
                    # Take first sample only
                    reduced_inputs.append(inp[:1])
                else:
                    reduced_inputs.append(inp)
            
            return executor.run(reduced_inputs, debug=debug)
            
        except Exception as e2:
            if debug:
                print(f"   ❌ Strategy 2 failed: {e2}")
        
        # If all recovery attempts fail, raise the original error
        raise original_error

    def evaluation(self, mode="default", start_wt=[], multiplier=100.0, scaler=1.0, thresholding=0.5, task="binary-classification", target_token_ids=None, debug=False):
        evaluator = RelevancePropagator(
            graph=self.graph,
            node_io=self.node_io,
            activation_master=activation_master,
            get_layer_implementation=self.get_layer_implementation  # Pass the function instead of a static value
        )
        self.all_wt = evaluator.propagate(
            start_wt=start_wt,
            mode=mode,
            multiplier=multiplier,
            scaler=scaler,
            thresholding=thresholding,
            task=task,
            target_token_ids=target_token_ids,
            debug=debug
        )
        return self.all_wt

    def run_task(
        self,
        task="auto",
        inputs=None,
        tokenizer=None,
        mode="default",
        multiplier=100.0,
        scaler=1.0,
        thresholding=0.5,
        temperature=1.0,
        return_scores=False,
        return_relevance=False,
        return_layerwise_output=False,
        debug=False,
        **generation_kwargs
    ):
        """
        Unified method for running DL-Backtrace on different tasks.
        
        Args:
            task (str): Task type - "auto", "image-classification", "text-classification", or "generation"
                - "auto": Automatically detect task based on inputs
                - "image-classification": For image classification models (e.g., MobileNet, ResNet)
                - "text-classification": For text classification models (e.g., BERT sentiment)
                - "generation": For text generation models (e.g., GPT, LLaMA)
            
            inputs: Input data for the model
                - For image-classification: torch.Tensor of shape (B, C, H, W)
                - For text-classification: dict with 'input_ids' and 'attention_mask' or tuple of tensors
                - For generation: dict with 'input_ids' and 'attention_mask' or tuple of tensors
            
            tokenizer: Required for generation tasks (HuggingFace tokenizer)
            
            mode (str): Relevance propagation mode (default: "default")
            multiplier (float): Starting relevance value (default: 100.0)
            scaler (float): Relevance scaling factor (default: 1.0)
            thresholding (float): Relevance threshold (default: 0.5)
            temperature (float): Temperature for predict() call (default: 1.0)
            return_scores (bool): Return scores trace (default: False)
            return_relevance (bool): Return relevance trace (default: False)
            return_layerwise_output (bool): Return layer-wise output trace (default: False)
            debug (bool): Enable debug logging (default: False)
            
            **generation_kwargs: Additional kwargs for generation task (passed to sample_auto)
                - max_new_tokens, top_k, top_p, num_beams, etc.
        
        Returns:
            dict: Results containing:
                - 'task': Detected or specified task type
                - 'node_io': Layer-wise outputs from predict()
                - 'relevance': Relevance scores from evaluation()
                - 'predictions': Model predictions (logits for classification)
                - 'generated_ids': (generation only) Generated token IDs
                - 'scores_trace': (if return_scores=True) Scores trace
                - 'relevance_trace': (if return_relevance=True) Relevance trace
                - 'layerwise_output_trace': (if return_layerwise_output=True) Layer-wise output trace
        
        Examples:
            # Image classification
            results = dlb.run_task(
                task="image-classification",
                inputs=image_tensor
            )
            
            # Text classification with traces
            results = dlb.run_task(
                task="text-classification",
                inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
                return_relevance=True,
                return_layerwise_output=True
            )
            
            # Text generation with traces
            results = dlb.run_task(
                task="generation",
                inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
                tokenizer=tokenizer,
                max_new_tokens=50,
                temperature=0.7,
                return_relevance=True,
                return_scores=True
            )
        """
        
        # Auto-detect task if needed
        if task == "auto":
            task = self._detect_task(inputs, tokenizer)
            if debug:
                print(f"🔍 Auto-detected task: {task}")
        
        # Validate task type
        valid_tasks = ["image-classification", "text-classification", "generation"]
        if task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got: {task}")
        
        # Prepare inputs based on task
        if task == "generation":
            # Generation task - delegate to sample_auto
            if tokenizer is None:
                raise ValueError("tokenizer is required for generation task")
            
            # Extract input_ids and attention_mask
            if isinstance(inputs, dict):
                input_ids = inputs.get("input_ids")
                attention_mask = inputs.get("attention_mask")
            elif isinstance(inputs, (tuple, list)):
                input_ids = inputs[0]
                attention_mask = inputs[1] if len(inputs) > 1 else None
            else:
                raise ValueError("For generation, inputs must be dict or tuple of (input_ids, attention_mask)")
            
            if debug:
                print(f"🚀 Running generation task with sample_auto...")
            
            # Call sample_auto with generation kwargs and trace flags
            generated_output = self.sample_auto(
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_scores=return_scores,
                return_relevance=return_relevance,
                return_layerwise_output=return_layerwise_output,
                debug=debug,
                **generation_kwargs
            )
            
            # Parse output (can be tensor or tuple)
            result = {
                'task': task,
                'node_io': self.node_io if hasattr(self, 'node_io') else None,
                'relevance': self.all_wt if hasattr(self, 'all_wt') else None,
            }
            
            if isinstance(generated_output, tuple):
                # (generated_ids, info_dict)
                result['generated_ids'] = generated_output[0]
                info_dict = generated_output[1]
                if 'scores_trace' in info_dict:
                    result['scores_trace'] = info_dict['scores_trace']
                if 'relevance_trace' in info_dict:
                    result['relevance_trace'] = info_dict['relevance_trace']
                if 'layerwise_output_trace' in info_dict:
                    result['layerwise_output_trace'] = info_dict['layerwise_output_trace']
            else:
                # Just generated_ids
                result['generated_ids'] = generated_output
            
            return result
        
        else:
            # Classification tasks (image or text)
            if debug:
                print(f"🚀 Running {task} task...")
            
            # Prepare model inputs
            if task == "image-classification":
                # Image input: single tensor
                if isinstance(inputs, torch.Tensor):
                    model_inputs = (inputs,)
                elif isinstance(inputs, (tuple, list)):
                    model_inputs = inputs
                else:
                    raise ValueError("For image-classification, inputs must be a torch.Tensor or tuple")
                
                predict_inputs = model_inputs
                eval_task = "multi-class classification"
            
            elif task == "text-classification":
                # Text input: input_ids and attention_mask
                if isinstance(inputs, dict):
                    input_ids = inputs.get("input_ids")
                    attention_mask = inputs.get("attention_mask")
                    model_inputs = (input_ids, attention_mask)
                elif isinstance(inputs, (tuple, list)):
                    model_inputs = inputs
                else:
                    raise ValueError("For text-classification, inputs must be dict or tuple of (input_ids, attention_mask)")
                
                predict_inputs = model_inputs
                eval_task = "multi-class classification"
            
            # Step 1: Forward pass
            if debug:
                print(f"   📊 Running forward pass (predict)...")
            
            node_io = self.predict(*predict_inputs, temperature=temperature, debug=debug)
            
            # Extract predictions
            if "output" in node_io and "output_values" in node_io["output"]:
                predictions = node_io["output"]["output_values"]
            else:
                # Fallback: get last node's output
                last_node = list(node_io.keys())[-1]
                predictions = node_io[last_node].get("output_values")
            
            if debug:
                if isinstance(predictions, torch.Tensor):
                    print(f"   ✅ Predictions shape: {predictions.shape}")
                elif isinstance(predictions, np.ndarray):
                    print(f"   ✅ Predictions shape: {predictions.shape}")
            
            # Step 2: Relevance propagation
            if debug:
                print(f"   🔬 Running relevance propagation (evaluation)...")
            
            relevance = self.evaluation(
                mode=mode,
                start_wt=[],
                multiplier=multiplier,
                scaler=scaler,
                thresholding=thresholding,
                task=eval_task,
                target_token_ids=None,
                debug=debug
            )
            
            if debug:
                print(f"   ✅ Relevance computed for {len(relevance)} nodes")
            
            # Build result dict
            result = {
                'task': task,
                'node_io': node_io,
                'relevance': relevance,
                'predictions': predictions,
            }
            
            # Add traces if requested (for classification tasks, these are single-step)
            if return_scores:
                # For classification, scores are the logits/predictions
                result['scores_trace'] = [predictions]
            
            if return_relevance:
                # For classification, relevance trace is the computed relevance
                result['relevance_trace'] = [relevance]
            
            if return_layerwise_output:
                # For classification, layerwise output is the node_io
                result['layerwise_output_trace'] = [node_io]
            
            return result
    
    def _detect_task(self, inputs, tokenizer):
        """
        Auto-detect task type based on inputs and model characteristics.
        
        Returns:
            str: Detected task type
        """
        # If tokenizer provided and inputs have input_ids, likely text-based
        if tokenizer is not None:
            if isinstance(inputs, dict) and "input_ids" in inputs:
                # Check if model is generative (has generate method)
                if hasattr(self.model, "generate"):
                    return "generation"
                else:
                    return "text-classification"
            elif isinstance(inputs, (tuple, list)) and len(inputs) >= 2:
                # Assume (input_ids, attention_mask)
                if hasattr(self.model, "generate"):
                    return "generation"
                else:
                    return "text-classification"
        
        # If single tensor input, likely image classification
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 4:  # (B, C, H, W)
                return "image-classification"
            elif inputs.dim() == 2:  # Could be input_ids (B, seq_len)
                return "text-classification"
        
        # Default fallback
        return "image-classification"

    def sample_auto(self, tokenizer, input_ids, attention_mask=None, **kwargs):
        """
        Wrapper for DLB-based generation (greedy / sampling / beam).
        Accepts kwargs: temperature, top_k, top_p, max_new_tokens, min_new_tokens, max_time,
                        early_stopping, repetition_penalty, no_repeat_ngram_size,
                        bad_words_ids, bos_token_id, eos_token_id, pad_token_id,
                        num_beams, num_return_sequences, length_penalty, return_scores, debug

        Returns:
            - Always a single sequence with shape [1, T_total]
            (If return_scores=True on sampling path, returns (sequence, scores_trace))
            - Beam path returns the top-1 sequence (num_return_sequences is ignored on return).
        """

        # --- helpers ---
        def _pick_device():
            mdl = getattr(self, "model", None)
            dev = getattr(mdl, "device", None)
            if isinstance(dev, torch.device):
                return dev
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _to_long_tensor(x, device):
            if isinstance(x, torch.Tensor):
                t = x
            elif isinstance(x, (list, tuple)):
                t = torch.tensor(x)
            else:
                try:
                    import numpy as np  # noqa: F401
                    if isinstance(x, np.ndarray):
                        t = torch.from_numpy(x)
                    else:
                        raise TypeError
                except Exception:
                    raise TypeError(
                        f"input must be Tensor/list/tuple/ndarray; got {type(x).__name__}"
                    )
            if t.dim() == 1:
                t = t.unsqueeze(0)  # ensure [B=1, T]
            return t.to(device=device, dtype=torch.long, non_blocking=True)

        def _ensure_mask(mask, ids, pad_id, device):
            if mask is None:
                if pad_id is not None:
                    m = (ids != int(pad_id)).long()
                else:
                    m = torch.ones_like(ids, dtype=torch.long, device=device)
            else:
                m = _to_long_tensor(mask, device)
                if m.shape != ids.shape:
                    raise ValueError(
                        f"attention_mask shape {tuple(m.shape)} does not match input_ids shape {tuple(ids.shape)}"
                    )
            return m

        # --- device & core tensors ---
        device = _pick_device()
        input_ids = _to_long_tensor(input_ids, device)

        # Enforce B=1 (DLB export/engine assumes batch-static B=1)
        if input_ids.size(0) != 1:
            raise AssertionError("DLBAutoSampler currently assumes batch size = 1. Provide a single prompt.")

        # --- auto-fill token IDs from tokenizer if absent ---
        for kid in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if kwargs.get(kid) is None and hasattr(tokenizer, kid):
                v = getattr(tokenizer, kid)
                if v is not None:
                    kwargs[kid] = v

        pad_id = kwargs.get("pad_token_id", None)
        attention_mask = _ensure_mask(attention_mask, input_ids, pad_id, device)

        # --- sanitize knobs / lengths ---
        temperature = kwargs.get("temperature", None)
        if temperature is not None and float(temperature) <= 0.0:
            raise ValueError("temperature must be > 0 when provided")

        top_k = kwargs.get("top_k", None)
        if top_k is not None and int(top_k) < 0:
            raise ValueError("top_k must be >= 0")

        top_p = kwargs.get("top_p", None)
        if top_p is not None and not (0.0 < float(top_p) <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

        max_new_tokens = kwargs.get("max_new_tokens", 50)
        if int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        kwargs["max_new_tokens"] = int(max_new_tokens)

        mnt = kwargs.get("min_new_tokens", None)
        if mnt is not None:
            mnt = int(mnt)
            if mnt < 0:
                raise ValueError("min_new_tokens must be >= 0")
            if mnt > max_new_tokens:
                mnt = max_new_tokens
            kwargs["min_new_tokens"] = mnt

        # --- sanitize beams ---
        num_beams = int(kwargs.get("num_beams", 1))
        if num_beams < 1:
            raise ValueError("num_beams must be >= 1")
        kwargs["num_beams"] = num_beams

        # even though the generator returns top-1 for beams, keep HF internals happy:
        nret = int(kwargs.get("num_return_sequences", 1))
        if nret < 1:
            nret = 1
        if nret > num_beams:
            nret = num_beams
        kwargs["num_return_sequences"] = nret

        # --- normalize bad_words_ids to List[List[int]] ---
        bwi = kwargs.get("bad_words_ids", None)
        if bwi is not None:
            norm = []
            if isinstance(bwi, (list, tuple)):
                for seq in bwi:
                    if isinstance(seq, (list, tuple)):
                        norm.append([int(t) for t in seq])
                    else:
                        norm.append([int(seq)])
            else:
                norm.append([int(bwi)])
            kwargs["bad_words_ids"] = norm

        # --- create engine & dispatch ---
        eng = DLBAutoSampler(self, tokenizer)  # `self` is the DLB engine (has .model and .predict)
        return eng.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs) 

    def print_all_relevance_info(self):
        """ 
            print shape and sum of all relevance values in `self.all_wt`.
        """ 
        if not hasattr(self, "all_wt"):
            print("❌ self.all_wt not found. Run evaluation() first.")
            return

        for key, val in self.all_wt.items():
            if isinstance(val, (list, tuple)):
                for i, v in enumerate(val):
                    if hasattr(v, "shape") and hasattr(v, "sum"):
                        print(f"[{key}][{i}] shape: {v.shape}, sum: {np.sum(v):.4f}")
                    else:
                        print(f"[{key}][{i}] is not a NumPy array or tensor.")
            elif hasattr(val, "shape") and hasattr(val, "sum"):
                print(f"[{key}] shape: {val.shape}, sum: {np.sum(val):.4f}")
            else:
                print(f"[{key}] is not a NumPy array or tensor.")

    def visualize(self, save_path="graph.png"):
        visualize_graph(self.graph, save_path)

    def visualize_dlbacktrace(self, output_path="backtrace_graph", top_k=None, relevance_threshold=None, engine_auto_threshold=1500):
        visualize_relevance_auto(
            self.graph,
            self.all_wt,
            output_path=output_path,          # pretty path for small graphs
            node_threshold=500,
            engine_auto_threshold=engine_auto_threshold,
            fast_output_path="backtrace_collapsed_fast",  # path for large graphs
            show=True,                        # ⬅️ show in Colab
            inline_format="svg",              # or "png" if SVG too heavy
        )

    def visualize_tokenwise_relevance_map(
        self,
        timewise_relevance_out,     # list[dict], one dict per generated timestep
        input_ids,
        tokenizer,
        *,
        generated_ids=None,
        input_key="input_ids",
        figsize=(12, 6),
        **kwargs,
    ):
        """
        Thin wrapper that forwards to core.token_relevance_visuals.plot_tokenwise_relevance_map_swapped.
        x-axis: generated tokens (time), y-axis: input tokens. Normalized to [0,1].
        """
        return plot_tokenwise_relevance_map_swapped(
            timewise_relevance_out,
            input_ids,
            tokenizer,
            generated_ids=generated_ids,
            input_key=input_key,
            figsize=figsize,
            **kwargs,
        )

    def visualize_input_heatmap_for_token(
        self,
        timewise_relevance_out,
        n,                      # index of generated token (0-based)
        input_ids,
        tokenizer,
        *,
        generated_ids=None,
        input_key="input_ids",
        figsize=(10, 3),
        **kwargs,
    ):
        """
        Thin wrapper that forwards to core.token_relevance_visuals.plot_input_heatmap_for_token.
        Shows normalized relevance of input tokens for the n-th generated token.
        """
        return plot_input_heatmap_for_token(
            timewise_relevance_out,
            n,
            input_ids,
            tokenizer,
            generated_ids=generated_ids,
            input_key=input_key,
            figsize=figsize,
            **kwargs,
        )
    
    def debug_execution_differences(self, *test_inputs):
        """
        Debug why DLB execution differs from direct model execution
        """
        print("🔍 Debugging execution differences...")
        
        # Get direct model output
        with torch.no_grad():
            direct_output = self.model(*test_inputs)
            if isinstance(direct_output, (list, tuple)):
                direct_output = direct_output[0]
        
        # Get DLB predict output with debug info
        dlb_node_io = self.predict(*test_inputs, debug=True)
        
        # Find the final output node
        final_output = None
        final_node_name = None
        for node_name, node_data in dlb_node_io.items():
            if node_data.get('layer_type') == 'Output' or 'output' in node_name.lower():
                final_output = node_data['output_values']
                final_node_name = node_name
                break
        
        if final_output is None:
            # Fallback: use the last node's output
            final_node_name = list(dlb_node_io.keys())[-1]
            final_output = dlb_node_io[final_node_name]['output_values']
        
        # 🔧 FIX: Ensure final_output is a tensor, not a dict
        if isinstance(final_output, dict):
            if 'output_values' in final_output:
                final_output = final_output['output_values']
            else:
                raise ValueError(f"Expected tensor in final_output, got dict with keys: {list(final_output.keys())}")
        
        # Ensure both outputs are tensors
        if isinstance(final_output, (list, tuple)):
            final_output = final_output[0]
        
        print(f"🔍 Final output analysis:")
        print(f"   Direct model output: {direct_output.shape} on {direct_output.device}, dtype: {direct_output.dtype}")
        print(f"   DLB output ({final_node_name}): {final_output.shape} on {final_output.device}, dtype: {final_output.dtype}")
        
        # Calculate differences
        if isinstance(direct_output, torch.Tensor) and isinstance(final_output, torch.Tensor):
            max_diff = torch.max(torch.abs(direct_output - final_output)).item()
            mean_diff = torch.mean(torch.abs(direct_output - final_output)).item()
            
            print(f"   Max difference: {max_diff:.2e}")
            print(f"   Mean difference: {mean_diff:.2e}")
            
            # Check for NaN or Inf values
            if torch.isnan(direct_output).any():
                print("   ⚠️  Direct output contains NaN values")
            if torch.isnan(final_output).any():
                print("   ⚠️  DLB output contains NaN values")
            if torch.isinf(direct_output).any():
                print("   ⚠️  Direct output contains Inf values")
            if torch.isinf(final_output).any():
                print("   ⚠️  DLB output contains Inf values")
            
            return {
                'max_difference': max_diff,
                'mean_difference': mean_diff,
                'direct_output': direct_output,
                'dlb_output': final_output,
                'final_node_name': final_node_name
            }
        else:
            print(f"   ❌ Type mismatch: direct={type(direct_output)}, dlb={type(final_output)}")
            return {'error': 'Type mismatch'}

    def diagnose_model_compatibility(self):
        """Diagnose potential issues with model tracing"""
        print("🔍 Diagnosing model compatibility...")
        
        issues = []
        warnings = []
        
        # Check model type
        model_type = self._infer_model_type()
        print(f"   🏷️  Model type: {model_type}")
        
        # Check for common problematic patterns
        for name, module in self.model.named_modules():
            # Check for unsupported operations
            if hasattr(module, 'forward'):
                # This is a basic check - in practice you'd inspect the forward method
                if 'custom' in str(type(module)).lower():
                    warnings.append(f"Custom module detected: {name} ({type(module)})")
        
        # Check input requirements
        try:
            signature = inspect.signature(self.model.forward)
            params = list(signature.parameters.keys())
            print(f"   📝 Forward signature: {params}")
            
            if len(params) > 3:
                warnings.append(f"Complex forward signature with {len(params)} parameters")
                
        except Exception as e:
            issues.append(f"Could not inspect forward signature: {e}")
        
        # Check if model has been exported successfully
        if hasattr(self, 'exported_program') and self.exported_program is not None:
            print("   ✅ Model export successful")
        else:
            issues.append("Model export failed or not attempted")
        
        # Check graph complexity
        if hasattr(self, 'graph') and hasattr(self, 'layer_stack'):
            print(f"   📊 Graph nodes: {len(self.graph.nodes) if hasattr(self.graph, 'nodes') else 'unknown'}")
            print(f"   📊 Layer stack: {len(self.layer_stack)}")
            
            if len(self.layer_stack) > 1000:
                warnings.append("Very large computation graph - may be slow")
        
        # Report findings
        if issues:
            print(f"\n❌ Issues found ({len(issues)}):")
            for issue in issues:
                print(f"   • {issue}")
        
        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"   • {warning}")
        
        if not issues and not warnings:
            print("\n✅ No obvious compatibility issues detected")
        
        return {
            'model_type': model_type,
            'issues': issues,
            'warnings': warnings,
            'forward_params': signature.parameters if 'signature' in locals() else None
        }
    
    def _setup_deterministic_environment(self, seed: int = 42, verbose: bool = True, strict_cpu: bool = True):
        """
        Deterministic environment for export/replay parity on CUDA *and* CPU.
        - No global default dtype changes (let model dtype decide).
        - No global grad mode changes (use torch.no_grad() at call sites).
        - On CPU, pins threads and (optionally) disables MKL-DNN for stricter equality.
        """
        import os, warnings, random
        import numpy as np
        import torch

        if verbose:
            print("🔧 Setting up deterministic execution environment...")

        # Quiet some noise (don't hide RuntimeWarnings/NaNs)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # ---- Environment seeds (set env first where relevant) ----
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ---- Determinism: fail if a nondeterministic kernel sneaks in ----
        torch.use_deterministic_algorithms(True, warn_only=False)

        # ---- Common precision policy ----
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        # ---- Autocast policy: force disabled for parity ----
        try:
            torch.set_autocast_enabled(False)
        except Exception:
            pass

        # ---- CUDA path ----
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Force math SDPA (avoid Flash/ME drift)
            try:
                torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
            except Exception:
                pass
            # Ensure CUDA autocast is disabled
            try:
                torch.set_autocast_enabled(False)
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if verbose:
                    print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass

        # ---- CPU path ----
        else:
            # 1) Pin threads for all BLAS/OpenMP stacks to 1
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
            os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            
            if strict_cpu:
                try:
                    torch.backends.mkldnn.enabled = False
                except Exception:
                    pass
            # Ensure CPU autocast is disabled
            try:
                torch.set_autocast_enabled(False)
            except Exception:
                pass

        if verbose:
            print("✅ Deterministic environment setup complete!")