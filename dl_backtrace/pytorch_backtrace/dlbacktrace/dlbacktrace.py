# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/dlbacktrace.py

from .core.graph_builder import build_graph
from .core.execution_engine_noncache import ExecutionEngineNoCache
from .core.trace_utils import (
    extract_placeholders,
    map_placeholders_to_state_dict,
    get_weight_from_placeholder
)
from .core.config import activation_master
from .core.relevance_propagation import RelevancePropagator
from .core.visualization import visualize_graph, visualize_relevance

import numpy as np 
import torch
import inspect

class DLBacktraceFX:
    def __init__(self, model, input_for_graph, dynamic_shapes=None, layer_implementation="original", verbose=False, strict_cpu=True):
        """
        Initialize DL-Backtrace FX for model tracing and explainability.
        
        Args:
            model: PyTorch model to trace
            input_for_graph: Sample input for graph tracing
            dynamic_shapes: Optional dynamic shape constraints
            layer_implementation: Choice of layer implementation. Can be:
                - String: Global implementation ("original", "cuda", "pytorch", "refactored")
                - Dict: Layer-specific implementations, e.g.:
                  {
                      "linear": "cuda",        # Use CUDA for linear layers
                      "conv2d": "pytorch",     # Use PyTorch for conv2d layers  
                      "attention": "cuda",     # Use CUDA for attention layers
                      "default": "original"    # Use original for all other layers
                  }
            verbose (bool): Enable verbose initialization logs.
            strict_cpu (bool): When running on CPU, disable MKL-DNN and pin threads for stricter determinism.
        """
        # 🔧 CRITICAL: Set up deterministic environment for consistent tracing
        self.verbose = verbose
        self.strict_cpu = strict_cpu
        self._setup_deterministic_environment(seed=42, verbose=self.verbose, strict_cpu=self.strict_cpu)
        
        self.model = model
        if self.verbose:
            print("---------------------------v1------------------------------------------")
        self.input_for_graph = input_for_graph
        # Normalize sample inputs to a tuple for exporter compatibility
        if isinstance(self.input_for_graph, torch.Tensor):
            self.input_for_graph = (self.input_for_graph,)
        elif not isinstance(self.input_for_graph, (tuple, list)):
            self.input_for_graph = (self.input_for_graph,)
        self.dynamic_shapes = dynamic_shapes
        self.model.eval()
        self.model.requires_grad_(False)
        # Handle both string and dict layer implementation configurations
        self.layer_implementation = self._parse_layer_implementation(layer_implementation)
        
        if self.verbose:
            print(f"🚀 Layer Implementation Configuration:")
            if isinstance(self.layer_implementation, str):
                print(f"   Global: {self.layer_implementation.upper()}")
            else:
                print(f"   Layer-specific configuration:")
                for layer_type, impl in self.layer_implementation.items():
                    print(f"     {layer_type}: {impl.upper()}")
        
        # Cache manager removed - using non-cache execution only
        if self.verbose:
            print("---------------------------v2------------------------------------------")
        # Export and trace model
        self._trace_model()
        if self.verbose:
            print("---------------------------v3------------------------------------------")
        # Placeholder mapping
        self.fx_placeholders = extract_placeholders(self.exported_program)
        if self.verbose:
            print("---------------------------v4------------------------------------------")
        self.placeholder_to_real_name = map_placeholders_to_state_dict(
            self.exported_program, self.model
        )
        if self.verbose:
            print("---------------------------v5------------------------------------------")
        self.extracted_weights = {
            p: get_weight_from_placeholder(
                p, self.exported_program, self.model, self.placeholder_to_real_name
            ) for p in self.fx_placeholders
        }
        if self.verbose:
            print("---------------------------v6------------------------------------------")

        # Graph + metadata
        self.graph, self.layer_stack = build_graph(
            self.tracer, self.extracted_weights
        )
        if self.verbose:
            print("---------------------------v7------------------------------------------")
        # I/O and bookkeeping
        self.node_io = {}
        self.activation_dict = {}
        if self.verbose:
            print("---------------------------v8------------------------------------------")
    
    def _parse_layer_implementation(self, layer_implementation):
        """Parse and validate layer implementation configuration."""
        valid_implementations = ["original", "cuda", "pytorch", "refactored"]
        
        if isinstance(layer_implementation, str):
            # Global configuration
            if layer_implementation not in valid_implementations:
                raise ValueError(f"layer_implementation must be one of {valid_implementations}, got: {layer_implementation}")
            
            # Check CUDA availability if requested
            if layer_implementation == "cuda" and not torch.cuda.is_available():
                print("⚠️  CUDA implementation requested but CUDA not available. Falling back to 'original' implementation.")
                return "original"
            
            return layer_implementation
            
        elif isinstance(layer_implementation, dict):
            # Layer-specific configuration
            valid_layer_types = ["linear", "conv2d", "attention", "embedding", "wt_add_equal", "wt_mul", "default"]
            
            # Validate all implementations
            for layer_type, impl in layer_implementation.items():
                if layer_type not in valid_layer_types:
                    print(f"⚠️  Unknown layer type '{layer_type}'. Valid types: {valid_layer_types}")
                
                if impl not in valid_implementations:
                    raise ValueError(f"Implementation for '{layer_type}' must be one of {valid_implementations}, got: {impl}")
                
                # Check CUDA availability
                if impl == "cuda" and not torch.cuda.is_available():
                    print(f"⚠️  CUDA requested for '{layer_type}' but not available. Using 'original' instead.")
                    layer_implementation[layer_type] = "original"
            
            # Ensure default is specified
            if "default" not in layer_implementation:
                layer_implementation["default"] = "original"
                print("ℹ️  No default implementation specified. Using 'original' as default.")
            
            return layer_implementation
        
        else:
            raise TypeError(f"layer_implementation must be string or dict, got: {type(layer_implementation)}")
    
    def get_layer_implementation(self, layer_type):
        """Get the implementation to use for a specific layer type."""
        if isinstance(self.layer_implementation, str):
            return self.layer_implementation
        else:
            # Map common layer names to our keys
            layer_mapping = {
                "MLP_Layer": "linear",
                "Linear": "linear", 
                "DL_Layer": "conv2d",  # Assuming DL_Layer is primarily conv2d
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

    def predict(self, *inputs, debug=None):
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

    def evaluation(self, mode="default", start_wt=[], multiplier=100.0, scaler=1.0, thresholding=0.5, task="binary-classification", debug=False):
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
            debug=debug
        )
        return self.all_wt

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

    def visualize_dlbacktrace(self, output_path="backtrace_graph", top_k=None, relevance_threshold=None):
        visualize_relevance(self.graph, self.all_wt, output_path, top_k, relevance_threshold)
    
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