# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/dlbacktrace.py

from .core.graph_builder import build_graph
from .core.execution_engine import ExecutionEngine, DiskCacheManager
from .core.execution_engine_noncache import ExecutionEngineNoCache
from .core.trace_utils import (
    extract_placeholders,
    map_placeholders_to_state_dict,
    get_weight_from_placeholder
)
from .core.config import ATEN_HYPERPARAMS, ATEN_DEFAULTS, activation_master
from .core.relevance_propagation import RelevancePropagator
from .core.visualization import visualize_graph, visualize_relevance
from .core.io_utils import tensor_to_numpy

import numpy as np 
import torch
from torch.export import export, default_decompositions, export_for_training

class DLBacktraceFX:
    def __init__(self, model, input_for_graph, dynamic_shapes=None, use_disk_cache=False, layer_implementation="original"):
        """
        Initialize DL-Backtrace FX for model tracing and explainability.
        
        Args:
            model: PyTorch model to trace
            input_for_graph: Sample input for graph tracing
            dynamic_shapes: Optional dynamic shape constraints
            use_disk_cache: Whether to use disk caching for execution
            layer_implementation: Choice of layer implementation. Can be:
                - String: Global implementation ("original", "cuda", "pytorch", "refactored")
                - Dict: Layer-specific implementations, e.g.:
                  {
                      "linear": "cuda",        # Use CUDA for linear layers
                      "conv2d": "pytorch",     # Use PyTorch for conv2d layers  
                      "attention": "cuda",     # Use CUDA for attention layers
                      "default": "original"    # Use original for all other layers
                  }
        """
        self.model = model.eval()
        print("---------------------------v1------------------------------------------")
        self.input_for_graph = input_for_graph
        self.dynamic_shapes = dynamic_shapes
        self.use_disk_cache = use_disk_cache
        
        # Handle both string and dict layer implementation configurations
        self.layer_implementation = self._parse_layer_implementation(layer_implementation)
        
        print(f"🚀 Layer Implementation Configuration:")
        if isinstance(self.layer_implementation, str):
            print(f"   Global: {self.layer_implementation.upper()}")
        else:
            print(f"   Layer-specific configuration:")
            for layer_type, impl in self.layer_implementation.items():
                print(f"     {layer_type}: {impl.upper()}")
        
        if self.use_disk_cache:
            self.cache_manager = DiskCacheManager()
        else:
            self.cache_manager = None  # not used
        print("---------------------------v2------------------------------------------")
        # Export and trace model
        self._trace_model()
        print("---------------------------v3------------------------------------------")
        # Placeholder mapping
        self.fx_placeholders = extract_placeholders(self.exported_program)
        print("---------------------------v4------------------------------------------")
        self.placeholder_to_real_name = map_placeholders_to_state_dict(
            self.exported_program, self.model
        )
        print("---------------------------v5------------------------------------------")
        self.extracted_weights = {
            p: get_weight_from_placeholder(
                p, self.exported_program, self.model, self.placeholder_to_real_name
            ) for p in self.fx_placeholders
        }
        print("---------------------------v6------------------------------------------")

        # Graph + metadata
        self.graph, self.layer_stack = build_graph(
            self.tracer, self.extracted_weights
        )
        print("---------------------------v7------------------------------------------")
        # I/O and bookkeeping
        self.node_io = {}
        self.activation_dict = {}
        self.model_resource = {}
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
            valid_layer_types = ["linear", "conv2d", "attention", "embedding", "wt_add_equal", "wt_mul", "pooling", "default"]
            
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
                "Mathematical_Operation_add": "wt_add_equal",
                "Pooling": "pooling"
            }
            
            mapped_type = layer_mapping.get(layer_type, layer_type.lower())
            return self.layer_implementation.get(mapped_type, self.layer_implementation["default"])

    def _trace_model(self):
        #if self.dynamic_shapes:
        program = export_for_training(
            self.model,
            self.input_for_graph,
            dynamic_shapes=self.dynamic_shapes,
        )
        self.exported_program = program.run_decompositions(
            decomp_table={}
            #default_decompositions()
        )
        #else:
        #self.exported_program = export(self.model, self.input_for_graph)

        self.tracer = self.exported_program.graph_module

    def predict(self, *inputs, debug=False):
        """
        Execute the model with the given inputs and return node I/O data.
        
        Args:
            *inputs: Input tensors for the model
            debug (bool): Enable debug mode for detailed execution logging
            
        Returns:
            dict: Node I/O data containing execution results
        """
        if debug:
            print(f"🔧 DLB Predict: Debug mode enabled")
            print(f"   Input count: {len(inputs)}")
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    print(f"   Input {i}: {inp.shape} on {inp.device}, dtype: {inp.dtype}")
                else:
                    print(f"   Input {i}: {type(inp)}")
            print(f"   Use disk cache: {self.use_disk_cache}")
            print(f"   Layer stack length: {len(self.layer_stack)}")
        
        if self.use_disk_cache:
            executor = ExecutionEngine(
                model=self.model,
                extracted_weights=self.extracted_weights,
                fx_graph=self.graph,
                layer_stack=self.layer_stack,
                tracer=self.tracer,
                exported_program=self.exported_program,
                cache_manager=self.cache_manager
            )
        else:
            executor = ExecutionEngineNoCache(
                model=self.model,
                extracted_weights=self.extracted_weights,
                fx_graph=self.graph,
                layer_stack=self.layer_stack,
                tracer=self.tracer,
                exported_program=self.exported_program,
            )
        
        if debug:
            print(f"🔧 Starting execution with {type(executor).__name__}")
        
        self.node_io = executor.run(inputs, debug=debug)
        
        if debug:
            print(f"🔧 Execution completed successfully")
            print(f"   Output nodes: {len(self.node_io)}")
        
        return self.node_io

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
    
    def verify_model_consistency(self, test_inputs, tolerance=1e-6):
        """
        Verify that DLB predict produces identical results to direct model inference.
        
        Args:
            test_inputs: Input tensors for testing
            tolerance: Numerical tolerance for comparison
            
        Returns:
            dict: Comparison results with max difference and success status
        """
        print("🔍 Verifying model consistency...")
        
        # Ensure model is in eval mode
        self.model.eval()
        torch.set_grad_enabled(False)
        
        # Get direct model output
        with torch.no_grad():
            direct_output = self.model(*test_inputs)
            if isinstance(direct_output, (list, tuple)):
                direct_output = direct_output[0]  # Take first output for comparison
        
        # Get DLB predict output
        dlb_node_io = self.predict(*test_inputs, debug=False)
        
        # Find the final output node
        final_output = None
        for node_name, node_data in dlb_node_io.items():
            if node_data.get('layer_type') == 'Output' or 'output' in node_name.lower():
                final_output = node_data['output_values']
                break
        
        if final_output is None:
            # Fallback: use the last node's output
            last_node = list(dlb_node_io.keys())[-1]
            final_output = dlb_node_io[last_node]['output_values']
        
        # Ensure both outputs are tensors
        if isinstance(final_output, (list, tuple)):
            final_output = final_output[0]
        
        # Compare outputs
        if isinstance(direct_output, torch.Tensor) and isinstance(final_output, torch.Tensor):
            # Ensure same device and dtype for comparison
            if direct_output.device != final_output.device:
                final_output = final_output.to(device=direct_output.device)
            if direct_output.dtype != final_output.dtype:
                final_output = final_output.to(dtype=direct_output.dtype)
            
            # Calculate differences
            max_diff = torch.max(torch.abs(direct_output - final_output)).item()
            mean_diff = torch.mean(torch.abs(direct_output - final_output)).item()
            
            # Check if outputs are identical within tolerance
            is_consistent = max_diff < tolerance
            
            result = {
                'consistent': is_consistent,
                'max_difference': max_diff,
                'mean_difference': mean_diff,
                'tolerance': tolerance,
                'direct_output_shape': direct_output.shape,
                'dlb_output_shape': final_output.shape,
                'direct_output_sum': torch.sum(direct_output).item(),
                'dlb_output_sum': torch.sum(final_output).item()
            }
            
            if is_consistent:
                print(f"✅ Model consistency verified! Max difference: {max_diff:.2e}")
            else:
                print(f"❌ Model inconsistency detected! Max difference: {max_diff:.2e} (tolerance: {tolerance:.2e})")
                print(f"   Direct output sum: {result['direct_output_sum']:.6f}")
                print(f"   DLB output sum: {result['dlb_output_sum']:.6f}")
            
            return result
        else:
            print(f"❌ Cannot compare outputs: direct={type(direct_output)}, dlb={type(final_output)}")
            return {'consistent': False, 'error': 'Output type mismatch'}