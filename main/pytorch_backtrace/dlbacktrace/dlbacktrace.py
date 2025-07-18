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
    def __init__(self, model, input_for_graph, dynamic_shapes=None,use_disk_cache=False):
        self.model = model.eval()
        print("---------------------------v1------------------------------------------")
        self.input_for_graph = input_for_graph
        self.dynamic_shapes = dynamic_shapes
        self.use_disk_cache = use_disk_cache
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

    def predict(self, *inputs):
        if self.use_disk_cache:
            executor = ExecutionEngine(
                model=self.model,
                extracted_weights=self.extracted_weights,
                fx_graph=self.graph,
                layer_stack=self.layer_stack,
                tracer=self.tracer,
                exported_program=self.exported_program,
                cache_manager=self.cache_manager  # ✅ ADD THIS
            )
        else:
            executor = ExecutionEngineNoCache(
                model=self.model,
                extracted_weights=self.extracted_weights,
                fx_graph=self.graph,
                layer_stack=self.layer_stack,
                tracer=self.tracer,
                exported_program=self.exported_program
            )
        self.node_io = executor.run(inputs)
        return self.node_io


    def evaluation(self, mode="default", start_wt=[], multiplier=100.0, scaler=1.0, thresholding=0.5, task="binary-classification"):
        evaluator = RelevancePropagator(
            graph=self.graph,
            node_io=self.node_io,
            activation_master=activation_master
        )
        self.all_wt = evaluator.propagate(
            start_wt=start_wt,
            mode=mode,
            multiplier=multiplier,
            scaler=scaler,
            thresholding=thresholding,
            task=task
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
