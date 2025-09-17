#DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/aten_operations.py
from typing import Any
import torch

ATEN_HYPERPARAMS: dict[str, Any] = {
            "conv1d": ["weight","bias","stride", "padding", "dilation", "groups"],
            "conv2d": ["weight","bias","stride", "padding", "dilation", "groups",],
            "conv3d": ["weight","bias","stride", "padding", "dilation", "groups"],
            "conv_transpose1d": ["weight","bias","stride", "padding", "output_padding", "groups", "dilation"],
            "conv_transpose2d": ["weight","bias","stride", "padding", "output_padding", "groups", "dilation"],
            "conv_transpose3d": ["weight","bias","stride", "padding", "output_padding", "groups", "dilation"],
            # Fully Connected (Linear)
            "linear": ["weight","bias"],
            "addmm": ["beta", "alpha"],
            "lstm": ["hx", "params", "has_biases", "num_layers", "dropout", "train", "bidirectional", "batch_first"],
            "embedding": ["weight", "indices", "padding_idx", "scale_grad_by_freq", "sparse"],
            # Normalization Layers
            "batch_norm": ["weight","bias","running_mean","running_var","training","momentum", "eps", "cudnn_enabled","affine", "track_running_stats"],
            "layer_norm": ["normalized_shape","weight","bias", "eps", "elementwise_affine"],
            "group_norm": ["num_groups", "eps", "affine"],
            "instance_norm": ["momentum", "eps", "affine", "track_running_stats"],
            # Pooling Operations
            "max_pool1d": ["kernel_size", "stride", "padding", "dilation", "ceil_mode"],
            "max_pool2d": ["kernel_size", "stride", "padding", "dilation", "ceil_mode"],
            "max_pool3d": ["kernel_size", "stride", "padding", "dilation", "ceil_mode"],
            "avg_pool1d": ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"],
            "avg_pool2d": ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"],
            "avg_pool3d": ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"],
            "adaptive_avg_pool2d": ["output_size"],
            # Activation Functions (Some may not need hyperparams, but included for consistency)
            "relu": [],
            "relu_": [],
            "sigmoid": [],
            "tanh": [],
            "softmax": ["dim"],
            # Tensor Operations
            "view": ["shape"],
            "reshape": ["shape"],
            "permute": ["dims"],
            "transpose": ["dim0", "dim1"],
            "squeeze": ["dim"],
            "unsqueeze": ["dim"],
            "select": ["dim", "index"],
            "slice": ["dim", "start", "end", "step"],
            "unflatten": ["dim", "sizes"],
            "contiguous": ["memory_format"],
            "scaled_dot_product_attention": ["attn_mask", "dropout_p", "is_causal"],
            # Additional attention operations for transformers
            "addmv": ["beta", "alpha"],  # Matrix-vector multiplication with bias
            "baddbmm": ["beta", "alpha"],  # Batch matrix-matrix multiplication with bias
            "split": ["split_size_or_sections", "dim"],  # Split tensor into chunks
            "chunk": ["chunks", "dim"],  # Split tensor into equal chunks
            "stack": ["tensors", "dim"],  # Stack tensors along new dimension
            "gather": ["dim", "index"],  # Gather values along dimension
            "scatter": ["dim", "index", "src"],  # Scatter values along dimension
            "index_add": ["dim", "index", "source"],  # Add values at indices
            "native_layer_norm": ["normalized_shape", "weight", "bias", "eps"],  # Native layer norm
            "dropout": ["p", "train"],
            "add": ["other", "alpha"],
            "add_": ["other", "alpha"],
            "mean": ["dim", "keepdim", "dtype"],
            # Tensor Operations
            "expand": ["sizes"],
            "expand_as": ["other"],
            "repeat": ["sizes"],
            "zeros":["sizes"],
            # Reductions
            "sum": ["dim", "keepdim", "dtype"],
            "prod": ["dim", "keepdim", "dtype"],
            "min": ["dim", "keepdim"],
            "max": ["dim", "keepdim"],
            # Comparison
            "eq": ["other"],
            "ne": ["other"],
            "gt": ["other"],
            "lt": ["other"],
            # Element-wise Ops
            "mul": ["other"],
            "div": ["other", "rounding_mode"],
            "sub": ["other", "alpha"],
            # Misc
            "sym_size": ["dim"],  # Extracts tensor dimension dynamically
            "to": ["dtype", "non_blocking", "copy", "memory_format"],
            "rsub": ["other", "alpha"],
            "masked_fill": ["mask", "value"],
            "gelu": [],  # No additional hyperparameters needed
            "expand": ["sizes", "implicit"],  # Expands tensor to target shape
            "masked_fill" : ["input_ids","mask_ids","value"],
            "flatten": ["start_dim", "end_dim"],
            "permute": ["dims"],
            "dropout": ["p", "train"],
            "mean": ["dim", "keepdim", "dtype"],
            "softmax": ["dim"],
            "arange": ["start", "end", "step", "dtype", "layout", "device", "pin_memory"],
            "triu": ["diagonal"],
            "full": ["size", "fill_value", "dtype", "layout", "device", "pin_memory"],
            "clone": ["memory_format"],
            "copy_": ["src", "non_blocking"],
            "pow": ["exponent"],
            "rsqrt": [],
            "neg": [],
            "silu": [],
            "mul_": ["other"],
            "cat": ["tensors", "dim"],
            # View / Memory Ops
            "_to_copy": ["memory_format", "non_blocking", "copy"],
            "_unsafe_view": ["shape"],
            "copy": ["src", "non_blocking"],
            "clone": ["memory_format"],  # already present
            "slice_scatter": ["dim", "start", "end", "step"],
            "to": ["dtype", "non_blocking", "copy", "memory_format"],  # already present

            # Math Ops (no hyperparams needed)
            "cos": [],
            "sin": [],
            "neg": [],
            "rsqrt": [],
            "pow": ["exponent"],  # already present
            "matmul": [],
            "logical_not": ["input"],
            "full_like": ["input", "fill_value", "dtype", "layout", "device", "pin_memory"],
            "where": ["condition", "input", "other"],
            "bmm": ["input", "mat2"],
            "mm": ["input", "mat2"],
            "_softmax": ["input", "dim", "half_to_float"],
            "any": ["input", "dim", "keepdim"],
            "ge": ["other"],
            "scalar_tensor": ["value", "dtype", "device"],



        }

ATEN_DEFAULTS: dict[str, Any] = {
            # Convolution Defaults (Dynamically adjust based on layer dimensionality)
            "stride": 1,  # Can be (1,1) or (1,1,1) for 2D/3D
            "padding": 0,  # Can be (0,0) or (0,0,0) for 2D/3D
            "dilation": 1,  # Can be (1,1) or (1,1,1) for 2D/3D
            "output_padding": 0,  # Needed for ConvTranspose (adjust dynamically)
            "groups": 1,
            "bias": True,
            # Normalization Defaults
            "momentum": 0.1,
            "eps": 1e-5,
            "normalized_shape": None,  # Needed for LayerNorm
            "num_groups": 1,  # Needed for GroupNorm
            "affine": True,  # Needed for BatchNorm, LayerNorm
            "track_running_stats": True,  # Needed for BatchNorm, InstanceNorm
            "elementwise_affine": True,  # Needed for LayerNorm
            # Pooling Defaults
            "kernel_size": 2,  # Default kernel size for pooling
            "stride_pool": None,  # Defaults to kernel_size if not specified
            "padding_pool": 0,
            "dilation_pool": 1,  # Used in max pooling
            "ceil_mode": False,
            "count_include_pad": True,  # Needed for AvgPool
            # Tensor Operations Defaults
            "dim": 1,  # Default dimension for reduction and activation ops
            "keepdim": False,  # Used in mean, sum, etc.
            "dtype": None,  # Explicit dtype handling
            # Slicing & Indexing
            "start": 0,  # Default start index for slice
            "end": None,  # Default None (full slice)
            "step": 1,  # Default step size for slice
            "index": 0,  # Default index for select operations
            # Reduction Defaults
            "alpha": 1.0,  # Default scaling factor for add/sub/mul operations
            # for addmm
            "beta" : 1.0,
            # Dropout
            "p": 0.5,  # Default dropout probability
            "train": True,  # Apply dropout only in training mode
            # Attention
            "attn_mask": None,  # No mask by default
            "dropout_p": 0.0,  # Default dropout probability in attention
            "is_causal": False,  # Default non-causal attention
            # Tensor Layout
            "memory_format": torch.contiguous_format,  # Default contiguous format
            "non_blocking": False,  # Default to synchronous conversion
            "copy": False,  # Only copy if necessary
            "has_biases": True,  # Default LSTM includes biases
            # Misc
            "num_layers": 1,  # Default number of layers
            "bidirectional": False,  # Default is not bidirectional
            "batch_first": False,  # Default batch_first=False (batch dim second)
            "padding_idx": -1,  # Default padding index (-1 means ignored)
            "scale_grad_by_freq": False,  # Default is False
            "sparse": False,  # Default is False (dense gradients),
            "value": 0.0,  # Default fill value for masked_fill
            "implicit": False,  # Default implicit broadcasting
            "start": 0,
            "end": None,  # No hardcoded default for end
            "step": 1,
            "fill_value": 0,
            "diagonal": 0,
            "memory_format": torch.contiguous_format,
            "non_blocking": False,
            "exponent": 2.0,
            "src": None,
            "shape": None,
            "fill_value": 0,
            "dtype": torch.float32,
            "layout": torch.strided,
            "device": torch.device("cpu"),
            "pin_memory": False,
            "half_to_float": False,
            "keepdim": False


        }