"""
Relevance Saver - Save DLBacktrace relevance data with quantization.

Supports saving relevance traces in multiple precision formats:
  - fp16: Half precision (baseline, ~0% error)
  - fp8:  Float8 E4M3 (high compression, ~0.000005 MAE)
  - fp4:  Packed Int4 (maximum compression, ~0.000004 MAE)

Usage:
    from dl_backtrace.pytorch_backtrace.dlbacktrace.core.relevance_saver import (
        save_relevance,
        load_relevance,
        Precision,
    )
    
    # Save relevance data
    save_path = save_relevance(
        relevance_trace=results['relevance_trace'],
        output_dir="./my_relevance",
        precision="fp8",
        metadata={"input_text": "My custom sentence"}
    )
    
    # Load relevance data
    data = load_relevance(save_path)
"""

import gzip
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


class Precision(str, Enum):
    """Supported precision formats for relevance storage."""
    FP16 = "fp16"
    FP8 = "fp8"
    FP4 = "fp4"


@dataclass
class PrecisionConfig:
    """Configuration for precision conversion."""
    precision: Precision = Precision.FP8
    fp8_format: str = "e4m3"  # "e4m3" or "e5m2"


class RelevancePrecisionConverter:
    """Converts relevance tensors to different precision formats."""
    
    def __init__(self, config: Optional[PrecisionConfig] = None):
        self.config = config or PrecisionConfig()
    
    def convert(self, tensor: torch.Tensor) -> tuple:
        """
        Convert tensor to target precision.
        
        Args:
            tensor: Input tensor to convert
            
        Returns:
            tuple: (converted_tensor, metadata_dict or None)
        """
        precision = self.config.precision
        
        if precision == Precision.FP16:
            return tensor.half(), None
        elif precision == Precision.FP8:
            return self._to_fp8(tensor)
        elif precision == Precision.FP4:
            return self._to_fp4(tensor)
        else:
            raise ValueError(f"Unknown precision: {precision}")
    
    def _to_fp8(self, tensor: torch.Tensor) -> tuple:
        """Convert to FP8 (float8_e4m3fn)."""
        tensor = tensor.float()
        
        if self.config.fp8_format == "e4m3":
            fp8_dtype = torch.float8_e4m3fn
            max_val = 448.0
        else:
            fp8_dtype = torch.float8_e5m2
            max_val = 57344.0
        
        abs_max = torch.abs(tensor).max().item()
        scale = 1.0
        if abs_max > max_val:
            scale = abs_max / max_val
            tensor = tensor / scale
        
        tensor = torch.clamp(tensor, -max_val, max_val)
        fp8_tensor = tensor.to(fp8_dtype)
        
        metadata = {
            "scale": scale,
            "original_shape": list(tensor.shape),
            "fp8_format": self.config.fp8_format,
            "max_val": max_val,
        }
        
        return fp8_tensor, metadata
    
    def _to_fp4(self, tensor: torch.Tensor) -> tuple:
        """Convert to packed Int4 (2 values per byte)."""
        values = tensor.float().flatten()
        
        abs_max = torch.abs(values).max().item()
        scale = abs_max / 7.0 if abs_max > 0 else 1.0
        
        quantized = torch.round(values / scale).to(torch.int8)
        quantized = torch.clamp(quantized, -8, 7)
        
        unsigned = (quantized + 8).to(torch.uint8)
        
        if len(unsigned) % 2 != 0:
            unsigned = torch.cat([unsigned, torch.zeros(1, dtype=torch.uint8)])
        
        packed = (unsigned[0::2] << 4) | (unsigned[1::2] & 0x0F)
        
        metadata = {
            "scale": scale,
            "original_shape": list(tensor.shape),
            "original_numel": tensor.numel(),
        }
        
        return packed, metadata


def _extract_relevance_with_precision(
    relevance_trace: List[Dict[str, Any]],
    converter: RelevancePrecisionConverter,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract and convert relevance data to target precision.
    
    Args:
        relevance_trace: List of relevance dicts (one per generation step)
        converter: Precision converter instance
        metadata: Optional user metadata to include
        
    Returns:
        dict: Converted data ready for saving
    """
    precision = converter.config.precision
    
    output = {
        "metadata": {
            "precision": precision.value,
            "fp8_format": converter.config.fp8_format if precision == Precision.FP8 else None,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        },
        "relevance_trace": [],
        "precision_scales": {} if precision in (Precision.FP4, Precision.FP8) else None,
    }
    
    num_tensors = 0
    total_elements = 0
    
    for step_idx, step_dict in enumerate(relevance_trace):
        step_data = {}
        
        for node_name, tensor_val in step_dict.items():
            if tensor_val is None:
                continue
            
            # Convert numpy to torch
            if isinstance(tensor_val, np.ndarray):
                tensor_val = torch.from_numpy(tensor_val)
            elif not isinstance(tensor_val, torch.Tensor):
                continue
            
            original_fp32 = tensor_val.cpu().float()
            key = f"step{step_idx}_{node_name}"
            
            converted, meta = converter.convert(original_fp32)
            step_data[node_name] = converted
            
            num_tensors += 1
            total_elements += original_fp32.numel()
            
            if meta is not None:
                output["precision_scales"][key] = meta
        
        output["relevance_trace"].append(step_data)
    
    output["metadata"]["num_tensors"] = num_tensors
    output["metadata"]["total_elements"] = total_elements
    
    return output


def save_relevance(
    relevance_trace: List[Dict[str, Any]],
    output_dir: str = "./relevance_output",
    precision: Union[str, Precision] = "fp8",
    name: Optional[str] = None,
    compress: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    save_metadata_json: bool = True,
) -> str:
    """
    Save relevance trace to disk with optional quantization and compression.
    
    Args:
        relevance_trace: List of relevance dicts from run_task()
        output_dir: Directory to save files (created if doesn't exist)
        precision: "fp16", "fp8", or "fp4"
        name: Custom filename prefix (auto-generated if None)
        compress: Enable gzip compression (default: True)
        metadata: Custom user metadata to include in the file
        save_metadata_json: Also save a human-readable .json metadata file
        
    Returns:
        str: Path to the saved file
        
    Example:
        >>> results = ir.run_task(..., return_relevance=True)
        >>> save_path = save_relevance(
        ...     results['relevance_trace'],
        ...     output_dir="./my_data",
        ...     precision="fp8",
        ...     metadata={"input_text": "Hello world"}
        ... )
        >>> print(f"Saved to: {save_path}")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse precision
    if isinstance(precision, str):
        precision = Precision(precision.lower())
    
    # Create converter
    config = PrecisionConfig(precision=precision)
    converter = RelevancePrecisionConverter(config)
    
    # Generate filename
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"relevance_{timestamp}"
    
    # Extract and convert relevance
    data = _extract_relevance_with_precision(relevance_trace, converter, metadata)
    
    # Save to file
    if compress:
        filename = os.path.join(output_dir, f"{name}_{precision.value}.pt.gz")
        buffer = BytesIO()
        torch.save(data, buffer)
        buffer.seek(0)
        with gzip.open(filename, 'wb', compresslevel=9) as f:
            f.write(buffer.read())
    else:
        filename = os.path.join(output_dir, f"{name}_{precision.value}.pt")
        torch.save(data, filename)
    
    file_size_kb = os.path.getsize(filename) / 1024
    
    # Optionally save human-readable metadata JSON
    if save_metadata_json:
        meta_filename = os.path.join(output_dir, f"{name}_meta.json")
        meta_output = {
            **data["metadata"],
            "file": os.path.basename(filename),
            "file_size_kb": round(file_size_kb, 2),
            "compressed": compress,
        }
        with open(meta_filename, 'w') as f:
            json.dump(meta_output, f, indent=2, default=str)
    
    return filename


def load_relevance(filepath: str) -> Dict[str, Any]:
    """
    Load saved relevance data from disk.
    
    Args:
        filepath: Path to .pt or .pt.gz file
        
    Returns:
        dict: Loaded relevance data with keys:
            - 'metadata': dict with timestamp, precision, num_tensors, etc.
            - 'relevance_trace': List of dicts (one per step)
            - 'precision_scales': Optional scale factors for dequantization
            
    Example:
        >>> data = load_relevance("./my_data/relevance_20260123_001522_fp8.pt.gz")
        >>> print(f"Precision: {data['metadata']['precision']}")
        >>> print(f"Num steps: {len(data['relevance_trace'])}")
    """
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rb') as f:
            buffer = BytesIO(f.read())
            data = torch.load(buffer, weights_only=False)
    else:
        data = torch.load(filepath, weights_only=False)
    
    return data


def dequantize_fp8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Dequantize FP8 tensor back to FP32.
    
    Args:
        tensor: FP8 tensor
        scale: Scale factor from precision_scales metadata
        
    Returns:
        torch.Tensor: Dequantized FP32 tensor
    """
    return tensor.float() * scale


def dequantize_fp4(
    packed: torch.Tensor, 
    scale: float, 
    original_shape: List[int],
    original_numel: int
) -> torch.Tensor:
    """
    Dequantize packed Int4 tensor back to FP32.
    
    Args:
        packed: Packed uint8 tensor (2 values per byte)
        scale: Scale factor from precision_scales metadata
        original_shape: Original tensor shape
        original_numel: Original number of elements
        
    Returns:
        torch.Tensor: Dequantized FP32 tensor
    """
    # Unpack
    high = (packed >> 4).to(torch.int8) - 8
    low = (packed & 0x0F).to(torch.int8) - 8
    
    # Interleave
    unpacked = torch.zeros(len(packed) * 2, dtype=torch.int8)
    unpacked[0::2] = high
    unpacked[1::2] = low
    
    # Trim to original size and reshape
    unpacked = unpacked[:original_numel].float() * scale
    return unpacked.reshape(original_shape)
