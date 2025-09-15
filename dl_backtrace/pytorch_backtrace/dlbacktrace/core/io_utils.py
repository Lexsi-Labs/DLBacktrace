# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/io_utils.py

import logging
import numpy as np
import torch
from typing import Union, List, Tuple, Any, Optional

# Set up logger for this module
logger = logging.getLogger(__name__)

def tensor_to_numpy(tensor: Union[torch.Tensor, np.ndarray, List, int, float]) -> np.ndarray:
    """
    Convert Tensor or list of tensors to NumPy format.
    
    Args:
        tensor: Input tensor, numpy array, list, or scalar value
        
    Returns:
        NumPy array representation of the input
        
    Raises:
        TypeError: If input type is not supported
        RuntimeError: If tensor conversion fails
    """
    if tensor is None:
        raise ValueError("Input tensor cannot be None")
    
    try:
        if isinstance(tensor, np.ndarray):
            logger.debug(f"Input is already numpy array with shape {tensor.shape}")
            return tensor
            
        elif isinstance(tensor, list):
            if not tensor:
                logger.warning("Empty list provided, returning empty numpy array")
                return np.array([], dtype=np.float32)
                
            tensor_list = []
            for i, item in enumerate(tensor):
                if isinstance(item, torch.Tensor):
                    tensor_list.append(item.detach().cpu().numpy())
                elif isinstance(item, (int, float, np.ndarray)):
                    tensor_list.append(item)
                else:
                    logger.warning(f"Unsupported item type {type(item)} at index {i}, skipping")
            
            if not tensor_list:
                raise ValueError("No valid items found in input list")
            
            # 🔧 FIX: Handle lists with different tensor shapes
            try:
                # Try to create a regular numpy array first
                result = np.array(tensor_list, dtype=np.float32)
                logger.debug(f"Converted list of {len(tensor_list)} items to numpy array with shape {result.shape}")
                return result
            except ValueError as shape_error:
                # If shapes are incompatible, return as object array or flatten
                logger.warning(f"Incompatible tensor shapes in list: {shape_error}")
                logger.warning("Returning as object array to preserve individual tensors")
                
                # Convert each tensor to numpy and store in object array
                numpy_list = []
                for item in tensor_list:
                    if isinstance(item, np.ndarray):
                        numpy_list.append(item)
                    else:
                        numpy_list.append(np.array(item, dtype=np.float32))
                
                result = np.array(numpy_list, dtype=object)
                logger.debug(f"Converted list to object array with {len(numpy_list)} items")
                return result
            
        elif isinstance(tensor, torch.Tensor):
            result = tensor.detach().cpu().numpy()
            logger.debug(f"Converted torch tensor {tensor.shape} to numpy array {result.shape}")
            return result
            
        elif isinstance(tensor, (int, float)):
            result = np.array(tensor, dtype=np.float32)
            logger.debug(f"Converted scalar {tensor} to numpy array")
            return result
            
        else:
            raise TypeError(f"Unsupported type for tensor conversion: {type(tensor)}")
            
    except Exception as e:
        logger.error(f"Failed to convert tensor to numpy: {e}")
        raise RuntimeError(f"Tensor conversion failed: {e}") from e


def eval_process_input_to_numpy(x: Union[List, np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Ensure input is converted to NumPy array for evaluation.
    
    Args:
        x: Input data (list, numpy array, or torch tensor)
        
    Returns:
        NumPy array representation
        
    Raises:
        TypeError: If input type is not supported
        ValueError: If input is empty or invalid
    """
    if x is None:
        raise ValueError("Input cannot be None")
    
    try:
        if isinstance(x, list):
            if not x:
                raise ValueError("Input list is empty")
            x = x[0]  # Take first element
            logger.debug("Extracted first element from input list")
            
        if isinstance(x, np.ndarray):
            logger.debug(f"Input is numpy array with shape {x.shape}")
            return x
            
        if isinstance(x, torch.Tensor):
            result = x.detach().cpu().numpy()
            logger.debug(f"Converted torch tensor {x.shape} to numpy array {result.shape}")
            return result
            
        raise TypeError(f"Input must be list, NumPy array, or torch.Tensor, got {type(x)}")
        
    except Exception as e:
        logger.error(f"Failed to process input to numpy: {e}")
        raise


def eval_normalize_tuple(param: Union[Tuple, Any]) -> Union[Tuple, Any]:
    """
    Reduce duplicated nested tuple (e.g. ((1,1),(1,1)) → (1,1)).
    
    Args:
        param: Input parameter (preferably a tuple)
        
    Returns:
        Normalized tuple or original parameter
        
    Raises:
        TypeError: If input is not a tuple when expected
    """
    if param is None:
        logger.warning("Input parameter is None")
        return param
    
    try:
        if isinstance(param, tuple):
            if all(isinstance(p, tuple) for p in param) and len(param) == 2 and param[0] == param[1]:
                logger.debug(f"Normalized duplicated tuple {param} to {param[0]}")
                return param[0]
            logger.debug(f"Tuple {param} does not need normalization")
            return param
            
        # For non-tuple inputs, log but don't raise error (more flexible)
        logger.debug(f"Input is not a tuple: {type(param)}, returning as-is")
        return param
        
    except Exception as e:
        logger.error(f"Failed to normalize tuple: {e}")
        raise TypeError(f"Error processing parameter: {e}") from e


def sanitize_input_tensor(t: Union[torch.Tensor, torch.nn.Parameter, Any]) -> Union[torch.Tensor, Any]:
    """
    Detach Parameter or Tensor for safe evaluation.
    
    Args:
        t: Input tensor, parameter, or other value
        
    Returns:
        Detached tensor or original value
    """
    if t is None:
        logger.warning("Input tensor is None")
        return t
    
    try:
        if isinstance(t, torch.nn.Parameter):
            result = t.detach()
            logger.debug(f"Detached parameter with shape {result.shape}")
            return result
        elif isinstance(t, torch.Tensor):
            logger.debug(f"Input is already a tensor with shape {t.shape}")
            return t
        else:
            logger.debug(f"Input is not a tensor/parameter: {type(t)}, returning as-is")
            return t
            
    except Exception as e:
        logger.error(f"Failed to sanitize input tensor: {e}")
        return t  # Return original on error


def process_output_tuple(output: Union[Tuple, List, torch.Tensor, int, Any]) -> Union[Tuple, List, torch.Tensor, int, Any]:
    """
    Handle nested tuples/lists with torch.Tensor or scalar outputs.
    
    Args:
        output: Output data structure to process
        
    Returns:
        Processed output with tensors converted appropriately
    """
    if output is None:
        logger.warning("Output is None")
        return output
    
    try:
        if isinstance(output, tuple):
            result = tuple(process_output_tuple(item) for item in output)
            logger.debug(f"Processed tuple with {len(result)} items")
            return result
            
        if isinstance(output, list):
            result = [process_output_tuple(item) for item in output]
            logger.debug(f"Processed list with {len(result)} items")
            return result
            
        if isinstance(output, torch.Tensor):
            logger.debug(f"Output is tensor with shape {output.shape}")
            return output
            
        if isinstance(output, int):
            logger.debug(f"Output is integer: {output}")
            return output
            
        # Try to convert to tensor
        try:
            result = torch.tensor(output)
            logger.debug(f"Converted output to tensor with shape {result.shape}")
            return result
        except Exception as tensor_error:
            logger.debug(f"Could not convert to tensor: {tensor_error}, returning original")
            return output
            
    except Exception as e:
        logger.error(f"Failed to process output tuple: {e}")
        return output  # Return original on error
