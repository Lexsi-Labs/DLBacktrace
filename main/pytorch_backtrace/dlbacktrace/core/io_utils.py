# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/io_utils.py
import numpy as np
import torch


def tensor_to_numpy(tensor):
    """🔁 Converts Tensor or list of tensors to NumPy format"""
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        tensor_list = []
        for item in tensor:
            if isinstance(item, torch.Tensor):
                tensor_list.append(item.detach().cpu().numpy())
            elif isinstance(item, (int, float, np.ndarray)):
                tensor_list.append(item)
        return np.array(tensor_list, dtype=np.float32)
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float)):
       return np.array(tensor, dtype=np.float32)
    else:
        raise TypeError(f"Unsupported type for tensor conversion: {type(tensor)}")


def eval_process_input_to_numpy(x):
    """Ensures input is converted to NumPy array"""
    if isinstance(x, list):
        x = x[0]
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError("Input must be list, NumPy array, or torch.Tensor")


def eval_normalize_tuple(param):
    """Reduces duplicated nested tuple (e.g. ((1,1),(1,1)) → (1,1))"""
    if isinstance(param, tuple):
        if all(isinstance(p, tuple) for p in param) and len(param) == 2 and param[0] == param[1]:
            return param[0]
        return param
    raise TypeError(f"Expected tuple, got {type(param)}")


def sanitize_input_tensor(t):
    """Detach Parameter or Tensor for safe evaluation"""
    if isinstance(t, torch.nn.Parameter):
        return t.detach()
    return t


def process_output_tuple(output):
    """Handles nested tuples/lists with torch.Tensor or scalar outputs"""
    if isinstance(output, tuple):
        return tuple(process_output_tuple(item) for item in output)
    if isinstance(output, list):
        return [process_output_tuple(item) for item in output]
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, int):
        return output
    try:
        return torch.tensor(output)
    except:
        return output
