import torch
from typing import List

@torch.compile
def calculate_wt_add_equal_vectorized(
    R: torch.Tensor,
    inp: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Highly optimized vectorized version of calculate_wt_add_equal.
    
    This version pre-computes reduction operations and uses more efficient
    tensor operations to minimize memory allocations and improve performance.
    
    Args:
        R: Relevance tensor of shape ``(batch, d1, d2, …)``.
        inp: List of input tensors, each with shape ``(batch, …)``. The first
            dimension of every tensor must equal ``R.shape[0]``.

    Returns:
        List of relevance tensors, one per entry in ``inp``, each having the
        same shape as the corresponding entry in ``inp``.
    """
    num_inputs = len(inp)
    device = R.device
    dtype = R.dtype
    
    # Pre-compute equal relevance once
    equal_relevance = R.div_(num_inputs) if R.requires_grad else R / num_inputs
    
    result: List[torch.Tensor] = []
    
    for tensor in inp:
        # Start with broadcasted relevance
        reduced_rel = equal_relevance
        
        # Collect all dimensions that need reduction
        dims_to_reduce = []
        for axis in range(1, len(tensor.shape)):
            if tensor.shape[axis] == 1 and reduced_rel.shape[axis] > 1:
                dims_to_reduce.append(axis)
        
        # Perform all reductions at once if possible
        if dims_to_reduce:
            # Sort dimensions in descending order to avoid index shifting
            dims_to_reduce.sort(reverse=True)
            
            for dim in dims_to_reduce:
                reduced_rel = torch.sum(reduced_rel, dim=dim, keepdim=True)
        
        result.append(reduced_rel)
    
    return result
