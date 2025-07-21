import torch
from typing import Optional, Union


@torch.compile
def calculate_wt_embedding(
    R_out: torch.Tensor, 
    input_ids: torch.Tensor, 
    vocab_size: Optional[int] = None, 
    aggregate: str = "sum"
) -> torch.Tensor:
    """
    Backtrace relevance for embedding layers using vectorized PyTorch operations.
    
    This function accumulates relevance scores from the output of an embedding layer
    back to individual vocabulary tokens, supporting both sum and mean aggregation.
    Optimized for GPU acceleration and autograd compatibility.
    
    Args:
        R_out (torch.Tensor): Relevance at output of embedding layer with shape [B, T, D],
                             where B is batch size, T is sequence length, D is embedding dimension.
        input_ids (torch.Tensor): Input token IDs with shape [B, T], containing integer token indices.
        vocab_size (Optional[int]): Optional vocabulary size. If None, inferred as max(input_ids) + 1.
        aggregate (str): Aggregation method, either 'sum' (returns [vocab_size, D]) or 
                        'mean' (returns [vocab_size] with mean across embedding dimension).
    
    Returns:
        torch.Tensor: If aggregate='sum', returns [vocab_size, D] tensor with
                     accumulated relevance vectors per token. If aggregate='mean',
                     returns [vocab_size] tensor with mean relevance per token.
    
    Raises:
        ValueError: If aggregate parameter is not 'sum' or 'mean'.
    """
    B, T, D = R_out.shape
    
    # Ensure input_ids is long type for indexing
    input_ids = input_ids.long()
    
    # Infer vocabulary size if not provided
    if vocab_size is None:
        vocab_size = int(torch.max(input_ids).item()) + 1
    
    # Initialize relevance matrix with same dtype and device as R_out
    relevance_matrix = torch.zeros(
        (vocab_size, D), 
        dtype=R_out.dtype, 
        device=R_out.device
    )
    
    # Flatten input arrays for vectorized indexing - avoid unnecessary copies
    flat_input_ids = input_ids.view(-1)  # [B*T] - more efficient than flatten()
    flat_R_out = R_out.view(-1, D)       # [B*T, D] - more efficient than reshape()
    
    # Vectorized accumulation using scatter_add for handling duplicate indices
    # This is the PyTorch equivalent of np.add.at and is autograd-compatible
    relevance_matrix.scatter_add_(0, flat_input_ids.unsqueeze(1).expand(-1, D), flat_R_out)
    
    # Apply aggregation based on the specified method
    if aggregate == "sum":
        return relevance_matrix  # [vocab_size, D]
    elif aggregate == "mean":
        return torch.mean(relevance_matrix, dim=-1)  # [vocab_size]
    else:
        raise ValueError(f"Unsupported aggregation: {aggregate}")
