import numpy as np
from typing import Optional, Union


def calculate_wt_embedding(
    R_out: np.ndarray, 
    input_ids: np.ndarray, 
    vocab_size: Optional[int] = None, 
    aggregate: str = "sum"
) -> np.ndarray:
    """
    Backtrace relevance for embedding layers using vectorized operations.
    
    This function accumulates relevance scores from the output of an embedding layer
    back to individual vocabulary tokens, supporting both sum and mean aggregation.
    
    Args:
        R_out: Relevance at output of embedding layer with shape [B, T, D],
               where B is batch size, T is sequence length, D is embedding dimension.
        input_ids: Input token IDs with shape [B, T], containing integer token indices.
        vocab_size: Optional vocabulary size. If None, inferred as max(input_ids) + 1.
        aggregate: Aggregation method, either 'sum' (returns [vocab_size, D]) or 
                  'mean' (returns [vocab_size] with mean across embedding dimension).
    
    Returns:
        relevance_matrix: If aggregate='sum', returns [vocab_size, D] array with
                         accumulated relevance vectors per token. If aggregate='mean',
                         returns [vocab_size] array with mean relevance per token.
    
    Raises:
        ValueError: If aggregate parameter is not 'sum' or 'mean'.
    """
    B, T, D = R_out.shape
    
    # Infer vocabulary size if not provided
    if vocab_size is None:
        vocab_size = int(np.max(input_ids)) + 1
    
    # Initialize relevance matrix with same dtype as R_out for consistency
    relevance_matrix = np.zeros((vocab_size, D), dtype=np.float32)
    
    # Flatten input arrays for vectorized indexing
    # This replaces the nested loops with a single vectorized operation
    flat_input_ids = input_ids.flatten()  # [B*T]
    flat_R_out = R_out.reshape(-1, D)     # [B*T, D]
    
    # Vectorized accumulation using np.add.at for handling duplicate indices
    # This is equivalent to the original nested loop accumulation
    np.add.at(relevance_matrix, flat_input_ids, flat_R_out)
    
    # Apply aggregation based on the specified method
    if aggregate == "sum":
        return relevance_matrix  # [vocab_size, D]
    elif aggregate == "mean":
        return np.mean(relevance_matrix, axis=-1)  # [vocab_size]
    else:
        raise ValueError(f"Unsupported aggregation: {aggregate}")
