import numpy as np

def calculate_wt_embedding(R_out, input_ids, vocab_size=None, aggregate="sum"):
    """
    Backtrace relevance for embedding layers.
    
    Args:
        R_out: relevance at output of embedding [B, T, D]
        input_ids: input token ids [B, T] (int)
        vocab_size: optional; default inferred from max token ID
        aggregate: 'sum' or 'mean' for token-wise relevance
    
    Returns:
        relevance_matrix: [vocab_size, D] or [B, T] if reduced
    """
    B, T, D = R_out.shape
    if vocab_size is None:
        vocab_size = np.max(input_ids) + 1

    relevance_matrix = np.zeros((vocab_size, D), dtype=np.float32)

    for b in range(B):
        for t in range(T):
            token_id = input_ids[b, t]
            relevance_matrix[token_id] += R_out[b, t]  # accumulate vector relevance

    if aggregate == "sum":
        return relevance_matrix  # [V, D]
    elif aggregate == "mean":
        return np.mean(relevance_matrix, axis=-1)  # [V]
    else:
        raise ValueError(f"Unsupported aggregation: {aggregate}")
