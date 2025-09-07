from typing import Tuple
import numpy as np

def calculate_wt_mul(
    R: np.ndarray, 
    X: np.ndarray, 
    Y: np.ndarray, 
    epsilon: float = 1e-12, 
    clip_negative: bool = True, 
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stable and safe distribution of relevance for elementwise multiplication.
    
    This function distributes relevance scores equally between two input arrays
    that participate in elementwise multiplication, effectively splitting the
    relevance 50-50 between X and Y inputs.
    
    Parameters
    ----------
    R : np.ndarray
        Relevance from the output. Will be converted to float32.
    X : np.ndarray
        First input to the multiplication. Will be converted to float32.
    Y : np.ndarray
        Second input to the multiplication. Will be converted to float32.
    epsilon : float, optional
        Small constant to avoid divide-by-zero (default: 1e-12).
        Note: Currently unused in this implementation.
    clip_negative : bool, optional
        If True, clips negative relevance to zero (default: True).
        Note: Currently unused in this implementation.
    normalize : bool, optional
        If True, ensures Rx + Ry ≈ sum(R) (default: True).
        Note: Currently unused in this implementation.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        R_x : Relevance assigned to X (R / 2)
        R_y : Relevance assigned to Y (R / 2)
    
    Notes
    -----
    The current implementation performs a simple 50-50 split of relevance,
    ignoring the epsilon, clip_negative, and normalize parameters.
    """
    # Convert inputs to float32 arrays 
    R = np.asarray(R, dtype=np.float32)
    
    # Vectorized 50-50 relevance split using broadcasting
    relevance_half = R * 0.5
    
    return relevance_half, relevance_half
