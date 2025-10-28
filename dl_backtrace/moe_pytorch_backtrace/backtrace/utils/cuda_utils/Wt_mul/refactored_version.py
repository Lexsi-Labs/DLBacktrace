from typing import Tuple
import numpy as np

def calculate_wt_mul(
    R: np.ndarray, 
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
    R_half = np.multiply(R, 0.5)
    return R_half, R_half
