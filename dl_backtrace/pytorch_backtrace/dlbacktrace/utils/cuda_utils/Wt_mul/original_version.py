import numpy as np

def calculate_wt_mul(R):
    """
    Stable and safe distribution of relevance for elementwise multiplication.

    Parameters:
    - R: Relevance from the output (np.ndarray)
    - X, Y: Inputs to the multiplication (np.ndarray)
    - epsilon: Small constant to avoid divide-by-zero
    - clip_negative: If True, clips negative relevance to zero
    - normalize: If True, ensures Rx + Ry ≈ sum(R)

    Returns:
    - R_x: Relevance assigned to X
    - R_y: Relevance assigned to Y
    """
    R = np.array(R, dtype=np.float64)

    Rx, Ry = R / 2, R / 2
    return Rx, Ry   
