import numpy as np
from typing import List


def calculate_wt_add_equal(
    R: np.ndarray,
    inp: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Distribute relevance equally across inputs and restore original broadcasting
    dimensions.

    The relevance ``R`` is split evenly among all inputs.  For any input that
    originally had a broadcasted dimension (i.e. a length-1 axis that was
    later broadcast to a larger size), the corresponding slice of relevance
    is summed back along that axis so that its shape is identical to the
    original input tensor.

    Parameters
    ----------
    R : np.ndarray
        Relevance tensor of shape ``(batch, d1, d2, …)``.
    inp : List[np.ndarray]
        List of input tensors, each with shape ``(batch, …)``.  The first
        dimension of every tensor must equal ``R.shape[0]``.

    Returns
    -------
    List[np.ndarray]
        List of relevance tensors, one per entry in ``inp``, each having the
        same shape as the corresponding entry in ``inp``.
    """
    num_inputs = len(inp)
    input_shapes = [x.shape for x in inp]

    # Split relevance equally across inputs (vectorized broadcast)
    equal_relevance = R / num_inputs  # shape == R.shape

    result: List[np.ndarray] = []
    for orig_shape in input_shapes:
        # Start with the broadcasted relevance slice for this input
        reduced_rel = equal_relevance  # shape == R.shape

        # Iterate over non-batch dimensions and sum where the original
        # tensor had a length-1 axis but the broadcasted tensor did not.
        for axis, orig_dim in enumerate(orig_shape[1:], start=1):
            if orig_dim == 1 and reduced_rel.shape[axis] > 1:
                # Sum back along the broadcasted axis, keeping dims for
                # subsequent broadcasting checks
                reduced_rel = reduced_rel.sum(axis=axis, keepdims=True)

        result.append(reduced_rel)

    return result
