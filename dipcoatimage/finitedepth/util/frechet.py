import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from typing import Tuple


__all__ = [
    "dfd",
    "dfd_pair",
]


def dfd(
    P: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute discrete Fréchet distance with Euclidean metric.

    Implements the algorithm from [1]_, with modification from [2]_.

    Parameters
    ----------
    P, Q : ndarray
        2D Numpy array with ``np.float64`` dtype. Rows are the points and
        columns are the dimensions.

    Returns
    -------
    ca: ndarray
        Accumulated cost array for the Fréchet distance.
        The Fréchet distance is the last element of the array i.e. `ca[-1, -1]`.
        If *P* or *Q* is empty, return value is an empty array.

    References
    ----------
    .. [1] Eiter, Thomas, and Heikki Mannila. "Computing discrete Fréchet
       distance." (1994).

    .. [2] https://pypi.org/project/similaritymeasures/

    """
    # TODO: use subquadratic algorithm (Agarwal et al, 2014)
    return _dfd(cdist(P, Q))


@njit
def _dfd(freespace: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    p, q = freespace.shape
    ca = np.zeros((p, q), dtype=np.float64)
    if p == 0 or q == 0:
        return ca
    ca.fill(-1.0)

    ca[0, 0] = np.sqrt(freespace[0, 0])

    for i in range(1, p):
        ca[i, 0] = max(ca[i - 1, 0], freespace[i, 0])

    for j in range(1, q):
        ca[0, j] = max(ca[0, j - 1], freespace[0, j])

    for i in range(1, p):
        for j in range(1, q):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]),
                freespace[i, j],
            )
    return ca


def dfd_pair(ca: npt.NDArray[np.float64]) -> Tuple[np.int64, np.int64]:
    """
    Find the point pair which determines the discrete Fréchet distance.

    Parameters
    ----------
    ca: ndarray
        Accumulated cost array of discrete Fréchet distance between two curves.

    Returns
    -------
    tuple
        Indices for the two curves to get the point pair.

    See Also
    --------
    dfd
    """
    i, j = np.nonzero(ca == ca[-1, -1])
    return (i[0], j[0])
