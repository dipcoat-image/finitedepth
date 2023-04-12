import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore


__all__ = [
    "df",
]


def df(
    P: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
) -> np.float64:
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
    np.float64
        Discrete Fréchet distance value.
        If *P* or *Q* is empty, return value is zero.

    References
    ----------
    .. [1] Eiter, Thomas, and Heikki Mannila. "Computing discrete Fréchet
       distance." (1994).

    .. [2] https://pypi.org/project/similaritymeasures/

    """
    freespace = cdist(P, Q)
    return _df(freespace)


@njit
def _df(fs: npt.NDArray[np.float64]) -> np.float64:
    p, q = fs.shape
    if p == 0 or q == 0:
        return np.float64(0)

    ca = np.zeros((p, q), dtype=np.float64)
    ca.fill(-1.0)

    ca[0, 0] = np.sqrt(fs[0, 0])

    for i in range(1, p):
        ca[i, 0] = max(ca[i - 1, 0], fs[i, 0])

    for j in range(1, q):
        ca[0, j] = max(ca[0, j - 1], fs[0, j])

    for i in range(1, p):
        for j in range(1, q):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]),
                fs[i, j],
            )
    return ca[p - 1, q - 1]
