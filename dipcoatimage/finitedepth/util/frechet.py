import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from typing import Tuple


__all__ = [
    "dfd",
    "dfd_pair",
    "sfd",
    "sfd_path",
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
    ndarray
        Accumulated cost array for the Fréchet distance.
        The distance is the last element of the array i.e. `ca[-1, -1]`.
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

    ca[0, 0] = freespace[0, 0]

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


def sfd(
    P: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute summed Fréchet distance (equivalent to dynamic time warping) with
    Euclidean metric.

    Implements the algorithm from [1]_, with modification from [2]_.

    Parameters
    ----------
    P, Q : ndarray
        2D Numpy array with ``np.float64`` dtype. Rows are the points and
        columns are the dimensions.

    Returns
    -------
    ndarray
        Accumulated cost array for the summed Fréchet distance.
        The distance is the last element of the array i.e. `ca[-1, -1]`.
        If *P* or *Q* is empty, return value is an empty array.

    References
    ----------
    .. [1] Senin, Pavel. "Dynamic time warping algorithm review." Information and
       Computer Science Department University of Hawaii at Manoa Honolulu,
       USA 855.1-23 (2008): 40.

    .. [2] https://pypi.org/project/similaritymeasures/

    """
    return _sfd(cdist(P, Q))


@njit
def _sfd(freespace: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    p, q = freespace.shape
    ca = np.zeros((p, q), dtype=np.float64)
    if p == 0 or q == 0:
        return ca

    ca[0, 0] = freespace[0, 0]

    for i in range(1, p):
        ca[i, 0] = ca[i - 1, 0] + freespace[i, 0]

    for j in range(1, q):
        ca[0, j] = ca[0, j - 1] + freespace[0, j]

    for i in range(1, p):
        for j in range(1, q):
            ca[i, j] = (
                min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]) + freespace[i, j]
            )

    return ca


def sfd_path(ca: npt.NDArray[np.float64]):
    """
    Compute path for optimal summed Fréchet distance in the free space.

    Implements the algorithm from [1]_, with modification from [2]_.

    Parameters
    ----------
    ca: ndarray
        Accumulated cost array of summed Fréchet distance between two curves.

    Returns
    -------
    ndarray
        Indices for the two curves to get the path.

    References
    ----------
    .. [1] Senin, Pavel. "Dynamic time warping algorithm review." Information and
       Computer Science Department University of Hawaii at Manoa Honolulu,
       USA 855.1-23 (2008): 40.

    .. [2] https://pypi.org/project/similaritymeasures/

    See Also
    --------
    sfd

    """
    path, path_len = _sfd_path(ca)
    return path[-(len(path) - path_len + 1) :: -1]


@njit
def _sfd_path(ca: npt.NDArray[np.float64]):
    p, q = ca.shape
    if p == 0 or q == 0:
        return np.empty((0, 2), dtype=np.int32), np.int32(0)

    path = np.zeros((p + q - 1, 2), dtype=np.int32)
    path_len = np.int32(0)

    i, j = p - 1, q - 1
    path[path_len] = [i, j]
    path_len += 1

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            d = min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1])
            if ca[i - 1, j] == d:
                i -= 1
            elif ca[i, j - 1] == d:
                j -= 1
            else:
                i -= 1
                j -= 1

        path[path_len] = [i, j]
        path_len += 1

    return path, path_len
