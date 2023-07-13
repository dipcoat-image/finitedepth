import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from typing import Tuple


__all__ = [
    "sfd",
    "sfd_path",
    "ssfd",
    "ssfd_path",
]


def sfd(
    P: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute summed Fréchet distance between two series, using L2 norm.

    The summed Fréchet distance is equivalent to the dynamic time warping[1]_.
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

    See Also
    --------
    sfd_path

    """
    return _sfd(cdist(P, Q))


@njit(cache=True)
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


def sfd_path(ca: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    """
    Compute optimal path for summed Fréchet distance in the free space.

    Implements the algorithm from [1]_, with modification from [2]_.

    Parameters
    ----------
    ca: ndarray
        Accumulated cost array of summed Fréchet distance between two series.

    Returns
    -------
    ndarray
        Indices for the two series to get the path.

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
    path, path_len = _ca_path(ca)
    return path[-(len(path) - path_len + 1) :: -1]


@njit(cache=True)
def _ca_path(ca: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.int32], np.int32]:
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


def ssfd(
    P: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute summed square Fréchet distance between two series, using L2 norm.

    Parameters
    ----------
    P, Q : ndarray
        2D Numpy array with ``np.float64`` dtype. Rows are the points and
        columns are the dimensions.

    Returns
    -------
    ndarray
        Accumulated cost array for the summed square Fréchet distance.
        The distance is the last element of the array i.e. `ca[-1, -1]`.
        If *P* or *Q* is empty, return value is an empty array.

    See Also
    --------
    ssfd_path

    """
    return _sfd(cdist(P, Q) ** 2)


def ssfd_path(ca: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    """
    Compute optimal path for summed square Fréchet distance in the free space.

    Parameters
    ----------
    ca: ndarray
        Accumulated cost array of summed square Fréchet distance between two
        series.

    Returns
    -------
    ndarray
        Indices for the two series to get the path.

    See Also
    --------
    ssfd

    """
    path, path_len = _ca_path(ca)
    return path[-(len(path) - path_len + 1) :: -1]
