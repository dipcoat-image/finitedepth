import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from typing import Tuple


__all__ = [
    "dfd",
    "dfd_pair",
    "dtw",
    "dtw_path",
    "sfd",
    "sfd_path",
    "ssfd",
    "ssfd_path",
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


@njit(cache=True)
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


def dfd_pair(ca: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    """
    Find the point pair which determines the discrete Fréchet distance.

    Parameters
    ----------
    ca: ndarray
        Accumulated cost array of discrete Fréchet distance between two curves.

    Returns
    -------
    ndarray
        Indices for the two curves to get the point pair.

    See Also
    --------
    dfd
    """
    i, j = np.nonzero(ca == ca[-1, -1])
    return np.array([[i[0], j[0]]], dtype=np.int32)


def dtw(
    P: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute dynamic time warping[1]_ with L2 norm.

    Implements the algorithm from [1]_, with modification from [2]_.

    Parameters
    ----------
    P, Q : ndarray
        2D Numpy array with ``np.float64`` dtype. Rows are the points and
        columns are the dimensions.

    Returns
    -------
    ndarray
        Accumulated cost array for dynamic time warping.
        The distance is the last element of the array i.e. `ca[-1, -1]`.
        If *P* or *Q* is empty, return value is an empty array.

    References
    ----------
    .. [1] Senin, Pavel. "Dynamic time warping algorithm review." Information and
       Computer Science Department University of Hawaii at Manoa Honolulu,
       USA 855.1-23 (2008): 40.

    .. [2] https://pypi.org/project/similaritymeasures/

    """
    return _dtw(cdist(P, Q))


@njit(cache=True)
def _dtw(freespace: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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


def dtw_path(ca: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    """
    Compute optimal path for dynamic time warping in the free space.

    Implements the algorithm from [1]_, with modification from [2]_.

    Parameters
    ----------
    ca: ndarray
        Accumulated cost array of dynamic time warping between two curves.

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
    dtw

    """
    path, path_len = _dtw_path(ca)
    return path[-(len(path) - path_len + 1) :: -1]


@njit(cache=True)
def _dtw_path(ca: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.int32], np.int32]:
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


def sfd(
    P: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute summed Fréchet distance[1]_ using L2 norm for both curve space and
    paramer space.

    The points are parameterized by the arc length.

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
    .. [1] Brakatsoulas, Sotiris, et al. "On map-matching vehicle tracking data."
       Proceedings of the 31st international conference on Very large data bases.
       2005.

    """
    P_dists = np.linalg.norm(np.diff(P, axis=0), axis=-1)
    Q_dists = np.linalg.norm(np.diff(Q, axis=0), axis=-1)
    return _sfd(cdist(P, Q), P_dists / np.sum(P_dists), Q_dists / np.sum(Q_dists))


@njit(cache=True)
def _sfd(
    freespace: npt.NDArray[np.float64],
    param1: npt.NDArray[np.float64],
    param2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    p, q = freespace.shape
    ca = np.zeros((p, q), dtype=np.float64)
    if p == 0 or q == 0:
        return ca

    ca[0, 0] = freespace[0, 0]

    for i in range(1, p):
        ca[i, 0] = ca[i - 1, 0] + freespace[i, 0] * param1[i - 1]

    for j in range(1, q):
        ca[0, j] = ca[0, j - 1] + freespace[0, j] * param2[j - 1]

    for i in range(1, p):
        for j in range(1, q):
            dx = param1[i - 1]
            dy = param2[j - 1]
            ca[i, j] = min(
                ca[i - 1, j] + freespace[i, j] * dx,
                ca[i, j - 1] + freespace[i, j] * dy,
                ca[i - 1, j - 1] + freespace[i, j] * np.sqrt(dx**2 + dy**2),
            )

    return ca


def sfd_path(ca: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    """
    Compute optimal path for summed Fréchet distance variants in the free space.

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


@njit(cache=True)
def _sfd_path(ca: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.int32], np.int32]:
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
    Compute summed square Fréchet distance with Euclidean metric.

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

    """
    return _sfd(cdist(P, Q), 2)


# SSFD path can be acquired using algorithm identical to SFD path.
# We still provide easy aliasing.
ssfd_path = sfd_path
