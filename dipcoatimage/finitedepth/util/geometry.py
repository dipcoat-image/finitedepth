import numpy as np
import numpy.typing as npt

__all__ = [
    "project_on_lines",
    "lines_points",
    "project_on_polyline",
    "polyline_points",
    "polyline_parallel_area",
]


def project_on_lines(
    points: npt.NDArray, lines: npt.NDArray
) -> npt.NDArray[np.float64]:
    """
    Find orthogonal projections[1]_ of points onto lines.

    Parameters
    ----------
    points: ndarray
        Coordinate of the points which are projected to *lines*.
        The shape must be `(N, 1, D)` where `N` is the number of points and `D`
        is the dimension.
    lines: ndarray
        Coordinate of the points representing infinite lines.
        The shape must be `(M, 2, D)` where `M` is the number of lines and `D` is
        the dimension. The second axis is for the two points which constitute a
        line.

    Returns
    -------
    ndarray
        Parameters for the projection points on each line.
        The shape is `(N, M)`.

    Notes
    -----
    This function takes orthogonal projection onto the infinite lines represented
    by two points, not onto the line segment.

    See Also
    --------
    lines_points : Converts the line parameters to line points.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Projection_(linear_algebra)

    """
    A, B = lines.transpose(1, 0, 2)  # shape: (M, D)
    Ap = points - A  # shape: (N, M, D)
    AB = B - A  # shape: (M, D)
    return np.sum(Ap * AB, axis=-1) / np.sum(AB * AB, axis=-1)  # shape: (N, M)


def lines_points(
    parameters: npt.NDArray, lines: npt.NDArray
) -> npt.NDArray[np.float64]:
    """
    Get the coordinates of points in infinite lines from the parameters.

    Parameters
    ----------
    parameters: ndarray
        Parameters for the points in lines.
        The shape must be `(N, M)` where `N` is the number of points and `M` is
        the number of lines.
    lines: ndarray
        Coordinate of the points representing infinite lines.
        The shape must be `(M, 2, D)` where `M` is the number of lines and `D` is
        the dimension. The second axis is for the two points which constitute a
        line.

    Returns
    -------
    ndarray
        Coordinates of the points in each line.
        The shape is `(N, M, D)`.

    Notes
    -----
    A parameter is a real number which specifies the position of a point in the
    infinite line represented by two points. For example, a parameter $1.2$
    represents $P_1 + 1.2(P_2 - P_1)$ on the line which passes through $P_1$ and
    $P_2$.
    """
    A, B = lines.transpose(1, 0, 2)  # shape: (M, D)
    AB = B - A  # shape: (M, D)
    idx = parameters.astype(int)
    idx_shape = idx.shape + (-1,)
    return A[idx] + AB * (parameters - idx).reshape(idx_shape)


def project_on_polyline(
    points: npt.NDArray, line: npt.NDArray
) -> npt.NDArray[np.float64]:
    """
    Find the projections[1]_ of points onto a polyline[2]_.

    The resulting projections are the points in the *line* with the smallest
    distance to each *points*.

    Parameters
    ----------
    points: ndarray
        Coordinate of the points.
        The shape must be `(N, 1, D)` where `N` is the number of points and `D`
        is the dimension.
    line: ndarray
        Vertices of a polyline.
        The shape must be `(M, 1, D)` where `M` is the number of vertices and
        `D` is the dimension.

    Returns
    -------
    ndarray
        Parameters for the projection points on the polyline.

    Notes
    -----
    This function does not take the projection onto the extension of the line
    segment. All projections are confined in the polyline. If an orthogonal
    projection lies outside of the line segment, the closest point in the segment
    is returned.

    See Also
    --------
    polyline_points : Converts the polyline parameters to polyline points.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Projection_(mathematics)
    .. [2] https://en.wikipedia.org/wiki/Polygonal_chain
    """
    line = line.transpose(1, 0, 2)
    Ap = (points - line)[:, :-1]
    AB = np.diff(line, axis=1)
    t = np.clip(np.sum(Ap * AB, axis=-1) / np.sum(AB * AB, axis=-1), 0, 1)
    Projs = line[:, :-1] + (t[..., np.newaxis] * AB)
    dists = np.linalg.norm(Projs - points, axis=-1)
    closest_lines = np.argmin(dists, axis=1)
    return closest_lines + t[np.arange(len(points)), closest_lines]


def polyline_points(
    parameters: npt.NDArray, line: npt.NDArray
) -> npt.NDArray[np.float64]:
    """
    Get the coordinates of points in a polyline[1]_ from the parameters.

    Parameters
    ----------
    parameters: ndarray
        1-D array of non-negative real number.
    line: ndarray
        Vertices of a polyline.
        Length of the first axis must be the number of vertices.

    Returns
    -------
    ndarray
        Coordinates of the points in the polyline.

    Notes
    -----
    A parameter is a non-negative real number which specifies the position of a
    point in the polyline. The integer part describes which line segment the
    point belongs to and the decimal parts where the point is in the segment.
    For example, a parameter $1.2$ represents $P_1 + 0.2(P_2 - P_1)$ where
    $P_i$ is i-th vertex of the polyline.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Polygonal_chain

    """
    idx = parameters.astype(int)
    vec = np.diff(line, axis=0)[idx]
    idx_shape = (-1,) + (1,) * (len(vec.shape) - 1)
    return line[idx] + vec * (parameters - idx).reshape(idx_shape)


def polyline_parallel_area(line: npt.NDArray, t: float) -> np.float64:
    """
    Calculate the area formed by convex polyline[1]_ and its parallel curve[2]_.

    Parameters
    ----------
    line : ndarray
        Vertices of a polyline.
        The first dimension must be the number of vertices and the last dimension
        must be the dimension of the manifold.
    t : float
        Thickness between *line* and its parallel curve.

    Returns
    -------
    area : float

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Polygonal_chain
    .. [2] https://en.wikipedia.org/wiki/Parallel_curve
    """
    vec = np.diff(line, axis=0)
    d_l = np.linalg.norm(vec, axis=-1)
    d_theta = np.abs(np.diff(np.arctan2(vec[..., 1], vec[..., 0])))
    return np.float64(np.sum(d_l) * t + np.sum(d_theta) * (t**2) / 2)
