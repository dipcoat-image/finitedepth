import numpy as np
import numpy.typing as npt

__all__ = [
    "closest_points",
    "project_on_polylines",
    "polylines_external_points",
    "closest_in_polylines",
    "polylines_internal_points",
    "polyline_parallel_area",
    "equidistant_interpolate",
]


def closest_points(
    points1: npt.NDArray, points2: npt.NDArray
) -> npt.NDArray[np.int64]:
    """
    For each point in *points1*, find the closest point in *points2*.

    Parameters
    ----------
    points1: ndarray
        Coordinates of the points.
        The shape must be `(N, 1, D)` where `N` is the number of points and `D`
        is the dimension.
    points2: ndarray
        Coordinates of the points.
        The shape must be `(M, 1, D)` where `M` is the number of points and `D`
        is the dimension.

    Returns
    -------
    ndarray
        Indices of the closest points in *points2*.
        The shape is `(N,)`.

    """
    dist = np.linalg.norm(points1 - points2.transpose(1, 0, 2), axis=-1)
    return np.argmin(dist, axis=-1)


def project_on_polylines(
    points: npt.NDArray, polylines: npt.NDArray
) -> npt.NDArray[np.float64]:
    """
    Find orthogonal projections[1]_ of points onto extended edges of the
    polylines[2]_ which have same number of vertices.

    Parameters
    ----------
    points: ndarray
        Coordinates of the points which are projected to *polylines*.
        The shape must be `(N, 1, D)` where `N` is the number of points and `D`
        is the dimension.
    polylines: ndarray
        Coordinates of the polyline vertices.
        The shape must be `(M, V, D)` where `M` is the number of polylines,
        `V` is the number of vertices and `D` is the dimension.

    Returns
    -------
    ndarray
        Parameters for the projection points on each line.
        The shape is `(N, M, V - 1)`.

    Notes
    -----
    This function takes orthogonal projection onto the infinite lines represented
    by two points, not onto the line segment.

    A parameter is a real number which specifies the position of a point in the
    infinite line. For example, if a point on the line which passes through
    points $P_1$ and $P_2$ is specified the parameter $1.2$, its location is
    $P_1 + 1.2(P_2 - P_1)$.

    See Also
    --------
    polylines_external_points : Converts the parameters to points on extended
    edges of polylines.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Projection_(linear_algebra)
    .. [2] https://en.wikipedia.org/wiki/Polygonal_chain
    """
    A = polylines[:, :-1, :]  # shape: (M, V - 1, D)
    Ap = points[:, np.newaxis] - A[np.newaxis, :]  # shape: (N, M, V - 1, D)
    AB = np.diff(polylines, axis=1)[np.newaxis, ...]  # shape: (1, M, V - 1, D)
    return np.sum(Ap * AB, axis=-1) / np.sum(AB * AB, axis=-1)


def polylines_external_points(
    parameters: npt.NDArray, polylines: npt.NDArray
) -> npt.NDArray[np.float64]:
    """
    Get the coordinates of points in extended edges of polylines[1]_ from the
    parameters.

    Parameters
    ----------
    parameters: ndarray
        Parameters for the points in polylines.
        The shape must be `(N, M, V - 1)` where `N` is the number of points,
        where `M` is the number of polylines and `V` is the number of vertices.
    polylines: ndarray
        Coordinates of the polyline vertices.
        The shape must be `(M, V, D)` where `M` is the number of polylines,
        `V` is the number of vertices and `D` is the dimension.

    Returns
    -------
    ndarray
        Coordinates of the points in extended edges of polylines.
        The shape is `(N, M, V - 1, D)`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Polygonal_chain

    """
    A = polylines[:, :-1, :]  # shape: (M, V - 1, D)
    AB = np.diff(polylines, axis=1)[np.newaxis, ...]  # shape: (1, M, V - 1, D)
    t = parameters[..., np.newaxis]  # shape: (N, M, V - 1, 1)
    return A + AB * t


def closest_in_polylines(
    points: npt.NDArray, polylines: npt.NDArray
) -> npt.NDArray[np.float64]:
    """
    Find the projections[1]_ of points with the smallest distance onto
    polylines[2]_.

    Parameters
    ----------
    points: ndarray
        Coordinates of the points which are projected to *polylines*.
        The shape must be `(N, 1, D)` where `N` is the number of points and `D`
        is the dimension.
    polylines: ndarray
        Coordinates of the polyline vertices.
        The shape must be `(M, V, D)` where `M` is the number of polylines,
        `V` is the number of vertices and `D` is the dimension.

    Returns
    -------
    ndarray
        Parameters for the projection points on polylines.
        The shape is `(N, M)`.

    Notes
    -----
    This function does not take the projection onto the extension of the line
    segment. All projections are confined in each polyline. If an orthogonal
    projection lies outside of the line segment, the closest point in the segment
    is returned.

    See Also
    --------
    polylines_internal_points : Converts the parameters to points on internal
    edges of polylines.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Projection_(mathematics)
    .. [2] https://en.wikipedia.org/wiki/Polygonal_chain
    """
    t = np.clip(project_on_polylines(points, polylines), 0, 1)  # shape: (N, M, V - 1)
    prj = polylines_external_points(t, polylines)  # (N, M, V - 1, D)
    dists = np.linalg.norm(points[..., np.newaxis, :] - prj, axis=-1)  # (N, M, V - 1)
    closest_lines = np.argmin(dists, axis=-1)  # (N, M)
    idx_arr = np.ix_(*[np.arange(dim) for dim in closest_lines.shape])
    return closest_lines + t[idx_arr + (closest_lines,)]


def polylines_internal_points(
    parameters: npt.NDArray, polylines: npt.NDArray
) -> npt.NDArray[np.float64]:
    """
    Get the coordinates of points in edges of polylines[1]_ from the parameters.

    Parameters
    ----------
    parameters: ndarray
        Parameters for the points in polylines.
        The shape must be `(N, M)` where `N` is the number of points and `M` is
        the number of polylines.
    polylines: ndarray
        Coordinates of the polyline vertices.
        The shape must be `(M, V, D)` where `M` is the number of polylines,
        `V` is the number of vertices and `D` is the dimension.

    Returns
    -------
    ndarray
        Coordinates of the points in polylines.
        The shape is `(N, M, D)`.

    Notes
    -----
    The parameters are non-negative real numbers which specify the position of
    points in the polylines. The integer part describes which line segment the
    point belongs to and the decimal parts where the point is in the segment.
    For example, a parameter $1.2$ represents $P_1 + 0.2(P_2 - P_1)$ where
    $P_i$ is i-th vertex of the polyline.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Polygonal_chain

    """
    A = polylines[:, :-1, :].transpose(1, 0, 2)  # shape: (V - 1, M, D)
    AB = np.diff(polylines, axis=1).transpose(1, 0, 2)  # shape: (V - 1, M, D)
    idx = parameters.astype(int)  # shape: (N, M)
    idx_arr = np.ix_(*[np.arange(dim) for dim in idx.shape])
    t = (parameters - idx)[..., np.newaxis]
    return A[(idx,) + idx_arr[1:]] + t * AB[(idx,) + idx_arr[1:]]


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


def equidistant_interpolate(points, n) -> npt.NDArray[np.float64]:
    """
    Interpolate *points* with *n* number of points with same distances.

    Parameters
    ----------
    points: ndarray
        Points that are interpolated.
        The shape must be `(N, 1, D)` where `N` is the number of points and `D`
        is the dimension.
    n: int
        Number of new points.

    Returns
    -------
    ndarray
        Interpolated points with same distances.
        If `N` is positive number, the shape is `(n, 1, D)`. If `N` is zero,
        the shape is `(n, 0, D)`.
    """
    # https://stackoverflow.com/a/19122075
    if points.size == 0:
        return np.empty((n, 0, points.shape[-1]), dtype=np.float64)
    vec = np.diff(points, axis=0)
    dist = np.linalg.norm(vec, axis=-1)
    u = np.insert(np.cumsum(dist), 0, 0)
    t = np.linspace(0, u[-1], n)
    ret = np.column_stack([np.interp(t, u, a) for a in np.squeeze(points, axis=1).T])
    return ret.reshape((n,) + points.shape[1:])
