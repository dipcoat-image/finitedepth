import numpy as np
import numpy.typing as npt

__all__ = [
    "project_on_polyline",
    "polyline_points",
]


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
    Transform the polyline parameters to points in the polyline.

    Parameters
    ----------
    parameters: ndarray
        1-D array of non-negative real number.
    line: ndarray
        Vertices of a polyline.
        The first dimension must be the number of vertices.

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

    """
    idx = parameters.astype(int)
    vec = np.diff(line, axis=0)[idx]
    idx_shape = (-1,) + (1,) * (len(vec.shape) - 1)
    return line[idx] + vec * (parameters - idx).reshape(idx_shape)
