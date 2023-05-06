import numpy as np
import numpy.typing as npt

__all__ = [
    "find_polyline_projections",
]


def find_polyline_projections(points, lines) -> npt.NDArray[np.float64]:
    """
    Find the projection relation between points and a polyline.

    Parameters
    ----------
    points: ndarray
        Coordinate of the points.
        The shape must be `(N, 1, D)` where `N` is the number of points and `D`
        is the dimension.
    lines: ndarray
        Vertices of a polyline.
        The shape must be `(M, 1, D)` where `M` is the number of vertices and
        `D` is the dimension.

    Returns
    -------
    ndarray
        Projection relation between points and polyline segments.
        The shape is `(N, 2)`, where `N` is same as the number of points.
        Index of the row represents the index of point in *points*. Columns
        represent the index of segment where the point is projected to, and
        scalar to find the projection.
        For example, let `n`-th row is `[m, t]`. This means that `n`-th point in
        *point* is projected to the line between `m`-th vertex (A) and `m + 1`-th
        vertex (B) in *lines*. The projection point is `A + t*(B - A)`.
    """
    lines = lines.transpose(1, 0, 2)
    Ap = (points - lines)[:, :-1]
    AB = np.diff(lines, axis=1)
    t = np.clip(np.sum(Ap * AB, axis=-1) / np.sum(AB * AB, axis=-1), 0, 1)
    Projs = lines[:, :-1] + (t[..., np.newaxis] * AB)
    dists = np.linalg.norm(Projs - points, axis=-1)

    closest_lines = np.argmin(dists, axis=1)
    return np.stack([closest_lines, t[np.arange(len(points)), closest_lines]]).T
