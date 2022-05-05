"""
Geometry functions
==================

:mod:`dipcoatimage.finitedepth.util.geometry` provides geometry functions.

"""

import numpy as np
from numpy.linalg import inv
from typing import Tuple


__all__ = [
    "get_extended_line",
    "intrsct_pt_polar",
]


def get_extended_line(
    frame_shape: Tuple[int, int], p1: Tuple[int, int], p2: Tuple[int, int]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Extends the line and return its cross points with frame edge.

    Parameters
    ==========

    frame_shape
        Shape of the frame in ``(height, width)``.

    p1, p2
        ``(x, y)`` coorindates of the two points of line.

    Returns
    =======

    ext_p1, ext_p2
        ``(x, y)`` coorindates of the two points of extended.

    Raises
    ======

    ZeroDivisionError
        *p1* and *p2* are same.

    Examples
    ========

    >>> from dipcoatimage.finitedepth.util import get_extended_line
    >>> get_extended_line((1080, 1920), (192, 108), (1920, 1080))
    ((0, 0), (1920, 1080))
    >>> get_extended_line((1080, 1920), (300, 500), (400, 500))
    ((0, 500), (1920, 500))
    >>> get_extended_line((1080, 1920), (900, 400), (900, 600))
    ((900, 0), (900, 1080))

    """
    h, w = frame_shape
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and y1 == y2:
        raise ZeroDivisionError("Duplicate points: %s and %s" % (p1, p2))

    elif x1 != x2 and y1 != y2:
        candidates = (
            (int((x2 - x1) / (y2 - y1) * (0 - y1) + x1), 0),
            (int((x2 - x1) / (y2 - y1) * (h - y1) + x1), h),
            (0, int((y2 - y1) / (x2 - x1) * (0 - x1) + y1)),
            (w, int((y2 - y1) / (x2 - x1) * (w - x1) + y1)),
        )

        ret = []
        for x, y in set(candidates):
            if 0 <= x <= w and 0 <= y <= h:
                ret.append((x, y))
        ret.sort()

        ext_p1, ext_p2 = ret

    elif x1 == x2 and y1 != y2:
        ext_p1 = (x1, 0)
        ext_p2 = (x1, h)

    elif x1 != x2 and y1 == y2:
        ext_p1 = (0, y1)
        ext_p2 = (w, y1)

    return ext_p1, ext_p2


def intrsct_pt_polar(r1: float, t1: float, r2: float, t2: float) -> Tuple[float, float]:
    """
    Find the Cartesian coordinates of the intersecting point of two
    lines by their polar parameters.

    Parameters
    ==========

    r1, t1, r2, t2
        Radius and angle for the first and second line.

    Returns
    =======

    x, y
        Cartesian coordinates of the intersecting point.

    Examples
    ========

    >>> from dipcoatimage.finitedepth.util import intrsct_pt_polar
    >>> from numpy import pi
    >>> x, y = intrsct_pt_polar(10, pi/3, 5, pi/6)
    >>> round(x, 2)
    -1.34
    >>> round(y, 2)
    12.32

    """
    mat = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    vec = np.array([r1, r2])
    ret = inv(mat) @ vec
    return tuple(float(i) for i in ret)  # type: ignore
