"""
Geometry functions
==================

:mod:`dipcoatimage.finitedepth.util.geometry` provides geometry functions.

"""

import numpy as np
from numpy.linalg import inv
from typing import Tuple


__all__ = [
    "intrsct_pt_polar",
]


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
