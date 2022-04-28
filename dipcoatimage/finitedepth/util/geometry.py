"""
Geometry functions
==================

:mod:`dipcoatimage.finitedepth.util.geometry` provides geometry functions.

"""

from typing import Tuple


__all__ = [
    "get_extended_line",
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
