"""
Type annotation
===============

:mod:`dipcoatimage.finitedepth.util.typing` provides type annotation objects.

"""

from typing import Optional, Tuple


__all__ = [
    "OptionalROI",
    "IntROI",
    "sanitize_ROI",
]


OptionalROI = Tuple[int, int, Optional[int], Optional[int]]
IntROI = Tuple[int, int, int, int]


def sanitize_ROI(roi: OptionalROI, h: int, w: int) -> IntROI:
    full_roi = (0, 0, w, h)
    max_vars = (w, h, w, h)

    ret = list(roi)
    for i, var in enumerate(roi):
        if var is None:
            ret[i] = full_roi[i]
        elif var < 0:
            ret[i] = max_vars[i] + var
    return tuple(ret)  # type: ignore[return-value]
