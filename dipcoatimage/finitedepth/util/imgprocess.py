"""
Image processing functions
==========================

:mod:`dipcoatimage.finitedepth.util.imgprocess` provides functions to process
image.

"""

import cv2  # type: ignore
import numpy as np
from numpy.linalg import inv
import numpy.typing as npt
from typing import Tuple


__all__ = [
    "intrsct_pt_polar",
    "binarize",
    "colorize",
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


def binarize(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    if image.size == 0:
        return np.empty((0, 0), dtype=np.uint8)
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3:
        ch = image.shape[-1]
        if ch == 1:
            gray = image
        elif ch == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            raise TypeError(f"Image with invalid channel: {ch}")
    else:
        raise TypeError(f"Invalid image shape: {image}")
    _, ret = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if ret is None:
        ret = np.empty((0, 0), dtype=np.uint8)
    return ret


def colorize(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    if image.size == 0:
        return np.empty((0, 0, 0), dtype=np.uint8)
    if len(image.shape) == 2:
        ret = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        ch = image.shape[-1]
        if ch == 1:
            ret = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif ch == 3:
            ret = image.copy()
        else:
            raise TypeError(f"Image with invalid channel: {ch}")
    else:
        raise TypeError(f"Invalid image shape: {image}")
    return ret
