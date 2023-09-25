"""
Image processing functions
==========================

:mod:`dipcoatimage.finitedepth.util.imgprocess` provides functions to process
image.

"""

import cv2
import numpy as np
import numpy.typing as npt


__all__ = [
    "binarize",
    "colorize",
]


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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        ret = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3:
        ch = image.shape[-1]
        if ch == 1:
            ret = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif ch == 3:
            ret = image.copy()
        else:
            raise TypeError(f"Image with invalid channel: {ch}")
    else:
        raise TypeError(f"Invalid image shape: {image}")
    return ret
