"""
Image processing functions
==========================

:mod:`dipcoatimage.finitedepth.util.imgprocess` provides functions to process
image.

"""

import cv2  # type: ignore
import numpy as np
import numpy.typing as npt
from typing import Tuple


__all__ = [
    "match_template",
    "images_XOR",
    "images_ANDXOR",
    "binarize",
    "colorize",
]


def match_template(
    image: npt.NDArray[np.uint8], template: npt.NDArray[np.uint8]
) -> Tuple[float, Tuple[int, int]]:
    """Perform template matching using :obj:`cv2.TM_SQDIFF_NORMED`."""
    res = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
    score, _, loc, _ = cv2.minMaxLoc(res)
    return (score, loc)


def images_XOR(
    img1: npt.NDArray[np.bool_],
    img2: npt.NDArray[np.bool_],
    point: Tuple[int, int] = (0, 0),
) -> npt.NDArray[np.bool_]:
    """
    Subtract *img2* from *img1* at *point* by XOR operation.

    This function leaves the pixels that exist either in *img1* or *img2*. It
    can be used to visualize the template matching error.

    See Also
    --------

    images_ANDXOR
    """
    H, W = img1.shape
    h, w = img2.shape
    x0, y0 = point
    x1, y1 = x0 + w, y0 + h

    img1 = img1.copy()
    img1_crop = img1[max(y0, 0) : min(y1, H), max(x0, 0) : min(x1, W)]
    img2_crop = img2[max(-y0, 0) : min(H - y0, h), max(-x0, 0) : min(W - x0, w)]
    img1_crop ^= img2_crop
    return img1


def images_ANDXOR(
    img1: npt.NDArray[np.bool_],
    img2: npt.NDArray[np.bool_],
    point: Tuple[int, int] = (0, 0),
) -> npt.NDArray[np.bool_]:
    """
    Subtract *img2* from *img1* at *point* by AND and XOR operation.

    This function leaves the pixels that exist in *img1* but not in *img2*. It
    can be used to extract the coating layer pixels.

    See Also
    --------

    images_XOR
    """
    H, W = img1.shape
    h, w = img2.shape
    x0, y0 = point
    x1, y1 = x0 + w, y0 + h

    img1 = img1.copy()
    img1_crop = img1[max(y0, 0) : min(y1, H), max(x0, 0) : min(x1, W)]
    img2_crop = img2[max(-y0, 0) : min(H - y0, h), max(-x0, 0) : min(W - x0, w)]
    common = img1_crop & img2_crop
    img1_crop ^= common
    return img1


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
