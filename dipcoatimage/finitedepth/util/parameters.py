"""
Parameter dataclasses
=====================

:mod:`dipcoatimage.finitedepth.util.parameters` provides common dataclasses for
*parameters* argument of image analysis classes.

"""

import cv2  # type: ignore
import dataclasses
import numpy as np


__all__ = [
    "ThresholdParameters",
    "CannyParameters",
    "HoughLinesParameters",
]


@dataclasses.dataclass(frozen=True)
class ThresholdParameters:
    """
    Parameters for :func:`cv2.threshold`.

    This parameters defaults to Otsu's binarization.

    Attributes
    ==========

    thresh

    maxval

    type
        Default is ``cv2.THRESH_BINARY | cv2.THRESH_OTSU`` value.

    """

    thresh: int = 0
    maxval: int = 255
    type: int = cv2.THRESH_BINARY | cv2.THRESH_OTSU


@dataclasses.dataclass(frozen=True)
class CannyParameters:
    """Parameters for :func:`cv2.Canny`."""

    threshold1: float
    threshold2: float
    apertureSize: int = 3
    L2gradient: bool = False


@dataclasses.dataclass(frozen=True)
class HoughLinesParameters:
    """Parameters for :func:`cv2.HoughLines`."""

    rho: float
    theta: float
    threshold: int
    srn: float = 0
    stn: float = 0
    min_theta: float = 0
    max_theta: float = np.pi
