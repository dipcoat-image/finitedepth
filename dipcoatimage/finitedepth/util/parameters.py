"""
Parameter classes
=================

:mod:`dipcoatimage.finitedepth.util.parameters` provides common classes to
construct dataclasses for image analysis classes.

"""

import dataclasses
import enum
import numpy as np


__all__ = [
    "CannyParameters",
    "HoughLinesParameters",
    "BinaryImageDrawMode",
]


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


class BinaryImageDrawMode(enum.Enum):
    """
    Option to determine if the image is drawn in original/binary.

    Attributes
    ==========

    ORIGINAL
        Show the original image.

    BINARY
        Show the binarized image.

    """

    ORIGINAL = "ORIGINAL"
    BINARY = "BINARY"
