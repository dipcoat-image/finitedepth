"""
Parameter classes
=================

:mod:`dipcoatimage.finitedepth.util.parameters` provides common classes to
construct dataclasses for image analysis classes.

"""

import dataclasses
import enum
import numpy as np
from typing import Tuple


__all__ = [
    "HoughLinesParameters",
    "BinaryImageDrawMode",
    "MorphologyClosingParameters",
]


@dataclasses.dataclass(frozen=True)
class HoughLinesParameters:
    """Parameters for :func:`cv2.HoughLines`."""

    rho: float
    theta: float
    threshold: int
    srn: float = 0.0
    stn: float = 0.0
    min_theta: float = 0.0
    max_theta: float = np.pi


@dataclasses.dataclass(frozen=True)
class MorphologyClosingParameters:
    kernelSize: Tuple[int, int]
    anchor: Tuple[int, int] = (-1, -1)
    iterations: int = 1


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
