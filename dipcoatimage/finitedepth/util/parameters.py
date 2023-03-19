"""
Parameter classes
=================

:mod:`dipcoatimage.finitedepth.util.parameters` provides common classes to
construct dataclasses for image analysis classes.

"""

import dataclasses
import enum


__all__ = [
    "BinaryImageDrawMode",
    "Color",
    "FeatureDrawingOptions",
]


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


@dataclasses.dataclass
class Color:
    red: int = 0
    green: int = 0
    blue: int = 0


@dataclasses.dataclass
class FeatureDrawingOptions:
    """
    Parameters to paint the arbitrary feature in the image.

    Attributes
    ==========

    color
        Color in RGB

    thickness
        Thickness of the line.
        Zero value is the flag to not draw the feature.
        Negative value can be the flag to fill the feature interior.

    """

    color: Color = Color(0, 0, 0)
    thickness: int = 0
