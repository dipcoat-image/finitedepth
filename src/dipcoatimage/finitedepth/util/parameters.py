"""
Parameter classes
=================

:mod:`dipcoatimage.finitedepth.util.parameters` provides common classes to
construct dataclasses for image analysis classes.

"""

import dataclasses
import enum


__all__ = [
    "Color",
    "LineOptions",
    "PatchOptions",
    "MarkerTypes",
    "MarkerOptions",
]


@dataclasses.dataclass
class Color:
    red: int
    green: int
    blue: int


@dataclasses.dataclass
class LineOptions:
    """
    Parameters to draw a line in the image.

    Attributes
    ----------
    color : Color
    linewidth : int
        Width of the line.
        Zero value is the flag to not draw the line.

    """

    color: Color = dataclasses.field(default_factory=lambda: Color(0, 0, 0))
    linewidth: int = 1


@dataclasses.dataclass
class PatchOptions:
    """
    Parameters to draw a patch in the image.

    Attributes
    ----------
    fill : bool
        Whether to fill the patch with facecolor.
    edgecolor, facecolor : Color
    linewidth : int
        Width of the edge.
        Zero value is the flag to not draw the edge.

    """

    fill: bool = True
    edgecolor: Color = dataclasses.field(default_factory=lambda: Color(0, 0, 0))
    facecolor: Color = dataclasses.field(default_factory=lambda: Color(0, 0, 0))
    linewidth: int = 1


class MarkerTypes(enum.Enum):
    CROSS = "CROSS"
    TILTED_CROSS = "TILTED_CROSS"
    STAR = "STAR"
    DIAMOND = "DIAMOND"
    SQUARE = "SQUARE"
    TRIANGLE_UP = "TRIANGLE_UP"
    TRIANGLE_DOWN = "TRIANGLE_DOWN"


@dataclasses.dataclass
class MarkerOptions:
    """
    Parameters to draw a marker in the image.

    Attributes
    ----------
    color : Color
    marker : MarkerTypes
    linewidth : int
    markersize : int

    """

    color: Color = dataclasses.field(default_factory=lambda: Color(0, 0, 0))
    marker: MarkerTypes = MarkerTypes.CROSS
    linewidth: int = 1
    markersize: int = 20
