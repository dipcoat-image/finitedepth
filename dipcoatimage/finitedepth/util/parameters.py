"""
Parameter classes
=================

:mod:`dipcoatimage.finitedepth.util.parameters` provides common classes to
construct dataclasses for image analysis classes.

"""

import dataclasses


__all__ = [
    "Color",
    "LineOptions",
    "PatchOptions",
    "FeatureDrawingOptions",
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


@dataclasses.dataclass
class FeatureDrawingOptions:
    """
    Parameters to paint the arbitrary feature in the image.

    Attributes
    ==========

    color
        Color of the line or face in RGB

    thickness
        Thickness of the line.
        Zero value is the flag to not draw the feature.
        Negative value can be the flag to fill the feature interior.

    drawevery
        Every N-th feature will be drawn.
        This option pertains when drawing of multiple features is controlled by
        a single :class:`FeatureDrawingOptions` instance.

    """

    color: Color = dataclasses.field(default_factory=lambda: Color(0, 0, 0))
    thickness: int = 1
    drawevery: int = 1
