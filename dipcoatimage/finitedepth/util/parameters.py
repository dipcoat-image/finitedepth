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
    "FeatureDrawingOptions",
]


@dataclasses.dataclass
class Color:
    red: int = 0
    green: int = 0
    blue: int = 0


@dataclasses.dataclass
class LineOptions:
    """
    Parameters to draw a line in the image.

    Attributes
    ----------
    color : Color
        Color of the line.
    thickness : nonnegative int
        Thickness of the line.
        Zero value is the flag to not draw the feature.

    """

    color: Color = dataclasses.field(default_factory=Color)
    thickness: int = 1


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

    color: Color = dataclasses.field(default_factory=Color)
    thickness: int = 1
    drawevery: int = 1
