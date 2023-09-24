import dataclasses
import enum
import numpy as np
from .util.parameters import Color, MarkerOptions, LineOptions


__all__ = [
    "PaintMode",
    "DrawOptions",
    "Data",
]


class PaintMode(enum.Enum):
    """
    Option to determine how the substrate image is painted.

    Members
    -------
    ORIGINAL
        Show the original substrate image.
    BINARY
        Show the binarized substrate image.
    CONTOUR
        Show the contour of the substrate.

    """

    ORIGINAL = "ORIGINAL"
    BINARY = "BINARY"
    CONTOUR = "CONTOUR"


@dataclasses.dataclass
class DrawOptions:
    """
    Drawing options for :class:`RectSubstrate`.

    Attributes
    ----------
    paint : PaintMode
    vertices : MarkerOptions
    sidelines : LineOptions
    """

    paint: PaintMode = PaintMode.BINARY
    vertices: MarkerOptions = dataclasses.field(
        default_factory=lambda: MarkerOptions(color=Color(0, 255, 0))
    )
    sidelines: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=Color(0, 0, 255))
    )


@dataclasses.dataclass
class Data:
    """
    Analysis data for :class:`RectSubstrate`.

    - ChipWidth: Number of the pixels between lower vertices of the substrate.

    """

    ChipWidth: np.float32
