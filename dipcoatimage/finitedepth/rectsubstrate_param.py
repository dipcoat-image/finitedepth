import dataclasses
import enum
from .util.parameters import Color


__all__ = [
    "PaintMode",
    "SideLineOptions",
    "DrawOptions",
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
    EDGES
        Show the edges of the substrate image.

    """

    ORIGINAL = "ORIGINAL"
    BINARY = "BINARY"
    EDGES = "EDGES"


@dataclasses.dataclass
class SideLineOptions:
    """
    Parameters to draw the sidelines.

    Attributes
    ----------
    color
        Color of the line.
    thickness
        Thickness of the line.
        Zero value is the flag to not draw the feature.

    """

    color: Color = dataclasses.field(default_factory=Color)
    thickness: int = 1


@dataclasses.dataclass
class DrawOptions:
    """
    Drawing options for :class:`RectSubstrate`.

    Attributes
    ----------
    paint : PaintMode
    sidelines : SideLineOptions
    """

    paint: PaintMode = PaintMode.BINARY
    sidelines: SideLineOptions = dataclasses.field(
        default_factory=lambda: SideLineOptions(color=Color(0, 0, 255), thickness=1)
    )
