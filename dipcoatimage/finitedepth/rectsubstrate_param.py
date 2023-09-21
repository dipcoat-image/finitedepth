import dataclasses
import enum
from .util.parameters import Color, MarkerOptions, LineOptions


__all__ = [
    "PaintMode",
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
        default_factory=lambda: LineOptions(color=Color(0, 0, 255), linewidth=1)
    )
