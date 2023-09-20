import dataclasses
import enum
from .util.parameters import Color, LineOptions


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
    sidelines : LineOptions
    """

    paint: PaintMode = PaintMode.BINARY
    sidelines: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=Color(0, 0, 255), thickness=1)
    )
