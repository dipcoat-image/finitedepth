import dataclasses
import enum
from .util.parameters import Color, LineOptions


__all__ = [
    "Parameters",
    "PaintMode",
    "DrawOptions",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Additional parameters for :class:`SubstrateReference` instance."""

    pass


class PaintMode(enum.Enum):
    """
    Option to determine if the reference image is painted in original/binary.

    Members
    -------
    ORIGINAL
        Show the original image.
    BINARY
        Show the binarized image.

    """

    ORIGINAL = "ORIGINAL"
    BINARY = "BINARY"


@dataclasses.dataclass
class DrawOptions:
    """
    Drawing options for :class:`SubstrateReference`.

    Attributes
    ----------
    paint : PaintMode
    templateROI, substrateROI : LineOptions
        Determines how the ROIs are drawn.
    """

    paint: PaintMode = PaintMode.BINARY
    templateROI: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=Color(0, 255, 0), linewidth=1)
    )
    substrateROI: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=Color(255, 0, 0), linewidth=1)
    )
