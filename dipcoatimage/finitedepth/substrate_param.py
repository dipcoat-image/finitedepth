import dataclasses
import enum


__all__ = [
    "Parameters",
    "PaintMode",
    "DrawOptions",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Additional parameters for :class:`Substrate` instance."""

    pass


class PaintMode(enum.Enum):
    """
    Option to determine if the substrate image is painted in original/binary.

    Attributes
    ----------
    ORIGINAL
        Show the original image.
    BINARY
        Show the binarized image.

    """

    ORIGINAL = "ORIGINAL"
    BINARY = "BINARY"


@dataclasses.dataclass
class DrawOptions:
    """Drawing options for :class:`SubstrateReference`."""

    paint: PaintMode = PaintMode.BINARY
