import dataclasses
import enum
from .util.parameters import Color


__all__ = [
    "Parameters",
    "PaintMode",
    "ROIOptions",
    "DrawOptions",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Additional parameters for :class:`SubstrateReference` instance."""

    pass


class PaintMode(enum.Enum):
    """
    Option to determine if the reference image is painted in original/binary.

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
class ROIOptions:
    """
    Parameters to draw the ROI box in the image.

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
    """Drawing options for :class:`SubstrateReference`."""

    paint: PaintMode = PaintMode.BINARY
    templateROI: ROIOptions = dataclasses.field(
        default_factory=lambda: ROIOptions(color=Color(0, 255, 0), thickness=1)
    )
    substrateROI: ROIOptions = dataclasses.field(
        default_factory=lambda: ROIOptions(color=Color(255, 0, 0), thickness=1)
    )
