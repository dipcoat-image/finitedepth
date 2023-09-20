import dataclasses
import enum
from typing import Tuple


__all__ = [
    "DistanceMeasure",
    "Parameters",
    "PaintMode",
    "SubtractionMode",
    "DrawOptions",
]


class DistanceMeasure(enum.Enum):
    """
    Distance measures to compute the curve similarity.

    Members
    -------
    DTW
        Dynamic time warping.
    SDTW
        Squared dynamic time warping.
    """

    DTW = "DTW"
    SDTW = "SDTW"


@dataclasses.dataclass(frozen=True)
class Parameters:
    """
    Analysis parameters for :class:`RectLayerShape` instance.

    Attributes
    ----------
    KernelSize : tuple
        Size of the kernel for morphological operation to remove noises.
    ReconstructRadius : int
        Connected components outside of this radius from bottom corners of the
        substrate are regarded as image artifacts.
    RoughnessMeasure : DistanceMeasure
        Measure to compute layer roughness.

    """

    KernelSize: Tuple[int, int]
    ReconstructRadius: int
    RoughnessMeasure: DistanceMeasure


class PaintMode(enum.Enum):
    """
    Option to determine how the coating layer image is painted.

    Members
    -------
    ORIGINAL
        Show the original image.
    BINARY
        Show the binarized image.
    EMPTY
        Show empty image. Only the layer will be drawn.
    """

    ORIGINAL = "ORIGINAL"
    BINARY = "BINARY"
    EMPTY = "EMPTY"


class SubtractionMode(enum.Flag):
    """
    Option to determine how the template matching result will be displayed.

    Template matching result is shown by subtracting the pixels from the
    background.

    Members
    -------
    NONE
        Do not show the template matching result.
    TEMPLATE
        Subtract the template ROI.
    SUBSTRRATE
        Subtract the substrate ROI.
    FULL
        Subtract both template and substrate ROIs.

    """

    NONE = 0
    TEMPLATE = 1
    SUBSTRATE = 2
    FULL = TEMPLATE | SUBSTRATE


@dataclasses.dataclass
class DrawOptions:
    """
    Drawing options for :class:`RectLayerShape` instance.

    Attributes
    ----------
    paint : PaintMode
    sidelines : SideLineOptions
    """

    paint: PaintMode = PaintMode.BINARY
    subtraction: SubtractionMode = SubtractionMode.NONE
