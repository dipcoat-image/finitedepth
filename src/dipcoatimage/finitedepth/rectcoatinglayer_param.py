import dataclasses
import enum
import numpy as np
from .coatinglayer_param import SubtractionMode
from .util.parameters import LineOptions, PatchOptions
from typing import Tuple


__all__ = [
    "DistanceMeasure",
    "Parameters",
    "PaintMode",
    "DrawOptions",
    "LinesOptions",
    "DecoOptions",
    "Data",
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
    EMPTY
        Show empty image. Only the layer will be drawn.
    """

    ORIGINAL = "ORIGINAL"
    EMPTY = "EMPTY"


@dataclasses.dataclass
class DrawOptions:
    """
    Drawing options for :class:`RectLayerShape` instance.

    Attributes
    ----------
    paint : PaintMode
    subtraction : SubtractionMode
    """

    paint: PaintMode = PaintMode.ORIGINAL
    subtraction: SubtractionMode = SubtractionMode.NONE


@dataclasses.dataclass
class LinesOptions:
    """
    Parameters to draw lines in the image.

    Attributes
    ----------
    color : tuple
        Color of the lines in RGB
    linewidth : int
        Width of the line.
        Zero value is the flag to not draw the line.
    step : int
        Steps to jump the lines. `1` draws every line.

    """

    color: Tuple[int, int, int] = (0, 0, 0)
    linewidth: int = 1
    step: int = 1


@dataclasses.dataclass
class DecoOptions:
    """
    Options to show the analysis result on :class:`RectLayerShape`.

    Attributes
    ----------
    layer : PatchOptions
    contact_line, thickness, uniform_layer : LineOptions
    conformality, roughness : LinesOptions

    """

    layer: PatchOptions = dataclasses.field(
        default_factory=lambda: PatchOptions(
            fill=True,
            edgecolor=(0, 0, 255),
            facecolor=(255, 255, 255),
            linewidth=1,
        )
    )
    contact_line: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(0, 0, 255), linewidth=1)
    )
    thickness: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(0, 0, 255), linewidth=1)
    )
    uniform_layer: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(255, 0, 0), linewidth=1)
    )
    conformality: LinesOptions = dataclasses.field(
        default_factory=lambda: LinesOptions(color=(0, 255, 0), linewidth=1, step=10)
    )
    roughness: LinesOptions = dataclasses.field(
        default_factory=lambda: LinesOptions(color=(255, 0, 0), linewidth=1, step=10)
    )


@dataclasses.dataclass
class Data:
    """
    Analysis data for :class:`RectLayerShape` instance.

    - LayerLength_{Left, Right}: Distance between the bottom sideline of the
      substrate and the upper limit of the coating layer.
    - Conformality: Conformality of the coating layer.
    - AverageThickness: Average thickness of the coating layer.
    - Roughness: Roughness of the coating layer.
    - MaxThickness_{Left, Bottom, Right}: Number of the pixels for the maximum
      thickness on each region.

    The following data are the metadata for the analysis.

    - MatchError: Template matching error between 0 to 1. 0 means perfect match.

    """

    LayerLength_Left: np.float64
    LayerLength_Right: np.float64

    Conformality: float
    AverageThickness: np.float64
    Roughness: float

    MaxThickness_Left: np.float64
    MaxThickness_Bottom: np.float64
    MaxThickness_Right: np.float64

    MatchError: float
