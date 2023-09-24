import dataclasses
import enum
from .util.parameters import PatchOptions

__all__ = [
    "Parameters",
    "SubtractionMode",
    "DrawOptions",
    "DecoOptions",
    "Data",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Additional parameters for :class:`CoatingLayer` instance."""

    pass


class SubtractionMode(enum.Enum):
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

    NONE = "NONE"
    TEMPLATE = "TEMPLATE"
    SUBSTRATE = "SUBSTRATE"
    FULL = "FULL"


@dataclasses.dataclass
class DrawOptions:
    """
    Drawing options for :class:`CoatingLayer` instance.

    Attributes
    ----------
    subtraction : SubtractionMode
    """

    subtraction: SubtractionMode = SubtractionMode.NONE


@dataclasses.dataclass
class DecoOptions:
    """
    Options to show the coating layer of :class:`CoatingLayer`.

    Attributes
    ----------
    layer : PatchOptions

    """

    layer: PatchOptions = dataclasses.field(
        default_factory=lambda: PatchOptions(
            fill=True,
            edgecolor=(0, 0, 255),
            facecolor=(255, 255, 255),
            linewidth=1,
        )
    )


@dataclasses.dataclass
class Data:
    """Analysis data for :class:`CoatingLayer`."""

    pass
