import dataclasses
from .util.parameters import LineOptions


__all__ = [
    "Parameters",
    "DrawOptions",
    "Data",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Additional parameters for :class:`Reference` instance."""

    pass


@dataclasses.dataclass
class DrawOptions:
    """
    Drawing options for :class:`Reference`.

    Attributes
    ----------
    templateROI, substrateROI : LineOptions
        Determines how the ROIs are drawn.
    """

    templateROI: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(0, 255, 0), linewidth=1)
    )
    substrateROI: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(255, 0, 0), linewidth=1)
    )


@dataclasses.dataclass
class Data:
    """Analysis data for :class:`Reference`."""

    pass
