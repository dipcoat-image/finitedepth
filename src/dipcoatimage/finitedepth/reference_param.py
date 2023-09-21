import dataclasses
from .util.parameters import Color, LineOptions


__all__ = [
    "Parameters",
    "DrawOptions",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Additional parameters for :class:`SubstrateReference` instance."""

    pass


@dataclasses.dataclass
class DrawOptions:
    """
    Drawing options for :class:`SubstrateReference`.

    Attributes
    ----------
    templateROI, substrateROI : LineOptions
        Determines how the ROIs are drawn.
    """

    templateROI: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=Color(0, 255, 0), linewidth=1)
    )
    substrateROI: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=Color(255, 0, 0), linewidth=1)
    )
