import dataclasses

__all__ = [
    "Parameters",
    "DrawOptions",
    "Data",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Additional parameters for :class:`Substrate` instance."""

    pass


@dataclasses.dataclass
class DrawOptions:
    """Drawing options for :class:`Substrate`."""

    pass


@dataclasses.dataclass
class Data:
    """Analysis data for :class:`Substrate`."""

    pass
