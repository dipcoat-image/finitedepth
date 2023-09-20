import dataclasses


__all__ = [
    "Parameters",
    "DrawOptions",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Additional parameters for :class:`Substrate` instance."""

    pass


@dataclasses.dataclass
class DrawOptions:
    """Drawing options for :class:`SubstrateReference`."""

    pass
