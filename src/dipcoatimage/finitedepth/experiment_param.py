import dataclasses


__all__ = [
    "Parameters",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Additional parameters for :class:`Experiment` instance."""
