import dataclasses


__all__ = [
    "Parameters",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """
    Additional parameters for :class:`Experiment` instance.

    Attributes
    ----------
    fast : bool
        If True, optimization is used to speed up template matching.
        If False, brute force method is used (slow but reliable).
    """

    fast: bool = True
