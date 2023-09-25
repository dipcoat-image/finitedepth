import dataclasses
from typing import Tuple


__all__ = [
    "Parameters",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """
    Additional parameters for :class:`FastExperiment` instance.

    Attributes
    ----------
    window : tuple
        Restricts the possible location of template to boost speed.
        Negative value turns off the restriction.
    """

    window: Tuple[int, int] = (-1, -1)
