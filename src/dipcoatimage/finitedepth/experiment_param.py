import dataclasses
from typing import Tuple


__all__ = [
    "Parameters",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """
    Additional parameters for :class:`Experiment` instance.

    Attributes
    ----------
    window : tuple
        Restricts the possible location of template to boost speed.
        Negative value means no restriction in corresponding axis.
    """

    window: Tuple[int, int] = (-1, -1)
