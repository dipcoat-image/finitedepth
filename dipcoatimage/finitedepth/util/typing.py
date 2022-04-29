"""
Type annotation
===============

:mod:`dipcoatimage.finitedepth.util.geometry` provides type annotation objects.

"""

from typing import Protocol, Dict, Optional, Callable, Tuple


__all__ = [
    "DataclassProtocol",
    "OptionalROI",
    "IntROI",
]


class DataclassProtocol(Protocol):
    """Type annotation for dataclass type object."""

    # https://stackoverflow.com/a/70114354/11501976
    __dataclass_fields__: Dict
    __dataclass_params__: Dict
    __post_init__: Optional[Callable]


OptionalROI = Tuple[int, int, Optional[int], Optional[int]]
IntROI = Tuple[int, int, int, int]
