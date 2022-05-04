"""
Type annotation
===============

:mod:`dipcoatimage.finitedepth.util.geometry` provides type annotation objects.

"""

from typing import Protocol, Dict, Optional, Tuple


__all__ = [
    "DataclassProtocol",
    "OptionalROI",
    "IntROI",
]


class DataclassProtocol(Protocol):
    """Type annotation for dataclass type object."""

    # https://stackoverflow.com/a/55240861/11501976
    __dataclass_fields__: Dict


OptionalROI = Tuple[int, int, Optional[int], Optional[int]]
IntROI = Tuple[int, int, int, int]
