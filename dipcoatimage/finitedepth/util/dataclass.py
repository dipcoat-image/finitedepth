"""
Dataclass utilities
===================

:mod:`dipcoatimage.finitedepth.util.dataclass` provides ``data_converter``, which
is :class:`cattrs.Converter` instance dedicated to convert the dataclasses in
:mod:`dipcoatimage.finitedepth`.

"""
from cattrs import Converter

__all__ = [
    "data_converter",
]


data_converter = Converter()
