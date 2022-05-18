"""
Utilities
=========

:mod:`dipcoatimage.finitedepth.util` provides utility functions.

.. automodule:: dipcoatimage.finitedepth.util.dataclass
   :members:

.. automodule:: dipcoatimage.finitedepth.util.geometry
   :members:

.. automodule:: dipcoatimage.finitedepth.util.importing
   :members:

.. automodule:: dipcoatimage.finitedepth.util.path
   :members:

.. automodule:: dipcoatimage.finitedepth.util.typing
   :members:

.. automodule:: dipcoatimage.finitedepth.util.parameters
   :members:

.. automodule:: dipcoatimage.finitedepth.util.testing
   :members:

"""

from .dataclass import data_converter
from .geometry import get_extended_line, intrsct_pt_polar
from .importing import import_variable
from .path import get_samples_path, cwd
from .typing import DataclassProtocol, OptionalROI, IntROI
from .parameters import (
    CannyParameters,
    HoughLinesParameters,
    BinaryImageDrawMode,
)
from .testing import dict_includes

__all__ = [
    "data_converter",
    "get_extended_line",
    "intrsct_pt_polar",
    "import_variable",
    "get_samples_path",
    "cwd",
    "DataclassProtocol",
    "OptionalROI",
    "IntROI",
    "CannyParameters",
    "HoughLinesParameters",
    "BinaryImageDrawMode",
    "dict_includes",
]
