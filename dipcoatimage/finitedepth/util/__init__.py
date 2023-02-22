"""
Utilities
=========

:mod:`dipcoatimage.finitedepth.util` provides utility functions.

.. automodule:: dipcoatimage.finitedepth.util.imgprocess
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

from .imgprocess import intrsct_pt_polar, binarize, colorize
from .importing import import_variable, ImportStatus, Importer
from .path import get_samples_path, cwd
from .typing import DataclassProtocol, OptionalROI, IntROI
from .parameters import (
    CannyParameters,
    HoughLinesParameters,
    MorphologyClosingParameters,
    BinaryImageDrawMode,
)
from .testing import dict_includes

__all__ = [
    "intrsct_pt_polar",
    "binarize",
    "colorize",
    "import_variable",
    "ImportStatus",
    "Importer",
    "get_samples_path",
    "cwd",
    "DataclassProtocol",
    "OptionalROI",
    "IntROI",
    "CannyParameters",
    "HoughLinesParameters",
    "MorphologyClosingParameters",
    "BinaryImageDrawMode",
    "dict_includes",
]
