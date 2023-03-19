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

"""

from .imgprocess import binarize, colorize
from .importing import import_variable, ImportStatus, Importer
from .path import get_samples_path, cwd
from .typing import DataclassProtocol, OptionalROI, IntROI
from .parameters import (
    HoughLinesParameters,
    MorphologyClosingParameters,
    BinaryImageDrawMode,
    SubstrateSubtractionMode,
    Color,
    FeatureDrawingOptions,
)

__all__ = [
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
    "HoughLinesParameters",
    "MorphologyClosingParameters",
    "BinaryImageDrawMode",
    "SubstrateSubtractionMode",
    "Color",
    "FeatureDrawingOptions",
]
