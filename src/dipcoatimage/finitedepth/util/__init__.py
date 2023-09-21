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

from .imgprocess import match_template, images_XOR, images_ANDXOR, binarize, colorize
from .importing import import_variable, ImportStatus, Importer
from .path import get_samples_path
from .typing import DataclassProtocol, OptionalROI, IntROI, sanitize_ROI
from .parameters import Color

__all__ = [
    "match_template",
    "images_XOR",
    "images_ANDXOR",
    "binarize",
    "colorize",
    "import_variable",
    "ImportStatus",
    "Importer",
    "get_samples_path",
    "DataclassProtocol",
    "OptionalROI",
    "IntROI",
    "sanitize_ROI",
    "Color",
]
