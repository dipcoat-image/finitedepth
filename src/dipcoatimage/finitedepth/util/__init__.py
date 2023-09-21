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

.. automodule:: dipcoatimage.finitedepth.util.parameters
   :members:

"""

from .importing import import_variable, ImportStatus, Importer
from .path import get_data_path
from .parameters import Color

__all__ = [
    "import_variable",
    "ImportStatus",
    "Importer",
    "get_data_path",
    "Color",
]
