"""
GUI package for :mod:`dipcoatimage.finitedepth`.
"""

from dipcoatimage.finitedepth import __version__  # noqa

from .roi import (
    ROIModel,
    ROIWidget,
)


__all__ = ["ROIModel", "ROIWidget"]
