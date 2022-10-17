"""
GUI package for :mod:`dipcoatimage.finitedepth`.
"""

from dipcoatimage.finitedepth import __version__  # noqa

from .analysisgui import AnalysisGUI
from .mainwindow import MainWindow


__all__ = [
    "AnalysisGUI",
    "MainWindow",
]
