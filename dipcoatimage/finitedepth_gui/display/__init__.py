"""
Displaying widgets
==================

This module provides the widgets to display the visualized results.

"""


from .maindisplay import (
    MainDisplayWindow,
)
from .roidisplay import (
    NDArrayROILabel,
    coords_label2pixmap,
    coords_pixmap2label,
)
from .toolbar import (
    DisplayWidgetToolBar,
    get_icons_path,
)
from .videostream import PreviewableNDArrayVideoPlayer


__all__ = [
    "PreviewableNDArrayVideoPlayer",
    "NDArrayROILabel",
    "coords_label2pixmap",
    "coords_pixmap2label",
    "DisplayWidgetToolBar",
    "get_icons_path",
    "MainDisplayWindow",
]
