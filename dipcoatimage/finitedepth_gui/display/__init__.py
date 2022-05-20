"""
Displaying widgets
==================

This module provides the widgets to display the visualized results.

"""


from .maindisplay import (
    PreviewableNDArrayVideoPlayer,
    ROIVideoWidget,
    ROICameraWidget,
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


__all__ = [
    "PreviewableNDArrayVideoPlayer",
    "NDArrayROILabel",
    "coords_label2pixmap",
    "coords_pixmap2label",
    "DisplayWidgetToolBar",
    "get_icons_path",
    "ROIVideoWidget",
    "ROICameraWidget",
    "MainDisplayWindow",
]
