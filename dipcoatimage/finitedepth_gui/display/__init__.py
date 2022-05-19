"""
Displaying widgets
==================

This module provides the widgets to display the visualized results.

"""


from .display import (
    PreviewableNDArrayVideoPlayer,
    NDArrayROILabel,
    coords_label2pixmap,
    coords_pixmap2label,
    DisplayWidgetToolBar,
    get_icons_path,
    ROIVideoWidget,
    ROICameraWidget,
    MainDisplayWindow,
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
