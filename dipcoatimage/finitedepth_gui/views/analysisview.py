"""
Analysis view
=============

V2 for controlwidgets/analysiswidget.py
"""

from PySide6.QtWidgets import (
    QWidget,
    QStyledItemDelegate,
)

__all__ = [
    "AnalysisView",
    "AnalysisArgsDelegate",
]


class AnalysisView(QWidget):
    ...


class AnalysisArgsDelegate(QStyledItemDelegate):
    ...
