"""
Import data view
================

V2 for importwidget.py
"""

from PySide6.QtWidgets import QGroupBox, QLineEdit, QVBoxLayout


__all__ = [
    "ImportDataView",
]


class ImportDataView(QGroupBox):
    """Widget to display import data."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._varNameLineEdit = QLineEdit()
        self._moduleNameLineEdit = QLineEdit()

        self._varNameLineEdit.setPlaceholderText("Variable name")
        self._moduleNameLineEdit.setPlaceholderText("Module")

        layout = QVBoxLayout()
        layout.addWidget(self._varNameLineEdit)
        layout.addWidget(self._moduleNameLineEdit)
        self.setLayout(layout)
