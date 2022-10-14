"""
Import data view
================

V2 for importwidget.py
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLineEdit,
    QVBoxLayout,
)


__all__ = [
    "ImportDataView",
]


class ImportDataView(QGroupBox):
    """Widget to display import data."""

    editingFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._varNameLineEdit = QLineEdit()
        self._moduleNameLineEdit = QLineEdit()

        self._varNameLineEdit.editingFinished.connect(self.editingFinished)
        self._moduleNameLineEdit.editingFinished.connect(self.editingFinished)

        self._varNameLineEdit.setPlaceholderText("Variable name")
        self._moduleNameLineEdit.setPlaceholderText("Module")

        layout = QVBoxLayout()
        layout.addWidget(self._varNameLineEdit)
        layout.addWidget(self._moduleNameLineEdit)
        self.setLayout(layout)

    def variableName(self) -> str:
        return self._varNameLineEdit.text()

    def setVariableName(self, name: str):
        self._varNameLineEdit.setText(name)

    def moduleName(self) -> str:
        return self._moduleNameLineEdit.text()

    def setModuleName(self, name: str):
        self._moduleNameLineEdit.setText(name)
