"""
Import data view
================

V2 for importwidget.py
"""

from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex
from PySide6.QtWidgets import QGroupBox, QLineEdit, QDataWidgetMapper, QVBoxLayout
from typing import Optional


__all__ = [
    "ImportDataView",
]


class ImportDataView(QGroupBox):
    """
    Widget to display import data.

    >>> from PySide6.QtCore import Qt
    >>> from PySide6.QtWidgets import QApplication, QWidget, QTreeView, QHBoxLayout
    >>> from PySide6.QtGui import QStandardItemModel, QStandardItem
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.views import ImportDataView
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = QStandardItemModel()
    ...     parent = QStandardItem()
    ...     parent.appendRows([QStandardItem() for _ in range(2)])
    ...     model.appendRow(parent)
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     treeView = QTreeView()
    ...     treeView.setModel(model)
    ...     layout.addWidget(treeView)
    ...     importView = ImportDataView()
    ...     importView.setModel(model)
    ...     importView.setRootIndex(parent.index())
    ...     importView.setOrientation(Qt.Vertical)
    ...     layout.addWidget(importView)
    ...     window.setLayout(layout)
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._varNameLineEdit = QLineEdit()
        self._moduleNameLineEdit = QLineEdit()
        self._mapper = QDataWidgetMapper()

        self._varNameLineEdit.setPlaceholderText("Variable name")
        self._moduleNameLineEdit.setPlaceholderText("Module")

        layout = QVBoxLayout()
        layout.addWidget(self._varNameLineEdit)
        layout.addWidget(self._moduleNameLineEdit)
        self.setLayout(layout)

    def initMapper(self):
        self._mapper.addMapping(self._varNameLineEdit, 0)
        self._mapper.addMapping(self._moduleNameLineEdit, 1)
        self._mapper.setCurrentIndex(0)

    def model(self) -> Optional[QAbstractItemModel]:
        return self._mapper.model()

    def setModel(self, model: Optional[QAbstractItemModel]):
        self._mapper.setModel(model)
        self.initMapper()

    def rootIndex(self) -> QModelIndex:
        return self._mapper.rootIndex()

    def setRootIndex(self, index: QModelIndex):
        self._mapper.setRootIndex(index)
        self.initMapper()

    def orientation(self) -> Qt.Orientation:
        return self._mapper.orientation()

    def setOrientation(self, orientation: Qt.Orientation):
        self._mapper.setOrientation(orientation)
        self.initMapper()
