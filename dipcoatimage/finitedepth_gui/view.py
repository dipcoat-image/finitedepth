"""
Experiment data view
====================

V2 for inventory.py
"""

from PySide6.QtCore import (
    QModelIndex,
    Qt,
    Slot,
)
from PySide6.QtWidgets import (
    QStyledItemDelegate,
    QWidget,
    QListView,
    QToolButton,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMenu,
    QSizePolicy,
)
from .model import ExperimentDataModel
from typing import Optional


__all__ = [
    "QStyledItemDelegate",
    "ExperimentListWidget",
]


class ExperimentListDelegate(QStyledItemDelegate):
    """Delegate to mark activated item."""

    ACTIVATED_INDENT = 10
    ACTIVATED_MARKER_RADIUS = 2

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            if index == model.activatedIndex():
                option.font.setBold(True)
                option.rect.adjust(self.ACTIVATED_INDENT, 0, 0, 0)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            if index == model.activatedIndex():
                markerSpace = option.rect.adjusted(
                    0, 0, -option.rect.width() + self.ACTIVATED_INDENT, 0
                )
                w, h = markerSpace.width(), markerSpace.height()
                dx = w // 2 - self.ACTIVATED_MARKER_RADIUS
                dy = h // 2 - self.ACTIVATED_MARKER_RADIUS
                markerRect = markerSpace.adjusted(dx, dy, -dx, -dy)

                painter.save()
                painter.setBrush(Qt.black)
                painter.drawEllipse(markerRect)
                painter.restore()


class ExperimentListWidget(QWidget):
    """
    Widget to display the list of experiments.

    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
    >>> from dipcoatimage.finitedepth_gui.view import ExperimentListWidget
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     window = ExperimentListWidget()
    ...     window.setModel(ExperimentDataModel())
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._listView = QListView()
        self._addButton = QToolButton()
        self._addButton.setMenu(QMenu(self))
        copyAction = self._addButton.menu().addAction("Copy selected items")
        self._deleteButton = QPushButton()

        self._listView.setItemDelegate(ExperimentListDelegate())
        self._listView.setSelectionMode(QListView.ExtendedSelection)
        self._listView.activated.connect(self.onIndexActivated)
        self._addButton.clicked.connect(self.appendNewRow)
        copyAction.triggered.connect(self.copySelectedRows)
        self._deleteButton.clicked.connect(self.deleteSelectedRows)

        self._addButton.setPopupMode(QToolButton.MenuButtonPopup)
        self._addButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._deleteButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._addButton.setText("Add")
        self._deleteButton.setText("Delete")

        layout = QVBoxLayout()
        layout.addWidget(self._listView)
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self._addButton)
        buttonLayout.addWidget(self._deleteButton)
        layout.addLayout(buttonLayout)
        self.setLayout(layout)

    def model(self) -> Optional[ExperimentDataModel]:
        return self._listView.model()

    def setModel(self, model: Optional[ExperimentDataModel]):
        oldModel = self.model()
        if oldModel is not None:
            oldModel.activatedIndexChanged.disconnect(self._listView.viewport().update)
        self._listView.setModel(model)
        if model is not None:
            model.activatedIndexChanged.connect(self._listView.viewport().update)

    @Slot()
    def appendNewRow(self):
        model = self.model()
        if model is not None:
            rowNum = model.rowCount()
            success = model.insertRow(model.rowCount())
            if success:
                index = model.index(rowNum, 0)
                model.setData(index, "New Experiment", role=Qt.DisplayRole)

    @Slot()
    def copySelectedRows(self):
        model = self.model()
        if model is not None:
            for index in self._listView.selectedIndexes():
                parent = index.parent()
                model.copyRows(parent, index.row(), 1, parent, model.rowCount(parent))

    @Slot()
    def deleteSelectedRows(self):
        model = self.model()
        if model is not None:
            rows = [idx.row() for idx in self._listView.selectedIndexes()]
            for i in reversed(sorted(rows)):
                model.removeRow(i)

    @Slot(QModelIndex)
    def onIndexActivated(self, index: QModelIndex):
        model = self.model()
        if model is not None:
            model.setActivatedIndex(index)
