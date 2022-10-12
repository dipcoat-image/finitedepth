"""
Experiment data view
====================

V2 for inventory.py
"""

from PySide6.QtCore import (
    QModelIndex,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
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
    "ExperimentListWidget",
]


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

    activated = Signal(QModelIndex)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._listView = QListView()
        self._addButton = QToolButton()
        self._addButton.setMenu(QMenu(self))
        copyAction = self._addButton.menu().addAction("Copy selected items")
        self._deleteButton = QPushButton()

        self._listView.setSelectionMode(QListView.ExtendedSelection)
        self._listView.activated.connect(self.activated)
        self._addButton.clicked.connect(self.addNewExperiment)
        copyAction.triggered.connect(self.copySelectedExperiments)
        self._deleteButton.clicked.connect(self.deleteSelectedExperiments)

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
        self._listView.setModel(model)

    @Slot()
    def addNewExperiment(self):
        model = self.model()
        if model is not None:
            rowNum = model.rowCount()
            success = model.insertRow(model.rowCount())
            if success:
                index = model.index(rowNum, 0)
                model.setData(index, "New Experiment", role=Qt.DisplayRole)

    @Slot()
    def copySelectedExperiments(self):
        model = self.model()
        if model is not None:
            for index in self._listView.selectedIndexes():
                parent = index.parent()
                model.copyRows(parent, index.row(), 1, parent, model.rowCount(parent))

    @Slot()
    def deleteSelectedExperiments(self):
        model = self.model()
        if model is not None:
            rows = [idx.row() for idx in self._listView.selectedIndexes()]
            for i in reversed(sorted(rows)):
                model.removeRow(i)
