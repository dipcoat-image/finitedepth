"""
Experiment view
===============

V2 for controlwidgets/exptwidget.py
"""

from PySide6.QtCore import Slot, QModelIndex, Qt
from PySide6.QtWidgets import (
    QWidget,
    QLineEdit,
    QListView,
    QPushButton,
    QDataWidgetMapper,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
)
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from typing import Optional


__all__ = [
    "ExperimentWidget",
]


class ExperimentWidget(QWidget):
    """
    Widget to display experiment name, coating layer file paths and
    :class:`ExperimentArgs`.

    >>> from PySide6.QtWidgets import QApplication, QWidget, QTreeView, QHBoxLayout
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
    >>> from dipcoatimage.finitedepth_gui.views import (
    ...     ExperimentListWidget,
    ...     ExperimentWidget
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentListWidget()
    ...     exptListWidget.setModel(model)
    ...     layout.addWidget(exptListWidget)
    ...     treeView = QTreeView()
    ...     treeView.setModel(model)
    ...     layout.addWidget(treeView)
    ...     exptWidget = ExperimentWidget()
    ...     exptWidget.setModel(model)
    ...     layout.addWidget(exptWidget)
    ...     window.setLayout(layout)
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None
        self._nameLineEdit = QLineEdit()
        self._pathsListView = QListView()
        self._addButton = QPushButton("Add")
        self._deleteButton = QPushButton("Delete")
        self._mapper = QDataWidgetMapper()

        self._pathsListView.setSelectionMode(QListView.ExtendedSelection)
        self._pathsListView.setEditTriggers(QListView.SelectedClicked)
        self._addButton.clicked.connect(self.appendNewPath)
        self._deleteButton.clicked.connect(self.deleteSelectedPaths)

        layout = QVBoxLayout()
        layout.addWidget(self._nameLineEdit)
        pathsGroupBox = QGroupBox("Coating layer files path")
        pathsLayout = QVBoxLayout()
        pathsLayout.addWidget(self._pathsListView)
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addWidget(self._addButton)
        buttonsLayout.addWidget(self._deleteButton)
        pathsLayout.addLayout(buttonsLayout)
        pathsGroupBox.setLayout(pathsLayout)
        layout.addWidget(pathsGroupBox)
        self.setLayout(layout)

    def model(self) -> Optional[ExperimentDataModel]:
        return self._model

    def setModel(self, model: Optional[ExperimentDataModel]):
        oldModel = self.model()
        if oldModel is not None:
            oldModel.activatedIndexChanged.disconnect(self.setActivatedIndex)
        self._model = model
        if model is not None:
            model.activatedIndexChanged.connect(self.setActivatedIndex)

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        self._pathsListView.setModel(model)
        if model is not None:
            self._pathsListView.setRootIndex(model.index(model.ROW_COATPATHS, 0, index))

    @Slot()
    def appendNewPath(self):
        model = self._pathsListView.model()
        parent = self._pathsListView.rootIndex()
        if model is not None and parent.isValid():
            rowNum = model.rowCount(parent)
            success = model.insertRow(rowNum, parent)
            if success:
                index = model.index(rowNum, 0, parent)
                model.setData(index, "New path", role=Qt.DisplayRole)

    @Slot()
    def deleteSelectedPaths(self):
        model = self._pathsListView.model()
        if model is not None:
            rows = [idx.row() for idx in self._pathsListView.selectedIndexes()]
            for i in reversed(sorted(rows)):
                model.removeRow(i, self._pathsListView.rootIndex())
