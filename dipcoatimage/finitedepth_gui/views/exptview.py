"""
Experiment view
===============

"""

import dataclasses
import dawiq
from itertools import groupby
from operator import itemgetter
from PySide6.QtCore import Qt, Slot, QModelIndex
from PySide6.QtWidgets import (
    QWidget,
    QLineEdit,
    QListView,
    QPushButton,
    QDataWidgetMapper,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFileDialog,
)
from dipcoatimage.finitedepth_gui.core import DataArgFlag
from dipcoatimage.finitedepth_gui.worker import WorkerUpdateFlag
from dipcoatimage.finitedepth_gui.model import (
    ExperimentDataModel,
    getTopLevelIndex,
    IndexRole,
    ExperimentSignalBlocker,
)
from .importview import ImportDataView, ImportArgsDelegate
from typing import Optional, List


__all__ = [
    "ExperimentView",
    "ExperimentArgsDelegate",
]


class ExperimentView(QWidget):
    """
    Widget to display experiment name, coating layer file paths and
    :class:`ExperimentArgs`.

    >>> from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
    >>> from dipcoatimage.finitedepth_gui.views import (
    ...     ExperimentDataListView,
    ...     ExperimentView
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentDataListView()
    ...     exptListWidget.setModel(model)
    ...     layout.addWidget(exptListWidget)
    ...     exptWidget = ExperimentView()
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
        self._importView = ImportDataView()
        self._pathsListView = QListView()
        self._addButton = QPushButton("Add")
        self._deleteButton = QPushButton("Delete")
        self._browseButton = QPushButton("Browse")
        self._paramStackWidget = dawiq.DataclassStackedWidget()

        self._nameMapper = QDataWidgetMapper()
        self._typeMapper = QDataWidgetMapper()
        self._argsMapper = QDataWidgetMapper()

        self._importView.editingFinished.connect(self._typeMapper.submit)
        self._pathsListView.setSelectionMode(QListView.ExtendedSelection)
        self._pathsListView.setEditTriggers(QListView.SelectedClicked)
        self._addButton.clicked.connect(self.appendNewPath)
        self._deleteButton.clicked.connect(self.deleteSelectedPaths)
        self._browseButton.clicked.connect(self.openBrowseDialog)
        self._paramStackWidget.currentDataEdited.connect(self._argsMapper.submit)
        self._typeMapper.setOrientation(Qt.Orientation.Vertical)
        self._typeMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._typeMapper.setItemDelegate(ImportArgsDelegate())
        self._argsMapper.setOrientation(Qt.Orientation.Vertical)
        self._argsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._argsMapper.setItemDelegate(ExperimentArgsDelegate())

        self._nameLineEdit.setPlaceholderText("Experiment name")
        self._importView.setTitle("Experiment type")
        self._paramStackWidget.addWidget(
            QGroupBox("Parameters")
        )  # default empty widget

        layout = QVBoxLayout()
        layout.addWidget(self._nameLineEdit)
        layout.addWidget(self._importView)
        pathsGroupBox = QGroupBox("Coating layer files path")
        pathsLayout = QVBoxLayout()
        pathsLayout.addWidget(self._pathsListView)
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addWidget(self._addButton)
        buttonsLayout.addWidget(self._deleteButton)
        pathsLayout.addLayout(buttonsLayout)
        pathsLayout.addWidget(self._browseButton)
        pathsGroupBox.setLayout(pathsLayout)
        layout.addWidget(pathsGroupBox)
        layout.addWidget(self._paramStackWidget)
        self.setLayout(layout)

    def model(self) -> Optional[ExperimentDataModel]:
        return self._model

    def setModel(self, model: Optional[ExperimentDataModel]):
        oldModel = self.model()
        if oldModel is not None:
            oldModel.activatedIndexChanged.disconnect(self.setActivatedIndex)
        self._model = model
        # no pathsListView.setModel here (if not, expt list will be displayed)
        self._nameMapper.clearMapping()
        self._typeMapper.clearMapping()
        self._argsMapper.clearMapping()
        self._nameMapper.setModel(model)
        self._typeMapper.setModel(model)
        self._argsMapper.setModel(model)
        if model is not None:
            model.activatedIndexChanged.connect(self.setActivatedIndex)
            self._nameMapper.addMapping(self._nameLineEdit, 0)
            self._typeMapper.addMapping(self._importView, model.Row_ExptArgs)
            self._argsMapper.addMapping(
                self._paramStackWidget, model.Row_ExptParameters
            )

    @Slot()
    def appendNewPath(self):
        model = self._pathsListView.model()
        if not isinstance(model, ExperimentDataModel):
            return
        parent = self._pathsListView.rootIndex()
        self.insertPaths(model.rowCount(parent), ["New path"])

    @Slot()
    def openBrowseDialog(self):
        model = self._pathsListView.model()
        if not isinstance(model, ExperimentDataModel):
            return
        parent = self._pathsListView.rootIndex()
        if not model.whatsThisIndex(parent) == IndexRole.COATPATHS:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select experiment files",
            "./",
            options=QFileDialog.DontUseNativeDialog,
        )
        self.insertPaths(model.rowCount(parent), paths)

    def insertPaths(self, row, paths: List[str]):
        model = self._pathsListView.model()
        if not isinstance(model, ExperimentDataModel):
            return
        parent = self._pathsListView.rootIndex()
        if not model.whatsThisIndex(parent) == IndexRole.COATPATHS:
            return

        count = len(paths)
        with ExperimentSignalBlocker(model):
            model.insertRows(row, count, parent)
            for i, path in enumerate(paths):
                index = model.index(row + i, 0, parent)
                model.setData(index, path, role=model.Role_CoatPath)

        topLevelIndex = getTopLevelIndex(parent)
        model.updateWorker(topLevelIndex, WorkerUpdateFlag.ANALYSIS)
        model.emitExperimentDataChanged(topLevelIndex, DataArgFlag.COATPATHS)

    @Slot()
    def deleteSelectedPaths(self):
        model = self._pathsListView.model()
        if not isinstance(model, ExperimentDataModel):
            return
        parent = self._pathsListView.rootIndex()
        if not model.whatsThisIndex(parent) == IndexRole.COATPATHS:
            return

        rows = [idx.row() for idx in self._pathsListView.selectedIndexes()]
        continuous_rows = [
            list(map(itemgetter(1), g))
            for k, g in groupby(enumerate(sorted(rows)), lambda i_x: i_x[0] - i_x[1])
        ]
        with ExperimentSignalBlocker(model):
            for row_list in reversed(continuous_rows):
                model.removeRows(row_list[0], len(row_list), parent)

        topLevelIndex = getTopLevelIndex(parent)
        model.updateWorker(topLevelIndex, WorkerUpdateFlag.ANALYSIS)
        model.emitExperimentDataChanged(topLevelIndex, DataArgFlag.COATPATHS)

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            self._nameMapper.setCurrentModelIndex(index)
            self._typeMapper.setRootIndex(index)
            self._typeMapper.toFirst()
            self._pathsListView.setModel(model)
            coatPathsIndex = model.getIndexFor(IndexRole.COATPATHS, index)
            self._pathsListView.setRootIndex(coatPathsIndex)
            exptArgsIndex = model.getIndexFor(IndexRole.EXPTARGS, index)
            self._argsMapper.setRootIndex(exptArgsIndex)
            self._argsMapper.toFirst()
        else:
            self._nameMapper.setCurrentModelIndex(QModelIndex())
            self._typeMapper.setCurrentModelIndex(QModelIndex())
            self._argsMapper.setCurrentModelIndex(QModelIndex())
            self._nameLineEdit.clear()
            self._importView.clear()
            self._pathsListView.setModel(None)
            self._paramStackWidget.setCurrentIndex(0)


class ExperimentArgsDelegate(dawiq.DataclassDelegate):
    TypeRole = ExperimentDataModel.Role_DataclassType
    DataRole = ExperimentDataModel.Role_DataclassData

    def ignoreMissing(self) -> bool:
        return False

    def cacheModelData(cls, model, index, value, role):
        if isinstance(model, ExperimentDataModel):
            model.cacheData(index, value, role)
        else:
            super().cacheModelData(model, index, value, role)

    def setEditorData(self, editor, index):
        model = index.model()
        if isinstance(model, ExperimentDataModel) and isinstance(
            editor, dawiq.DataclassStackedWidget
        ):
            # add data widget if absent
            dclsType = model.data(index, role=self.TypeRole)
            if isinstance(dclsType, type) and dataclasses.is_dataclass(dclsType):
                dclsIdx = editor.indexOfDataclass(dclsType)
                if dclsIdx == -1:
                    widget = dawiq.dataclass2Widget(dclsType)
                    indexRole = model.whatsThisIndex(index)
                    if indexRole == IndexRole.EXPT_PARAMETERS:
                        title = "Parameters"
                    else:
                        title = ""
                    widget.setTitle(title)
                    dclsIdx = editor.addDataWidget(widget, dclsType)
            else:
                dclsIdx = 0
            editor.setCurrentIndex(dclsIdx)
            self.setEditorData(editor.currentWidget(), index)
        else:
            super().setEditorData(editor, index)
