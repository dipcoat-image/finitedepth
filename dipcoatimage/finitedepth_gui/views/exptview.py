"""
Experiment view
===============

V2 for controlwidgets/exptwidget.py
"""

import dataclasses
import dawiq
from itertools import groupby
from operator import itemgetter
from PySide6.QtCore import Slot, QModelIndex
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
from dipcoatimage.finitedepth import ExperimentBase, ImportArgs
from dipcoatimage.finitedepth.util import DataclassProtocol, Importer
from dipcoatimage.finitedepth_gui.core import DataArgs
from dipcoatimage.finitedepth_gui.worker import WorkerUpdateFlag
from dipcoatimage.finitedepth_gui.model import (
    ExperimentDataModel,
    IndexRole,
    ExperimentSignalBlocker,
)
from .importview import ImportDataView
from typing import Optional, Type, Union, List


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
        self._exptArgsMapper = QDataWidgetMapper()

        self._importView.editingFinished.connect(self._exptArgsMapper.submit)
        self._pathsListView.setSelectionMode(QListView.ExtendedSelection)
        self._pathsListView.setEditTriggers(QListView.SelectedClicked)
        self._addButton.clicked.connect(self.appendNewPath)
        self._deleteButton.clicked.connect(self.deleteSelectedPaths)
        self._browseButton.clicked.connect(self.openBrowseDialog)
        self._paramStackWidget.currentDataEdited.connect(self._exptArgsMapper.submit)
        self._exptArgsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._exptArgsMapper.setItemDelegate(ExperimentArgsDelegate())

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
        self._nameMapper.setModel(model)
        self._nameMapper.addMapping(self._nameLineEdit, 0)
        self._exptArgsMapper.setModel(model)
        self._exptArgsMapper.addMapping(self, 0)
        if model is not None:
            model.activatedIndexChanged.connect(self.setActivatedIndex)

    def typeName(self) -> str:
        return self._importView.variableName()

    def setTypeName(self, name: str):
        self._importView.setVariableName(name)

    def moduleName(self) -> str:
        return self._importView.moduleName()

    def setModuleName(self, name: str):
        self._importView.setModuleName(name)

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

        topLevelIndex = model.getTopLevelIndex(parent)
        model.updateWorker(topLevelIndex, WorkerUpdateFlag.ANALYSIS)
        model.emitExperimentDataChanged(topLevelIndex, DataArgs.COATPATHS)

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

        topLevelIndex = model.getTopLevelIndex(parent)
        model.updateWorker(topLevelIndex, WorkerUpdateFlag.ANALYSIS)
        model.emitExperimentDataChanged(topLevelIndex, DataArgs.COATPATHS)

    def parametersStackedWidget(self) -> dawiq.DataclassStackedWidget:
        return self._paramStackWidget

    def currentParametersWidget(self) -> Union[dawiq.DataWidget, QGroupBox]:
        return self._paramStackWidget.currentWidget()

    def indexOfParametersType(self, paramType: Type[DataclassProtocol]) -> int:
        return self._paramStackWidget.indexOfDataclass(paramType)

    def addParametersType(self, paramType: Type[DataclassProtocol]) -> int:
        widget = dawiq.dataclass2Widget(paramType)
        widget.setTitle("Parameters")
        index = self._paramStackWidget.addDataWidget(widget, paramType)
        return index

    def setCurrentParametersIndex(self, index: int):
        self._paramStackWidget.setCurrentIndex(index)

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            self._nameMapper.setCurrentModelIndex(index)
            self._pathsListView.setModel(model)
            coatPathIndex = model.getIndexFor(IndexRole.COATPATHS, index)
            self._pathsListView.setRootIndex(coatPathIndex)
            self._exptArgsMapper.setRootIndex(index)
            exptIndex = model.getIndexFor(IndexRole.EXPTARGS, index)
            self._exptArgsMapper.setCurrentModelIndex(exptIndex)
        else:
            self._nameLineEdit.clear()
            self._importView.clear()
            self._nameMapper.setCurrentModelIndex(QModelIndex())
            self._pathsListView.setModel(None)
            self.setCurrentParametersIndex(0)
            self._exptArgsMapper.setCurrentModelIndex(QModelIndex())


class ExperimentArgsDelegate(dawiq.DataclassDelegate):

    TypeRole = ExperimentDataModel.Role_DataclassType
    DataRole = ExperimentDataModel.Role_DataclassData

    def ignoreMissing(self) -> bool:
        return False

    def setModelData(self, editor, model, index):
        if isinstance(model, ExperimentDataModel):
            indexRole = model.whatsThisIndex(index)
            if indexRole == IndexRole.EXPTARGS and isinstance(editor, ExperimentView):
                with ExperimentSignalBlocker(model):
                    # set ImportArgs for experiment type to model
                    importArgs = ImportArgs(editor.typeName(), editor.moduleName())
                    model.setData(
                        model.getIndexFor(IndexRole.EXPT_TYPE, index),
                        importArgs,
                        role=model.Role_ImportArgs,
                    )

                    # set dataclass type to model
                    paramIndex = model.getIndexFor(IndexRole.EXPT_PARAMETERS, index)
                    exptType, _ = Importer(
                        importArgs.name, importArgs.module
                    ).try_import()
                    if isinstance(exptType, type) and issubclass(
                        exptType, ExperimentBase
                    ):
                        paramType = exptType.Parameters
                    else:
                        paramType = None
                    model.setData(paramIndex, paramType, role=self.TypeRole)

                    # set dataclass data to model
                    self.setModelData(
                        editor.currentParametersWidget(), model, paramIndex
                    )

                topLevelIndex = model.getTopLevelIndex(index)
                model.updateWorker(topLevelIndex, WorkerUpdateFlag.EXPERIMENT)
                model.emitExperimentDataChanged(topLevelIndex, DataArgs.EXPERIMENT)

        super().setModelData(editor, model, index)

    def setEditorData(self, editor, index):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            indexRole = model.whatsThisIndex(index)
            if indexRole == IndexRole.EXPTARGS and isinstance(editor, ExperimentView):
                # set import args for experiment type to editor
                importArgs = model.data(
                    model.getIndexFor(IndexRole.EXPT_TYPE, index),
                    role=model.Role_ImportArgs,
                )
                editor.setTypeName(importArgs.name)
                editor.setModuleName(importArgs.module)

                # add data widget if absent
                paramIndex = model.getIndexFor(IndexRole.EXPT_PARAMETERS, index)
                paramType = model.data(paramIndex, role=self.TypeRole)
                if isinstance(paramType, type) and dataclasses.is_dataclass(paramType):
                    paramWidgetIdx = editor.indexOfParametersType(paramType)
                    if paramWidgetIdx == -1:
                        paramWidgetIdx = editor.addParametersType(paramType)
                else:
                    paramWidgetIdx = -1

                # set dataclass type and data to editor
                self.setEditorData(
                    editor.parametersStackedWidget(),
                    model.getIndexFor(IndexRole.EXPT_PARAMETERS, index),
                )

                # show default widget for invalid index
                if paramWidgetIdx == -1:
                    editor.setCurrentParametersIndex(0)

        super().setEditorData(editor, index)
