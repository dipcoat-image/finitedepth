"""
Experiment view
===============

V2 for controlwidgets/exptwidget.py
"""

import dawiq
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
    QStyledItemDelegate,
)
from dipcoatimage.finitedepth import Experiment
from dipcoatimage.finitedepth.analysis import ImportArgs, ExperimentArgs
from dipcoatimage.finitedepth.util import DataclassProtocol, Importer
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from .importview import ImportDataView
from typing import Optional, Type


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
    ...     ExperimentListView,
    ...     ExperimentView
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentListView()
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
        self._nameMapper = QDataWidgetMapper()
        self._importView = ImportDataView()
        self._pathsListView = QListView()
        self._addButton = QPushButton("Add")
        self._deleteButton = QPushButton("Delete")
        self._parametersView = dawiq.DataclassStackedWidget()
        self._exptArgsDelegate = ExperimentArgsDelegate()
        self._exptArgsMapper = QDataWidgetMapper()

        self._pathsListView.setSelectionMode(QListView.ExtendedSelection)
        self._pathsListView.setEditTriggers(QListView.SelectedClicked)
        self._addButton.clicked.connect(self.appendNewPath)
        self._deleteButton.clicked.connect(self.deleteSelectedPaths)
        self._exptArgsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._importView.editingFinished.connect(self._exptArgsMapper.submit)
        self._parametersView.addWidget(QWidget())  # default empty widget
        self._exptArgsMapper.setItemDelegate(self._exptArgsDelegate)

        self._nameLineEdit.setPlaceholderText("Experiment name")
        self._importView.setTitle("Experiment type")

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
        pathsGroupBox.setLayout(pathsLayout)
        layout.addWidget(pathsGroupBox)
        paramsGroupBox = QGroupBox("Parameters")
        paramsLayout = QVBoxLayout()
        paramsLayout.addWidget(self._parametersView)
        paramsGroupBox.setLayout(paramsLayout)
        layout.addWidget(paramsGroupBox)
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

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            self._nameMapper.setCurrentModelIndex(index)
            self._pathsListView.setModel(model)
            coatPathIndex = model.index(model.ROW_COATPATHS, 0, index)
            self._pathsListView.setRootIndex(coatPathIndex)
            self._exptArgsMapper.setRootIndex(index)
            exptIndex = model.index(model.ROW_EXPERIMENT, 0, index)
            self._exptArgsMapper.setCurrentModelIndex(exptIndex)
        else:
            self._nameLineEdit.clear()
            self._importView.clear()
            self._nameMapper.setCurrentModelIndex(QModelIndex())
            self._pathsListView.setModel(None)
            self._parametersView.setCurrentIndex(0)
            self._exptArgsMapper.setCurrentModelIndex(QModelIndex())

    def variableName(self) -> str:
        return self._importView.variableName()

    def setVariableName(self, name: str):
        self._importView.setVariableName(name)

    def moduleName(self) -> str:
        return self._importView.moduleName()

    def setModuleName(self, name: str):
        self._importView.setModuleName(name)

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

    def indexOfParameterType(self, paramType: Type[DataclassProtocol]) -> int:
        return self._parametersView.indexOfDataclass(paramType)

    def addParameterType(self, paramType: Type[DataclassProtocol]) -> int:
        index = self._parametersView.addDataWidget(
            dawiq.dataclass2Widget(paramType),
            paramType,
        )
        return index

    def setCurrentParameterIndex(self, index: int):
        self._parametersView.setCurrentIndex(index)


class ExperimentArgsDelegate(QStyledItemDelegate):
    def setModelData(self, editor, model, index):
        if isinstance(editor, ExperimentView):
            importArgs = ImportArgs(editor.variableName(), editor.moduleName())
            exptArgs = ExperimentArgs(importArgs)
            model.setData(index, exptArgs, Qt.UserRole)
        super().setModelData(editor, model, index)

    def setEditorData(self, editor, index):
        data = index.data(Qt.UserRole)
        if isinstance(editor, ExperimentView) and isinstance(data, ExperimentArgs):
            editor.setVariableName(data.type.name)
            editor.setModuleName(data.type.module)

            typeVar, _ = Importer(data.type.name, data.type.module).try_import()
            if isinstance(typeVar, type) and issubclass(typeVar, Experiment):
                paramType = typeVar.Parameters
                paramIdx = editor.indexOfParameterType(paramType)
                if paramIdx == -1:
                    paramIdx = editor.addParameterType(paramType)
            else:
                paramIdx = 0
            editor.setCurrentParameterIndex(paramIdx)

        super().setEditorData(editor, index)
