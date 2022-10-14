"""
Substrate view
==============

V2 for controlwidgets/substwidget.py
"""

import dawiq
from PySide6.QtCore import Qt, Slot, QModelIndex
from PySide6.QtWidgets import (
    QWidget,
    QDataWidgetMapper,
    QVBoxLayout,
    QGroupBox,
    QStyledItemDelegate,
)
from dipcoatimage.finitedepth import SubstrateBase
from dipcoatimage.finitedepth.analysis import ImportArgs, SubstrateArgs
from dipcoatimage.finitedepth.util import DataclassProtocol, Importer
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from .importview import ImportDataView
from typing import Optional, Type


__all__ = [
    "SubstrateView",
    "SubstrateArgsDelegate",
]


class SubstrateView(QWidget):
    """
    Widget to :class:`SubstrateArgs`.

    >>> from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
    >>> from dipcoatimage.finitedepth_gui.views import (
    ...     ExperimentListView,
    ...     SubstrateView,
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentListView()
    ...     exptListWidget.setModel(model)
    ...     layout.addWidget(exptListWidget)
    ...     substWidget = SubstrateView()
    ...     substWidget.setModel(model)
    ...     layout.addWidget(substWidget)
    ...     window.setLayout(layout)
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None
        self._importView = ImportDataView()
        self._parametersView = dawiq.DataclassStackedWidget()
        self._drawOptionsView = dawiq.DataclassStackedWidget()
        self._substArgsDelegate = SubstrateArgsDelegate()
        self._substArgsMapper = QDataWidgetMapper()

        self._substArgsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._importView.editingFinished.connect(self._substArgsMapper.submit)
        self._substArgsMapper.setItemDelegate(self._substArgsDelegate)

        self._importView.setTitle("Substrate type")
        self._parametersView.addWidget(
            QGroupBox("Parameters")  # default empty widget
        )
        self._drawOptionsView.addWidget(
            QGroupBox("Draw options")  # default empty widget
        )

        layout = QVBoxLayout()
        layout.addWidget(self._importView)
        layout.addWidget(self._parametersView)
        layout.addWidget(self._drawOptionsView)
        self.setLayout(layout)

    def model(self) -> Optional[ExperimentDataModel]:
        return self._model

    def setModel(self, model: Optional[ExperimentDataModel]):
        oldModel = self.model()
        if oldModel is not None:
            oldModel.activatedIndexChanged.disconnect(self.setActivatedIndex)
        self._model = model
        self._substArgsMapper.setModel(model)
        self._substArgsMapper.addMapping(self, 0)
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

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            self._substArgsMapper.setRootIndex(index)
            substIndex = model.index(model.ROW_SUBSTRATE, 0, index)
            self._substArgsMapper.setCurrentModelIndex(substIndex)
        else:
            self._importView.clear()
            self._parametersView.setCurrentIndex(0)
            self._drawOptionsView.setCurrentIndex(0)
            self._substArgsMapper.setCurrentModelIndex(QModelIndex())

    def indexOfParameterType(self, paramType: Type[DataclassProtocol]) -> int:
        return self._parametersView.indexOfDataclass(paramType)

    def addParameterType(self, paramType: Type[DataclassProtocol]) -> int:
        widget = dawiq.dataclass2Widget(paramType)
        widget.setTitle("Parameters")
        index = self._parametersView.addDataWidget(widget, paramType)
        return index

    def setCurrentParameterIndex(self, index: int):
        self._parametersView.setCurrentIndex(index)

    def indexOfDrawOptionsType(self, drawOptType: Type[DataclassProtocol]) -> int:
        return self._drawOptionsView.indexOfDataclass(drawOptType)

    def addDrawOptionsType(self, drawOptType: Type[DataclassProtocol]) -> int:
        widget = dawiq.dataclass2Widget(drawOptType)
        widget.setTitle("Draw options")
        index = self._drawOptionsView.addDataWidget(widget, drawOptType)
        return index

    def setCurrentDrawOptionsIndex(self, index: int):
        self._drawOptionsView.setCurrentIndex(index)


class SubstrateArgsDelegate(QStyledItemDelegate):
    def setModelData(self, editor, model, index):
        if isinstance(editor, SubstrateView):
            importArgs = ImportArgs(editor.typeName(), editor.moduleName())
            substArgs = SubstrateArgs(importArgs)
            model.setData(index, substArgs, Qt.UserRole)
        super().setModelData(editor, model, index)

    def setEditorData(self, editor, index):
        data = index.data(Qt.UserRole)
        if isinstance(editor, SubstrateView) and isinstance(data, SubstrateArgs):
            editor.setTypeName(data.type.name)
            editor.setModuleName(data.type.module)

            typeVar, _ = Importer(data.type.name, data.type.module).try_import()
            if isinstance(typeVar, type) and issubclass(typeVar, SubstrateBase):
                paramType = typeVar.Parameters
                paramIdx = editor.indexOfParameterType(paramType)
                if paramIdx == -1:
                    paramIdx = editor.addParameterType(paramType)
                drawOptType = typeVar.DrawOptions
                drawOptIdx = editor.indexOfDrawOptionsType(drawOptType)
                if drawOptIdx == -1:
                    drawOptIdx = editor.addDrawOptionsType(drawOptType)
            else:
                paramIdx = 0
                drawOptIdx = 0
            editor.setCurrentParameterIndex(paramIdx)
            editor.setCurrentDrawOptionsIndex(drawOptIdx)

        super().setEditorData(editor, index)
