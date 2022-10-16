"""
Coating layer view
==================

V2 for controlwidgets/layerwidget.py
"""

import dawiq
from PySide6.QtCore import Qt, Slot, QModelIndex
from PySide6.QtWidgets import (
    QWidget,
    QDataWidgetMapper,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
)
from dipcoatimage.finitedepth import CoatingLayerBase
from dipcoatimage.finitedepth.analysis import ImportArgs, CoatingLayerArgs
from dipcoatimage.finitedepth.util import DataclassProtocol, Importer
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from .importview import ImportDataView
from typing import Optional, Type, Union


__all__ = [
    "CoatingLayerView",
    "CoatingLayerArgsDelegate",
]


class CoatingLayerView(QWidget):
    """
    Widget to :class:`CoatingLayerArgs`.

    >>> from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
    >>> from dipcoatimage.finitedepth_gui.views import (
    ...     ExperimentListView,
    ...     CoatingLayerView,
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentListView()
    ...     exptListWidget.setModel(model)
    ...     layout.addWidget(exptListWidget)
    ...     layerWidget = CoatingLayerView()
    ...     layerWidget.setModel(model)
    ...     layout.addWidget(layerWidget)
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
        self._paramStackWidget = dawiq.DataclassStackedWidget()
        self._drawOptStackWidget = dawiq.DataclassStackedWidget()
        self._decoOptStackWidget = dawiq.DataclassStackedWidget()
        self._layerArgsMapper = QDataWidgetMapper()

        self._layerArgsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._importView.editingFinished.connect(self._layerArgsMapper.submit)
        self._paramStackWidget.currentDataValueChanged.connect(
            self._layerArgsMapper.submit
        )
        self._drawOptStackWidget.currentDataValueChanged.connect(
            self._layerArgsMapper.submit
        )
        self._decoOptStackWidget.currentDataValueChanged.connect(
            self._layerArgsMapper.submit
        )
        self._layerArgsMapper.setItemDelegate(CoatingLayerArgsDelegate())

        self._importView.setTitle("CoatingLayer type")
        self._paramStackWidget.addWidget(
            QGroupBox("Parameters")  # default empty widget
        )
        self._drawOptStackWidget.addWidget(
            QGroupBox("Draw options")  # default empty widget
        )
        self._decoOptStackWidget.addWidget(
            QGroupBox("Decorate options")  # default empty widget
        )

        layout = QVBoxLayout()
        layout.addWidget(self._importView)
        layout.addWidget(self._paramStackWidget)
        dataLayout = QHBoxLayout()
        dataLayout.addWidget(self._drawOptStackWidget)
        dataLayout.addWidget(self._decoOptStackWidget)
        layout.addLayout(dataLayout)
        self.setLayout(layout)

    def model(self) -> Optional[ExperimentDataModel]:
        return self._model

    def setModel(self, model: Optional[ExperimentDataModel]):
        oldModel = self.model()
        if oldModel is not None:
            oldModel.activatedIndexChanged.disconnect(self.setActivatedIndex)
        self._model = model
        self._layerArgsMapper.setModel(model)
        self._layerArgsMapper.addMapping(self, 0)
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

    def currentParametersWidget(self) -> Union[dawiq.DataWidget, QGroupBox]:
        return self._paramStackWidget.currentWidget()

    def indexOfParameterType(self, paramType: Type[DataclassProtocol]) -> int:
        return self._paramStackWidget.indexOfDataclass(paramType)

    def addParameterType(self, paramType: Type[DataclassProtocol]) -> int:
        widget = dawiq.dataclass2Widget(paramType)
        widget.setTitle("Parameters")
        index = self._paramStackWidget.addDataWidget(widget, paramType)
        return index

    def setCurrentParametersIndex(self, index: int):
        self._paramStackWidget.setCurrentIndex(index)

    def currentDrawOptionsWidget(self) -> Union[dawiq.DataWidget, QGroupBox]:
        return self._drawOptStackWidget.currentWidget()

    def indexOfDrawOptionsType(self, drawOptType: Type[DataclassProtocol]) -> int:
        return self._drawOptStackWidget.indexOfDataclass(drawOptType)

    def addDrawOptionsType(self, drawOptType: Type[DataclassProtocol]) -> int:
        widget = dawiq.dataclass2Widget(drawOptType)
        widget.setTitle("Draw options")
        index = self._drawOptStackWidget.addDataWidget(widget, drawOptType)
        return index

    def setCurrentDrawOptionsIndex(self, index: int):
        self._drawOptStackWidget.setCurrentIndex(index)

    def currentDecoOptionsWidget(self) -> Union[dawiq.DataWidget, QGroupBox]:
        return self._decoOptStackWidget.currentWidget()

    def indexOfDecoOptionsType(self, decoOptType: Type[DataclassProtocol]) -> int:
        return self._decoOptStackWidget.indexOfDataclass(decoOptType)

    def addDecoOptionsType(self, decoOptType: Type[DataclassProtocol]) -> int:
        widget = dawiq.dataclass2Widget(decoOptType)
        widget.setTitle("Decorate options")
        index = self._decoOptStackWidget.addDataWidget(widget, decoOptType)
        return index

    def setCurrentDecoOptionsIndex(self, index: int):
        self._decoOptStackWidget.setCurrentIndex(index)

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            self._layerArgsMapper.setRootIndex(index)
            layerIndex = model.index(model.ROW_COATINGLAYER, 0, index)
            self._layerArgsMapper.setCurrentModelIndex(layerIndex)
        else:
            self._importView.clear()
            self._paramStackWidget.setCurrentIndex(0)
            self._drawOptStackWidget.setCurrentIndex(0)
            self._decoOptStackWidget.setCurrentIndex(0)
            self._layerArgsMapper.setCurrentModelIndex(QModelIndex())


class CoatingLayerArgsDelegate(dawiq.DataclassDelegate):
    def ignoreMissing(self) -> bool:
        return False

    def setModelData(self, editor, model, index):
        if isinstance(editor, CoatingLayerView):
            importArgs = ImportArgs(editor.typeName(), editor.moduleName())
            paramWidget = editor.currentParametersWidget()
            if isinstance(paramWidget, dawiq.DataWidget):
                parameters = paramWidget.dataValue()
            else:
                parameters = {}
            drawOptWidget = editor.currentDrawOptionsWidget()
            if isinstance(drawOptWidget, dawiq.DataWidget):
                drawOpt = drawOptWidget.dataValue()
            else:
                drawOpt = {}
            decoOptWidget = editor.currentDecoOptionsWidget()
            if isinstance(decoOptWidget, dawiq.DataWidget):
                decoOpt = decoOptWidget.dataValue()
            else:
                decoOpt = {}
            typeVar, _ = Importer(importArgs.name, importArgs.module).try_import()
            if isinstance(typeVar, type) and issubclass(typeVar, CoatingLayerBase):
                paramType = typeVar.Parameters
                drawOptType = typeVar.DrawOptions
                decoOptType = typeVar.DecoOptions
                parameters = dawiq.convertFromQt(
                    paramType, parameters, self.ignoreMissing()
                )
                drawOpt = dawiq.convertFromQt(
                    drawOptType, drawOpt, self.ignoreMissing()
                )
                decoOpt = dawiq.convertFromQt(
                    decoOptType, decoOpt, self.ignoreMissing()
                )
            layerArgs = CoatingLayerArgs(importArgs, parameters, drawOpt, decoOpt)
            model.setData(index, layerArgs, Qt.UserRole)
        super().setModelData(editor, model, index)

    def setEditorData(self, editor, index):
        data = index.data(Qt.UserRole)
        if isinstance(editor, CoatingLayerView) and isinstance(data, CoatingLayerArgs):
            editor.setTypeName(data.type.name)
            editor.setModuleName(data.type.module)

            typeVar, _ = Importer(data.type.name, data.type.module).try_import()
            if isinstance(typeVar, type) and issubclass(typeVar, CoatingLayerBase):
                paramType = typeVar.Parameters
                paramIdx = editor.indexOfParameterType(paramType)
                if paramIdx == -1:
                    paramIdx = editor.addParameterType(paramType)
                editor.setCurrentParametersIndex(paramIdx)
                drawOptType = typeVar.DrawOptions
                drawOptIdx = editor.indexOfDrawOptionsType(drawOptType)
                if drawOptIdx == -1:
                    drawOptIdx = editor.addDrawOptionsType(drawOptType)
                editor.setCurrentDrawOptionsIndex(drawOptIdx)
                decoOptType = typeVar.DecoOptions
                decoOptIdx = editor.indexOfDecoOptionsType(decoOptType)
                if decoOptIdx == -1:
                    decoOptIdx = editor.addDecoOptionsType(decoOptType)
                editor.setCurrentDecoOptionsIndex(decoOptIdx)

                self.setEditorDataclassData(
                    editor.currentParametersWidget(),
                    paramType,
                    data.parameters,
                )
                self.setEditorDataclassData(
                    editor.currentDrawOptionsWidget(),
                    drawOptType,
                    data.draw_options,
                )
                self.setEditorDataclassData(
                    editor.currentDecoOptionsWidget(),
                    decoOptType,
                    data.deco_options,
                )
            else:
                editor.setCurrentParametersIndex(0)
                editor.setCurrentDrawOptionsIndex(0)
                editor.setCurrentDecoOptionsIndex(0)
        super().setEditorData(editor, index)
