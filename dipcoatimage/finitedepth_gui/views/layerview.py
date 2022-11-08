"""
Coating layer view
==================

V2 for controlwidgets/layerwidget.py
"""

import dataclasses
import dawiq
from PySide6.QtCore import Slot, QModelIndex
from PySide6.QtWidgets import (
    QWidget,
    QDataWidgetMapper,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
)
from dipcoatimage.finitedepth import CoatingLayerBase
from dipcoatimage.finitedepth.analysis import ImportArgs
from dipcoatimage.finitedepth.util import DataclassProtocol, Importer
from dipcoatimage.finitedepth_gui.model import (
    ExperimentDataModel,
    IndexRole,
    WorkerUpdateBlocker,
)
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
    ...     ExperimentDataListView,
    ...     CoatingLayerView,
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentDataListView()
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

        self._importView.editingFinished.connect(self._layerArgsMapper.submit)
        self._paramStackWidget.currentDataEdited.connect(self._layerArgsMapper.submit)
        self._drawOptStackWidget.currentDataEdited.connect(self._layerArgsMapper.submit)
        self._decoOptStackWidget.currentDataEdited.connect(self._layerArgsMapper.submit)
        self._layerArgsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
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

    def drawOptionsStackedWidget(self) -> dawiq.DataclassStackedWidget:
        return self._drawOptStackWidget

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

    def decoOptionsStackedWidget(self) -> dawiq.DataclassStackedWidget:
        return self._decoOptStackWidget

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
            layerIndex = model.getIndexFor(IndexRole.LAYERARGS, index)
            self._layerArgsMapper.setCurrentModelIndex(layerIndex)
        else:
            self._importView.clear()
            self._paramStackWidget.setCurrentIndex(0)
            self._drawOptStackWidget.setCurrentIndex(0)
            self._decoOptStackWidget.setCurrentIndex(0)
            self._layerArgsMapper.setCurrentModelIndex(QModelIndex())


class CoatingLayerArgsDelegate(dawiq.DataclassDelegate):
    TypeRole = ExperimentDataModel.Role_DataclassType
    DataRole = ExperimentDataModel.Role_DataclassData

    def ignoreMissing(self) -> bool:
        return False

    def setModelData(self, editor, model, index):
        if isinstance(model, ExperimentDataModel):
            indexRole = model.whatsThisIndex(index)
            if indexRole == IndexRole.LAYERARGS and isinstance(
                editor, CoatingLayerView
            ):
                with WorkerUpdateBlocker(model):
                    # set ImportArgs for layer type to model
                    importArgs = ImportArgs(editor.typeName(), editor.moduleName())
                    model.setData(
                        model.getIndexFor(IndexRole.LAYER_TYPE, index),
                        importArgs,
                        role=model.Role_ImportArgs,
                    )

                    # set dataclasses types to model
                    paramIndex = model.getIndexFor(IndexRole.LAYER_PARAMETERS, index)
                    drawOptIndex = model.getIndexFor(IndexRole.LAYER_DRAWOPTIONS, index)
                    decoOptIndex = model.getIndexFor(IndexRole.LAYER_DECOOPTIONS, index)
                    layerType, _ = Importer(
                        importArgs.name, importArgs.module
                    ).try_import()
                    if isinstance(layerType, type) and issubclass(
                        layerType, CoatingLayerBase
                    ):
                        paramType = layerType.Parameters
                        drawOptType = layerType.DrawOptions
                        decoOptType = layerType.DecoOptions
                    else:
                        paramType = None
                        drawOptType = None
                        decoOptType = None
                    model.setData(paramIndex, paramType, role=self.TypeRole)
                    model.setData(drawOptIndex, drawOptType, role=self.TypeRole)
                    model.setData(decoOptIndex, decoOptType, role=self.TypeRole)

                    # set dataclasses data to model
                    self.setModelData(
                        editor.currentParametersWidget(), model, paramIndex
                    )
                    self.setModelData(
                        editor.currentDrawOptionsWidget(), model, drawOptIndex
                    )
                    self.setModelData(
                        editor.currentDecoOptionsWidget(), model, decoOptIndex
                    )

                model.updateWorker(model.getTopLevelIndex(index))

        super().setModelData(editor, model, index)

    def setEditorData(self, editor, index):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            indexRole = model.whatsThisIndex(index)
            if indexRole == IndexRole.LAYERARGS and isinstance(
                editor, CoatingLayerView
            ):
                # set import args for layer type to editor
                importArgs = model.data(
                    model.getIndexFor(IndexRole.LAYER_TYPE, index),
                    role=model.Role_ImportArgs,
                )
                editor.setTypeName(importArgs.name)
                editor.setModuleName(importArgs.module)

                # add data widget if absent
                paramIndex = model.getIndexFor(IndexRole.LAYER_PARAMETERS, index)
                paramType = model.data(paramIndex, role=self.TypeRole)
                if isinstance(paramType, type) and dataclasses.is_dataclass(paramType):
                    paramWidgetIdx = editor.indexOfParametersType(paramType)
                    if paramWidgetIdx == -1:
                        paramWidgetIdx = editor.addParametersType(paramType)
                else:
                    paramWidgetIdx = -1
                drawOptIndex = model.getIndexFor(IndexRole.LAYER_DRAWOPTIONS, index)
                drawOptType = model.data(drawOptIndex, role=self.TypeRole)
                if isinstance(drawOptType, type) and dataclasses.is_dataclass(
                    drawOptType
                ):
                    drawOptWidgetIdx = editor.indexOfDrawOptionsType(drawOptType)
                    if drawOptWidgetIdx == -1:
                        drawOptWidgetIdx = editor.addDrawOptionsType(drawOptType)
                else:
                    drawOptWidgetIdx = -1
                decoOptIndex = model.getIndexFor(IndexRole.LAYER_DECOOPTIONS, index)
                decoOptType = model.data(decoOptIndex, role=self.TypeRole)
                if isinstance(decoOptType, type) and dataclasses.is_dataclass(
                    decoOptType
                ):
                    decoOptWidgetIdx = editor.indexOfDecoOptionsType(decoOptType)
                    if decoOptWidgetIdx == -1:
                        decoOptWidgetIdx = editor.addDecoOptionsType(decoOptType)
                else:
                    decoOptWidgetIdx = -1

                # set dataclasses type and data to editor
                self.setEditorData(editor.parametersStackedWidget(), paramIndex)
                self.setEditorData(editor.drawOptionsStackedWidget(), drawOptIndex)
                self.setEditorData(editor.decoOptionsStackedWidget(), decoOptIndex)

                # show default widget for invalid index
                if paramWidgetIdx == -1:
                    editor.setCurrentParametersIndex(0)
                if drawOptWidgetIdx == -1:
                    editor.setCurrentDrawOptionsIndex(0)
                if decoOptWidgetIdx == -1:
                    editor.setCurrentDecoOptionsIndex(0)

        super().setEditorData(editor, index)
