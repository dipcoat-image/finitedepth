"""
Substrate view
==============

V2 for controlwidgets/substwidget.py
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
from dipcoatimage.finitedepth import SubstrateBase
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
    ...     ExperimentDataListView,
    ...     SubstrateView,
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentDataListView()
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
        self._paramStackWidget = dawiq.DataclassStackedWidget()
        self._drawOptStackWidget = dawiq.DataclassStackedWidget()

        self._substArgsMapper = QDataWidgetMapper()

        self._importView.editingFinished.connect(self._substArgsMapper.submit)
        self._paramStackWidget.currentDataEdited.connect(self._substArgsMapper.submit)
        self._drawOptStackWidget.currentDataEdited.connect(self._substArgsMapper.submit)
        self._substArgsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._substArgsMapper.setItemDelegate(SubstrateArgsDelegate())

        self._importView.setTitle("Substrate type")
        self._paramStackWidget.addWidget(
            QGroupBox("Parameters")  # default empty widget
        )
        self._drawOptStackWidget.addWidget(
            QGroupBox("Draw options")  # default empty widget
        )

        layout = QVBoxLayout()
        layout.addWidget(self._importView)
        dataLayout = QHBoxLayout()
        dataLayout.addWidget(self._paramStackWidget)
        dataLayout.addWidget(self._drawOptStackWidget)
        layout.addLayout(dataLayout)
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

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            self._substArgsMapper.setRootIndex(index)
            substIndex = model.getIndexFor(IndexRole.SUBSTARGS, index)
            self._substArgsMapper.setCurrentModelIndex(substIndex)
        else:
            self._importView.clear()
            self._paramStackWidget.setCurrentIndex(0)
            self._drawOptStackWidget.setCurrentIndex(0)
            self._substArgsMapper.setCurrentModelIndex(QModelIndex())


class SubstrateArgsDelegate(dawiq.DataclassDelegate):

    TypeRole = ExperimentDataModel.Role_DataclassType
    DataRole = ExperimentDataModel.Role_DataclassData

    def ignoreMissing(self) -> bool:
        return False

    def setModelData(self, editor, model, index):
        if isinstance(model, ExperimentDataModel):
            indexRole = model.whatsThisIndex(index)
            if indexRole == IndexRole.SUBSTARGS and isinstance(editor, SubstrateView):
                with WorkerUpdateBlocker(model):
                    # set ImportArgs for substtrate type to model
                    importArgs = ImportArgs(editor.typeName(), editor.moduleName())
                    model.setData(
                        model.getIndexFor(IndexRole.SUBST_TYPE, index),
                        importArgs,
                        role=model.Role_ImportArgs,
                    )

                    # set dataclasses types to model
                    paramIndex = model.getIndexFor(IndexRole.SUBST_PARAMETERS, index)
                    drawOptIndex = model.getIndexFor(IndexRole.SUBST_DRAWOPTIONS, index)
                    substType, _ = Importer(
                        importArgs.name, importArgs.module
                    ).try_import()
                    if isinstance(substType, type) and issubclass(
                        substType, SubstrateBase
                    ):
                        paramType = substType.Parameters
                        drawOptType = substType.DrawOptions
                    else:
                        paramType = None
                        drawOptType = None
                    model.setData(paramIndex, paramType, role=self.TypeRole)
                    model.setData(drawOptIndex, drawOptType, role=self.TypeRole)

                    # set dataclasses data to model
                    self.setModelData(
                        editor.currentParametersWidget(), model, paramIndex
                    )
                    self.setModelData(
                        editor.currentDrawOptionsWidget(), model, drawOptIndex
                    )

                model.updateWorker(model.getTopLevelIndex(index))

        super().setModelData(editor, model, index)

    def setEditorData(self, editor, index):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            indexRole = model.whatsThisIndex(index)
            if indexRole == IndexRole.SUBSTARGS and isinstance(editor, SubstrateView):
                # set import args for substrate type to editor
                importArgs = model.data(
                    model.getIndexFor(IndexRole.SUBST_TYPE, index),
                    role=model.Role_ImportArgs,
                )
                editor.setTypeName(importArgs.name)
                editor.setModuleName(importArgs.module)

                # add data widget if absent
                paramIndex = model.getIndexFor(IndexRole.SUBST_PARAMETERS, index)
                paramType = model.data(paramIndex, role=self.TypeRole)
                if isinstance(paramType, type) and dataclasses.is_dataclass(paramType):
                    paramWidgetIdx = editor.indexOfParametersType(paramType)
                    if paramWidgetIdx == -1:
                        paramWidgetIdx = editor.addParametersType(paramType)
                else:
                    paramWidgetIdx = -1
                drawOptIndex = model.getIndexFor(IndexRole.SUBST_DRAWOPTIONS, index)
                drawOptType = model.data(drawOptIndex, role=self.TypeRole)
                if isinstance(drawOptType, type) and dataclasses.is_dataclass(
                    drawOptType
                ):
                    drawOptWidgetIdx = editor.indexOfDrawOptionsType(drawOptType)
                    if drawOptWidgetIdx == -1:
                        drawOptWidgetIdx = editor.addDrawOptionsType(drawOptType)
                else:
                    drawOptWidgetIdx = -1

                # set dataclasses type and data to editor
                self.setEditorData(editor.parametersStackedWidget(), paramIndex)
                self.setEditorData(editor.drawOptionsStackedWidget(), drawOptIndex)

                # show default widget for invalid index
                if paramWidgetIdx == -1:
                    editor.setCurrentParametersIndex(0)
                if drawOptWidgetIdx == -1:
                    editor.setCurrentDrawOptionsIndex(0)

        super().setEditorData(editor, index)
