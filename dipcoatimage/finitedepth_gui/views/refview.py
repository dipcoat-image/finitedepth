"""
Reference view
==============

V2 for controlwidgets/refwidget.py
"""

import dataclasses
import dawiq
import enum
import imagesize  # type: ignore
from PySide6.QtCore import Qt, Signal, Slot, QModelIndex
from PySide6.QtWidgets import (
    QWidget,
    QLineEdit,
    QPushButton,
    QDataWidgetMapper,
    QSizePolicy,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QStyledItemDelegate,
)
from dipcoatimage.finitedepth import SubstrateReferenceBase, ImportArgs
from dipcoatimage.finitedepth.util import DataclassProtocol, Importer
from dipcoatimage.finitedepth_gui.core import DataArgFlag
from dipcoatimage.finitedepth_gui.worker import WorkerUpdateFlag
from dipcoatimage.finitedepth_gui.model import (
    ExperimentDataModel,
    IndexRole,
    ExperimentSignalBlocker,
)
from .importview import ImportDataView
from .roiview import ROIView
from typing import Optional, Type, Union


__all__ = [
    "ROIDrawFlag",
    "ReferenceView",
    "ReferencePathDelegate",
    "ReferenceArgsDelegate",
]


class ROIDrawFlag(enum.IntFlag):
    NONE = 0
    TEMPLATE = 1
    SUBSTRATE = 2


class ReferenceView(QWidget):
    """
    Widget to display reference file path and :class:`ReferenceArgs`.

    >>> from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
    >>> from dipcoatimage.finitedepth_gui.views import (
    ...     ExperimentDataListView,
    ...     ReferenceView,
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentDataListView()
    ...     exptListWidget.setModel(model)
    ...     layout.addWidget(exptListWidget)
    ...     refWidget = ReferenceView()
    ...     refWidget.setModel(model)
    ...     layout.addWidget(refWidget)
    ...     window.setLayout(layout)
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    roiDrawFlagChanged = Signal(ROIDrawFlag)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None

        self._refPathLineEdit = QLineEdit()
        self._browseButton = QPushButton()
        self._importView = ImportDataView()
        self._tempROIView = ROIView()
        self._tempROIDrawButton = QPushButton()
        self._substROIView = ROIView()
        self._substROIDrawButton = QPushButton()
        self._paramStackWidget = dawiq.DataclassStackedWidget()
        self._drawOptStackWidget = dawiq.DataclassStackedWidget()

        self._refPathMapper = QDataWidgetMapper()
        self._refArgsMapper = QDataWidgetMapper()
        self._refROIsMapper = QDataWidgetMapper()

        self._browseButton.clicked.connect(self.browseReferenceImage)
        self._importView.editingFinished.connect(self._refArgsMapper.submit)
        self._tempROIView.editingFinished.connect(self._refROIsMapper.submit)
        self._tempROIDrawButton.setCheckable(True)
        self._tempROIDrawButton.clicked.connect(self._onTempROIDrawClick)
        self._substROIView.editingFinished.connect(self._refROIsMapper.submit)
        self._substROIDrawButton.setCheckable(True)
        self._substROIDrawButton.clicked.connect(self._onSubstROIDrawClick)
        self._paramStackWidget.currentDataEdited.connect(self._refArgsMapper.submit)
        self._drawOptStackWidget.currentDataEdited.connect(self._refArgsMapper.submit)
        refPathDelegate = ReferencePathDelegate()
        refPathDelegate.roiMaximumChanged.connect(self.setROIMaximum)
        self._refPathMapper.setItemDelegate(refPathDelegate)
        self._refArgsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._refArgsMapper.setItemDelegate(ReferenceArgsDelegate())
        self._refROIsMapper.setOrientation(Qt.Orientation.Vertical)
        self._refROIsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._refROIsMapper.setItemDelegate(ReferenceROIDelegate())

        self._refPathLineEdit.setPlaceholderText("Path for the reference image file")
        self._browseButton.setText("Browse")
        self._importView.setTitle("Reference type")
        self._tempROIDrawButton.setText("Draw template ROI")
        self._tempROIDrawButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._substROIDrawButton.setText("Draw substrate ROI")
        self._substROIDrawButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._paramStackWidget.addWidget(
            QGroupBox("Parameters")  # default empty widget
        )
        self._drawOptStackWidget.addWidget(
            QGroupBox("Draw options")  # default empty widget
        )

        layout = QVBoxLayout()
        pathLayout = QHBoxLayout()
        pathLayout.addWidget(self._refPathLineEdit)
        pathLayout.addWidget(self._browseButton)
        layout.addLayout(pathLayout)
        layout.addWidget(self._importView)
        tempROIGroupBox = QGroupBox("Template ROI")
        tempROILayout = QHBoxLayout()
        tempROILayout.addWidget(self._tempROIView)
        tempROILayout.addWidget(self._tempROIDrawButton)
        tempROIGroupBox.setLayout(tempROILayout)
        layout.addWidget(tempROIGroupBox)
        substROIGroupBox = QGroupBox("Substrate ROI")
        substROILayout = QHBoxLayout()
        substROILayout.addWidget(self._substROIView)
        substROILayout.addWidget(self._substROIDrawButton)
        substROIGroupBox.setLayout(substROILayout)
        layout.addWidget(substROIGroupBox)
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
        self._refPathMapper.setModel(model)
        self._refPathMapper.addMapping(self._refPathLineEdit, 0)
        self._refArgsMapper.setModel(model)
        self._refArgsMapper.addMapping(self, 0)
        self._refROIsMapper.setModel(model)
        self._refROIsMapper.addMapping(self._tempROIView, 1)
        self._refROIsMapper.addMapping(self._substROIView, 2)
        if model is not None:
            model.activatedIndexChanged.connect(self.setActivatedIndex)

    @Slot()
    def browseReferenceImage(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select reference image file",
            "./",
            options=QFileDialog.DontUseNativeDialog,
        )
        if path:
            self._refPathLineEdit.setText(path)
            self._refPathMapper.submit()

    def typeName(self) -> str:
        return self._importView.variableName()

    def setTypeName(self, name: str):
        self._importView.setVariableName(name)

    def moduleName(self) -> str:
        return self._importView.moduleName()

    def setModuleName(self, name: str):
        self._importView.setModuleName(name)

    @Slot(int, int)
    def setROIMaximum(self, w: int, h: int):
        self._tempROIView.setROIMaximum(w, h)
        self._substROIView.setROIMaximum(w, h)

    def _onTempROIDrawClick(self, checked: bool):
        if checked:
            self._substROIDrawButton.setChecked(False)
            self.roiDrawFlagChanged.emit(ROIDrawFlag.TEMPLATE)
        else:
            self.roiDrawFlagChanged.emit(ROIDrawFlag.NONE)

    def _onSubstROIDrawClick(self, checked: bool):
        if checked:
            self._tempROIDrawButton.setChecked(False)
            self.roiDrawFlagChanged.emit(ROIDrawFlag.SUBSTRATE)
        else:
            self.roiDrawFlagChanged.emit(ROIDrawFlag.NONE)

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
            self._refPathMapper.setRootIndex(index)
            refPathIndex = model.getIndexFor(IndexRole.REFPATH, index)
            self._refPathMapper.setCurrentModelIndex(refPathIndex)
            self._refArgsMapper.setRootIndex(index)
            refArgsIndex = model.getIndexFor(IndexRole.REFARGS, index)
            self._refArgsMapper.setCurrentModelIndex(refArgsIndex)
            self._refROIsMapper.setRootIndex(refArgsIndex)
            self._refROIsMapper.toFirst()
        else:
            self._refPathMapper.setCurrentModelIndex(QModelIndex())
            self._refArgsMapper.setCurrentModelIndex(QModelIndex())
            self._refROIsMapper.setCurrentModelIndex(QModelIndex())
            self._refPathLineEdit.clear()
            self._importView.clear()
            self._tempROIView.clear()
            self._substROIView.clear()
            self._paramStackWidget.setCurrentIndex(0)
            self._drawOptStackWidget.setCurrentIndex(0)


class ReferencePathDelegate(QStyledItemDelegate):

    roiMaximumChanged = Signal(int, int)

    def setEditorData(self, editor, index):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            indexRole = model.whatsThisIndex(index)
            if indexRole == IndexRole.REFPATH:
                path = model.data(index, role=model.Role_RefPath)
                try:
                    w, h = imagesize.get(path)
                except (FileNotFoundError, PermissionError):
                    w, h = (-1, -1)
                self.roiMaximumChanged.emit(w, h)
        super().setEditorData(editor, index)


class ReferenceArgsDelegate(dawiq.DataclassDelegate):

    TypeRole = ExperimentDataModel.Role_DataclassType
    DataRole = ExperimentDataModel.Role_DataclassData

    def ignoreMissing(self) -> bool:
        return False

    def setModelData(self, editor, model, index):
        if isinstance(model, ExperimentDataModel):
            indexRole = model.whatsThisIndex(index)
            if indexRole == IndexRole.REFARGS and isinstance(editor, ReferenceView):
                with ExperimentSignalBlocker(model):
                    # set ImportArgs for reference type to model
                    importArgs = ImportArgs(editor.typeName(), editor.moduleName())
                    model.setData(
                        model.getIndexFor(IndexRole.REF_TYPE, index),
                        importArgs,
                        role=model.Role_ImportArgs,
                    )

                    # set dataclasses types to model
                    paramIndex = model.getIndexFor(IndexRole.REF_PARAMETERS, index)
                    drawOptIndex = model.getIndexFor(IndexRole.REF_DRAWOPTIONS, index)
                    refType, _ = Importer(
                        importArgs.name, importArgs.module
                    ).try_import()
                    if isinstance(refType, type) and issubclass(
                        refType, SubstrateReferenceBase
                    ):
                        paramType = refType.Parameters
                        drawOptType = refType.DrawOptions
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

                topLevelIndex = model.getTopLevelIndex(index)
                flag = (
                    WorkerUpdateFlag.REFERENCE
                    | WorkerUpdateFlag.SUBSTRATE
                    | WorkerUpdateFlag.EXPERIMENT
                )
                model.updateWorker(topLevelIndex, flag)
                model.emitExperimentDataChanged(topLevelIndex, DataArgFlag.REFERENCE)

        super().setModelData(editor, model, index)

    def setEditorData(self, editor, index):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            indexRole = model.whatsThisIndex(index)
            if indexRole == IndexRole.REFARGS and isinstance(editor, ReferenceView):
                # set import args for reference type to editor
                importArgs = model.data(
                    model.getIndexFor(IndexRole.REF_TYPE, index),
                    role=model.Role_ImportArgs,
                )
                editor.setTypeName(importArgs.name)
                editor.setModuleName(importArgs.module)

                # add data widget if absent
                paramIndex = model.getIndexFor(IndexRole.REF_PARAMETERS, index)
                paramType = model.data(paramIndex, role=self.TypeRole)
                if isinstance(paramType, type) and dataclasses.is_dataclass(paramType):
                    paramWidgetIdx = editor.indexOfParametersType(paramType)
                    if paramWidgetIdx == -1:
                        paramWidgetIdx = editor.addParametersType(paramType)
                else:
                    paramWidgetIdx = -1
                drawOptIndex = model.getIndexFor(IndexRole.REF_DRAWOPTIONS, index)
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


class ReferenceROIDelegate(QStyledItemDelegate):
    def setModelData(self, editor, model, index):
        if isinstance(model, ExperimentDataModel) and isinstance(editor, ROIView):
            model.setData(index, editor.roi(), role=model.Role_ROI)
        super().setModelData(editor, model, index)

    def setEditorData(self, editor, index):
        model = index.model()
        if isinstance(model, ExperimentDataModel) and isinstance(editor, ROIView):
            roi = index.data(model.Role_ROI)
            editor.setROI(roi)
        super().setEditorData(editor, index)
