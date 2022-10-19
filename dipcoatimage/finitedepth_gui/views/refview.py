"""
Reference view
==============

V2 for controlwidgets/refwidget.py
"""

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
from dipcoatimage.finitedepth import SubstrateReferenceBase
from dipcoatimage.finitedepth.analysis import ImportArgs, ReferenceArgs
from dipcoatimage.finitedepth.util import DataclassProtocol, Importer, OptionalROI
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel, IndexRole
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
    ...     ExperimentListView,
    ...     ReferenceView,
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentListView()
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
        self._refPathMapper = QDataWidgetMapper()
        self._browseButton = QPushButton()
        self._importView = ImportDataView()
        self._tempROIView = ROIView()
        self._tempROIDrawButton = QPushButton()
        self._substROIView = ROIView()
        self._substROIDrawButton = QPushButton()
        self._paramStackWidget = dawiq.DataclassStackedWidget()
        self._drawOptStackWidget = dawiq.DataclassStackedWidget()
        self._refArgsMapper = QDataWidgetMapper()

        self._refPathMapper.setItemDelegate(ReferencePathDelegate())
        self._refPathMapper.itemDelegate().roiMaximumChanged.connect(self.setROIMaximum)
        self._browseButton.clicked.connect(self.browseReferenceImage)
        self._refArgsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._importView.editingFinished.connect(self._refArgsMapper.submit)
        self._tempROIView.editingFinished.connect(self._refArgsMapper.submit)
        self._tempROIDrawButton.setCheckable(True)
        self._tempROIDrawButton.clicked.connect(self._onTempROIDrawClick)
        self._substROIView.editingFinished.connect(self._refArgsMapper.submit)
        self._substROIDrawButton.setCheckable(True)
        self._substROIDrawButton.clicked.connect(self._onSubstROIDrawClick)
        self._paramStackWidget.currentDataValueChanged.connect(
            self._refArgsMapper.submit
        )
        self._drawOptStackWidget.currentDataValueChanged.connect(
            self._refArgsMapper.submit
        )
        self._refArgsMapper.setItemDelegate(ReferenceArgsDelegate())

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

    def templateROI(self) -> OptionalROI:
        return self._tempROIView.roi()

    def setTemplateROI(self, roi: OptionalROI):
        self._tempROIView.setROI(roi)

    def _onTempROIDrawClick(self, checked: bool):
        if checked:
            self._substROIDrawButton.setChecked(False)
            self.roiDrawFlagChanged.emit(ROIDrawFlag.TEMPLATE)
        else:
            self.roiDrawFlagChanged.emit(ROIDrawFlag.NONE)

    def substrateROI(self) -> OptionalROI:
        return self._substROIView.roi()

    def setSubstrateROI(self, roi: OptionalROI):
        self._substROIView.setROI(roi)

    def _onSubstROIDrawClick(self, checked: bool):
        if checked:
            self._tempROIDrawButton.setChecked(False)
            self.roiDrawFlagChanged.emit(ROIDrawFlag.SUBSTRATE)
        else:
            self.roiDrawFlagChanged.emit(ROIDrawFlag.NONE)

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

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            self._refPathMapper.setRootIndex(index)
            refPathIndex = model.getIndexFor(IndexRole.REFPATH, index)
            self._refPathMapper.setCurrentModelIndex(refPathIndex)
            self._refArgsMapper.setRootIndex(index)
            refIndex = model.getIndexFor(IndexRole.REFARGS, index)
            self._refArgsMapper.setCurrentModelIndex(refIndex)
        else:
            self._refPathLineEdit.clear()
            self._importView.clear()
            self._tempROIView.clear()
            self._substROIView.clear()
            self._refPathMapper.setCurrentModelIndex(QModelIndex())
            self._paramStackWidget.setCurrentIndex(0)
            self._drawOptStackWidget.setCurrentIndex(0)
            self._refArgsMapper.setCurrentModelIndex(QModelIndex())


class ReferencePathDelegate(QStyledItemDelegate):

    roiMaximumChanged = Signal(int, int)

    def setEditorData(self, editor, index):
        super().setEditorData(editor, index)
        path = index.data(Qt.DisplayRole)
        try:
            w, h = imagesize.get(path)
        except (FileNotFoundError, PermissionError):
            w, h = (-1, -1)
        self.roiMaximumChanged.emit(w, h)


class ReferenceArgsDelegate(dawiq.DataclassDelegate):
    def ignoreMissing(self) -> bool:
        return False

    def setModelData(self, editor, model, index):
        if isinstance(editor, ReferenceView):
            importArgs = ImportArgs(editor.typeName(), editor.moduleName())
            tempROI = editor.templateROI()
            substROI = editor.substrateROI()
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
            typeVar, _ = Importer(importArgs.name, importArgs.module).try_import()
            if isinstance(typeVar, type) and issubclass(
                typeVar, SubstrateReferenceBase
            ):
                paramType = typeVar.Parameters
                drawOptType = typeVar.DrawOptions
                parameters = dawiq.convertFromQt(
                    paramType, parameters, self.ignoreMissing()
                )
                drawOpt = dawiq.convertFromQt(
                    drawOptType, drawOpt, self.ignoreMissing()
                )
            refArgs = ReferenceArgs(importArgs, tempROI, substROI, parameters, drawOpt)
            model.setData(index, refArgs, Qt.UserRole)
        super().setModelData(editor, model, index)

    def setEditorData(self, editor, index):
        data = index.data(Qt.UserRole)
        if isinstance(editor, ReferenceView) and isinstance(data, ReferenceArgs):
            refArgs = data
            editor.setTypeName(refArgs.type.name)
            editor.setModuleName(refArgs.type.module)

            editor.setTemplateROI(refArgs.templateROI)
            editor.setSubstrateROI(refArgs.substrateROI)

            typeVar, _ = Importer(refArgs.type.name, refArgs.type.module).try_import()
            if isinstance(typeVar, type) and issubclass(
                typeVar, SubstrateReferenceBase
            ):
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

                self.setEditorDataclassData(
                    editor.currentParametersWidget(),
                    paramType,
                    refArgs.parameters,
                )
                self.setEditorDataclassData(
                    editor.currentDrawOptionsWidget(),
                    drawOptType,
                    refArgs.draw_options,
                )
            else:
                editor.setCurrentParametersIndex(0)
                editor.setCurrentDrawOptionsIndex(0)
        super().setEditorData(editor, index)
