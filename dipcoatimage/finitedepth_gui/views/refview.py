"""
Reference view
==============

V2 for controlwidgets/refwidget.py
"""

import cv2
import dawiq
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
    QStyledItemDelegate,
)
from dipcoatimage.finitedepth import SubstrateReferenceBase
from dipcoatimage.finitedepth.analysis import ImportArgs, ReferenceArgs
from dipcoatimage.finitedepth.util import DataclassProtocol, Importer, IntROI
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from .importview import ImportDataView
from .roiview import ROIView
from typing import Optional, Type


__all__ = [
    "ReferenceView",
    "ReferencePathDelegate",
    "ReferenceArgsDelegate",
]


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
        self._refArgsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._importView.editingFinished.connect(self._refArgsMapper.submit)
        self._tempROIView.editingFinished.connect(self._refArgsMapper.submit)
        self._substROIView.editingFinished.connect(self._refArgsMapper.submit)
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

    def parametersStackedWidget(self) -> dawiq.DataclassStackedWidget:
        return self._paramStackWidget

    def drawOptionsStackedWidget(self) -> dawiq.DataclassStackedWidget:
        return self._drawOptStackWidget

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

    def templateROI(self) -> IntROI:
        return self._tempROIView.roi()

    def substrateROI(self) -> IntROI:
        return self._substROIView.roi()

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            self._refPathMapper.setRootIndex(index)
            refPathIndex = model.index(model.ROW_REFPATH, 0, index)
            self._refPathMapper.setCurrentModelIndex(refPathIndex)
            self._refArgsMapper.setRootIndex(index)
            refIndex = model.index(model.ROW_REFERENCE, 0, index)
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

    def addParameterType(self, paramType: Type[DataclassProtocol]) -> int:
        widget = dawiq.dataclass2Widget(paramType)
        widget.setTitle("Parameters")
        index = self._paramStackWidget.addDataWidget(widget, paramType)
        return index

    def addDrawOptionsType(self, drawOptType: Type[DataclassProtocol]) -> int:
        widget = dawiq.dataclass2Widget(drawOptType)
        widget.setTitle("Draw options")
        index = self._drawOptStackWidget.addDataWidget(widget, drawOptType)
        return index


class ReferencePathDelegate(QStyledItemDelegate):

    roiMaximumChanged = Signal(int, int)

    def setEditorData(self, editor, index):
        super().setEditorData(editor, index)
        path = index.data(Qt.DisplayRole)
        img = cv2.imread(path)
        if img is not None:
            w, h = (img.shape[1], img.shape[0])
            self.roiMaximumChanged.emit(w, h)


class ReferenceArgsDelegate(dawiq.DataclassDelegate):
    def ignoreMissing(self) -> bool:
        return False

    def setModelData(self, editor, model, index):
        if isinstance(editor, ReferenceView):
            importArgs = ImportArgs(editor.typeName(), editor.moduleName())
            tempROI = editor.templateROI()
            substROI = editor.substrateROI()
            paramWidget = editor.parametersStackedWidget().currentWidget()
            if isinstance(paramWidget, dawiq.DataWidget):
                parameters = paramWidget.dataValue()
            else:
                parameters = {}
            drawOptWidget = editor.drawOptionsStackedWidget().currentWidget()
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
                drawOptType = dawiq.convertFromQt(
                    drawOptType, drawOpt, self.ignoreMissing()
                )
            refArgs = ReferenceArgs(importArgs, tempROI, substROI, parameters, drawOpt)
            model.setData(index, refArgs, Qt.UserRole)
        super().setModelData(editor, model, index)

    def setEditorData(self, editor, index):
        data = index.data(Qt.UserRole)
        if isinstance(editor, ReferenceView) and isinstance(data, ReferenceArgs):
            editor.setTypeName(data.type.name)
            editor.setModuleName(data.type.module)

            typeVar, _ = Importer(data.type.name, data.type.module).try_import()
            if isinstance(typeVar, type) and issubclass(
                typeVar, SubstrateReferenceBase
            ):
                paramType = typeVar.Parameters
                paramIdx = editor.parametersStackedWidget().indexOfDataclass(paramType)
                if paramIdx == -1:
                    paramIdx = editor.addParameterType(paramType)
                drawOptType = typeVar.DrawOptions
                drawOptIdx = editor.drawOptionsStackedWidget().indexOfDataclass(
                    drawOptType
                )
                if drawOptIdx == -1:
                    drawOptIdx = editor.addDrawOptionsType(drawOptType)
                editor.parametersStackedWidget().setCurrentIndex(paramIdx)
                editor.drawOptionsStackedWidget().setCurrentIndex(drawOptIdx)

                self.setEditorDataclassData(
                    editor.parametersStackedWidget().currentWidget(),
                    paramType,
                    data.parameters,
                )
                self.setEditorDataclassData(
                    editor.drawOptionsStackedWidget().currentWidget(),
                    drawOptType,
                    data.draw_options,
                )
            else:
                editor.parametersStackedWidget().setCurrentIndex(0)
                editor.drawOptionsStackedWidget().setCurrentIndex(0)
        super().setEditorData(editor, index)
