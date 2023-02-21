"""
Reference view
==============

"""

import dataclasses
import dawiq
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
from dipcoatimage.finitedepth_gui.core import ROIDrawMode
from dipcoatimage.finitedepth_gui.model import (
    ExperimentDataModel,
    IndexRole,
)
from .importview import ImportDataView, ImportArgsDelegate
from .roiview import ROIView
from typing import Optional


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

    roiDrawModeChanged = Signal(ROIDrawMode)

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

        self._typeMapper = QDataWidgetMapper()
        self._pathMapper = QDataWidgetMapper()
        self._argsMapper = QDataWidgetMapper()

        self._browseButton.clicked.connect(self.browseReferenceImage)
        self._importView.editingFinished.connect(self._typeMapper.submit)
        self._tempROIView.editingFinished.connect(self._argsMapper.submit)
        self._tempROIDrawButton.setCheckable(True)
        self._tempROIDrawButton.clicked.connect(self._onTempROIDrawClick)
        self._substROIView.editingFinished.connect(self._argsMapper.submit)
        self._substROIDrawButton.setCheckable(True)
        self._substROIDrawButton.clicked.connect(self._onSubstROIDrawClick)
        self._paramStackWidget.currentDataEdited.connect(self._argsMapper.submit)
        self._drawOptStackWidget.currentDataEdited.connect(self._argsMapper.submit)
        self._typeMapper.setOrientation(Qt.Orientation.Vertical)
        self._typeMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._typeMapper.setItemDelegate(ImportArgsDelegate())
        refPathDelegate = ReferencePathDelegate()
        refPathDelegate.roiMaximumChanged.connect(self.setROIMaximum)
        self._pathMapper.setItemDelegate(refPathDelegate)
        self._argsMapper.setOrientation(Qt.Orientation.Vertical)
        self._argsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._argsMapper.setItemDelegate(ReferenceArgsDelegate())

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
        self._typeMapper.clearMapping()
        self._pathMapper.clearMapping()
        self._argsMapper.clearMapping()
        self._typeMapper.setModel(model)
        self._pathMapper.setModel(model)
        self._argsMapper.setModel(model)
        if model is not None:
            model.activatedIndexChanged.connect(self.setActivatedIndex)
            self._typeMapper.addMapping(self._importView, model.Row_RefArgs)
            self._pathMapper.addMapping(self._refPathLineEdit, model.Row_RefPath)
            self._argsMapper.addMapping(self._tempROIView, model.Row_RefTemplateROI)
            self._argsMapper.addMapping(self._substROIView, model.Row_RefSubstrateROI)
            self._argsMapper.addMapping(self._paramStackWidget, model.Row_RefParameters)
            self._argsMapper.addMapping(
                self._drawOptStackWidget, model.Row_RefDrawOptions
            )

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
            self._pathMapper.submit()

    @Slot(int, int)
    def setROIMaximum(self, w: int, h: int):
        self._tempROIView.setROIMaximum(w, h)
        self._substROIView.setROIMaximum(w, h)

    def _onTempROIDrawClick(self, checked: bool):
        if checked:
            self._substROIDrawButton.setChecked(False)
            self.roiDrawModeChanged.emit(ROIDrawMode.TEMPLATE)
        else:
            self.roiDrawModeChanged.emit(ROIDrawMode.NONE)

    def _onSubstROIDrawClick(self, checked: bool):
        if checked:
            self._tempROIDrawButton.setChecked(False)
            self.roiDrawModeChanged.emit(ROIDrawMode.SUBSTRATE)
        else:
            self.roiDrawModeChanged.emit(ROIDrawMode.NONE)

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            self._typeMapper.setRootIndex(index)
            self._typeMapper.toFirst()
            self._pathMapper.setRootIndex(index)
            self._pathMapper.toFirst()
            refArgsIndex = model.getIndexFor(IndexRole.REFARGS, index)
            self._argsMapper.setRootIndex(refArgsIndex)
            self._argsMapper.toFirst()
        else:
            self._typeMapper.setCurrentModelIndex(QModelIndex())
            self._pathMapper.setCurrentModelIndex(QModelIndex())
            self._argsMapper.setCurrentModelIndex(QModelIndex())
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

    def cacheModelData(cls, model, index, value, role):
        if isinstance(model, ExperimentDataModel):
            model.cacheData(index, value, role)
        else:
            super().cacheModelData(model, index, value, role)

    def setModelData(self, editor, model, index):
        if isinstance(model, ExperimentDataModel) and isinstance(editor, ROIView):
            model.cacheData(index, editor.roi(), model.Role_ROI)
        super().setModelData(editor, model, index)

    def setEditorData(self, editor, index):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            if isinstance(editor, ROIView):
                roi = index.data(model.Role_ROI)
                editor.setROI(roi)
            elif isinstance(editor, dawiq.DataclassStackedWidget):
                # add data widget if absent
                dclsType = model.data(index, role=self.TypeRole)
                if isinstance(dclsType, type) and dataclasses.is_dataclass(dclsType):
                    dclsIdx = editor.indexOfDataclass(dclsType)
                    if dclsIdx == -1:
                        widget = dawiq.dataclass2Widget(dclsType)
                        indexRole = model.whatsThisIndex(index)
                        if indexRole == IndexRole.REF_PARAMETERS:
                            title = "Parameters"
                        elif indexRole == IndexRole.REF_DRAWOPTIONS:
                            title = "Draw options"
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
