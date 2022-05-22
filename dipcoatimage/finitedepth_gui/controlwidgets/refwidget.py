import cv2  # type: ignore
from dataclass2PySide6 import DataclassWidget, StackedDataclassWidget
from dipcoatimage.finitedepth import SubstrateReferenceBase, data_converter
from dipcoatimage.finitedepth.analysis import ImportArgs, ReferenceArgs
from dipcoatimage.finitedepth_gui.core import StructuredReferenceArgs
from dipcoatimage.finitedepth_gui.importwidget import ImportWidget
from dipcoatimage.finitedepth_gui.roimodel import ROIWidget
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QDataWidgetMapper,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QGroupBox,
    QSizePolicy,
)
from typing import Any, List
from .base import ControlWidget


__all__ = [
    "ReferenceWidget",
]


class ReferenceWidget(ControlWidget):
    """
    View-like widget to control data for substrate reference object.

    .. rubric:: Substrate reference data

    Data consists of substrate reference type which is a concrete subclass of
    :class:`.SubstrateReferenceBase`, its image and its parameters.

    Image is emitted by :attr:`imageChanged` signal.

    .. rubric:: Setting type

    Substrate reference type can be specified by :meth:`typeWidget`.

    When current class changes, current indices of :meth:`parametersWidget` and
    :meth:`drawOptionsWidget` are changed to show the new dataclass widget.

    .. rubric:: Setting image

    Image must be loaded from local file. File path can be set by directly
    passing to :meth:`pathLineEdit`, or selecting from file dialog open by
    clicking :meth:`browseButton`.

    Changing the image updates the maximum ROI value of :meth:`templateROIWidget`
    and :meth:`substrateROIWidget`.

    .. rubric:: Setting ROI

    :meth:`templateROIWidget` and :meth:`substrateROIWidget` set the ROI values.

    :meth:`templateROIDrawButton` and :meth:`substrateROIDrawButton` are the API
    for higher widget. When toggled, external object controls the ROI models of
    ROI widgets, e.g. drawing on the displaying widget.

    Examples
    ========

    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.controlwidgets import ReferenceWidget
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     widget = ReferenceWidget()
    ...     widget.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    imageChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._blockModelUpdate = False

        self._refpath_mapper = QDataWidgetMapper()
        self._refpath_lineedit = QLineEdit()
        self._browse_button = QPushButton()
        self._importwidget = ImportWidget()
        self._temproi_widget = ROIWidget()
        self._temproi_draw_button = QPushButton()
        self._substroi_widget = ROIWidget()
        self._substroi_draw_button = QPushButton()
        self._param_widget = StackedDataclassWidget()
        self._drawopt_widget = StackedDataclassWidget()

        self.pathLineEdit().editingFinished.connect(self.onPathEditFinished)
        self.browseButton().clicked.connect(self.browseReferenceImage)
        self.typeWidget().variableChanged.connect(self.onReferenceTypeChange)
        self.templateROIWidget().roiModel().roiChanged.connect(self.commitToCurrentItem)
        self.templateROIDrawButton().setCheckable(True)
        self.templateROIDrawButton().toggled.connect(self.onTemplateROIDrawButtonToggle)
        self.substrateROIWidget().roiModel().roiChanged.connect(
            self.commitToCurrentItem
        )
        self.substrateROIDrawButton().setCheckable(True)
        self.substrateROIDrawButton().toggled.connect(
            self.onSubstrateROIDrawButtonToggle
        )
        self.parametersWidget().dataValueChanged.connect(self.commitToCurrentItem)
        self.drawOptionsWidget().dataValueChanged.connect(self.commitToCurrentItem)

        default_paramwdgt = DataclassWidget()  # default empty widget
        default_drawwdgt = DataclassWidget()  # default empty widget
        default_paramwdgt.setDataName("Parameters")
        default_drawwdgt.setDataName("Draw Options")
        self.parametersWidget().addWidget(default_paramwdgt)
        self.drawOptionsWidget().addWidget(default_drawwdgt)

        self.pathLineEdit().setPlaceholderText("Path for the reference image file")
        self.browseButton().setText("Browse")
        self.typeWidget().variableComboBox().setPlaceholderText(
            "Select reference type or specify import information"
        )
        self.typeWidget().variableNameLineEdit().setPlaceholderText(
            "Reference type name"
        )
        self.typeWidget().moduleNameLineEdit().setPlaceholderText(
            "Module name for the reference type"
        )
        self.templateROIDrawButton().setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.templateROIDrawButton().setText("Draw template ROI")
        self.substrateROIDrawButton().setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.substrateROIDrawButton().setText("Draw substrate ROI")

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.pathLineEdit())
        path_layout.addWidget(self.browseButton())

        temproi_groupbox = QGroupBox("Template ROI")
        temproi_layout = QHBoxLayout()
        temproi_layout.addWidget(self.templateROIWidget())
        temproi_layout.addWidget(self.templateROIDrawButton())
        temproi_groupbox.setLayout(temproi_layout)

        substroi_groupbox = QGroupBox("Substrate ROI")
        substroi_layout = QHBoxLayout()
        substroi_layout.addWidget(self.substrateROIWidget())
        substroi_layout.addWidget(self.substrateROIDrawButton())
        substroi_groupbox.setLayout(substroi_layout)

        options_layout = QHBoxLayout()
        options_layout.addWidget(self.parametersWidget())
        options_layout.addWidget(self.drawOptionsWidget())

        main_layout = QVBoxLayout()
        main_layout.addLayout(path_layout)
        main_layout.addWidget(self.typeWidget())
        main_layout.addWidget(temproi_groupbox)
        main_layout.addWidget(substroi_groupbox)
        main_layout.addLayout(options_layout)
        self.setLayout(main_layout)

    def pathMapper(self) -> QDataWidgetMapper:
        """
        Mapper to update :meth:`pathLineEdit` with reference path of currently
        activated item from :meth:`experimentItemModel`.
        """
        return self._refpath_mapper

    def pathLineEdit(self) -> QLineEdit:
        """Line edit for the path to reference file."""
        return self._refpath_lineedit

    def browseButton(self) -> QPushButton:
        """Button to browse reference file and set to :math:`pathLineEdit`."""
        return self._browse_button

    def typeWidget(self) -> ImportWidget:
        """Widget to specify the reference type."""
        return self._importwidget

    def templateROIWidget(self) -> ROIWidget:
        """Widget to specify the template ROI."""
        return self._temproi_widget

    def templateROIDrawButton(self) -> QPushButton:
        """Button to signal the external API to draw the template ROI."""
        return self._temproi_draw_button

    def substrateROIWidget(self) -> ROIWidget:
        """Widget to specify the substrate ROI."""
        return self._substroi_widget

    def substrateROIDrawButton(self) -> QPushButton:
        """Button to signal the external API to draw the substrate ROI."""
        return self._substroi_draw_button

    def parametersWidget(self) -> StackedDataclassWidget:
        """Widget to specify the reference parameters."""
        return self._param_widget

    def drawOptionsWidget(self) -> StackedDataclassWidget:
        """Widget to specify the reference drawing options."""
        return self._drawopt_widget

    def setExperimentItemModel(self, model):
        """Set :meth:`experimentItemModel` and remap :meth:`pathMapper`."""
        super().setExperimentItemModel(model)
        self.pathMapper().setModel(model)
        self.pathMapper().addMapping(
            self.pathLineEdit(),
            model.Col_ReferencePath,
        )

    def setCurrentExperimentRow(self, row: int):
        super().setCurrentExperimentRow(row)
        # update reference path
        self.pathMapper().setCurrentIndex(row)

        # update reference args
        self._blockModelUpdate = True
        model = self.experimentItemModel()
        args = model.data(
            model.index(row, model.Col_Reference),
            model.Role_Args,
        )
        self.typeWidget().variableNameLineEdit().setText(args.type.name)
        self.typeWidget().moduleNameLineEdit().setText(args.type.module)
        self.typeWidget().onInformationEdit()
        self.templateROIWidget().roiModel().setROI(*args.templateROI)
        self.substrateROIWidget().roiModel().setROI(*args.substrateROI)
        paramWidget = self.currentParametersWidget()
        try:
            paramWidget.setDataValue(
                data_converter.structure(args.parameters, paramWidget.dataclassType())
            )
        except TypeError:
            pass

        drawWidget = self.currentDrawOptionsWidget()
        try:
            drawWidget.setDataValue(
                data_converter.structure(args.draw_options, drawWidget.dataclassType())
            )
        except TypeError:
            pass
        self._blockModelUpdate = False

    def currentParametersWidget(self) -> DataclassWidget:
        """Currently displayed parameters widget."""
        widget = self.parametersWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

    def setCurrentParametersWidget(self, reftype: Any):
        """
        Update the parameters widget to display the parameters for *reftype*.
        """
        if isinstance(reftype, type) and issubclass(reftype, SubstrateReferenceBase):
            dcls = reftype.Parameters
            index = self.parametersWidget().indexOfDataclass(dcls)
            if index == -1:
                self.parametersWidget().addDataclass(dcls, "Parameters")
                index = self.parametersWidget().indexOfDataclass(dcls)
        else:
            index = 0
        self.parametersWidget().setCurrentIndex(index)

    def currentDrawOptionsWidget(self) -> DataclassWidget:
        """Currently displayed drawing options widget."""
        widget = self.drawOptionsWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

    def setCurrentDrawOptionsWidget(self, reftype: Any):
        """
        Update the drawing options widget to display the drawing option for
        *reftype*.
        """
        if isinstance(reftype, type) and issubclass(reftype, SubstrateReferenceBase):
            dcls = reftype.DrawOptions
            index = self.drawOptionsWidget().indexOfDataclass(dcls)
            if index == -1:
                self.drawOptionsWidget().addDataclass(dcls, "Draw options")
                index = self.drawOptionsWidget().indexOfDataclass(dcls)
        else:
            index = 0
        self.drawOptionsWidget().setCurrentIndex(index)

    @Slot()
    def onReferenceTypeChange(self):
        """
        Apply current variable from import widget to other widgets and emit data.
        """
        var = self.typeWidget().variable()
        self.setCurrentParametersWidget(var)
        self.setCurrentDrawOptionsWidget(var)
        self.commitToCurrentItem()

    def structuredReferenceArgs(self) -> StructuredReferenceArgs:
        """Return :class:`StructuredReferenceArgs` from current widget values."""
        ref_type = self.typeWidget().variable()
        templateROI = self.templateROIWidget().roiModel().roi()
        substrateROI = self.substrateROIWidget().roiModel().roi()
        try:
            param = self.currentParametersWidget().dataValue()
        except (TypeError, ValueError):
            param = None
        try:
            drawopt = self.currentDrawOptionsWidget().dataValue()
        except (TypeError, ValueError):
            drawopt = None

        data = StructuredReferenceArgs(
            ref_type, templateROI, substrateROI, param, drawopt
        )
        return data

    def referenceArgs(self) -> ReferenceArgs:
        """Return :class:`ReferenceArgs` from current widget values."""
        importArgs = ImportArgs(
            self.typeWidget().variableNameLineEdit().text(),
            self.typeWidget().moduleNameLineEdit().text(),
        )
        templateROI = self.templateROIWidget().roiModel().roi()
        substrateROI = self.substrateROIWidget().roiModel().roi()
        try:
            param = data_converter.unstructure(
                self.currentParametersWidget().dataValue()
            )
        except (TypeError, ValueError):
            param = dict()
        try:
            drawopt = data_converter.unstructure(
                self.currentDrawOptionsWidget().dataValue()
            )
        except (TypeError, ValueError):
            drawopt = dict()
        args = ReferenceArgs(importArgs, templateROI, substrateROI, param, drawopt)
        return args

    @Slot()
    def commitToCurrentItem(self):
        """
        Set :meth:`referenceArgs` and :meth:`structuredReferenceArgs`to
        currently activated item from :meth:`experimentItemModel`.
        """
        if not self._blockModelUpdate:
            model = self.experimentItemModel()
            row = self.currentExperimentRow()
            index = model.index(row, model.Col_Reference)
            if index.isValid():
                model.setData(index, self.referenceArgs(), model.Role_Args)
                model.setData(
                    index, self.structuredReferenceArgs(), model.Role_StructuredArgs
                )

    @Slot(str)
    def setReferencePath(self, path: str):
        """Update :meth:`pathLineEdit` with *path*."""
        self.pathLineEdit().setText(path)
        self.onPathEditFinished()
        self.pathMapper().submit()

    @Slot()
    def onPathEditFinished(self):
        """Update ROI widget, emit :attr:`imageChanged` and commit to model."""
        img = cv2.imread(self.pathLineEdit().text())
        if img is None:
            w, h = (0, 0)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            w, h = (img.shape[1], img.shape[0])
        self.templateROIWidget().setROIMaximum(w, h)
        self.substrateROIWidget().setROIMaximum(w, h)
        self.imageChanged.emit(img)

    @Slot()
    def browseReferenceImage(self):
        """Browse the reference file and set to :math:`pathLineEdit`"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select reference image file",
            "./",
            options=QFileDialog.DontUseNativeDialog,
        )
        if path:
            self.setReferencePath(path)

    def resetReferenceImage(self):
        """Apply the reference path in model to reference widget."""
        model = self.experimentItemModel()
        refpath = model.data(
            model.index(self.currentExperimentRow(), model.Col_ReferencePath)
        )
        self.setReferencePath(refpath)

    @Slot()
    def onTemplateROIDrawButtonToggle(self, state: bool):
        """Untoggle :meth:`substrateROIDrawButton`."""
        if state:
            self.substrateROIDrawButton().setChecked(False)

    @Slot()
    def onSubstrateROIDrawButtonToggle(self, state: bool):
        """Untoggle :meth:`templateROIDrawButton`."""
        if state:
            self.templateROIDrawButton().setChecked(False)

    def onExperimentsRemove(self, rows: List[int]):
        if self.currentExperimentRow() in rows:
            self._currentExperimentRow = -1
            self.pathLineEdit().clear()
            self.typeWidget().clear()
            self.templateROIWidget().roiModel().setROI(0, 0, 0, 0)
            self.substrateROIWidget().roiModel().setROI(0, 0, 0, 0)
            self.templateROIWidget().setROIMaximum(0, 0)
            self.substrateROIWidget().setROIMaximum(0, 0)
            self.parametersWidget().setCurrentIndex(0)
            self.drawOptionsWidget().setCurrentIndex(0)
