"""
Control Widgets
===============

This module provides widgets to control the analysis parameters.

"""


import cv2  # type: ignore
import dataclasses
from dataclass2PySide6 import DataclassWidget, StackedDataclassWidget
from dipcoatimage.finitedepth import SubstrateReferenceBase, data_converter
from dipcoatimage.finitedepth.util import OptionalROI
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QGroupBox,
    QSizePolicy,
)
from typing import Optional, Dict, Any

from .importwidget import ImportWidget
from .roimodel import ROIWidget


__all__ = [
    "ReferenceWidgetData",
    "ReferenceWidget",
]


@dataclasses.dataclass
class ReferenceWidgetData:
    """Data from reference widget to construct substrate reference object."""

    type: Any
    image: Optional[npt.NDArray[np.uint8]]
    templateROI: OptionalROI
    substrateROI: OptionalROI
    parameters: Any
    draw_options: Any


class ReferenceWidget(QWidget):
    """
    Widget to control data for substrate reference object.

    .. rubric:: Substrate reference data

    Data consists of substrate reference type, which is a concrete subclass of
    :class:`.SubstrateReferenceBase`, and its parameters.

    Data are wrapped by :class:`ReferenceWidgetData`. Whenever the widget values
    change :attr:`dataChanged` signal emits the data.

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

    dataChanged = Signal(ReferenceWidgetData)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._refpath_lineedit = QLineEdit()
        self._browse_button = QPushButton()
        self._importwidget = ImportWidget()
        self._temproi_widget = ROIWidget()
        self._temproi_draw_button = QPushButton()
        self._substroi_widget = ROIWidget()
        self._substroi_draw_button = QPushButton()
        self._param_widget = StackedDataclassWidget()
        self._drawopt_widget = StackedDataclassWidget()

        self.connectSignals()
        self.browseButton().clicked.connect(self.browseReferenceImage)
        self.templateROIDrawButton().setCheckable(True)
        self.substrateROIDrawButton().setCheckable(True)
        self.templateROIDrawButton().toggled.connect(self.onTemplateROIDrawButtonToggle)
        self.substrateROIDrawButton().toggled.connect(
            self.onSubstrateROIDrawButtonToggle
        )

        default_paramwdgt = DataclassWidget()  # default empty widget
        default_drawwdgt = DataclassWidget()  # default empty widget
        default_paramwdgt.setDataName("Parameters")
        default_drawwdgt.setDataName("Draw Options")
        self.parametersWidget().addWidget(default_paramwdgt)
        self.drawOptionsWidget().addWidget(default_drawwdgt)

        self.initUI()

    def connectSignals(self):
        """Connect the signals disconnected by :meth:`disconnectSignals`."""
        self._typeCnct = self.typeWidget().variableChanged.connect(
            self.onReferenceTypeChange
        )
        self._pathCnct = self.pathLineEdit().editingFinished.connect(
            self.onPathEditFinished
        )
        self._temproiCnct = self.templateROIWidget().roiChanged.connect(self.emitData)
        self._substroiCnct = self.substrateROIWidget().roiChanged.connect(self.emitData)
        self._paramCnct = self.parametersWidget().dataValueChanged.connect(
            self.emitData
        )
        self._drawoptCnct = self.drawOptionsWidget().dataValueChanged.connect(
            self.emitData
        )

    def disconnectSignals(self):
        """Disconnect the signals connected by :meth:`connectSignals`."""
        self.drawOptionsWidget().dataValueChanged.disconnect(self._drawoptCnct)
        self.parametersWidget().dataValueChanged.disconnect(self._paramCnct)
        self.substrateROIWidget().roiChanged.disconnect(self._substroiCnct)
        self.templateROIWidget().roiChanged.disconnect(self._temproiCnct)
        self.pathLineEdit().editingFinished.disconnect(self._pathCnct)
        self.typeWidget().variableChanged.disconnect(self._typeCnct)

    def initUI(self):
        self.pathLineEdit().setPlaceholderText("Path for the reference image file")
        self.browseButton().setText("Browse")
        self.typeWidget().variableComboBox().setPlaceholderText(
            "Select reference type or specify import info"
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

    def pathLineEdit(self) -> QLineEdit:
        return self._refpath_lineedit

    def browseButton(self) -> QPushButton:
        return self._browse_button

    def typeWidget(self) -> ImportWidget:
        return self._importwidget

    def templateROIWidget(self) -> ROIWidget:
        return self._temproi_widget

    def templateROIDrawButton(self) -> QPushButton:
        return self._temproi_draw_button

    def substrateROIWidget(self) -> ROIWidget:
        return self._substroi_widget

    def substrateROIDrawButton(self) -> QPushButton:
        return self._substroi_draw_button

    def parametersWidget(self) -> StackedDataclassWidget:
        return self._param_widget

    def drawOptionsWidget(self) -> StackedDataclassWidget:
        return self._drawopt_widget

    def currentParametersWidget(self) -> DataclassWidget:
        widget = self.parametersWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

    def currentDrawOptionsWidget(self) -> DataclassWidget:
        widget = self.drawOptionsWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

    @Slot(object)
    def onReferenceTypeChange(self):
        """
        Apply current variable from import widget to other widgets and emit data.
        """
        var = self.typeWidget().variable()
        self.setCurrentParametersWidget(var)
        self.setCurrentDrawOptionsWidget(var)
        self.emitData()

    def setCurrentParametersWidget(self, reftype: Any):
        if isinstance(reftype, type) and issubclass(reftype, SubstrateReferenceBase):
            dcls = reftype.Parameters
            index = self.parametersWidget().indexOfDataclass(dcls)
            if index == -1:
                self.parametersWidget().addDataclass(dcls, "Parameters")
                index = self.parametersWidget().indexOfDataclass(dcls)
        else:
            index = 0
        self.parametersWidget().setCurrentIndex(index)

    def setCurrentDrawOptionsWidget(self, reftype: Any):
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
    def onPathEditFinished(self):
        self.disconnectSignals()
        self.updateROIMaximum()
        self.connectSignals()
        self.emitData()

    def updateROIMaximum(self):
        path = self.pathLineEdit().text()
        img = cv2.imread(path)
        if img is None:
            w, h = (0, 0)
        else:
            w, h = (img.shape[1], img.shape[0])
        self.templateROIWidget().setROIMaximum(w, h)
        self.substrateROIWidget().setROIMaximum(w, h)

    def setReferencePath(self, path: str):
        self.pathLineEdit().setText(path)
        self.onPathEditFinished()

    def copyWidgetDataToDict(self, data: Dict[str, Any]):
        data["reference"]["type"]["name"] = (
            self.typeWidget().variableNameLineEdit().text()
        )
        data["reference"]["type"]["module"] = (
            self.typeWidget().moduleNameLineEdit().text()
        )
        data["reference"]["path"] = self.pathLineEdit().text()
        data["reference"]["templateROI"] = self.templateROIWidget().roiModel().roi()
        data["reference"]["substrateROI"] = self.substrateROIWidget().roiModel().roi()
        data["reference"]["parameters"] = data_converter.unstructure(
            self.currentParametersWidget().dataValue()
        )
        data["reference"]["draw_options"] = data_converter.unstructure(
            self.currentDrawOptionsWidget().dataValue()
        )

    def referenceWidgetData(self) -> ReferenceWidgetData:
        ref_type = self.typeWidget().variable()
        img = cv2.imread(self.pathLineEdit().text(), cv2.IMREAD_GRAYSCALE)
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

        data = ReferenceWidgetData(
            ref_type, img, templateROI, substrateROI, param, drawopt
        )
        return data

    @Slot()
    def emitData(self):
        self.dataChanged.emit(self.referenceWidgetData())

    def browseReferenceImage(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select reference image file",
            "./",
            options=QFileDialog.DontUseNativeDialog,
        )
        if path:
            self.setReferencePath(path)

    def onTemplateROIDrawButtonToggle(self, state: bool):
        if state:
            self.substrateROIDrawButton().setChecked(False)

    def onSubstrateROIDrawButtonToggle(self, state: bool):
        if state:
            self.templateROIDrawButton().setChecked(False)

    def setReferenceArgs(self, refargs: Dict[str, Any]):
        """Update the widgets data with *refargs*."""
        self.disconnectSignals()

        self.typeWidget().setImportInformation(
            refargs["type"]["name"], refargs["type"]["module"]
        )
        var = self.typeWidget().variable()
        self.setCurrentParametersWidget(var)
        self.setCurrentDrawOptionsWidget(var)

        self.pathLineEdit().setText(refargs["path"])
        self.updateROIMaximum()

        self.templateROIWidget().setROI(*refargs["templateROI"])
        self.substrateROIWidget().setROI(*refargs["substrateROI"])

        reftype = self.typeWidget().variable()
        if isinstance(reftype, type) and issubclass(reftype, SubstrateReferenceBase):
            params = data_converter.structure(refargs["parameters"], reftype.Parameters)
            self.currentParametersWidget().setDataValue(params)
            drawopts = data_converter.structure(
                refargs["draw_options"], reftype.DrawOptions
            )
            self.currentDrawOptionsWidget().setDataValue(drawopts)

        self.connectSignals()
