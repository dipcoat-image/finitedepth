from dataclass2PySide6 import DataclassWidget, StackedDataclassWidget
from dipcoatimage.finitedepth import (
    CoatingLayerBase,
    data_converter,
    ImportArgs,
    CoatingLayerArgs,
)
from dipcoatimage.finitedepth_gui.core import StructuredCoatingLayerArgs
from dipcoatimage.finitedepth_gui.importwidget import ImportWidget
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout
from typing import Any, List
from .base import ControlWidget


__all__ = [
    "CoatingLayerWidget",
]


class CoatingLayerWidget(ControlWidget):
    """
    View-like widget to control data for coating layer object.

    .. rubric:: Coating layer data

    Data consists of coating layer type which is a concrete subclass of
    :class:`.CoatingLayerBase`, its *parameters*, *draw_options* and
    *deco_options*. Note that coated substrate image is not specified by this
    widget, but by :class:`ExperimentWidget`.

    .. rubric:: Setting type

    Coating layer type can be specified by :meth:`typeWidget`.

    When current class changes, current indices of :meth:`parametersWidget`,
    :meth:`drawOptionsWidget`, and :meth:`decoOptionsWidget` are changed to show
    the new dataclass widget.

    Examples
    ========

    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.controlwidgets import (
    ...     CoatingLayerWidget
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     widget = CoatingLayerWidget()
    ...     widget.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._blockModelUpdate = False

        self._importwidget = ImportWidget()
        self._param_widget = StackedDataclassWidget()
        self._drawopt_widget = StackedDataclassWidget()
        self._decoopt_widget = StackedDataclassWidget()

        self.typeWidget().variableChanged.connect(self.onCoatingLayerTypeChange)
        self.parametersWidget().dataValueChanged.connect(self.commitToCurrentItem)
        self.drawOptionsWidget().dataValueChanged.connect(self.commitToCurrentItem)
        self.decoOptionsWidget().dataValueChanged.connect(self.commitToCurrentItem)

        default_paramwdgt = DataclassWidget()  # default empty widget
        default_drawwdgt = DataclassWidget()  # default empty widget
        default_decowdgt = DataclassWidget()  # default empty widget
        default_paramwdgt.setDataName("Parameters")
        default_drawwdgt.setDataName("Draw Options")
        default_decowdgt.setDataName("Decorate Options")
        self.parametersWidget().addWidget(default_paramwdgt)
        self.drawOptionsWidget().addWidget(default_drawwdgt)
        self.decoOptionsWidget().addWidget(default_decowdgt)

        self.typeWidget().variableComboBox().setPlaceholderText(
            "Select coating layer type or specify import information"
        )
        self.typeWidget().variableNameLineEdit().setPlaceholderText(
            "Coating layer type name"
        )
        self.typeWidget().moduleNameLineEdit().setPlaceholderText(
            "Module name for the coating layer type"
        )

        options_layout = QHBoxLayout()
        options_layout.addWidget(self.drawOptionsWidget())
        options_layout.addWidget(self.decoOptionsWidget())

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.typeWidget())
        main_layout.addWidget(self.parametersWidget())
        main_layout.addLayout(options_layout)
        self.setLayout(main_layout)

    def typeWidget(self) -> ImportWidget:
        """Widget to specify the coating layer type."""
        return self._importwidget

    def parametersWidget(self) -> StackedDataclassWidget:
        """Widget to specify the coating layer parameters."""
        return self._param_widget

    def drawOptionsWidget(self) -> StackedDataclassWidget:
        """Widget to specify the coating layer drawing options."""
        return self._drawopt_widget

    def decoOptionsWidget(self) -> StackedDataclassWidget:
        """Widget to specify the coating layer decorating options."""
        return self._decoopt_widget

    def currentParametersWidget(self) -> DataclassWidget:
        """Currently displayed parameters widget."""
        widget = self.parametersWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

    def setCurrentExperimentRow(self, row: int):
        super().setCurrentExperimentRow(row)

        self._blockModelUpdate = True
        model = self.experimentItemModel()
        args = model.data(
            model.index(row, model.Col_CoatingLayer),
            model.Role_Args,
        )
        self.typeWidget().variableNameLineEdit().setText(args.type.name)
        self.typeWidget().moduleNameLineEdit().setText(args.type.module)
        self.typeWidget().onInformationEdit()
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
        decoWidget = self.currentDecoOptionsWidget()
        try:
            decoWidget.setDataValue(
                data_converter.structure(args.deco_options, decoWidget.dataclassType())
            )
        except TypeError:
            pass
        self._blockModelUpdate = False

    def setCurrentParametersWidget(self, layertype: Any):
        """
        Update the parameters widget to display the parameters for *layertype*.
        """
        if isinstance(layertype, type) and issubclass(layertype, CoatingLayerBase):
            dcls = layertype.Parameters
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

    def setCurrentDrawOptionsWidget(self, layertype: Any):
        """
        Update the drawing options widget to display the drawing options for
        *layertype*.
        """
        if isinstance(layertype, type) and issubclass(layertype, CoatingLayerBase):
            dcls = layertype.DrawOptions
            index = self.drawOptionsWidget().indexOfDataclass(dcls)
            if index == -1:
                self.drawOptionsWidget().addDataclass(dcls, "Draw options")
                index = self.drawOptionsWidget().indexOfDataclass(dcls)
        else:
            index = 0
        self.drawOptionsWidget().setCurrentIndex(index)

    def currentDecoOptionsWidget(self) -> DataclassWidget:
        """Currently displayed decorating options widget."""
        widget = self.decoOptionsWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

    def setCurrentDecoOptionsWidget(self, layertype: Any):
        """
        Update the decorating options widget to display the decorating options
        for *layertype*.
        """
        if isinstance(layertype, type) and issubclass(layertype, CoatingLayerBase):
            dcls = layertype.DecoOptions
            index = self.decoOptionsWidget().indexOfDataclass(dcls)
            if index == -1:
                self.decoOptionsWidget().addDataclass(dcls, "Decorate options")
                index = self.decoOptionsWidget().indexOfDataclass(dcls)
        else:
            index = 0
        self.decoOptionsWidget().setCurrentIndex(index)

    @Slot(object)
    def onCoatingLayerTypeChange(self):
        """
        Apply current variable from import widget to other widgets and emit data.
        """
        var = self.typeWidget().variable()
        self.setCurrentParametersWidget(var)
        self.setCurrentDrawOptionsWidget(var)
        self.setCurrentDecoOptionsWidget(var)
        self.commitToCurrentItem()

    def structuredCoatingLayerArgs(self) -> StructuredCoatingLayerArgs:
        """
        Return :class:`StructuredCoatingLayerArgs` from current widget values.
        """
        layer_type = self.typeWidget().variable()
        try:
            param = self.currentParametersWidget().dataValue()
        except (TypeError, ValueError):
            param = None
        try:
            drawopt = self.currentDrawOptionsWidget().dataValue()
        except (TypeError, ValueError):
            drawopt = None
        try:
            decoopt = self.currentDecoOptionsWidget().dataValue()
        except (TypeError, ValueError):
            decoopt = None
        data = StructuredCoatingLayerArgs(layer_type, param, drawopt, decoopt)
        return data

    def coatingLayerArgs(self) -> CoatingLayerArgs:
        """Return :class:`CoatingLayerArgs` from current widget values."""
        importArgs = ImportArgs(
            self.typeWidget().variableNameLineEdit().text(),
            self.typeWidget().moduleNameLineEdit().text(),
        )
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
        try:
            decoopt = data_converter.unstructure(
                self.currentDecoOptionsWidget().dataValue()
            )
        except (TypeError, ValueError):
            decoopt = dict()
        args = CoatingLayerArgs(importArgs, param, drawopt, decoopt)
        return args

    @Slot()
    def commitToCurrentItem(self):
        """
        Set :meth:`coatingLayerArgs` and :meth:`structuredCoatingLayerArgs` to
        currently activated item from :meth:`experimentItemModel`.
        """
        if not self._blockModelUpdate:
            model = self.experimentItemModel()
            row = self.currentExperimentRow()
            index = model.index(row, model.Col_CoatingLayer)
            if index.isValid():
                model.setData(index, self.coatingLayerArgs(), model.Role_Args)
                model.setData(
                    index, self.structuredCoatingLayerArgs(), model.Role_StructuredArgs
                )

    def onExperimentsRemove(self, rows: List[int]):
        if self.currentExperimentRow() in rows:
            super().onExperimentsRemove(rows)
            self.typeWidget().clear()
            self.parametersWidget().setCurrentIndex(0)
            self.drawOptionsWidget().setCurrentIndex(0)
            self.decoOptionsWidget().setCurrentIndex(0)
