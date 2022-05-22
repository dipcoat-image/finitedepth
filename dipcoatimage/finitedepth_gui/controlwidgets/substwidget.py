from dataclass2PySide6 import DataclassWidget, StackedDataclassWidget
from dipcoatimage.finitedepth import SubstrateBase, data_converter
from dipcoatimage.finitedepth.analysis import ImportArgs, SubstrateArgs
from dipcoatimage.finitedepth_gui.core import StructuredSubstrateArgs
from dipcoatimage.finitedepth_gui.importwidget import ImportWidget
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout
from typing import Any, List
from .base import ControlWidget


__all__ = [
    "SubstrateWidget",
]


class SubstrateWidget(ControlWidget):
    """
    View-like widget to control data for substrate object.

    .. rubric:: Substrate data

    Data consists of substrate type which is a concrete subclass of
    :class:`.SubstrateBase`, its *parameters* and *draw_options*.
    Note that substrate image is not specified by this widget, but by
    :class:`ReferenceWidget`.

    .. rubric:: Setting type

    Substrate type can be specified by :meth:`typeWidget`.

    When current class changes, current indices of :meth:`parametersWidget` and
    :meth:`drawOptionsWidget` are changed to show the new dataclass widget.

    Examples
    ========

    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.controlwidgets import SubstrateWidget
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     widget = SubstrateWidget()
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

        self.typeWidget().variableChanged.connect(self.onSubstrateTypeChange)
        self.parametersWidget().dataValueChanged.connect(self.commitToCurrentItem)
        self.drawOptionsWidget().dataValueChanged.connect(self.commitToCurrentItem)

        default_paramwdgt = DataclassWidget()  # default empty widget
        default_drawwdgt = DataclassWidget()  # default empty widget
        default_paramwdgt.setDataName("Parameters")
        default_drawwdgt.setDataName("Draw Options")
        self.parametersWidget().addWidget(default_paramwdgt)
        self.drawOptionsWidget().addWidget(default_drawwdgt)

        self.typeWidget().variableComboBox().setPlaceholderText(
            "Select substrate type or specify import information"
        )
        self.typeWidget().variableNameLineEdit().setPlaceholderText(
            "Substrate type name"
        )
        self.typeWidget().moduleNameLineEdit().setPlaceholderText(
            "Module name for the substrate type"
        )

        options_layout = QHBoxLayout()
        options_layout.addWidget(self.parametersWidget())
        options_layout.addWidget(self.drawOptionsWidget())

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.typeWidget())
        main_layout.addLayout(options_layout)
        self.setLayout(main_layout)

    def typeWidget(self) -> ImportWidget:
        """Widget to specify the substrate type."""
        return self._importwidget

    def parametersWidget(self) -> StackedDataclassWidget:
        """Widget to specify the substrate parameters."""
        return self._param_widget

    def drawOptionsWidget(self) -> StackedDataclassWidget:
        """Widget to specify the substrate drawing options."""
        return self._drawopt_widget

    def setCurrentExperimentRow(self, row: int):
        super().setCurrentExperimentRow(row)

        self._blockModelUpdate = True
        model = self.experimentItemModel()
        args = model.data(
            model.index(row, model.Col_Substrate),
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
        self._blockModelUpdate = False

    def currentParametersWidget(self) -> DataclassWidget:
        """Currently displayed parameters widget."""
        widget = self.parametersWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

    def setCurrentParametersWidget(self, substtype: Any):
        """
        Update the parameters widget to display the parameters for *substtype*.
        """
        if isinstance(substtype, type) and issubclass(substtype, SubstrateBase):
            dcls = substtype.Parameters
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

    def setCurrentDrawOptionsWidget(self, substtype: Any):
        """
        Update the drawing options widget to display the drawing option for
        *substtype*.
        """
        if isinstance(substtype, type) and issubclass(substtype, SubstrateBase):
            dcls = substtype.DrawOptions
            index = self.drawOptionsWidget().indexOfDataclass(dcls)
            if index == -1:
                self.drawOptionsWidget().addDataclass(dcls, "Draw options")
                index = self.drawOptionsWidget().indexOfDataclass(dcls)
        else:
            index = 0
        self.drawOptionsWidget().setCurrentIndex(index)

    @Slot(object)
    def onSubstrateTypeChange(self):
        """
        Apply current variable from import widget to other widgets and emit data.
        """
        var = self.typeWidget().variable()
        self.setCurrentParametersWidget(var)
        self.setCurrentDrawOptionsWidget(var)
        self.commitToCurrentItem()

    def structuredSubstrateArgs(self) -> StructuredSubstrateArgs:
        """Return :class:`StructuredSubstrateArgs` from current widget values."""
        subst_type = self.typeWidget().variable()
        try:
            param = self.currentParametersWidget().dataValue()
        except (TypeError, ValueError):
            param = None
        try:
            drawopt = self.currentDrawOptionsWidget().dataValue()
        except (TypeError, ValueError):
            drawopt = None
        data = StructuredSubstrateArgs(subst_type, param, drawopt)
        return data

    def substrateArgs(self) -> SubstrateArgs:
        """Return :class:`SubstrateArgs` from current widget values."""
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
        args = SubstrateArgs(importArgs, param, drawopt)
        return args

    @Slot()
    def commitToCurrentItem(self):
        """
        Set :meth:`substrateArgs` and :meth:`structuredSubstrateArgs` to
        currently activated item from :meth:`experimentItemModel`.
        """
        if not self._blockModelUpdate:
            model = self.experimentItemModel()
            row = self.currentExperimentRow()
            index = model.index(row, model.Col_Substrate)
            if index.isValid():
                model.setData(index, self.substrateArgs(), model.Role_Args)
                model.setData(
                    index, self.structuredSubstrateArgs(), model.Role_StructuredArgs
                )

    def onExperimentsRemove(self, rows: List[int]):
        if self.currentExperimentRow() in rows:
            self._currentExperimentRow = -1
            self.typeWidget().clear()
            self.parametersWidget().setCurrentIndex(0)
            self.drawOptionsWidget().setCurrentIndex(0)
