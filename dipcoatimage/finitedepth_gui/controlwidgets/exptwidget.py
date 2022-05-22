from dataclass2PySide6 import DataclassWidget, StackedDataclassWidget
from dipcoatimage.finitedepth import ExperimentBase, data_converter
from dipcoatimage.finitedepth.analysis import ImportArgs, ExperimentArgs
from dipcoatimage.finitedepth_gui.core import StructuredExperimentArgs
from dipcoatimage.finitedepth_gui.importwidget import ImportWidget
from PySide6.QtCore import Slot
from PySide6.QtGui import QStandardItem
from PySide6.QtWidgets import (
    QDataWidgetMapper,
    QLineEdit,
    QListView,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QGroupBox,
)
from typing import Any, List
from .base import ControlWidget


__all__ = [
    "ExperimentWidget",
]


class ExperimentWidget(ControlWidget):
    """
    View-like widget for data from :meth:`experimentItemModel` which can
    construct experiment object.

    .. rubric:: Experiment data

    Data consists of experiment type which is a concrete subclass of
    :class:`.ExperimentBase` and its parameters.

    .. rubric:: Setting type

    Experiment type can be specified by :meth:`typeWidget`.

    When current class changes, current index of :meth:`parametersWidget` is
    changed to show the new dataclass widget.

    Examples
    ========

    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.controlwidgets import ExperimentWidget
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     widget = ExperimentWidget()
    ...     widget.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._blockModelUpdate = False

        self._exptname_mapper = QDataWidgetMapper()
        self._exptname_lineedit = QLineEdit()
        self._paths_listview = QListView()
        self._add_button = QPushButton("Add")
        self._delete_button = QPushButton("Delete")
        self._browse_button = QPushButton("Browse")
        self._importinfo_widget = ImportWidget()
        self._param_widget = StackedDataclassWidget()

        self.typeWidget().variableChanged.connect(self.onExperimentTypeChange)
        self.pathsView().setSelectionMode(QListView.ExtendedSelection)
        self.pathsView().setEditTriggers(QListView.SelectedClicked)
        self.pathAddButton().clicked.connect(self.onAddButtonClicked)
        self.pathDeleteButton().clicked.connect(self.onDeleteButtonClicked)
        self.pathBrowseButton().clicked.connect(self.onBrowseButtonClicked)
        self.parametersWidget().dataValueChanged.connect(self.commitToCurrentItem)

        default_paramwdgt = DataclassWidget()  # default empty widget
        default_paramwdgt.setDataName("Parameters")
        self.parametersWidget().addWidget(default_paramwdgt)

        self.typeWidget().variableComboBox().setPlaceholderText(
            "Select experiment type or specify import information"
        )
        self.typeWidget().variableNameLineEdit().setPlaceholderText(
            "Experiment type name"
        )
        self.typeWidget().moduleNameLineEdit().setPlaceholderText(
            "Module name for the experiment type"
        )

        path_groupbox = QGroupBox("Experiment files path")
        path_layout = QVBoxLayout()
        path_layout.addWidget(self.pathsView())
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.pathAddButton())
        buttons_layout.addWidget(self.pathDeleteButton())
        path_layout.addLayout(buttons_layout)
        path_layout.addWidget(self.pathBrowseButton())
        path_groupbox.setLayout(path_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.experimentNameLineEdit())
        main_layout.addWidget(self.typeWidget())
        main_layout.addWidget(path_groupbox)
        main_layout.addWidget(self.parametersWidget())
        self.setLayout(main_layout)

    def experimentNameMapper(self) -> QDataWidgetMapper:
        """
        Mapper to update :meth:`experimentNameLineEdit` with experiment name of
        currently activated item from :meth:`experimentItemModel`.
        """
        return self._exptname_mapper

    def experimentNameLineEdit(self) -> QLineEdit:
        """Line edit for the experiment name."""
        return self._exptname_lineedit

    def pathsView(self) -> QListView:
        """List view to display the paths to coated substrate files."""
        return self._paths_listview

    def pathAddButton(self) -> QPushButton:
        """Push button to add a new item to :meth:`pathsView`."""
        return self._add_button

    def pathDeleteButton(self) -> QPushButton:
        """Push button to delete items from :meth:`pathsView`."""
        return self._delete_button

    def pathBrowseButton(self) -> QPushButton:
        """Button to browse and add a new item to :meth:`pathsView`."""
        return self._browse_button

    def typeWidget(self) -> ImportWidget:
        """Widget to specify the experiment type."""
        return self._importinfo_widget

    def parametersWidget(self) -> StackedDataclassWidget:
        """Widget to specify the experiment parameters."""
        return self._param_widget

    def setExperimentItemModel(self, model):
        """
        Set :meth:`experimentItemModel` and remap :meth:`experimentNameMapper`.
        """
        super().setExperimentItemModel(model)
        self.experimentNameMapper().setModel(model)
        self.experimentNameMapper().addMapping(
            self.experimentNameLineEdit(),
            model.Col_ExperimentName,
        )

    def setCurrentExperimentRow(self, row: int):
        super().setCurrentExperimentRow(row)
        # update experiment name and paths
        self.experimentNameMapper().setCurrentIndex(row)
        model = self.experimentItemModel()
        self.pathsView().setModel(model)
        self.pathsView().setRootIndex(model.index(row, model.Col_CoatPaths))

        # update experiment args
        self._blockModelUpdate = True
        args = model.data(
            model.index(row, model.Col_Experiment),
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
        self._blockModelUpdate = False

    def currentParametersWidget(self) -> DataclassWidget:
        """Currently displayed parameters widget."""
        widget = self.parametersWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

    def setCurrentParametersWidget(self, expttype: Any):
        """
        Update the parameters widget to display the parameters for *expttype*.
        """
        if isinstance(expttype, type) and issubclass(expttype, ExperimentBase):
            dcls = expttype.Parameters
            index = self.parametersWidget().indexOfDataclass(dcls)
            if index == -1:
                self.parametersWidget().addDataclass(dcls, "Parameters")
                index = self.parametersWidget().indexOfDataclass(dcls)
        else:
            index = 0
        self.parametersWidget().setCurrentIndex(index)

    @Slot()
    def onExperimentTypeChange(self):
        """
        Apply current variable from import widget to other widgets and emit data.
        """
        var = self.typeWidget().variable()
        self.setCurrentParametersWidget(var)
        self.commitToCurrentItem()

    def structuredExperimentArgs(self) -> StructuredExperimentArgs:
        """
        Return :class:`StructuredExperimentArgs` from current widget values.
        """
        expt_type = self.typeWidget().variable()
        try:
            param = self.currentParametersWidget().dataValue()
        except (TypeError, ValueError):
            param = None
        data = StructuredExperimentArgs(expt_type, param)
        return data

    def experimentArgs(self) -> ExperimentArgs:
        """Return :class:`ExperimentArgs` from current widget values."""
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
        args = ExperimentArgs(importArgs, param)
        return args

    @Slot()
    def commitToCurrentItem(self):
        """
        Set :meth:`experimentArgs` and :meth:`structuredExperimentArgs` to
        currently activated item from :meth:`experimentItemModel`.
        """
        if not self._blockModelUpdate:
            model = self.experimentItemModel()
            row = self.currentExperimentRow()
            index = model.index(row, model.Col_Experiment)
            if index.isValid():
                model.setData(index, self.experimentArgs(), model.Role_Args)
                model.setData(
                    index, self.structuredExperimentArgs(), model.Role_StructuredArgs
                )

    @Slot()
    def onAddButtonClicked(self):
        """Add new item to :meth:`pathsView`."""
        model = self.experimentItemModel()
        parentItem = model.itemFromIndex(self.pathsView().rootIndex())
        item = QStandardItem(f"Path {parentItem.rowCount()}")
        parentItem.appendRow(item)

    @Slot()
    def onDeleteButtonClicked(self):
        """Delete selected items from :meth:`pathsView`."""
        model = self.experimentItemModel()
        exptRow = self.pathsView().rootIndex().row()
        paths = model.coatPaths(exptRow)
        selectedRows = [idx.row() for idx in self.pathsView().selectedIndexes()]
        for i in reversed(sorted(selectedRows)):
            paths.pop(i)
            model.setCoatPaths(exptRow, paths)

    @Slot()
    def onBrowseButtonClicked(self):
        """Browse file and add their paths to :meth:`pathsView`."""
        model = self.experimentItemModel()
        parentItem = model.itemFromIndex(self.pathsView().rootIndex())
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select experiment files",
            "./",
            options=QFileDialog.DontUseNativeDialog,
        )
        for p in paths:
            parentItem.appendRow(QStandardItem(p))

    def onExperimentsRemove(self, rows: List[int]):
        if self.currentExperimentRow() in rows:
            super().onExperimentsRemove(rows)
            self.experimentNameLineEdit().clear()
            self.typeWidget().clear()
            self.pathsView().setModel(None)
            self.parametersWidget().setCurrentIndex(0)
