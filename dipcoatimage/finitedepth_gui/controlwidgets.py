"""
Control Widgets
===============

This module provides widgets to control the analysis parameters.

"""


import cv2  # type: ignore
from dataclass2PySide6 import DataclassWidget, StackedDataclassWidget
from dipcoatimage.finitedepth import (
    SubstrateReferenceBase,
    SubstrateBase,
    CoatingLayerBase,
    ExperimentBase,
    data_converter,
)
from dipcoatimage.finitedepth.analysis import (
    ImportArgs,
    ReferenceArgs,
    SubstrateArgs,
    CoatingLayerArgs,
    ExperimentArgs,
)
from PySide6.QtCore import QModelIndex, Qt, Signal, Slot, QSignalBlocker
from PySide6.QtGui import QStandardItem
from PySide6.QtWidgets import (
    QWidget,
    QDataWidgetMapper,
    QLineEdit,
    QListView,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QGroupBox,
    QSizePolicy,
)
from typing import Any
from .importwidget import ImportWidget
from .inventory import (
    ExperimentItemModel,
    StructuredExperimentArgs,
    StructuredReferenceArgs,
    StructuredSubstrateArgs,
    StructuredCoatingLayerArgs,
)
from .roimodel import ROIWidget


__all__ = [
    "ControlWidget",
    "ExperimentWidget",
    "ReferenceWidget",
    "SubstrateWidget",
    "CoatingLayerWidget",
]


class ControlWidget(QWidget):
    """Base class for all control widgets."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._exptitem_model = ExperimentItemModel()
        self._currentIndex = QModelIndex()

    def experimentItemModel(self) -> ExperimentItemModel:
        """Model which holds the experiment item data."""
        return self._exptitem_model

    def currentExperimentIndex(self) -> QModelIndex:
        """Currently activated index from the model."""
        return self._currentIndex

    def setExperimentItemModel(self, model: ExperimentItemModel):
        """Set :meth:`experimentItemModel`."""
        self._exptitem_model = model

    @Slot(QModelIndex)
    def setCurrentExperimentIndex(self, index: QModelIndex):
        """Set currently activated index from :meth:`experimentItemModel`."""
        if index.parent().isValid():
            raise TypeError("Only top-level index can be activated.")
        self._currentIndex = index


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

        self.initUI()

    def initUI(self):
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
        """Line edit to display the experiment name."""
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
        """Push button to browse and add a new item to :meth:`pathsView`."""
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
            ExperimentItemModel.Col_ExperimentName,
        )

    @Slot(QModelIndex)
    def setCurrentExperimentIndex(self, index: QModelIndex):
        """Set currently activated index from :meth:`experimentItemModel`."""
        super().setCurrentExperimentIndex(index)
        self.experimentNameMapper().setCurrentIndex(index.row())

        model = self.experimentItemModel()
        self.pathsView().setModel(model)
        self.pathsView().setRootIndex(
            model.index(index.row(), ExperimentItemModel.Col_CoatPaths)
        )

        self.setExperimentArgs(
            model.data(
                model.index(index.row(), ExperimentItemModel.Col_Experiment),
                Qt.UserRole,
            )[1]
        )

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

    @Slot(object)
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

    @Slot(ExperimentArgs)
    def setExperimentArgs(self, args: ExperimentArgs):
        """
        Update the widgets with *args*.

        This does not emit :attr:`dataChanged` signal.
        Run :meth:`commitToCurrentItem` manually after running this method.
        """
        self._blockModelUpdate = True

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

    @Slot()
    def commitToCurrentItem(self):
        """
        Set :meth:`structuredExperimentArgs` and :meth:`experimentArgs` to
        currently activated item from :meth:`experimentItemModel`.
        """
        index = self.currentExperimentIndex()
        if not self._blockModelUpdate and index.isValid():
            model = self.experimentItemModel()
            model.setData(
                model.index(index.row(), ExperimentItemModel.Col_Experiment),
                (self.structuredExperimentArgs(), self.experimentArgs()),
                Qt.UserRole,  # type: ignore[arg-type]
            )

    @Slot()
    def onAddButtonClicked(self):
        """Add new item to :meth:`pathsView`."""
        model = self.experimentItemModel()
        if model is not None:
            parentItem = model.itemFromIndex(self.pathsView().rootIndex())
            item = QStandardItem(f"Path {parentItem.rowCount()}")
            parentItem.appendRow(item)

    @Slot()
    def onDeleteButtonClicked(self):
        """Delete selected items from :meth:`pathsView`."""
        model = self.experimentItemModel()
        if model is not None:
            parentItem = model.itemFromIndex(self.pathsView().rootIndex())
            for items in reversed(sorted(self.pathsView().selectedIndexes())):
                parentItem.removeRow(items.row())

    @Slot()
    def onBrowseButtonClicked(self):
        """Browse file and add their paths to :meth:`pathsView`."""
        model = self.experimentItemModel()
        if model is not None:
            parentItem = model.itemFromIndex(self.pathsView().rootIndex())
            paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Select experiment files",
                "./",
                options=QFileDialog.DontUseNativeDialog,
            )
            for p in paths:
                parentItem.appendRow(QStandardItem(p))


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

        self.initUI()

    def initUI(self):
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
        Mapper to update :meth:`pathLineEdit` with reference path of
        currently activated item from :meth:`experimentItemModel`.
        """
        return self._refpath_mapper

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

    def setExperimentItemModel(self, model):
        """
        Set :meth:`experimentItemModel` and remap :meth:`pathMapper`.
        """
        super().setExperimentItemModel(model)
        self.pathMapper().setModel(model)
        self.pathMapper().addMapping(
            self.pathLineEdit(),
            ExperimentItemModel.Col_ReferencePath,
        )

    @Slot(QModelIndex)
    def setCurrentExperimentIndex(self, index: QModelIndex):
        super().setCurrentExperimentIndex(index)
        self.pathMapper().setCurrentIndex(index.row())

        model = self.experimentItemModel()
        self.setReferenceArgs(
            model.data(
                model.index(index.row(), ExperimentItemModel.Col_Reference),
                Qt.UserRole,
            )[1]
        )

    def currentParametersWidget(self) -> DataclassWidget:
        widget = self.parametersWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

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

    def currentDrawOptionsWidget(self) -> DataclassWidget:
        widget = self.drawOptionsWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

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

    @Slot(object)
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

    @Slot(ReferenceArgs)
    def setReferenceArgs(self, args: ReferenceArgs):
        self._blockModelUpdate = True

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

    @Slot()
    def commitToCurrentItem(self):
        """
        Set :meth:`structuredReferenceArgs` and :meth:`referenceArgs` to
        currently activated item from :meth:`experimentItemModel`.
        """
        index = self.currentExperimentIndex()
        if not self._blockModelUpdate and index.isValid():
            model = self.experimentItemModel()
            model.setData(
                model.index(index.row(), ExperimentItemModel.Col_Reference),
                (self.structuredReferenceArgs(), self.referenceArgs()),
                Qt.UserRole,  # type: ignore[arg-type]
            )

    @Slot(str)
    def setReferencePath(self, path: str):
        self.pathLineEdit().setText(path)
        self.onPathEditFinished()

    @Slot()
    def onPathEditFinished(self):
        self.updateROIMaximum()
        self.emitImage()
        self.commitToCurrentItem()

    def updateROIMaximum(self):
        path = self.pathLineEdit().text()
        img = cv2.imread(path)
        if img is None:
            w, h = (0, 0)
        else:
            w, h = (img.shape[1], img.shape[0])
        self.templateROIWidget().setROIMaximum(w, h)
        self.substrateROIWidget().setROIMaximum(w, h)

    @Slot()
    def emitImage(self):
        img = cv2.imread(self.pathLineEdit().text())
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.imageChanged.emit(img)

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


class SubstrateWidget(QWidget):
    """
    Widget to control data for substrate object.

    .. rubric:: Substrate data

    Data consists of substrate type which is a concrete subclass of
    :class:`.SubstrateBase`, its *parameters* and *draw_options*.
    Note that substrate image is not specified by this widget, but by
    :class:`ReferenceWidget`.

    Data are wrapped by :class:`StructuredSubstrateArgs`. Whenever the widget
    values change :attr:`dataChanged` signal emits the data.

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

    dataChanged = Signal(StructuredSubstrateArgs, SubstrateArgs)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._importwidget = ImportWidget()
        self._param_widget = StackedDataclassWidget()
        self._drawopt_widget = StackedDataclassWidget()

        self.typeWidget().variableChanged.connect(self.onSubstrateTypeChange)
        self.parametersWidget().dataValueChanged.connect(self.emitData)
        self.drawOptionsWidget().dataValueChanged.connect(self.emitData)

        default_paramwdgt = DataclassWidget()  # default empty widget
        default_drawwdgt = DataclassWidget()  # default empty widget
        default_paramwdgt.setDataName("Parameters")
        default_drawwdgt.setDataName("Draw Options")
        self.parametersWidget().addWidget(default_paramwdgt)
        self.drawOptionsWidget().addWidget(default_drawwdgt)

        self.initUI()

    def initUI(self):
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
        return self._importwidget

    def parametersWidget(self) -> StackedDataclassWidget:
        return self._param_widget

    def drawOptionsWidget(self) -> StackedDataclassWidget:
        return self._drawopt_widget

    def currentParametersWidget(self) -> DataclassWidget:
        widget = self.parametersWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

    def setCurrentParametersWidget(self, substtype: Any):
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
        widget = self.drawOptionsWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

    def setCurrentDrawOptionsWidget(self, substtype: Any):
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
        self.emitData()

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

    @Slot(SubstrateArgs)
    def setSubstrateArgs(self, args: SubstrateArgs):
        with QSignalBlocker(self):
            self.typeWidget().variableNameLineEdit().setText(args.type.name)
            self.typeWidget().moduleNameLineEdit().setText(args.type.module)
            self.typeWidget().onInformationEdit()

            paramWidget = self.currentParametersWidget()
            try:
                paramWidget.setDataValue(
                    data_converter.structure(
                        args.parameters, paramWidget.dataclassType()
                    )
                )
            except TypeError:
                pass

            drawWidget = self.currentDrawOptionsWidget()
            try:
                drawWidget.setDataValue(
                    data_converter.structure(
                        args.draw_options, drawWidget.dataclassType()
                    )
                )
            except TypeError:
                pass

    @Slot()
    def emitData(self):
        self.dataChanged.emit(self.structuredSubstrateArgs(), self.substrateArgs())


class CoatingLayerWidget(QWidget):
    """
    Widget to control data for coating layer object.

    .. rubric:: Coating layer data

    Data consists of coating layer type which is a concrete subclass of
    :class:`.CoatingLayerBase`, its *parameters*, *draw_options* and
    *deco_options*. Note that coated substrate image is not specified by this
    widget, but by :class:`ExperimentWidget`.

    Data are wrapped by :class:`StructuredCoatingLayerArgs`. Whenever the widget
    values change :attr:`dataChanged` signal emits the data.

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

    dataChanged = Signal(StructuredCoatingLayerArgs, SubstrateArgs)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._importwidget = ImportWidget()
        self._param_widget = StackedDataclassWidget()
        self._drawopt_widget = StackedDataclassWidget()
        self._decoopt_widget = StackedDataclassWidget()

        self.typeWidget().variableChanged.connect(self.onCoatingLayerTypeChange)
        self.parametersWidget().dataValueChanged.connect(self.emitData)
        self.drawOptionsWidget().dataValueChanged.connect(self.emitData)
        self.decoOptionsWidget().dataValueChanged.connect(self.emitData)

        default_paramwdgt = DataclassWidget()  # default empty widget
        default_drawwdgt = DataclassWidget()  # default empty widget
        default_decowdgt = DataclassWidget()  # default empty widget
        default_paramwdgt.setDataName("Parameters")
        default_drawwdgt.setDataName("Draw Options")
        default_decowdgt.setDataName("Decorate Options")
        self.parametersWidget().addWidget(default_paramwdgt)
        self.drawOptionsWidget().addWidget(default_drawwdgt)
        self.decoOptionsWidget().addWidget(default_decowdgt)

        self.initUI()

    def initUI(self):
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
        return self._importwidget

    def parametersWidget(self) -> StackedDataclassWidget:
        return self._param_widget

    def drawOptionsWidget(self) -> StackedDataclassWidget:
        return self._drawopt_widget

    def decoOptionsWidget(self) -> StackedDataclassWidget:
        return self._decoopt_widget

    def currentParametersWidget(self) -> DataclassWidget:
        widget = self.parametersWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

    def setCurrentParametersWidget(self, layertype: Any):
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
        widget = self.drawOptionsWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

    def setCurrentDrawOptionsWidget(self, layertype: Any):
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
        widget = self.decoOptionsWidget().currentWidget()
        if not isinstance(widget, DataclassWidget):
            raise TypeError(f"{widget} is not dataclass widget.")
        return widget

    def setCurrentDecoOptionsWidget(self, layertype: Any):
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
        self.emitData()

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

    @Slot(CoatingLayerArgs)
    def setCoatingLayerArgs(self, args: CoatingLayerArgs):
        with QSignalBlocker(self):
            self.typeWidget().variableNameLineEdit().setText(args.type.name)
            self.typeWidget().moduleNameLineEdit().setText(args.type.module)
            self.typeWidget().onInformationEdit()

            paramWidget = self.currentParametersWidget()
            try:
                paramWidget.setDataValue(
                    data_converter.structure(
                        args.parameters, paramWidget.dataclassType()
                    )
                )
            except TypeError:
                pass

            drawWidget = self.currentDrawOptionsWidget()
            try:
                drawWidget.setDataValue(
                    data_converter.structure(
                        args.draw_options, drawWidget.dataclassType()
                    )
                )
            except TypeError:
                pass

            decoWidget = self.currentDecoOptionsWidget()
            try:
                decoWidget.setDataValue(
                    data_converter.structure(
                        args.deco_options, decoWidget.dataclassType()
                    )
                )
            except TypeError:
                pass

    @Slot()
    def emitData(self):
        self.dataChanged.emit(
            self.structuredCoatingLayerArgs(), self.coatingLayerArgs()
        )
