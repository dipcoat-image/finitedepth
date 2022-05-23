from dipcoatimage.finitedepth_gui.core import ClassSelection
from dipcoatimage.finitedepth_gui.inventory import ExperimentItemModel
from dipcoatimage.finitedepth_gui.roimodel import ROIModel
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QTabWidget, QScrollArea
from typing import List
from .exptwidget import ExperimentWidget
from .refwidget import ReferenceWidget
from .substwidget import SubstrateWidget
from .layerwidget import CoatingLayerWidget
from .analysiswidget import AnalysisWidget


__all__ = [
    "MasterControlWidget",
]


class MasterControlWidget(QTabWidget):
    """Widget which contains control widgets."""

    referenceImageChanged = Signal(object)
    drawROIToggled = Signal(ROIModel, bool)
    selectedClassChanged = Signal(ClassSelection)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._expt_widget = ExperimentWidget()
        self._ref_widget = ReferenceWidget()
        self._subst_widget = SubstrateWidget()
        self._layer_widget = CoatingLayerWidget()
        self._anal_widget = AnalysisWidget()

        self.referenceWidget().imageChanged.connect(self.referenceImageChanged)
        self.referenceWidget().templateROIDrawButton().toggled.connect(
            self.onTemplateROIDrawButtonToggle
        )
        self.referenceWidget().substrateROIDrawButton().toggled.connect(
            self.onSubstrateROIDrawButtonToggle
        )
        self.currentChanged.connect(self.onCurrentTabChange)

        expt_scroll = QScrollArea()
        expt_scroll.setWidgetResizable(True)
        expt_scroll.setWidget(self.experimentWidget())
        self.addTab(expt_scroll, "Experiment")
        ref_scroll = QScrollArea()
        ref_scroll.setWidgetResizable(True)
        ref_scroll.setWidget(self.referenceWidget())
        self.addTab(ref_scroll, "Reference")
        subst_scroll = QScrollArea()
        subst_scroll.setWidgetResizable(True)
        subst_scroll.setWidget(self.substrateWidget())
        self.addTab(subst_scroll, "Substrate")
        layer_scroll = QScrollArea()
        layer_scroll.setWidgetResizable(True)
        layer_scroll.setWidget(self.coatingLayerWidget())
        self.addTab(layer_scroll, "Coating Layer")
        analyze_scroll = QScrollArea()
        analyze_scroll.setWidgetResizable(True)
        analyze_scroll.setWidget(self.analysisWidget())
        self.addTab(analyze_scroll, "Analyze")

    def experimentWidget(self) -> ExperimentWidget:
        return self._expt_widget

    def referenceWidget(self) -> ReferenceWidget:
        return self._ref_widget

    def substrateWidget(self) -> SubstrateWidget:
        return self._subst_widget

    def coatingLayerWidget(self) -> CoatingLayerWidget:
        return self._layer_widget

    def analysisWidget(self) -> AnalysisWidget:
        return self._anal_widget

    def setExperimentItemModel(self, model: ExperimentItemModel):
        """Set :meth:`experimentItemModel`."""
        self.experimentWidget().setExperimentItemModel(model)
        self.referenceWidget().setExperimentItemModel(model)
        self.substrateWidget().setExperimentItemModel(model)
        self.coatingLayerWidget().setExperimentItemModel(model)
        self.analysisWidget().setExperimentItemModel(model)

    @Slot(int)
    def setCurrentExperimentRow(self, row: int):
        """Set currently activated row from :meth:`experimentItemModel`."""
        self.experimentWidget().setCurrentExperimentRow(row)
        self.referenceWidget().setCurrentExperimentRow(row)
        self.substrateWidget().setCurrentExperimentRow(row)
        self.coatingLayerWidget().setCurrentExperimentRow(row)
        self.analysisWidget().setCurrentExperimentRow(row)

    @Slot(int, int)
    def setROIMaximum(self, width: int, height: int):
        self.referenceWidget().setROIMaximum(width, height)

    @Slot(bool)
    def onTemplateROIDrawButtonToggle(self, state: bool):
        self.drawROIToggled.emit(
            self.referenceWidget().templateROIWidget().roiModel(),
            state,
        )

    @Slot(bool)
    def onSubstrateROIDrawButtonToggle(self, state: bool):
        self.drawROIToggled.emit(
            self.referenceWidget().substrateROIWidget().roiModel(),
            state,
        )

    @Slot(ClassSelection)
    def setSelectedClass(self, select: ClassSelection):
        if select == ClassSelection.REFERENCE:
            index = 1
        elif select == ClassSelection.SUBSTRATE:
            index = 2
        else:
            index = 0
        self.setCurrentIndex(index)

    @Slot(int)
    def onCurrentTabChange(self, index: int):
        self.referenceWidget().templateROIDrawButton().setChecked(False)
        self.referenceWidget().substrateROIDrawButton().setChecked(False)

        widget = self.widget(index)
        if not isinstance(widget, QScrollArea):
            select = ClassSelection.UNKNOWN
        elif widget.widget() == self.referenceWidget():
            select = ClassSelection.REFERENCE
        elif widget.widget() == self.substrateWidget():
            select = ClassSelection.SUBSTRATE
        elif widget.widget() == self.coatingLayerWidget():
            select = ClassSelection.EXPERIMENT
        elif widget.widget() == self.experimentWidget():
            select = ClassSelection.EXPERIMENT
        elif widget.widget() == self.analysisWidget():
            select = ClassSelection.ANALYSIS
        else:
            select = ClassSelection.UNKNOWN
        self.selectedClassChanged.emit(select)

    @Slot()
    def resetReferenceImage(self):
        self.referenceWidget().resetReferenceImage()

    @Slot(list)
    def onExperimentsRemove(self, rows: List[int]):
        self.experimentWidget().onExperimentsRemove(rows)
        self.referenceWidget().onExperimentsRemove(rows)
        self.substrateWidget().onExperimentsRemove(rows)
        self.coatingLayerWidget().onExperimentsRemove(rows)
        self.analysisWidget().onExperimentsRemove(rows)
