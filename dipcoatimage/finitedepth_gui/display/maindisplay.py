from dipcoatimage.finitedepth.analysis import ExperimentKind
from dipcoatimage.finitedepth_gui.core import ClassSelection, VisualizationMode
from dipcoatimage.finitedepth_gui.inventory import ExperimentItemModel
from dipcoatimage.finitedepth_gui.roimodel import ROIModel
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from typing import Optional, List
from .toolbar import DisplayWidgetToolBar
from .roidisplay import NDArrayROILabel
from .videostream import MediaController


__all__ = [
    "MainDisplayWindow",
]


class MainDisplayWindow(QMainWindow):
    """Main window which includes various display widgets."""

    visualizationModeChanged = Signal(VisualizationMode)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._exptitem_model = None
        self._currentExperimentRow = -1
        self._expt_kind = ExperimentKind.NullExperiment
        self._selectedClass = ClassSelection.EXPERIMENT
        self._camera_on = False

        self._display_toolbar = DisplayWidgetToolBar()
        self._display_label = NDArrayROILabel()
        self._video_controller = MediaController()

        self.displayToolBar().visualizationModeChanged.connect(
            self.visualizationModeChanged
        )
        self.displayToolBar().cameraToggled.connect(self.onCameraToggle)

        self.addToolBar(self.displayToolBar())
        layout = QVBoxLayout()
        layout.addWidget(self.displayLabel())
        layout.addWidget(self.videoController())
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def experimentItemModel(self) -> Optional[ExperimentItemModel]:
        """Model which holds the experiment item data."""
        return self._exptitem_model

    def currentExperimentRow(self) -> int:
        """Currently activated row from :meth:`experimentItemModel`."""
        return self._currentExperimentRow

    def experimentKind(self) -> ExperimentKind:
        return self._expt_kind

    def selectedClass(self) -> ClassSelection:
        return self._selectedClass

    def cameraOn(self) -> bool:
        return self._camera_on

    def displayToolBar(self) -> DisplayWidgetToolBar:
        """Toolbar to control display options."""
        return self._display_toolbar

    def displayLabel(self) -> NDArrayROILabel:
        """Label to display the visualization result."""
        return self._display_label

    def videoController(self) -> MediaController:
        return self._video_controller

    def setExperimentItemModel(self, model: Optional[ExperimentItemModel]):
        """Set :meth:`experimentItemModel`."""
        old_model = self.experimentItemModel()
        if old_model is not None:
            self.disconnectModel(old_model)
        self._exptitem_model = model
        if model is not None:
            self.connectModel(model)

    def connectModel(self, model: ExperimentItemModel):
        model.coatPathsChanged.connect(self.onCoatPathsChange)

    def disconnectModel(self, model: ExperimentItemModel):
        model.coatPathsChanged.disconnect(self.onCoatPathsChange)

    @Slot(int, list, ExperimentKind)
    def onCoatPathsChange(self, row: int, paths: List[str], kind: ExperimentKind):
        self._expt_kind = kind
        self.updateControllerVisibility()

    @Slot(ClassSelection)
    def setSelectedClass(self, select: ClassSelection):
        self._selectedClass = select
        self.updateControllerVisibility()

    @Slot(int)
    def setCurrentExperimentRow(self, row: int):
        self._currentExperimentRow = row
        model = self.experimentItemModel()
        if model is None:
            return
        self._expt_kind = model.experimentKind(row)
        self.updateControllerVisibility()

    @Slot(bool)
    def onCameraToggle(self, toggled: bool):
        self._camera_on = toggled
        self.updateControllerVisibility()

    def updateControllerVisibility(self):
        if self.cameraOn() or self.selectedClass() != ClassSelection.EXPERIMENT:
            visible = False
        elif self.experimentKind() == ExperimentKind.VideoExperiment:
            visible = True
        else:
            visible = False
        self.videoController().setVisible(visible)

    @Slot(ROIModel, bool)
    def toggleROIDraw(self, model: ROIModel, state: bool):
        if state:
            self.displayLabel().addROIModel(model)
        else:
            self.displayLabel().removeROIModel(model)

    def setVisualizeActionToggleState(self, mode: VisualizationMode):
        self.displayToolBar().setVisualizeActionToggleState(mode)
