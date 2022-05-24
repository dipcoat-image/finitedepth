import cv2  # type: ignore[import]
from cv2PySide6 import NDArrayMediaCaptureSession
from dipcoatimage.finitedepth.analysis import ExperimentKind
from dipcoatimage.finitedepth_gui.core import ClassSelection, VisualizationMode
from dipcoatimage.finitedepth_gui.inventory import ExperimentItemModel
from dipcoatimage.finitedepth_gui.roimodel import ROIModel
from dipcoatimage.finitedepth_gui.workers import MasterWorker
from PySide6.QtCore import Signal, Slot, Qt, QUrl
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PySide6.QtMultimedia import QMediaPlayer, QCamera, QImageCapture, QMediaRecorder
from typing import Optional, List
from .toolbar import DisplayWidgetToolBar
from .roidisplay import NDArrayROILabel
from .videostream import (
    MediaController,
    PreviewableNDArrayVideoPlayer,
    VisualizeProcessor,
)


__all__ = [
    "MainDisplayWindow",
]


class MainDisplayWindow(QMainWindow):
    """Main window which includes various display widgets."""

    visualizationModeChanged = Signal(VisualizationMode)
    cameraTurnOn = Signal()
    cameraTurnOff = Signal()
    imageCaptured = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._exptitem_model = None
        self._currentExperimentRow = -1
        self._coat_paths = []
        self._expt_kind = ExperimentKind.NullExperiment
        self._selectedClass = ClassSelection.UNKNOWN

        self._display_toolbar = DisplayWidgetToolBar()
        self._display_label = NDArrayROILabel()
        self._video_controller = MediaController()

        self._video_player = PreviewableNDArrayVideoPlayer()
        self._camera = QCamera()
        self._capture_session = NDArrayMediaCaptureSession()
        self._image_capture = QImageCapture()
        self._media_recorder = QMediaRecorder()
        self._visualize_processor = VisualizeProcessor()

        self.displayToolBar().visualizationModeChanged.connect(
            self.visualizationModeChanged
        )
        self.displayToolBar().cameraChanged.connect(self.camera().setCameraDevice)
        self.displayToolBar().cameraToggled.connect(self.camera().setActive)
        self.displayToolBar().captureFormatChanged.connect(
            self.imageCapture().setFileFormat
        )
        self.displayToolBar().captureTriggered.connect(
            self.imageCapture().captureToFile
        )
        self.displayToolBar().recordFormatChanged.connect(
            self.mediaRecorder().mediaFormat().setFileFormat
        )
        self.videoController().setPlayer(self.videoPlayer())

        self.videoPlayer().arrayChanged.connect(self.visualizeProcessor().setArray)
        self.camera().activeChanged.connect(self.onCameraActiveChange)
        self.mediaCaptureSession().setCamera(self.camera())
        self.mediaCaptureSession().arrayChanged.connect(
            self.visualizeProcessor().setArray
        )
        self.mediaCaptureSession().setImageCapture(self.imageCapture())
        self.imageCapture().imageSaved.connect(self.onImageCapture)
        self.mediaCaptureSession().setRecorder(self.mediaRecorder())
        self.visualizeProcessor().arrayChanged.connect(self.displayLabel().setArray)

        self.addToolBar(self.displayToolBar())
        layout = QVBoxLayout()
        layout.addWidget(self.displayLabel())
        layout.addWidget(self.videoController())
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.displayLabel().setAlignment(Qt.AlignCenter)

    def experimentItemModel(self) -> Optional[ExperimentItemModel]:
        """Model which holds the experiment item data."""
        return self._exptitem_model

    def currentExperimentRow(self) -> int:
        """Currently activated row from :meth:`experimentItemModel`."""
        return self._currentExperimentRow

    def coatPaths(self) -> List[str]:
        return self._coat_paths

    def experimentKind(self) -> ExperimentKind:
        return self._expt_kind

    def selectedClass(self) -> ClassSelection:
        return self._selectedClass

    def displayToolBar(self) -> DisplayWidgetToolBar:
        """Toolbar to control display options."""
        return self._display_toolbar

    def displayLabel(self) -> NDArrayROILabel:
        """Label to display the visualization result."""
        return self._display_label

    def videoController(self) -> MediaController:
        return self._video_controller

    def videoPlayer(self) -> PreviewableNDArrayVideoPlayer:
        return self._video_player

    def camera(self) -> QCamera:
        return self._camera

    def mediaCaptureSession(self) -> NDArrayMediaCaptureSession:
        return self._capture_session

    def imageCapture(self) -> QImageCapture:
        return self._image_capture

    def mediaRecorder(self) -> QMediaRecorder:
        return self._media_recorder

    def visualizeProcessor(self) -> VisualizeProcessor:
        return self._visualize_processor

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
        # no need to wait for worker update, so visualization directly updated
        if row == self.currentExperimentRow():
            self._coat_paths = paths
            self._expt_kind = kind
            self.updateControllerVisibility()
            self.updateVisualization()

    @Slot(ClassSelection)
    def setSelectedClass(self, select: ClassSelection):
        # no need to wait for worker update, so visualization directly updated
        if select == ClassSelection.ANALYSIS:
            select = ClassSelection.EXPERIMENT
        if select != self.selectedClass():
            if self.videoPlayer().playbackState() == QMediaPlayer.PlayingState:
                self.videoPlayer().stop()
            self._selectedClass = select
            self.updateControllerVisibility()
            self.visualizeProcessor().setSelectedClass(select)
            self.updateVisualization()

    @Slot(int)
    def setCurrentExperimentRow(self, row: int):
        # visualization updated by worker signal
        self._currentExperimentRow = row
        model = self.experimentItemModel()
        if model is None:
            return
        self._coat_paths = model.coatPaths(row)
        self._expt_kind = model.experimentKind(row)
        self.updateControllerVisibility()

    @Slot(bool)
    def onCameraActiveChange(self, active: bool):
        self.updateControllerVisibility()
        if active:
            self.cameraTurnOn.emit()
        else:
            self.cameraTurnOff.emit()

    def updateControllerVisibility(self):
        if self.camera().isActive():
            visible = False
        elif self.selectedClass() in {
            ClassSelection.REFERENCE,
            ClassSelection.SUBSTRATE,
        }:
            visible = False
        elif self.experimentKind() == ExperimentKind.VideoExperiment:
            visible = True
        else:
            visible = False
        self.videoController().setVisible(visible)

    @Slot(ClassSelection)
    def onWorkersUpdate(self, changed: ClassSelection):
        if self.selectedClass() & changed:
            self.updateVisualization()

    def updateVisualization(self):
        if self.camera().isActive():
            pass
        elif self.selectedClass() in {
            ClassSelection.REFERENCE,
            ClassSelection.SUBSTRATE,
        }:  # directly update
            self.visualizeProcessor().emitVisualizationFromModel(self.selectedClass())
        elif self.experimentKind() == ExperimentKind.VideoExperiment:
            self.videoPlayer().setSource(QUrl.fromLocalFile(self.coatPaths()[0]))
        elif (
            self.experimentKind() == ExperimentKind.SingleImageExperiment
            or self.experimentKind() == ExperimentKind.MultiImageExperiment
        ):
            img = cv2.cvtColor(cv2.imread(self.coatPaths()[0]), cv2.COLOR_BGR2RGB)
            self.visualizeProcessor().setArray(img)
        else:  # flush image
            self.displayLabel().setPixmap(QPixmap())

    @Slot(ROIModel, bool)
    def toggleROIDraw(self, model: ROIModel, state: bool):
        if state:
            self.displayLabel().addROIModel(model)
        else:
            self.displayLabel().removeROIModel(model)

    @Slot(list)
    def onExperimentsRemove(self, rows: List[int]):
        if self.currentExperimentRow() in rows:
            self._coat_paths = []
            self._expt_kind = ExperimentKind.NullExperiment
            self.updateControllerVisibility()
            self.videoPlayer().setSource(QUrl())
            self.displayLabel().setPixmap(QPixmap())

    @Slot(int, str)
    def onImageCapture(self, id: int, path: str):
        if id != -1 and self.displayToolBar().captureAndAddAction().isChecked():
            self.imageCaptured.emit(path)

    def setVisualizeWorker(self, worker: Optional[MasterWorker]):
        self.visualizeProcessor().setVisualizeWorker(worker)

    def setVisualizeActionToggleState(self, mode: VisualizationMode):
        self.displayToolBar().setVisualizeActionToggleState(mode)
