from araviq6 import MediaController, NDArrayMediaCaptureSession
import cv2  # type: ignore[import]
import numpy as np
import numpy.typing as npt
from dipcoatimage.finitedepth.analysis import experiment_kind, ExperimentKind
from dipcoatimage.finitedepth_gui.core import (
    ClassSelection,
    VisualizationMode,
    DataMember,
    DataArgs,
)
from dipcoatimage.finitedepth_gui.inventory import ExperimentItemModel
from dipcoatimage.finitedepth_gui.roimodel import ROIModel
from dipcoatimage.finitedepth_gui.workers import MasterWorker
from dipcoatimage.finitedepth_gui.model import IndexRole, ExperimentDataModel
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt, QUrl, QModelIndex
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PySide6.QtMultimedia import QCamera, QImageCapture, QMediaRecorder, QMediaPlayer
from typing import Optional, List
from .toolbar import DisplayWidgetToolBar
from .roidisplay import NDArrayROILabel, NDArrayROILabel_V2
from .videostream import (
    PreviewableNDArrayVideoPlayer,
    VisualizeProcessor,
)


__all__ = [
    "MainDisplayWindow",
    "MainDisplayWindow_V2",
]


class SignalSender(QObject):
    """Object to send the signals to processor thread."""

    arrayChanged = Signal(np.ndarray)


class MainDisplayWindow(QMainWindow):
    """Main window which includes various display widgets."""

    visualizationModeChanged = Signal(VisualizationMode)
    cameraTurnOn = Signal()
    cameraTurnOff = Signal()
    imageCaptured = Signal(str)
    videoRecorded = Signal(QUrl)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._displayToolBar = DisplayWidgetToolBar()
        self._camera = QCamera()
        self._imageCapture = QImageCapture()
        self._mediaRecorder = QMediaRecorder()

        self._exptitem_model = None
        self._currentExperimentRow = -1
        self._coat_paths = []
        self._expt_kind = ExperimentKind.NullExperiment
        self._selectedClass = ClassSelection.UNKNOWN

        self._displayLabel = NDArrayROILabel()
        self._videoController = MediaController()

        self._video_player = PreviewableNDArrayVideoPlayer()
        self._capture_session = NDArrayMediaCaptureSession()
        self._signalSender = SignalSender()
        self._processorThread = QThread()
        self._visualize_processor = VisualizeProcessor()

        self.visualizeProcessor().moveToThread(self.processorThread())
        self.processorThread().start()

        self._displayToolBar.setCamera(self._camera)
        self._displayToolBar.setImageCapture(self._imageCapture)
        self._displayToolBar.setMediaRecorder(self._mediaRecorder)
        self._displayToolBar.visualizationModeChanged.connect(
            self.visualizationModeChanged
        )
        self._displayToolBar.imageCaptured.connect(self.imageCaptured)
        self._displayToolBar.videoRecorded.connect(self.videoRecorded)

        self.videoController().setPlayer(self.videoPlayer())

        self.videoPlayer().arrayChanged.connect(self.onArrayPassedFromSource)
        self._camera.activeChanged.connect(self.setCameraActive)
        self.mediaCaptureSession().setCamera(self._camera)
        self.mediaCaptureSession().arrayChanged.connect(self.onArrayPassedFromSource)
        self.mediaCaptureSession().setImageCapture(self._imageCapture)
        self.mediaCaptureSession().setRecorder(self._mediaRecorder)
        self._signalSender.arrayChanged.connect(self.visualizeProcessor().setArray)
        self.visualizeProcessor().arrayChanged.connect(self.displayLabel().setArray)

        self.addToolBar(self._displayToolBar)
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

    def displayLabel(self) -> NDArrayROILabel:
        """Label to display the visualization result."""
        return self._displayLabel

    def videoController(self) -> MediaController:
        return self._videoController

    def videoPlayer(self) -> PreviewableNDArrayVideoPlayer:
        return self._video_player

    def mediaCaptureSession(self) -> NDArrayMediaCaptureSession:
        return self._capture_session

    def processorThread(self) -> QThread:
        return self._processorThread

    def visualizeProcessor(self) -> VisualizeProcessor:
        return self._visualize_processor

    @Slot(np.ndarray)
    def onArrayPassedFromSource(self, array: np.ndarray):
        if self.visualizeProcessor().ready():
            self._signalSender.arrayChanged.emit(array)

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

    def setCoatPaths(self, paths: List[str], kind: ExperimentKind):
        self._coat_paths = paths
        self._expt_kind = kind
        self.updateControllerVisibility()
        if self._camera.isActive():
            pass
        elif self.selectedClass() in {
            ClassSelection.REFERENCE,
            ClassSelection.SUBSTRATE,
        }:
            pass
        elif self.experimentKind() == ExperimentKind.VideoExperiment:
            self.videoPlayer().setSource(QUrl.fromLocalFile(self.coatPaths()[0]))
        elif (
            self.experimentKind() == ExperimentKind.SingleImageExperiment
            or self.experimentKind() == ExperimentKind.MultiImageExperiment
        ):
            img = cv2.cvtColor(cv2.imread(self.coatPaths()[0]), cv2.COLOR_BGR2RGB)
            self.visualizeProcessor().setArray(img)
        else:
            img = np.empty((0, 0, 0), dtype=np.uint8)
            self.visualizeProcessor().setArray(img)

    @Slot(int, list, ExperimentKind)
    def onCoatPathsChange(self, row: int, paths: List[str], kind: ExperimentKind):
        if row != self.currentExperimentRow():
            return
        self.setCoatPaths(paths, kind)

    @Slot(ClassSelection)
    def setSelectedClass(self, select: ClassSelection):
        # no need to wait for worker update, so visualization directly updated
        if select == ClassSelection.ANALYSIS:
            select = ClassSelection.EXPERIMENT
        if select != self.selectedClass():
            if (
                self.videoPlayer().playbackState()
                == self.videoPlayer().PlaybackState.PlayingState
            ):
                self.videoPlayer().pause()
            self._selectedClass = select
            self.updateControllerVisibility()
            self.visualizeProcessor().setSelectedClass(select)
            self.updateVisualization()

    @Slot(int)
    def setCurrentExperimentRow(self, row: int):
        # visualization is not updated here but by onWorkersUpdate()
        self._currentExperimentRow = row
        model = self.experimentItemModel()
        if model is None:
            return
        paths = model.coatPaths(row)
        kind = model.experimentKind(row)
        self.setCoatPaths(paths, kind)

    @Slot(bool)
    def setCameraActive(self, active: bool):
        self.updateControllerVisibility()
        if active:
            self.cameraTurnOn.emit()
        else:
            self.cameraTurnOff.emit()

    def updateControllerVisibility(self):
        if self._camera.isActive():
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
        if self._camera.isActive():
            return
        self.visualizeProcessor().emitVisualizationFromModel(self.selectedClass())

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
        if id != -1 and self._displayToolBar._captureAndAddAction.isChecked():
            self.imageCaptured.emit(path)

    def setVisualizeWorker(self, worker: Optional[MasterWorker]):
        self.visualizeProcessor().setVisualizeWorker(worker)

    def setVisualizationMode(self, mode: VisualizationMode):
        self._displayToolBar.setVisualizationMode(mode)

    def closeEvent(self, event):
        self.processorThread().quit()
        self.processorThread().wait()
        super().closeEvent(event)


class MainDisplayWindow_V2(QMainWindow):
    """
    Window to display the visualization result.

    This widget is only a view. Visualization must be done by some other object.
    """

    visualizationModeChanged = Signal(VisualizationMode)
    imageCaptured = Signal(str)
    videoRecorded = Signal(QUrl)
    cameraActiveChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None
        self._currentModelIndex = QModelIndex()
        self._exptKind = ExperimentKind.NullExperiment
        self._currentView = DataMember.NULL
        self._camera = None

        self._displayLabel = NDArrayROILabel_V2()
        self._videoController = MediaController()
        self._displayToolBar = DisplayWidgetToolBar()

        self._imageCapture = QImageCapture()
        self._mediaRecorder = QMediaRecorder()
        self._displayLabel.setAlignment(Qt.AlignCenter)
        self._videoController.setVisible(False)

        self._displayToolBar.setImageCapture(self._imageCapture)
        self._displayToolBar.setMediaRecorder(self._mediaRecorder)
        self._displayToolBar.visualizationModeChanged.connect(
            self.visualizationModeChanged
        )
        self._displayToolBar.imageCaptured.connect(self.imageCaptured)
        self._displayToolBar.videoRecorded.connect(self.videoRecorded)

        self.addToolBar(self._displayToolBar)
        layout = QVBoxLayout()
        layout.addWidget(self._displayLabel)
        layout.addWidget(self._videoController)
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def model(self) -> Optional[ExperimentDataModel]:
        return self._model

    def setModel(self, model: Optional[ExperimentDataModel]):
        oldModel = self.model()
        if oldModel is not None:
            oldModel.activatedIndexChanged.disconnect(self.setActivatedIndex)
            oldModel.experimentDataChanged.disconnect(self._onExptDataChange)
        self._model = model
        if model is not None:
            model.activatedIndexChanged.connect(self.setActivatedIndex)
            model.experimentDataChanged.connect(self._onExptDataChange)

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            coatPathsIdx = model.getIndexFor(IndexRole.COATPATHS, index)
            coatPaths = [
                model.index(row, 0, coatPathsIdx).data(model.Role_CoatPath)
                for row in range(model.rowCount(coatPathsIdx))
            ]
            exptKind = experiment_kind(coatPaths)
        else:
            index = QModelIndex()
            exptKind = ExperimentKind.NullExperiment
        self._currentModelIndex = index
        self.setExperimentKind(exptKind)

    @Slot(QModelIndex, DataArgs)
    def _onExptDataChange(self, index: QModelIndex, flag: DataArgs):
        model = index.model()
        if not isinstance(model, ExperimentDataModel):
            return
        if index != self._currentModelIndex:
            return
        if flag & DataArgs.COATPATHS:
            coatPathsIdx = model.getIndexFor(IndexRole.COATPATHS, index)
            coatPaths = [
                model.index(row, 0, coatPathsIdx).data(model.Role_CoatPath)
                for row in range(model.rowCount(coatPathsIdx))
            ]
            exptKind = experiment_kind(coatPaths)
            self.setExperimentKind(exptKind)

    def setExperimentKind(self, exptKind: ExperimentKind):
        camera = self.camera()
        if camera is None:
            cameraActive = False
        else:
            cameraActive = camera.isActive()
        controllerVisible = self.isExperimentVideo(
            cameraActive, self._currentView, exptKind
        )
        self._videoController.setVisible(controllerVisible)
        self._exptKind = exptKind

    @Slot(DataMember)
    def setCurrentView(self, currentView: DataMember):
        camera = self.camera()
        if camera is None:
            cameraActive = False
        else:
            cameraActive = camera.isActive()
        controllerVisible = self.isExperimentVideo(
            cameraActive, currentView, self._exptKind
        )
        self._videoController.setVisible(controllerVisible)
        self._currentView = currentView

    def camera(self) -> Optional[QCamera]:
        return self._camera

    def setCamera(self, camera: Optional[QCamera]):
        oldCamera = self.camera()
        if oldCamera is not None:
            oldCamera.activeChanged.disconnect(  # type: ignore[attr-defined]
                self._onCameraActiveChange
            )
        self._camera = camera
        self._displayToolBar.setCamera(camera)
        if camera is not None:
            camera.activeChanged.connect(  # type: ignore[attr-defined]
                self._onCameraActiveChange
            )

    @Slot(bool)
    def _onCameraActiveChange(self, active: bool):
        controllerVisible = self.isExperimentVideo(
            active, self._currentView, self._exptKind
        )
        self._videoController.setVisible(controllerVisible)
        self.cameraActiveChanged.emit(active)

    def setPlayer(self, player: Optional[QMediaPlayer]):
        self._videoController.setPlayer(player)

    @staticmethod
    def isExperimentVideo(
        cameraActive: bool,
        currentView: DataMember,
        exptKind: ExperimentKind,
    ) -> bool:
        viewExclude = [
            DataMember.REFERENCE,
            DataMember.SUBSTRATE,
        ]
        ret = (
            not cameraActive
            and currentView not in viewExclude
            and exptKind == ExperimentKind.VideoExperiment
        )
        return ret

    @Slot(np.ndarray)
    def setArray(self, array: npt.NDArray[np.uint8]):
        self._displayLabel.setArray(array)
