import cv2  # type: ignore[import]
import enum
import numpy as np
import numpy.typing as npt
from dipcoatimage.finitedepth import ExperimentKind, experiment_kind
from dipcoatimage.finitedepth.reference import sanitize_ROI
from PySide6.QtCore import QObject, Signal, Slot, QUrl, QThread, QModelIndex
from PySide6.QtMultimedia import QMediaPlayer
from dipcoatimage.finitedepth_gui.core import (
    DataMember,
    DataArgFlag,
    FrameSource,
    VisualizationMode,
)
from dipcoatimage.finitedepth_gui.worker import ExperimentWorker
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from dipcoatimage.finitedepth_gui.util import (
    CameraProtocol,
    ImageCaptureProtocol,
    MediaRecorderProtocol,
)
from .imgprocess import ImageProcessor
from typing import Optional

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias


__all__ = [
    "VideoPlaybackState",
    "VisualizerBase",
]


class VideoPlaybackState(enum.Enum):
    StoppedState = 0
    PlayingState = 1
    PausedState = 2


class VisualizerBase(QObject):
    """
    Abstract base class for the visualization interface.
    """

    PlaybackState: TypeAlias = VideoPlaybackState

    _processRequested = Signal(np.ndarray)
    arrayChanged = Signal(np.ndarray)
    roiMaximumChanged = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None
        self._exptKind = ExperimentKind.NULL
        self._frameSource = FrameSource.NULL
        self._currentView = DataMember.NULL
        self._visualizeMode = VisualizationMode.OFF

        self._imageProcessor = ImageProcessor()

        self._processorThread = QThread()
        self._imageProcessor.moveToThread(self._processorThread)
        self._processorThread.start()

        self._processRequested.connect(self._imageProcessor.setArray)
        self._imageProcessor.arrayChanged.connect(self.arrayChanged)

    def model(self) -> Optional[ExperimentDataModel]:
        return self._model

    def setModel(self, model: Optional[ExperimentDataModel]):
        oldModel = self.model()
        if oldModel is not None:
            oldModel.activatedIndexChanged.disconnect(self._onActivatedIndexChange)
            oldModel.experimentDataChanged.disconnect(self._onExptDataChange)
        self._model = model
        if model is not None:
            model.activatedIndexChanged.connect(self._onActivatedIndexChange)
            model.experimentDataChanged.connect(self._onExptDataChange)

    def videoPlayer(self) -> QMediaPlayer:
        raise NotImplementedError

    def videoPlaybackState(self) -> PlaybackState:
        raise NotImplementedError

    def camera(self) -> CameraProtocol:
        """
        Abstract interface for camera.

        Implementation must detect the camera activation/deactivation and call
        :meth:`setFrameSource` with proper :class:`FrameSource`.
        """
        raise NotImplementedError

    def imageCapture(self) -> ImageCaptureProtocol:
        raise NotImplementedError

    def mediaRecorder(self) -> MediaRecorderProtocol:
        raise NotImplementedError

    def togglePlayerPipeline(self, toggle: bool):
        """
        Connect or disconnect the array pipeline from :meth:`videoPlayer` to
        :meth:`visualizeImageFromPlayer`
        """
        raise NotImplementedError

    def toggleCameraPipeline(self, toggle: bool):
        """
        Connect or disconnect the array pipeline from :meth:`camera` to
        :meth:`visualizeImageFromCamera`
        """
        raise NotImplementedError

    @Slot(QModelIndex)
    def _onActivatedIndexChange(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            worker = model.worker(index)
            if worker is None:
                coatPaths = []
            else:
                coatPaths = worker.exptData.coat_paths
        else:
            worker = None
            coatPaths = []
        self._imageProcessor.setWorker(worker)

        videoPlayer = self.videoPlayer()
        oldExptKind = self._exptKind
        if (
            oldExptKind == ExperimentKind.VIDEO
            and self._frameSource == FrameSource.FILE
            and self._currentView
            not in (
                DataMember.REFERENCE,
                DataMember.SUBSTRATE,
            )
        ):
            state = self.videoPlaybackState()
            if state == self.PlaybackState.PlayingState:
                videoPlayer.stop()
            self.togglePlayerPipeline(False)
        exptKind = experiment_kind(coatPaths)
        if exptKind == ExperimentKind.VIDEO:
            if self._frameSource == FrameSource.FILE and self._currentView not in (
                DataMember.REFERENCE,
                DataMember.SUBSTRATE,
            ):
                self.togglePlayerPipeline(True)
            source = QUrl.fromLocalFile(coatPaths[0])
            videoPlayer.setSource(source)
        else:
            videoPlayer.setSource(QUrl())
        self._exptKind = exptKind

        if self._frameSource == FrameSource.CAMERA:
            pass
        elif worker is None:
            img = np.empty((0, 0, 0), dtype=np.uint8)
            self.arrayChanged.emit(img)
        else:
            self.visualizeFromWorker(worker)

    @Slot(QModelIndex, DataArgFlag)
    def _onExptDataChange(self, index: QModelIndex, flag: DataArgFlag):
        model = index.model()
        if not isinstance(model, ExperimentDataModel):
            return
        if index != model.activatedIndex():
            return
        worker = model.worker(index)
        if worker is None:
            return

        if flag & DataArgFlag.COATPATHS:
            videoPlayer = self.videoPlayer()
            oldExptKind = self._exptKind
            if (
                oldExptKind == ExperimentKind.VIDEO
                and self._frameSource == FrameSource.FILE
                and self._currentView
                not in (
                    DataMember.REFERENCE,
                    DataMember.SUBSTRATE,
                )
            ):
                state = self.videoPlaybackState()
                if state == self.PlaybackState.PlayingState:
                    videoPlayer.stop()
                self.togglePlayerPipeline(False)
            coatPaths = worker.exptData.coat_paths
            exptKind = experiment_kind(coatPaths)
            if exptKind == ExperimentKind.VIDEO:
                if self._frameSource == FrameSource.FILE and self._currentView not in (
                    DataMember.REFERENCE,
                    DataMember.SUBSTRATE,
                ):
                    self.togglePlayerPipeline(True)
                source = QUrl.fromLocalFile(coatPaths[0])
                videoPlayer.setSource(source)
            else:
                videoPlayer.setSource(QUrl())
            self._exptKind = exptKind

        self._imageProcessor.setWorker(worker)

        if self._frameSource == FrameSource.CAMERA:
            return

        if self._currentView == DataMember.REFERENCE:
            if flag & (DataArgFlag.REFPATH | DataArgFlag.REFERENCE):
                self.visualizeFromWorker(worker)
        elif self._currentView == DataMember.SUBSTRATE:
            if flag & (
                DataArgFlag.REFPATH | DataArgFlag.REFERENCE | DataArgFlag.SUBSTRATE
            ):
                self.visualizeFromWorker(worker)
        else:
            if flag & (
                DataArgFlag.COATPATHS
                | DataArgFlag.REFPATH
                | DataArgFlag.REFERENCE
                | DataArgFlag.SUBSTRATE
                | DataArgFlag.COATINGLAYER
                | DataArgFlag.EXPERIMENT
            ):
                self.visualizeFromWorker(worker)

    def setFrameSource(self, frameSource: FrameSource):
        oldSource = self._frameSource
        if oldSource == FrameSource.CAMERA:
            self.toggleCameraPipeline(False)
        elif oldSource == FrameSource.FILE:
            if self._exptKind == ExperimentKind.VIDEO and self._currentView not in (
                DataMember.REFERENCE,
                DataMember.SUBSTRATE,
            ):
                videoPlayer = self.videoPlayer()
                state = self.videoPlaybackState()
                if state == self.PlaybackState.PlayingState:
                    videoPlayer.stop()
                self.togglePlayerPipeline(False)
        else:
            pass
        self._frameSource = frameSource
        if frameSource == FrameSource.CAMERA:
            self.toggleCameraPipeline(True)
        elif frameSource == FrameSource.FILE:
            if self._exptKind == ExperimentKind.VIDEO and self._currentView not in (
                DataMember.REFERENCE,
                DataMember.SUBSTRATE,
            ):
                self.togglePlayerPipeline(True)
        else:
            pass

        model = self.model()
        if model is None:
            return
        worker = model.worker(model.activatedIndex())

        if frameSource == FrameSource.CAMERA:
            pass
        elif worker is None:
            img = np.empty((0, 0, 0), dtype=np.uint8)
            self.arrayChanged.emit(img)
        else:
            self.visualizeFromWorker(worker)

    @Slot(DataMember)
    def setCurrentView(self, currentView: DataMember):
        def isExptView(view: DataMember):
            return view not in (DataMember.REFERENCE, DataMember.SUBSTRATE)

        oldView = self._currentView
        if isExptView(oldView) and not isExptView(currentView):
            videoPlayer = self.videoPlayer()
            state = self.videoPlaybackState()
            if (
                self._frameSource == FrameSource.FILE
                and self._exptKind == ExperimentKind.VIDEO
            ):
                if state == self.PlaybackState.PlayingState:
                    videoPlayer.stop()
                self.togglePlayerPipeline(False)
            if state == self.PlaybackState.PlayingState:
                videoPlayer.pause()
        elif not isExptView(oldView) and isExptView(currentView):
            if (
                self._frameSource == FrameSource.FILE
                and self._exptKind == ExperimentKind.VIDEO
            ):
                self.togglePlayerPipeline(True)
        self._currentView = currentView

        self._imageProcessor.setCurrentView(currentView)

        model = self.model()
        if model is None:
            return
        worker = model.worker(model.activatedIndex())

        if self._frameSource == FrameSource.CAMERA:
            pass
        elif worker is None:
            img = np.empty((0, 0, 0), dtype=np.uint8)
            self.arrayChanged.emit(img)
        else:
            self.visualizeFromWorker(worker)

    @Slot(VisualizationMode)
    def setVisualizationMode(self, mode: VisualizationMode):
        self._imageProcessor.setVisualizationMode(mode)
        self._visualizeMode = mode

        model = self.model()
        if model is None:
            return
        worker = model.worker(model.activatedIndex())

        if self._frameSource == FrameSource.CAMERA:
            pass
        elif worker is None:
            img = np.empty((0, 0, 0), dtype=np.uint8)
            self.arrayChanged.emit(img)
        else:
            self.visualizeFromWorker(worker)

    def visualizeFromWorker(self, worker: ExperimentWorker):
        currentView = self._currentView
        if currentView == DataMember.REFERENCE:
            if self._visualizeMode == VisualizationMode.FULL:
                ref = worker.reference
                if ref is not None:
                    img = ref.draw()
                else:
                    img = np.empty((0, 0, 0), dtype=np.uint8)
            else:
                img = worker.referenceImage
            self.arrayChanged.emit(img)
        elif currentView == DataMember.SUBSTRATE:
            if self._visualizeMode == VisualizationMode.FULL:
                subst = worker.substrate
            else:
                subst = None
            if subst is not None:
                img = subst.draw()
            else:
                img = worker.referenceImage
                roi = worker.exptData.reference.substrateROI
                h, w = img.shape[:2]
                x0, y0, x1, y1 = sanitize_ROI(roi, h, w)
                img = img[y0:y1, x0:x1]
            self.arrayChanged.emit(img)
        else:
            coatPaths = worker.exptData.coat_paths
            exptKind = experiment_kind(coatPaths)
            if exptKind in (
                ExperimentKind.SINGLE_IMAGE,
                ExperimentKind.MULTI_IMAGE,
            ):
                img = cv2.imread(coatPaths[0])
            elif exptKind == ExperimentKind.VIDEO:
                state = self.videoPlaybackState()
                if state == self.PlaybackState.PlayingState:
                    return
                elif state == self.PlaybackState.StoppedState:
                    cap = cv2.VideoCapture(coatPaths[0])
                    ok, img = cap.read()
                    cap.release()
                    if not ok:
                        img = None
                else:
                    img = self._lastVideoFrame
            else:
                img = None
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._imageProcessor.setArray(img)
            else:
                img = np.empty((0, 0, 0), dtype=np.uint8)
                self.arrayChanged.emit(img)

    @Slot(np.ndarray)
    def visualizeImageFromPlayer(self, array: npt.NDArray[np.uint8]):
        self._lastVideoFrame = array.copy()

        processor = self._imageProcessor
        if not processor.ready():
            return
        if array.size != 0:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        if self._currentView == DataMember.REFERENCE:
            h, w = array.shape[:2]
            self.roiMaximumChanged.emit(h, w)
        self._processRequested.emit(array)

    @Slot(np.ndarray)
    def visualizeImageFromCamera(self, array: npt.NDArray[np.uint8]):
        processor = self._imageProcessor
        if not processor.ready():
            return
        if array.size != 0:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        if self._currentView == DataMember.REFERENCE:
            h, w = array.shape[:2]
            self.roiMaximumChanged.emit(h, w)
        self._processRequested.emit(array)

    def stop(self):
        self._processorThread.quit()
        self._processorThread.wait()
