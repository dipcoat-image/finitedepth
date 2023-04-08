"""
Visualization
=============

"""

from araviq6 import ArrayWorker, ArrayProcessor, FrameToArrayConverter
import cv2  # type: ignore[import]
import numpy as np
import numpy.typing as npt
from dipcoatimage.finitedepth import ExperimentKind, experiment_kind
from dipcoatimage.finitedepth.util import (
    OptionalROI,
    sanitize_ROI,
    match_template,
    images_XOR,
    binarize,
)
from dipcoatimage.finitedepth_gui.core import (
    DataMember,
    DataArgFlag,
    FrameSource,
    VisualizationMode,
)
from dipcoatimage.finitedepth_gui.worker import ExperimentWorker
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from PySide6.QtCore import QObject, Signal, Slot, QUrl, QModelIndex
from PySide6.QtMultimedia import (
    QCamera,
    QMediaPlayer,
    QMediaCaptureSession,
    QVideoSink,
    QImageCapture,
    QMediaRecorder,
)
from typing import Optional


__all__ = [
    "VisualizeWorker",
    "VisualizeManager",
]


class VisualizeWorker(ArrayWorker):
    """
    Class to process the incoming array.

    :meth:`VisualizeWorker` manages the options and arguments for internal
    :meth:`ExperimentWorker` to visualize the array.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._prev = None
        self._currentView = DataMember.NULL
        self._visualizeMode = VisualizationMode.OFF

    def setWorker(self, worker: Optional[ExperimentWorker]):
        """Set internal worker which really performs visualization."""
        self._worker = worker
        self._prev = None

    def setCurrentView(self, currentView: DataMember):
        self._currentView = currentView

    def setVisualizationMode(self, mode: VisualizationMode):
        self._visualizeMode = mode

    def processArray(self, array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        worker = self._worker
        if worker is None:
            return array
        if self._currentView == DataMember.REFERENCE:
            if self._visualizeMode == VisualizationMode.FULL:
                ref = worker.constructReference(array, worker.exptData.reference)
                if ref is not None:
                    array = ref.draw()
        elif self._currentView == DataMember.SUBSTRATE:
            if self._visualizeMode == VisualizationMode.FULL:
                ref = worker.constructReference(array, worker.exptData.reference)
                subst = worker.constructSubstrate(ref, worker.exptData.substrate)
            else:
                subst = None
            if subst is not None:
                array = subst.draw()
            else:
                roi = worker.exptData.reference.substrateROI
                h, w = array.shape[:2]
                x0, y0, x1, y1 = sanitize_ROI(roi, h, w)
                array = array[y0:y1, x0:x1]
        else:
            if self._visualizeMode == VisualizationMode.FULL:
                expt = worker.experiment
                if expt is not None:
                    if array.size > 0:
                        layer = expt.construct_coatinglayer(array, self._prev)
                        if layer.valid():
                            array = layer.draw()
                            self._prev = layer
                        else:
                            self._prev = None
            elif self._visualizeMode == VisualizationMode.FAST:
                refImg = worker.referenceImage
                tempROI = worker.exptData.reference.templateROI
                substROI = worker.exptData.reference.substrateROI
                array = fastVisualize(refImg, array, tempROI, substROI)
        return array


class VisualizeManager(QObject):
    """
    Object to manage visualization pipeline.
    """

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

        self._videoPlayer = QMediaPlayer()
        self._playerSink = QVideoSink()
        self._lastVideoFrame = np.empty((0, 0, 0), dtype=np.uint8)

        self._camera = QCamera()
        self._captureSession = QMediaCaptureSession()
        self._imageCapture = QImageCapture()
        self._mediaRecorder = QMediaRecorder()
        self._cameraSink = QVideoSink()

        self._arrayConverter = FrameToArrayConverter()
        self._arrayProcessor = ArrayProcessor()
        self._visualizeWorker = VisualizeWorker()
        self._arrayProcessor.setWorker(self._visualizeWorker)

        self._videoPlayer.setVideoSink(self._playerSink)

        self._camera.activeChanged.connect(self._onCameraActiveChange)
        self._captureSession.setCamera(self._camera)
        self._captureSession.setImageCapture(self._imageCapture)
        self._captureSession.setRecorder(self._mediaRecorder)
        self._captureSession.setVideoSink(self._cameraSink)

        self._processRequested.connect(self._arrayProcessor.processArray)
        self._arrayProcessor.arrayProcessed.connect(self.arrayChanged)

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
        return self._videoPlayer

    def camera(self) -> QCamera:
        return self._camera

    def imageCapture(self) -> QImageCapture:
        return self._imageCapture

    def mediaRecorder(self) -> QMediaRecorder:
        return self._mediaRecorder

    def togglePlayerPipeline(self, toggle: bool):
        if toggle:
            self._playerSink.videoFrameChanged.connect(
                self._arrayConverter.convertVideoFrame
            )
            self._arrayConverter.arrayConverted.connect(self._processArrayFromPlayer)
        else:
            self._playerSink.videoFrameChanged.disconnect(
                self._arrayConverter.convertVideoFrame
            )
            self._arrayConverter.arrayConverted.disconnect(self._processArrayFromPlayer)

    def toggleCameraPipeline(self, toggle: bool):
        if toggle:
            self._cameraSink.videoFrameChanged.connect(
                self._arrayConverter.convertVideoFrame
            )
            self._arrayConverter.arrayConverted.connect(self._processArrayFromCamera)
        else:
            self._cameraSink.videoFrameChanged.disconnect(
                self._arrayConverter.convertVideoFrame
            )
            self._arrayConverter.arrayConverted.disconnect(self._processArrayFromCamera)

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
        self._visualizeWorker.setWorker(worker)

        if (
            self._exptKind == ExperimentKind.VIDEO
            and self._frameSource == FrameSource.FILE
            and self._currentView.displays() == DataMember.EXPERIMENT
        ):
            state = self._videoPlayer.playbackState()
            if state == QMediaPlayer.PlaybackState.PlayingState:
                self._videoPlayer.stop()
            self.togglePlayerPipeline(False)

        self._exptKind = experiment_kind(coatPaths)
        if self._exptKind == ExperimentKind.VIDEO:
            if (
                self._frameSource == FrameSource.FILE
                and self._currentView.displays() == DataMember.EXPERIMENT
            ):
                self.togglePlayerPipeline(True)
            self._videoPlayer.setSource(QUrl.fromLocalFile(coatPaths[0]))
        else:
            self._videoPlayer.setSource(QUrl())

        if self._frameSource == FrameSource.CAMERA:
            pass
        elif worker is None:
            img = np.empty((0, 0, 0), dtype=np.uint8)
            self.arrayChanged.emit(img)
        else:
            self._displayFromWorker(worker)

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
            oldExptKind = self._exptKind
            if (
                oldExptKind == ExperimentKind.VIDEO
                and self._frameSource == FrameSource.FILE
                and self._currentView.displays() == DataMember.EXPERIMENT
            ):
                state = self._videoPlayer.playbackState()
                if state == QMediaPlayer.PlaybackState.PlayingState:
                    self._videoPlayer.stop()
                self.togglePlayerPipeline(False)
            coatPaths = worker.exptData.coat_paths
            exptKind = experiment_kind(coatPaths)
            if exptKind == ExperimentKind.VIDEO:
                if (
                    self._frameSource == FrameSource.FILE
                    and self._currentView.displays() == DataMember.EXPERIMENT
                ):
                    self.togglePlayerPipeline(True)
                source = QUrl.fromLocalFile(coatPaths[0])
                self._videoPlayer.setSource(source)
            else:
                self._videoPlayer.setSource(QUrl())
            self._exptKind = exptKind

        self._visualizeWorker.setWorker(worker)

        if self._frameSource == FrameSource.CAMERA:
            return

        if self._currentView == DataMember.REFERENCE:
            if flag & (DataArgFlag.REFPATH | DataArgFlag.REFERENCE):
                self._displayFromWorker(worker)
        elif self._currentView == DataMember.SUBSTRATE:
            if flag & (
                DataArgFlag.REFPATH | DataArgFlag.REFERENCE | DataArgFlag.SUBSTRATE
            ):
                self._displayFromWorker(worker)
        else:
            if flag & (
                DataArgFlag.COATPATHS
                | DataArgFlag.REFPATH
                | DataArgFlag.REFERENCE
                | DataArgFlag.SUBSTRATE
                | DataArgFlag.COATINGLAYER
                | DataArgFlag.EXPERIMENT
            ):
                self._displayFromWorker(worker)

    @Slot(bool)
    def _onCameraActiveChange(self, active: bool):
        if active:
            frameSource = FrameSource.CAMERA
        else:
            frameSource = FrameSource.FILE
        self.setFrameSource(frameSource)

    def setFrameSource(self, frameSource: FrameSource):
        oldSource = self._frameSource
        if oldSource == FrameSource.CAMERA:
            self.toggleCameraPipeline(False)
        elif oldSource == FrameSource.FILE:
            if (
                self._exptKind == ExperimentKind.VIDEO
                and self._currentView.displays() == DataMember.EXPERIMENT
            ):
                state = self._videoPlayer.playbackState()
                if state == QMediaPlayer.PlaybackState.PlayingState:
                    self._videoPlayer.stop()
                self.togglePlayerPipeline(False)
        else:
            pass
        self._frameSource = frameSource
        if frameSource == FrameSource.CAMERA:
            self.toggleCameraPipeline(True)
        elif frameSource == FrameSource.FILE:
            if (
                self._exptKind == ExperimentKind.VIDEO
                and self._currentView.displays() == DataMember.EXPERIMENT
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
            self._displayFromWorker(worker)

    @Slot(DataMember)
    def setCurrentView(self, currentView: DataMember):
        oldView = self._currentView
        if (
            oldView.displays() == DataMember.EXPERIMENT
            and currentView.displays() != DataMember.EXPERIMENT
        ):
            if (
                self._frameSource == FrameSource.FILE
                and self._exptKind == ExperimentKind.VIDEO
            ):
                state = self._videoPlayer.playbackState()
                if state == QMediaPlayer.PlaybackState.PlayingState:
                    self._videoPlayer.stop()
                self.togglePlayerPipeline(False)
            if (
                self._videoPlayer.playbackState()
                == QMediaPlayer.PlaybackState.PlayingState
            ):
                self._videoPlayer.pause()
        elif (
            oldView.displays() != DataMember.EXPERIMENT
            and currentView.displays() == DataMember.EXPERIMENT
        ):
            if (
                self._frameSource == FrameSource.FILE
                and self._exptKind == ExperimentKind.VIDEO
            ):
                self.togglePlayerPipeline(True)
        self._currentView = currentView

        self._visualizeWorker.setCurrentView(currentView)

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
            self._displayFromWorker(worker)

    @Slot(VisualizationMode)
    def setVisualizationMode(self, mode: VisualizationMode):
        self._visualizeWorker.setVisualizationMode(mode)
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
            self._displayFromWorker(worker)

    def _displayFromWorker(self, worker: ExperimentWorker):
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
                state = self._videoPlayer.playbackState()
                if state == QMediaPlayer.PlaybackState.PlayingState:
                    return
                elif state == QMediaPlayer.PlaybackState.StoppedState:
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
                self._visualizeWorker.runProcess(img)
            else:
                img = np.empty((0, 0, 0), dtype=np.uint8)
                self.arrayChanged.emit(img)

    @Slot(np.ndarray)
    def _processArrayFromPlayer(self, array: npt.NDArray[np.uint8]):
        self._lastVideoFrame = array.copy()

        if array.size != 0:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        if self._currentView == DataMember.REFERENCE:
            h, w = array.shape[:2]
            self.roiMaximumChanged.emit(h, w)
        self._processRequested.emit(array)

    @Slot(np.ndarray)
    def _processArrayFromCamera(self, array: npt.NDArray[np.uint8]):
        if array.size != 0:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        if self._currentView == DataMember.REFERENCE:
            h, w = array.shape[:2]
            self.roiMaximumChanged.emit(h, w)
        self._processRequested.emit(array)

    def stop(self):
        self._arrayProcessor.stop()


def fastVisualize(
    refImg: npt.NDArray[np.uint8],
    layerImg: npt.NDArray[np.uint8],
    tempROI: OptionalROI,
    substROI: OptionalROI,
):
    ref_bin = binarize(refImg)
    if ref_bin.size == 0:
        return layerImg

    layer_bin = binarize(layerImg)
    if layer_bin.size == 0:
        return layerImg

    h, w = refImg.shape[:2]
    tempROI = sanitize_ROI(tempROI, h, w)
    substROI = sanitize_ROI(substROI, h, w)

    temp_x0, temp_y0, temp_x1, temp_y1 = tempROI
    template = ref_bin[temp_y0:temp_y1, temp_x0:temp_x1]
    subst_x0, subst_y0, subst_x1, subst_y1 = substROI
    substImg = ref_bin[subst_y0:subst_y1, subst_x0:subst_x1]

    _, (tx, ty) = match_template(layer_bin, template)
    dx, dy = (substROI[0] - tempROI[0], substROI[1] - tempROI[1])
    x0, y0 = (tx + dx, ty + dy)

    mask = images_XOR(~layer_bin.astype(bool), ~substImg.astype(bool), (x0, y0))
    H, W = substImg.shape
    x1, y1 = x0 + W, y0 + H
    layer_bin[y0:y1, x0:x1][mask] = 255

    return cv2.cvtColor(layer_bin, cv2.COLOR_GRAY2RGB)
