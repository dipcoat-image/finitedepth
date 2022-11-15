"""
Video streaming pipeline
"""

from araviq6 import NDArrayVideoPlayer, NDArrayMediaCaptureSession
import cv2  # type: ignore[import]
import numpy as np
import numpy.typing as npt
from dipcoatimage.finitedepth import SubstrateReferenceBase
from dipcoatimage.finitedepth.analysis import ExperimentKind, experiment_kind
from dipcoatimage.finitedepth_gui.core import DataMember, DataArgs, FrameSource
from dipcoatimage.finitedepth_gui.worker import ExperimentWorker
from dipcoatimage.finitedepth_gui.model import IndexRole, ExperimentDataModel
from PySide6.QtCore import QObject, Signal, Slot, QUrl, QThread, QModelIndex
from PySide6.QtMultimedia import QCamera, QMediaPlayer
from typing import Optional, Protocol


__all__ = [
    "ImageProcessor",
    "PreviewableNDArrayVideoPlayer",
    "DisplayProtocol",
    "VisualizeManager",
]


class ImageProcessor(QObject):
    """Object to process the incoming image from video or camera."""

    arrayChanged = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._currentView = DataMember.NULL
        self._ready = True

    def setWorker(self, worker: Optional[ExperimentWorker]):
        self._worker = worker

    def setCurrentView(self, currentView: DataMember):
        self._currentView = currentView

    @Slot(np.ndarray)
    def setArray(self, array: npt.NDArray[np.uint8]):
        array = array.copy()  # must detach array from the memory
        self._ready = False
        self.arrayChanged.emit(self.processArray(array))
        self._ready = True

    def processArray(self, array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        worker = self._worker
        if worker is None:
            return array
        if self._currentView == DataMember.REFERENCE:
            ref = worker.constructReference(array, worker.exptData.reference)
            if ref is not None:
                array = ref.draw()
        elif self._currentView == DataMember.SUBSTRATE:
            ref = worker.constructReference(array, worker.exptData.reference)
            subst = worker.constructSubstrate(ref, worker.exptData.substrate)
            if subst is not None:
                array = subst.draw()
            else:
                h, w = array.shape[:2]
                substROI = worker.exptData.reference.substrateROI
                x0, y0, x1, y1 = SubstrateReferenceBase.sanitize_ROI(substROI, h, w)
                array = array[y0:y1, x0:x1]
        else:
            expt = worker.experiment
            if expt is not None:
                if array.size > 0:
                    layer = expt.construct_coatinglayer(array)
                    if layer.valid():
                        array = layer.draw()
        return array

    def ready(self) -> bool:
        return self._ready


class PreviewableNDArrayVideoPlayer(NDArrayVideoPlayer):
    """
    Video player which emits first frame of the video on source change
    and on video stop.
    """

    @Slot(QUrl)
    def setSource(self, source: QUrl):
        super().setSource(source)
        self.arrayChanged.emit(self.previewImage())

    @Slot()
    def stop(self):
        super().stop()
        self.arrayChanged.emit(self.previewImage())

    def previewImage(self) -> npt.NDArray[np.uint8]:
        path = self.source().toLocalFile()
        if path:
            cap = cv2.VideoCapture(path)
            ok, img = cap.read()
            cap.release()
            if not ok:
                img = np.empty((0, 0, 0))
        else:
            img = np.empty((0, 0, 0))
        return img


class DisplayProtocol(Protocol):
    def setExperimentKind(self, exptKind: ExperimentKind):
        ...

    def setCurrentView(self, currentView: DataMember):
        ...

    def setFrameSource(self, frameSource: FrameSource):
        ...

    def setPlayer(self, player: Optional[QMediaPlayer]):
        ...

    def setArray(self, array: np.ndarray):
        ...


class VisualizeManager(QObject):

    _processRequested = Signal(np.ndarray)
    arrayChanged = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None
        self._frameSource = FrameSource.NULL
        self._videoPlayer = PreviewableNDArrayVideoPlayer()
        self._camera = None
        self._captureSession = NDArrayMediaCaptureSession()
        self._imageProcessor = ImageProcessor()
        self._display = None

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

    @Slot(QModelIndex)
    def _onActivatedIndexChange(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            worker = model.worker(index)
            coatPathsIdx = model.getIndexFor(IndexRole.COATPATHS, index)
            coatPaths = [
                model.index(row, 0, coatPathsIdx).data(model.Role_CoatPath)
                for row in range(model.rowCount(coatPathsIdx))
            ]
            exptKind = experiment_kind(coatPaths)
        else:
            worker = None
            exptKind = ExperimentKind.NullExperiment
        self._imageProcessor.setWorker(worker)
        display = self.display()
        if display is not None:
            display.setExperimentKind(exptKind)

    @Slot(QModelIndex, DataArgs)
    def _onExptDataChange(self, index: QModelIndex, flag: DataArgs):
        model = index.model()
        if not isinstance(model, ExperimentDataModel):
            return
        if index != model.activatedIndex():
            return
        if flag & DataArgs.COATPATHS:
            coatPathsIdx = model.getIndexFor(IndexRole.COATPATHS, index)
            coatPaths = [
                model.index(row, 0, coatPathsIdx).data(model.Role_CoatPath)
                for row in range(model.rowCount(coatPathsIdx))
            ]
            exptKind = experiment_kind(coatPaths)
            display = self.display()
            if display is not None:
                display.setExperimentKind(exptKind)

    def camera(self) -> Optional[QCamera]:
        return self._camera

    def setCamera(self, camera: Optional[QCamera]):
        oldCamera = self.camera()
        if oldCamera is not None:
            oldCamera.activeChanged.disconnect(  # type: ignore[attr-defined]
                self._onCameraActiveChange
            )
        self._camera = camera
        if camera is not None:
            camera.activeChanged.connect(  # type: ignore[attr-defined]
                self._onCameraActiveChange
            )

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
            self._captureSession.arrayChanged.disconnect(self._displayImage)
        elif oldSource == FrameSource.FILE:
            self._videoPlayer.arrayChanged.disconnect(self._displayImage)
        else:
            pass
        self._frameSource = frameSource
        if frameSource == FrameSource.CAMERA:
            self._captureSession.arrayChanged.connect(self._displayImage)
        elif oldSource == FrameSource.FILE:
            self._videoPlayer.arrayChanged.connect(self._displayImage)
        else:
            pass
        display = self.display()
        if display is not None:
            display.setFrameSource(frameSource)

    @Slot(DataMember)
    def setCurrentView(self, currentView: DataMember):
        if currentView in (DataMember.REFERENCE, DataMember.SUBSTRATE):
            self._videoPlayer.pause()
        self._imageProcessor.setCurrentView(currentView)
        model = self.model()
        if model is None:
            return
        if self._frameSource == FrameSource.FILE:
            worker = model.worker(model.activatedIndex())
            if worker is None:
                img = np.empty((0, 0, 0), dtype=np.uint8)
            else:
                if currentView == DataMember.REFERENCE:
                    ref = worker.reference
                    if ref is not None:
                        img = ref.draw()
                    else:
                        img = np.empty((0, 0, 0), dtype=np.uint8)
                elif currentView == DataMember.SUBSTRATE:
                    subst = worker.substrate
                    if subst is not None:
                        img = subst.draw()
                    else:
                        img = np.empty((0, 0, 0), dtype=np.uint8)
                else:
                    ...
                    img = np.empty((0, 0, 0), dtype=np.uint8)
            self.arrayChanged.emit(img)
        display = self.display()
        if display is not None:
            display.setCurrentView(currentView)

    @Slot(np.ndarray)
    def _displayImage(self, array: npt.NDArray[np.uint8]):
        processor = self._imageProcessor
        if not processor.ready():
            return
        self._processRequested.emit(array)

    def display(self) -> Optional[DisplayProtocol]:
        return self._display

    def setDisplay(self, display: Optional[DisplayProtocol]):
        oldDisplay = self.display()
        if oldDisplay is not None:
            oldDisplay.setPlayer(None)
            self.arrayChanged.disconnect(oldDisplay.setArray)
        self._display = display
        if display is not None:
            display.setPlayer(self._videoPlayer)
            self.arrayChanged.connect(display.setArray)

    def stop(self):
        self._processorThread.quit()
        self._processorThread.wait()
