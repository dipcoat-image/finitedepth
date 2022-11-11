"""
Video streaming pipeline
"""

from araviq6 import NDArrayVideoPlayer, NDArrayMediaCaptureSession
import cv2  # type: ignore[import]
import enum
import numpy as np
import numpy.typing as npt
from dipcoatimage.finitedepth_gui.core import DataMember
from dipcoatimage.finitedepth_gui.worker import ExperimentWorker
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from dipcoatimage.finitedepth_gui.display import MainDisplayWindow_V2
from PySide6.QtCore import QObject, Signal, Slot, QUrl, QThread, QModelIndex
from typing import Optional


__all__ = [
    "ImageProcessor",
    "PreviewableNDArrayVideoPlayer",
    "FrameSource",
    "VisualizeManager",
]


class ImageProcessor(QObject):
    """Object to process the incoming image."""

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
            worker.setReferenceImage(array)
            array = worker.drawReferenceImage()
        elif self._currentView == DataMember.SUBSTRATE:
            worker.setReferenceImage(array)
            array = worker.drawSubstrateImage()
        else:
            array = worker.drawCoatingLayerImage(array)
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


class FrameSource(enum.Enum):
    NULL = 0
    FILE = 1
    CAMERA = 2


class VisualizeManager(QObject):

    _processRequested = Signal(np.ndarray)
    arrayChanged = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None
        self._frameSource = FrameSource.NULL
        self._display = None

        self._videoPlayer = PreviewableNDArrayVideoPlayer()
        self._captureSession = NDArrayMediaCaptureSession()

        self._processorThread = QThread()
        self._imageProcessor = ImageProcessor()

        self._processRequested.connect(self._imageProcessor.setArray)
        self._imageProcessor.arrayChanged.connect(self.arrayChanged)

        self._imageProcessor.moveToThread(self._processorThread)
        self._processorThread.start()

    def setModel(self, model: Optional[ExperimentDataModel]):
        oldModel = self._model
        if oldModel is not None:
            oldModel.activatedIndexChanged.disconnect(self.setActivatedIndex)
        self._model = model
        if model is not None:
            model.activatedIndexChanged.connect(self.setActivatedIndex)

    def setDisplay(self, display: Optional[MainDisplayWindow_V2]):
        oldDisplay = self._display
        if oldDisplay is not None:
            oldDisplay.setPlayer(None)
            oldDisplay.cameraActiveChanged.disconnect(self.setCameraActive)
            self.arrayChanged.disconnect(oldDisplay.setArray)
        self._display = display
        if display is not None:
            display.setPlayer(self._videoPlayer)
            display.cameraActiveChanged.connect(self.setCameraActive)
            self.arrayChanged.connect(display.setArray)

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            worker = model.worker(index)
        else:
            worker = None
        self._imageProcessor.setWorker(worker)

    @Slot(bool)
    def setCameraActive(self, active: bool):
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

    @Slot(np.ndarray)
    def _displayImage(self, array: npt.NDArray[np.uint8]):
        processor = self._imageProcessor
        if not processor.ready():
            return
        self._processRequested.emit(array)

    def stop(self):
        self._processorThread.quit()
        self._processorThread.wait()
