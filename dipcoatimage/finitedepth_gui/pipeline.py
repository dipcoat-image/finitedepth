"""
Video streaming pipeline
"""

from araviq6 import NDArrayVideoPlayer
import cv2  # type: ignore[import]
import enum
import numpy as np
import numpy.typing as npt
from dipcoatimage.finitedepth_gui.core import DataMember
from dipcoatimage.finitedepth_gui.worker import ExperimentWorker
from PySide6.QtCore import QObject, Signal, Slot, QUrl, QThread
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
    def __init__(self, parent=None):
        super().__init__(parent)

        self._processorThread = QThread()
        self._imageProcessor = ImageProcessor()

        self._imageProcessor.moveToThread(self._processorThread)
        self._processorThread.start()

    def stop(self):
        self._processorThread.quit()
        self._processorThread.wait()
