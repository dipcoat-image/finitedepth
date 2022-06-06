from araviq6 import NDArrayVideoPlayer
import cv2  # type: ignore[import]
from dipcoatimage.finitedepth_gui.core import ClassSelection
from dipcoatimage.finitedepth_gui.workers import MasterWorker
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QUrl, Signal, Slot, QObject
from typing import Optional


__all__ = [
    "PreviewableNDArrayVideoPlayer",
    "VisualizeProcessor",
]


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


class VisualizeProcessor(QObject):
    """
    Video pipeline component to set the image to worker and visualize.
    """

    arrayChanged = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._selectedClass = ClassSelection.EXPERIMENT

    def visualizeWorker(self) -> Optional[MasterWorker]:
        return self._worker

    def selectedClass(self) -> ClassSelection:
        return self._selectedClass

    def setVisualizeWorker(self, worker: Optional[MasterWorker]):
        self._worker = worker

    def setSelectedClass(self, select: ClassSelection):
        self._selectedClass = select

    @Slot(np.ndarray)
    def setArray(self, array: npt.NDArray[np.uint8]):
        """
        Process *array* with :meth:`processArray` and emit to
        :attr:`arrayChanged`.
        """
        self.arrayChanged.emit(self.processArray(array))

    def processArray(self, array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        # only the selected class is updated, e.g. updated reference instance is not
        # applied to substrate worker when visualizing reference.
        worker = self.visualizeWorker()
        if worker is None:
            ret = array
        elif self.selectedClass() == ClassSelection.REFERENCE:
            worker.referenceWorker().setImage(array)
            worker.referenceWorker().updateReference()
            ret = worker.referenceWorker().visualizedImage()
        elif self.selectedClass() == ClassSelection.SUBSTRATE:
            worker.referenceWorker().setImage(array)
            worker.referenceWorker().updateReference()
            worker.substrateWorker().setReference(worker.referenceWorker().reference())
            worker.substrateWorker().updateSubstrate()
            ret = worker.substrateWorker().visualizedImage()
        else:
            ret = worker.experimentWorker().visualizeImage(array)
        return ret

    def emitVisualizationFromModel(self, select: ClassSelection):
        worker = self.visualizeWorker()
        if worker is None:
            return
        if select == ClassSelection.REFERENCE:
            ret = worker.referenceWorker().visualizedImage()
            self.arrayChanged.emit(ret)
        elif select == ClassSelection.SUBSTRATE:
            ret = worker.substrateWorker().visualizedImage()
            self.arrayChanged.emit(ret)
