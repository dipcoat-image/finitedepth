from araviq6 import NDArrayVideoPlayer
import cv2  # type: ignore[import]
import enum
import numpy as np
import numpy.typing as npt
from dipcoatimage.finitedepth import SubstrateReferenceBase
from dipcoatimage.finitedepth_gui.core import ClassSelection, DataMember
from dipcoatimage.finitedepth_gui.worker import ExperimentWorker
from dipcoatimage.finitedepth_gui.workers import MasterWorker
from PySide6.QtCore import QUrl, Signal, Slot, QObject
from typing import Optional


__all__ = [
    "PreviewableNDArrayVideoPlayer",
    "FrameSource",
    "VisualizeProcessor_V2",
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
        self._ready = True

    def visualizeWorker(self) -> Optional[MasterWorker]:
        return self._worker

    def selectedClass(self) -> ClassSelection:
        return self._selectedClass

    def ready(self) -> bool:
        return self._ready

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
        array = array.copy()  # must detach array from the memory
        self._ready = False
        self.arrayChanged.emit(self.processArray(array))
        self._ready = True

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
        elif select == ClassSelection.SUBSTRATE:
            ret = worker.substrateWorker().visualizedImage()
        else:
            ret = worker.experimentWorker().visualizeImage(
                worker.experimentWorker().coatingLayerImage()
            )
        self.arrayChanged.emit(ret)


class FrameSource(enum.Enum):
    NULL = 0
    FILE = 1
    CAMERA = 2


class VisualizeProcessor_V2(QObject):
    arrayChanged = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._currentView = DataMember.EXPERIMENT
        self._frameSource = FrameSource.NULL
        self._ready = True

    def setWorker(self, worker: Optional[ExperimentWorker]):
        self._worker = worker

    def setCurrentView(self, currentView: DataMember):
        self._currentView = currentView

    def setFrameSource(self, frameSource: FrameSource):
        self._frameSource = frameSource

    @Slot(np.ndarray)
    def setArray(self, array: npt.NDArray[np.uint8]):
        array = array.copy()  # must detach array from the memory
        self._ready = False
        self.arrayChanged.emit(self.processArray(array))
        self._ready = True

    def processArray(self, array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if self._worker is None:
            return array
        if self._currentView == DataMember.REFERENCE:
            self._worker.setReferenceImage(array)
            reference = self._worker.reference
            if reference is not None:
                array = reference.draw()
            else:
                array = self._worker.referenceImage
        elif self._currentView == DataMember.SUBSTRATE:
            self._worker.setReferenceImage(array)
            substrate = self._worker.substrate
            if substrate is not None:
                array = substrate.draw()
            else:
                h, w = array.shape[:2]
                substROI = self._worker.exptData.reference.substrateROI
                x0, y0, x1, y1 = SubstrateReferenceBase.sanitize_ROI(substROI, h, w)
                array = array[y0:y1, x0:x1]
        else:
            expt = self._worker.experiment
            if expt is not None:
                if array.size > 0:
                    layer = expt.construct_coatinglayer(array)
                    if layer.valid():
                        array = layer.draw()
        return array

    def ready(self) -> bool:
        return self._ready
