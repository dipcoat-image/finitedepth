"""
Video streaming pipeline
"""

import enum
import numpy as np
import numpy.typing as npt
from dipcoatimage.finitedepth_gui.core import DataMember
from dipcoatimage.finitedepth_gui.worker import ExperimentWorker
from PySide6.QtCore import QObject, Signal, Slot
from typing import Optional


__all__ = [
    "FrameSource",
    "VisualizeProcessor_V2",
]


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
