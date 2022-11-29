import cv2  # type: ignore[import]
import numpy as np
import numpy.typing as npt
from dipcoatimage.finitedepth.reference import sanitize_ROI
from dipcoatimage.finitedepth.coatinglayer import match_template, subtract_images
from dipcoatimage.finitedepth.util import OptionalROI, binarize
from dipcoatimage.finitedepth_gui.core import (
    DataMember,
    VisualizationMode,
)
from dipcoatimage.finitedepth_gui.worker import ExperimentWorker
from PySide6.QtCore import QObject, Signal, Slot
from typing import Optional


__all__ = [
    "ImageProcessor",
    "fastVisualize",
]


class ImageProcessor(QObject):
    """Object to process the incoming image from video or camera."""

    arrayChanged = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._prev = None
        self._currentView = DataMember.NULL
        self._visualizeMode = VisualizationMode.OFF
        self._ready = True

    def setWorker(self, worker: Optional[ExperimentWorker]):
        self._worker = worker
        self._prev = None

    def setCurrentView(self, currentView: DataMember):
        self._currentView = currentView

    def setVisualizationMode(self, mode: VisualizationMode):
        self._visualizeMode = mode

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

    def ready(self) -> bool:
        return self._ready


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

    ret = subtract_images(layer_bin, substImg, (x0, y0))
    return cv2.cvtColor(ret, cv2.COLOR_GRAY2RGB)
