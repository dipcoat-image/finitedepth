"""
Video streaming pipeline
"""

from araviq6 import NDArrayVideoPlayer, NDArrayMediaCaptureSession
import cv2  # type: ignore[import]
import numpy as np
import numpy.typing as npt
from dipcoatimage.finitedepth import ExperimentKind, experiment_kind
from dipcoatimage.finitedepth.reference import sanitize_ROI
from dipcoatimage.finitedepth.util import OptionalROI
from dipcoatimage.finitedepth_gui.core import (
    DataMember,
    DataArgs,
    FrameSource,
    VisualizationMode,
)
from dipcoatimage.finitedepth_gui.worker import ExperimentWorker
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from PySide6.QtCore import QObject, Signal, Slot, QUrl, QThread, QModelIndex
from PySide6.QtMultimedia import QCamera, QMediaPlayer
from typing import Optional, Protocol


__all__ = [
    "ImageProcessor",
    "DisplayProtocol",
    "VisualizeManager",
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


class DisplayProtocol(Protocol):
    def setActivatedIndex(self, index: QModelIndex):
        ...

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
    roiMaximumChanged = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None
        self._exptKind = ExperimentKind.NULL
        self._frameSource = FrameSource.NULL
        self._currentView = DataMember.NULL
        self._visualizeMode = VisualizationMode.OFF
        self._videoPlayer = NDArrayVideoPlayer()
        self._lastVideoFrame = np.empty((0, 0, 0), dtype=np.uint8)
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
            if worker is None:
                coatPaths = []
            else:
                coatPaths = worker.exptData.coat_paths
        else:
            worker = None
            coatPaths = []
        self._imageProcessor.setWorker(worker)

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
            self._videoPlayer.arrayChanged.disconnect(self._displayImageFromVideo)
        exptKind = experiment_kind(coatPaths)
        if exptKind == ExperimentKind.VIDEO:
            if self._frameSource == FrameSource.FILE and self._currentView not in (
                DataMember.REFERENCE,
                DataMember.SUBSTRATE,
            ):
                self._videoPlayer.arrayChanged.connect(self._displayImageFromVideo)
            source = QUrl.fromLocalFile(coatPaths[0])
            self._videoPlayer.setSource(source)
        else:
            self._videoPlayer.setSource(QUrl())
        self._exptKind = exptKind

        if self._frameSource == FrameSource.CAMERA:
            pass
        elif worker is None:
            img = np.empty((0, 0, 0), dtype=np.uint8)
            self.arrayChanged.emit(img)
        else:
            self._displayFromWorker(worker)

        display = self.display()
        if display is not None:
            display.setExperimentKind(exptKind)
            display.setActivatedIndex(index)

    @Slot(QModelIndex, DataArgs)
    def _onExptDataChange(self, index: QModelIndex, flag: DataArgs):
        model = index.model()
        if not isinstance(model, ExperimentDataModel):
            return
        if index != model.activatedIndex():
            return
        worker = model.worker(index)
        if worker is None:
            return

        if flag & DataArgs.COATPATHS:
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
                self._videoPlayer.arrayChanged.disconnect(self._displayImageFromVideo)
            coatPaths = worker.exptData.coat_paths
            exptKind = experiment_kind(coatPaths)
            if exptKind == ExperimentKind.VIDEO:
                if self._frameSource == FrameSource.FILE and self._currentView not in (
                    DataMember.REFERENCE,
                    DataMember.SUBSTRATE,
                ):
                    self._videoPlayer.arrayChanged.connect(self._displayImageFromVideo)
                source = QUrl.fromLocalFile(coatPaths[0])
                self._videoPlayer.setSource(source)
            else:
                self._videoPlayer.setSource(QUrl())
            self._exptKind = exptKind
            display = self.display()
            if display is not None:
                display.setExperimentKind(exptKind)

        self._imageProcessor.setWorker(worker)

        if self._frameSource == FrameSource.CAMERA:
            return

        if self._currentView == DataMember.REFERENCE:
            if flag & (DataArgs.REFPATH | DataArgs.REFERENCE):
                self._displayFromWorker(worker)
        elif self._currentView == DataMember.SUBSTRATE:
            if flag & (DataArgs.REFPATH | DataArgs.REFERENCE | DataArgs.SUBSTRATE):
                self._displayFromWorker(worker)
        else:
            if flag & (
                DataArgs.COATPATHS
                | DataArgs.REFPATH
                | DataArgs.REFERENCE
                | DataArgs.SUBSTRATE
                | DataArgs.COATINGLAYER
                | DataArgs.EXPERIMENT
            ):
                self._displayFromWorker(worker)

    def camera(self) -> Optional[QCamera]:
        return self._camera

    def setCamera(self, camera: Optional[QCamera]):
        oldCamera = self.camera()
        if oldCamera is not None:
            oldCamera.activeChanged.disconnect(  # type: ignore[attr-defined]
                self._onCameraActiveChange
            )
        self._camera = camera
        self._captureSession.setCamera(camera)
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
            self._captureSession.arrayChanged.disconnect(self._displayImageFromCamera)
        elif oldSource == FrameSource.FILE:
            if self._exptKind == ExperimentKind.VIDEO and self._currentView not in (
                DataMember.REFERENCE,
                DataMember.SUBSTRATE,
            ):
                self._videoPlayer.arrayChanged.disconnect(self._displayImageFromVideo)
        else:
            pass
        self._frameSource = frameSource
        if frameSource == FrameSource.CAMERA:
            self._captureSession.arrayChanged.connect(self._displayImageFromCamera)
        elif frameSource == FrameSource.FILE:
            if self._exptKind == ExperimentKind.VIDEO and self._currentView not in (
                DataMember.REFERENCE,
                DataMember.SUBSTRATE,
            ):
                self._videoPlayer.arrayChanged.connect(self._displayImageFromVideo)
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

        display = self.display()
        if display is not None:
            display.setFrameSource(frameSource)

    @Slot(DataMember)
    def setCurrentView(self, currentView: DataMember):
        def isExptView(view: DataMember):
            return view not in (DataMember.REFERENCE, DataMember.SUBSTRATE)

        oldView = self._currentView
        if isExptView(oldView) and not isExptView(currentView):
            if (
                self._frameSource == FrameSource.FILE
                and self._exptKind == ExperimentKind.VIDEO
            ):
                self._videoPlayer.arrayChanged.disconnect(self._displayImageFromVideo)
            if (
                self._videoPlayer.playbackState()
                == QMediaPlayer.PlaybackState.PlayingState
            ):
                self._videoPlayer.pause()
        elif not isExptView(oldView) and isExptView(currentView):
            if (
                self._frameSource == FrameSource.FILE
                and self._exptKind == ExperimentKind.VIDEO
            ):
                self._videoPlayer.arrayChanged.connect(self._displayImageFromVideo)
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
            self._displayFromWorker(worker)

        display = self.display()
        if display is not None:
            display.setCurrentView(currentView)

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
                if state == QMediaPlayer.PlaybackState.StoppedState:
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
    def _displayImageFromVideo(self, array: npt.NDArray[np.uint8]):
        self._lastVideoFrame = array.copy()
        if array.size != 0:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        if self._currentView == DataMember.REFERENCE:
            h, w = array.shape[:2]
            self.roiMaximumChanged.emit(h, w)
        self._displayImage(array)

    @Slot(np.ndarray)
    def _displayImageFromCamera(self, array: npt.NDArray[np.uint8]):
        if array.size != 0:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        if self._currentView == DataMember.REFERENCE:
            h, w = array.shape[:2]
            self.roiMaximumChanged.emit(h, w)
        self._displayImage(array)

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


def crop(img: npt.NDArray[np.uint8], roi: OptionalROI):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = sanitize_ROI(roi, h, w)
    return img[y0:y1, x0:x1]


def fastVisualize(
    refImg: npt.NDArray[np.uint8],
    layerImg: npt.NDArray[np.uint8],
    tempROI: OptionalROI,
    substROI: OptionalROI,
):
    if len(refImg.shape) == 2:
        ref_gray = refImg
    elif len(refImg.shape) == 3:
        ch = refImg.shape[-1]
        if ch == 1:
            ref_gray = refImg
        elif ch == 3:
            ref_gray = cv2.cvtColor(refImg, cv2.COLOR_RGB2GRAY)
        elif ch == 4:
            ref_gray = cv2.cvtColor(refImg, cv2.COLOR_RGBA2GRAY)
        else:
            raise TypeError(f"Reference image with invalid channel: {refImg.shape}")
    else:
        raise TypeError(f"Invalid reference image shape: {refImg.shape}")
    _, ref_bin = cv2.threshold(ref_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if ref_bin is None:
        return layerImg

    if len(layerImg.shape) == 2:
        gray = layerImg
    elif len(layerImg.shape) == 3:
        ch = layerImg.shape[-1]
        if ch == 1:
            gray = layerImg
        elif ch == 3:
            gray = cv2.cvtColor(layerImg, cv2.COLOR_RGB2GRAY)
        elif ch == 4:
            gray = cv2.cvtColor(layerImg, cv2.COLOR_RGBA2GRAY)
        else:
            raise TypeError(f"Layer image with invalid channel: {layerImg.shape}")
    else:
        raise TypeError(f"Invalid layer image shape: {layerImg.shape}")
    _, layer_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if layer_bin is None:
        return layerImg

    h, w = refImg.shape[:2]
    tempROI = sanitize_ROI(tempROI, h, w)
    substROI = sanitize_ROI(substROI, h, w)

    temp_x0, temp_y0, temp_x1, temp_y1 = tempROI
    template = ref_bin[temp_y0:temp_y1, temp_x0:temp_x1]
    subst_x0, subst_y0, subst_x1, subst_y1 = substROI
    substImg = ref_bin[subst_y0:subst_y1, subst_x0:subst_x1]

    res = cv2.matchTemplate(layer_bin, template, cv2.TM_SQDIFF_NORMED)
    _, _, (tx, ty), _ = cv2.minMaxLoc(res)
    dx, dy = (substROI[0] - tempROI[0], substROI[1] - tempROI[1])
    x0, y0 = (tx + dx, ty + dy)
    subst_h, subst_w = substImg.shape[:2]
    x1, y1 = (x0 + subst_w, y0 + subst_h)

    H, W = layer_bin.shape
    img_cropped = layer_bin[max(y0, 0) : min(y1, H), max(x0, 0) : min(x1, W)]
    subst_cropped = substImg[
        max(-y0, 0) : min(H - y0, subst_h),
        max(-x0, 0) : min(W - x0, subst_w),
    ]

    xor = cv2.bitwise_xor(img_cropped, subst_cropped)
    nxor = cv2.bitwise_not(xor)
    layer_bin[max(y0, 0) : min(y1, H), max(x0, 0) : min(x1, W)] = nxor
    return cv2.cvtColor(layer_bin, cv2.COLOR_GRAY2RGB)
