from araviq6 import MediaController
import numpy as np
import numpy.typing as npt
import os
from dipcoatimage.finitedepth import ExperimentKind
from dipcoatimage.finitedepth_gui.core import (
    VisualizationMode,
    DataMember,
    FrameSource,
    DataArgFlag,
)
from dipcoatimage.finitedepth_gui.worker import WorkerUpdateFlag
from dipcoatimage.finitedepth_gui.model import (
    IndexRole,
    ExperimentDataModel,
    ExperimentSignalBlocker,
)
from dipcoatimage.finitedepth_gui.views import ROIDrawFlag
from PySide6.QtCore import Signal, Slot, Qt, QUrl, QModelIndex
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PySide6.QtMultimedia import QCamera, QImageCapture, QMediaRecorder, QMediaPlayer
from typing import Optional
from .toolbar import DisplayWidgetToolBar
from .roidisplay import NDArrayROILabel


__all__ = [
    "MainDisplayWindow",
]


class MainDisplayWindow(QMainWindow):
    """
    Window to display the visualization result.

    This widget is only a view. Visualization must be done by some other object.
    """

    visualizationModeChanged = Signal(VisualizationMode)
    videoRecorded = Signal(QUrl)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._currentModelIndex = QModelIndex()
        self._exptKind = ExperimentKind.NULL
        self._currentView = DataMember.NULL
        self._frameSource = FrameSource.NULL

        self._displayLabel = NDArrayROILabel()
        self._videoController = MediaController()
        self._displayToolBar = DisplayWidgetToolBar()

        self._displayLabel.setAlignment(Qt.AlignCenter)
        self._videoController.setVisible(False)

        self._displayToolBar.visualizationModeChanged.connect(
            self.visualizationModeChanged
        )
        self._displayToolBar.imageCaptured.connect(self._onImageCapture)
        self._displayToolBar.videoRecorded.connect(self._onVideoRecord)

        self.addToolBar(self._displayToolBar)
        layout = QVBoxLayout()
        layout.addWidget(self._displayLabel)
        layout.addWidget(self._videoController)
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def setActivatedIndex(self, index: QModelIndex):
        self._currentModelIndex = index
        self._displayLabel.setActivatedIndex(index)

    def setExperimentKind(self, exptKind: ExperimentKind):
        controllerVisible = self.isExperimentVideo(
            self._frameSource, self._currentView, exptKind
        )
        self._videoController.setVisible(controllerVisible)
        self._exptKind = exptKind

    def setCurrentView(self, currentView: DataMember):
        controllerVisible = self.isExperimentVideo(
            self._frameSource, currentView, self._exptKind
        )
        self._videoController.setVisible(controllerVisible)
        self._currentView = currentView

    def setFrameSource(self, frameSource: FrameSource):
        controllerVisible = self.isExperimentVideo(
            frameSource, self._currentView, self._exptKind
        )
        self._videoController.setVisible(controllerVisible)
        self._frameSource = frameSource

    @staticmethod
    def isExperimentVideo(
        frameSource: FrameSource,
        currentView: DataMember,
        exptKind: ExperimentKind,
    ) -> bool:
        viewExclude = [
            DataMember.REFERENCE,
            DataMember.SUBSTRATE,
        ]
        ret = (
            frameSource != FrameSource.CAMERA
            and currentView not in viewExclude
            and exptKind == ExperimentKind.VIDEO
        )
        return ret

    @Slot(np.ndarray)
    def setArray(self, array: npt.NDArray[np.uint8]):
        self._displayLabel.setArray(array)

    def setPlayer(self, player: Optional[QMediaPlayer]):
        self._videoController.setPlayer(player)

    def setImageCapture(self, imageCapture: Optional[QImageCapture]):
        self._displayToolBar.setImageCapture(imageCapture)

    def setMediaRecorder(self, mediaRecorder: Optional[QMediaRecorder]):
        self._displayToolBar.setMediaRecorder(mediaRecorder)

    def setCamera(self, camera: Optional[QCamera]):
        self._displayToolBar.setCamera(camera)

    @Slot(ROIDrawFlag)
    def setROIDrawFlag(self, flag: ROIDrawFlag):
        self._displayLabel.setROIDrawFlag(flag)

    @Slot(str)
    def _onImageCapture(self, path: str):
        index = self._currentModelIndex
        if not index.isValid():
            return
        model = index.model()
        if not isinstance(model, ExperimentDataModel):
            return
        path = os.path.relpath(path)
        if self._currentView in (DataMember.REFERENCE, DataMember.SUBSTRATE):
            refPathIdx = model.getIndexFor(IndexRole.REFPATH, index)
            model.setData(refPathIdx, path, model.Role_RefPath)
        else:
            coatPathsIdx = model.getIndexFor(IndexRole.COATPATHS, index)
            with ExperimentSignalBlocker(model):
                row = model.rowCount(coatPathsIdx)
                model.insertRows(row, 1, coatPathsIdx)
                pathIdx = model.index(row, 0, coatPathsIdx)
                model.setData(pathIdx, path, role=model.Role_CoatPath)
            model.updateWorker(index, WorkerUpdateFlag.ANALYSIS)
            model.emitExperimentDataChanged(index, DataArgFlag.COATPATHS)

    @Slot(str)
    def _onVideoRecord(self, url: QUrl):
        index = self._currentModelIndex
        if not index.isValid():
            return
        model = index.model()
        if not isinstance(model, ExperimentDataModel):
            return
        path = os.path.relpath(url.toLocalFile())

        if self._currentView in (DataMember.REFERENCE, DataMember.SUBSTRATE):
            pass
        else:
            coatPathsIdx = model.getIndexFor(IndexRole.COATPATHS, index)
            with ExperimentSignalBlocker(model):
                row = model.rowCount(coatPathsIdx)
                model.insertRows(row, 1, coatPathsIdx)
                pathIdx = model.index(row, 0, coatPathsIdx)
                model.setData(pathIdx, path, role=model.Role_CoatPath)
            model.updateWorker(index, WorkerUpdateFlag.ANALYSIS)
            model.emitExperimentDataChanged(index, DataArgFlag.COATPATHS)
