import cv2  # type: ignore[import]
from cv2PySide6 import (
    NDArrayVideoPlayer,
    NDArrayVideoPlayerWidget,
    NDArrayCameraWidget,
)
from dipcoatimage.finitedepth.analysis import ExperimentKind
from dipcoatimage.finitedepth_gui.core import ClassSelection
from dipcoatimage.finitedepth_gui.roimodel import ROIModel
from dipcoatimage.finitedepth_gui.workers import (
    VisualizationMode,
)
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Signal, QUrl, Qt, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QWidget
from .roidisplay import NDArrayROILabel
from .toolbar import DisplayWidgetToolBar


__all__ = [
    "PreviewableNDArrayVideoPlayer",
    "ROIVideoWidget",
    "ROICameraWidget",
    "MainDisplayWindow",
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
        cap = cv2.VideoCapture(path)
        ok, img = cap.read()
        cap.release()
        if not ok:
            img = np.empty((0, 0, 0))
        return img


class ROIVideoWidget(NDArrayVideoPlayerWidget):
    def __init__(self, parent=None):
        self._roiVideoLabel = NDArrayROILabel()
        super().__init__(parent)

    def videoLabel(self) -> NDArrayROILabel:
        return self._roiVideoLabel


class ROICameraWidget(NDArrayCameraWidget):
    def __init__(self, parent=None):
        self._roiVideoLabel = NDArrayROILabel()
        super().__init__(parent)

    def videoLabel(self) -> NDArrayROILabel:
        return self._roiVideoLabel


class MainDisplayWindow(QMainWindow):
    """Main window which includes various display widgets."""

    visualizationModeChanged = Signal(VisualizationMode)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._selection = ClassSelection.EXPERIMENT
        self._expt_kind = ExperimentKind.VideoExperiment
        self._display_widget = QStackedWidget()
        self._img_display = NDArrayROILabel()
        self._vid_display = ROIVideoWidget()
        self._camera_display = ROICameraWidget()
        self._display_toolbar = DisplayWidgetToolBar()

        self.displayToolBar().visualizationModeChanged.connect(
            self.visualizationModeChanged
        )

        self.imageDisplayWidget().setAlignment(Qt.AlignCenter)
        self.videoDisplayWidget().setVideoPlayer(PreviewableNDArrayVideoPlayer(self))
        self.addToolBar(self.displayToolBar())
        self.setCentralWidget(self.displayStackWidget())
        self.centralWidget().addWidget(self.imageDisplayWidget())
        self.centralWidget().addWidget(self.videoDisplayWidget())
        self.centralWidget().addWidget(self.cameraDisplayWidget())

    def classSelection(self) -> ClassSelection:
        return self._selection

    def experimentKind(self) -> ExperimentKind:
        return self._expt_kind

    def displayStackWidget(self) -> QStackedWidget:
        return self._display_widget

    def imageDisplayWidget(self) -> NDArrayROILabel:
        """Widget to display single frame image."""
        return self._img_display

    def videoDisplayWidget(self) -> ROIVideoWidget:
        """Widget to display video."""
        return self._vid_display

    def cameraDisplayWidget(self) -> ROICameraWidget:
        """Widget to display camera stream."""
        return self._camera_display

    def displayToolBar(self) -> DisplayWidgetToolBar:
        """Toolbar to control display options."""
        return self._display_toolbar

    @Slot(ClassSelection)
    def setClassSelection(self, select: ClassSelection):
        self._selection = select
        self.exposeDisplayWidget(self.experimentKind(), self.classSelection())

    def exposeDisplayWidget(self, kind: ExperimentKind, selection: ClassSelection):
        """Determine the widget for *select* and expose to central area."""
        self.videoDisplayWidget().videoPlayer().pause()
        if (
            selection == ClassSelection.EXPERIMENT
            and kind == ExperimentKind.VideoExperiment
        ):
            widget: QWidget = self.videoDisplayWidget()
        else:
            widget = self.imageDisplayWidget()
        self.displayStackWidget().setCurrentWidget(widget)

    def exposedDisplayWidget(self) -> QWidget:
        """Return the display widget exposed to central area."""
        return self.displayStackWidget().currentWidget()

    def currentDisplayingLabel(self) -> NDArrayROILabel:
        """Return the displaying label in current widget."""
        widget = self.exposedDisplayWidget()
        if isinstance(widget, ROIVideoWidget):
            label = widget.videoLabel()
        elif isinstance(widget, ROICameraWidget):
            label = widget.videoLabel()
        else:
            raise TypeError("Unknown widget.")
        return label

    @Slot(np.ndarray)
    def displayImage(self, img: npt.NDArray[np.uint8]):
        """Display the image to :meth:`exposedDisplayWidget`."""
        label = self.currentDisplayingLabel()
        if img.size == 0:
            label.setPixmap(QPixmap())
        else:
            label.setArray(img)

    @Slot(ROIModel, bool)
    def toggleROIDraw(self, model: ROIModel, state: bool):
        label = self.currentDisplayingLabel()
        if state:
            label.addROIModel(model)
        else:
            label.removeROIModel(model)

    def setVisualizeActionToggleState(self, mode: VisualizationMode):
        self.displayToolBar().setVisualizeActionToggleState(mode)
