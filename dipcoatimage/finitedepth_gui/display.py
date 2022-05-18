"""
Displaying widgets
==================

This module provides the widgets to display the visualized results.

"""
import cv2  # type: ignore[import]
from cv2PySide6 import (
    NDArrayVideoPlayer,
    ArrayProcessor,
    NDArrayLabel,
    NDArrayVideoPlayerWidget,
    NDArrayCameraWidget,
)
import dipcoatimage.finitedepth_gui
import numpy as np
import numpy.typing as npt
import os
from PySide6.QtCore import Signal, QUrl, QSize, QRect, QPoint, Qt, Slot
from PySide6.QtGui import (
    QPaintEvent,
    QMouseEvent,
    QPainter,
    QBrush,
    QColor,
    QActionGroup,
    QAction,
    QIcon,
    QPixmap,
)
from PySide6.QtWidgets import QToolBar, QMainWindow, QStackedWidget, QWidget, QLabel
from typing import Tuple, List
from .roimodel import ROIModel
from .workers import ReferenceWorker, SubstrateWorker, ExperimentWorker


__all__ = [
    "PreviewableNDArrayVideoPlayer",
    "ReferenceArrayProcessor",
    "SubstrateArrayProcessor",
    "ExperimentArrayProcessor",
    "NDArrayROILabel",
    "coords_label2pixmap",
    "coords_pixmap2label",
    "DisplayWidgetToolBar",
    "get_icons_path",
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


class ReferenceArrayProcessor(ArrayProcessor):
    """
    Video pipeline component to convert bare substrate frame into
    visualized reference image.

    """

    imageSizeChanged = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._visualize_mode = True
        self._ref_worker = ReferenceWorker()
        self._imageSize = (0, 0)

    def visualizationMode(self) -> bool:
        """If False, frame is not passed to worker."""
        return self._visualize_mode

    def setVisualizationMode(self, mode: bool):
        """Update :meth:`visualizationMode` with *mode*."""
        self._visualize_mode = mode

    def referenceWorker(self) -> ReferenceWorker:
        return self._ref_worker

    def setReferenceWorker(self, worker: ReferenceWorker):
        self._ref_worker = worker

    def imageSize(self) -> Tuple[int, int]:
        return self._imageSize

    def processArray(self, array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        h, w = array.shape[:2]
        newsize = (w, h)
        if newsize != self.imageSize():
            self._imageSize = newsize
            self.imageSizeChanged.emit(w, h)

        if self.visualizationMode():
            if array.size == 0:
                gray = None
            elif len(array.shape) == 2 or array.shape[2] == 1:
                gray = array
            elif array.shape[2] == 3:
                gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
            self.referenceWorker().setImage(gray)
            self.referenceWorker().updateReference()
            ret = self.referenceWorker().visualizedImage()
        else:
            ret = array
        return ret


class SubstrateArrayProcessor(ArrayProcessor):
    """
    Video pipeline component to convert bare substrate frame into
    visualized substrate image.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._visualize_mode = True
        self._ref_worker = ReferenceWorker()
        self._subst_worker = SubstrateWorker()

    def visualizationMode(self) -> bool:
        """If False, frame is not passed to worker."""
        return self._visualize_mode

    def setVisualizationMode(self, mode: bool):
        """Update :meth:`visualizationMode` with *mode*."""
        self._visualize_mode = mode

    def referenceWorker(self) -> ReferenceWorker:
        return self._ref_worker

    def setReferenceWorker(self, worker: ReferenceWorker):
        self._ref_worker = worker

    def substrateWorker(self) -> SubstrateWorker:
        return self._subst_worker

    def setSubstrateWorker(self, worker: SubstrateWorker):
        self._subst_worker = worker

    def processArray(self, array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if self.visualizationMode():
            if array.size == 0:
                gray = None
            elif len(array.shape) == 2 or array.shape[2] == 1:
                gray = array
            elif array.shape[2] == 3:
                gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
            self.referenceWorker().setImage(gray)
            self.referenceWorker().updateReference()
            self.substrateWorker().setReference(self.referenceWorker().reference())
            self.substrateWorker().updateSubstrate()
            ret = self.substrateWorker().visualizedImage()
        else:
            ret = array
        return ret


class ExperimentArrayProcessor(ArrayProcessor):
    """
    Video pipeline component to convert coated substrate frame into
    visualized layer shape image.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._visualize_mode = True
        self._expt_worker = ExperimentWorker()

    def visualizationMode(self) -> bool:
        """If False, frame is not passed to worker."""
        return self._visualize_mode

    def setVisualizationMode(self, mode: bool):
        """Update :meth:`visualizationMode` with *mode*."""
        self._visualize_mode = mode

    def experimentWorker(self) -> ExperimentWorker:
        return self._expt_worker

    def setExperimentWorker(self, worker: ExperimentWorker):
        self._expt_worker = worker

    def processArray(self, array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if self.visualizationMode():
            if array.size == 0:
                gray = None
            elif len(array.shape) == 2 or array.shape[2] == 1:
                gray = array
            elif array.shape[2] == 3:
                gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
            self.experimentWorker().setImage(gray)
            ret = self.experimentWorker().visualizedImage()
        else:
            ret = array
        return ret


class NDArrayROILabel(NDArrayLabel):
    """
    ``cv2PySide6.NDArrayLabel`` which multiple ROIs can be set.

    .. rubric:: Model

    :class:`ROIModel` instance in :meth:`roiModels` are controlled and
    viewed by this class. Models can be added by :meth:`addROIModel`
    or :meth:`insertROIModel`. :meth:`removeROIModel` removes the model
    from :meth:`roiModels`.

    Change of :meth:`roiModels` emits :attr:`roiModelsChanged` signal.

    .. rubric:: View

    Each ROI data of added :class:`ROIModel` instances are displayed as
    rectangle on the label. When the label is in drawing state (i.e.
    mouse pressed and being moved), individual ROIs are not displayed.
    Instead new ROI candidate which is being drawn is displayed.

    .. rubric:: Control

    Clicking and dragging on the label's viewport sets the value of all
    added :class:`ROIModel` instances at the point of mouse release.

    Examples
    ========

    >>> import cv2
    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth import get_samples_path
    >>> from dipcoatimage.finitedepth_gui.roimodel import ROIModel
    >>> from dipcoatimage.finitedepth_gui.display import NDArrayROILabel
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     widget = NDArrayROILabel()
    ...     widget.setArray(cv2.imread(get_samples_path('coat1.png')))
    ...     widget.addROIModel(ROIModel())
    ...     widget.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    roiModelsChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._roi_models = []
        self._drawing = False
        self._temp_roi = (0, 0, 0, 0)  # displayed roi of original image

    def labelROI2OriginalROI(
        self, roi: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Convert ROI on the label to ROI on the original image."""
        # convert to roi of scaled pixmap
        x1, y1, x2, y2 = roi
        pixmap_size = self.pixmap().size()
        px1, py1 = coords_label2pixmap(
            (x1, y1), self.size(), pixmap_size, self.alignment()
        )
        px2, py2 = coords_label2pixmap(
            (x2, y2), self.size(), pixmap_size, self.alignment()
        )
        # convert to roi of original pixmap
        w, h = pixmap_size.width(), pixmap_size.height()
        original_size = self._original_pixmap.size()
        W, H = original_size.width(), original_size.height()
        if W == 0 or H == 0:
            ret = (0.0, 0.0, 0.0, 0.0)
        else:
            x1 = max(px1 / w * W, 0.0)
            y1 = max(py1 / h * H, 0.0)
            x2 = min(px2 / w * W, float(W))
            y2 = min(py2 / h * H, float(H))
            ret = (x1, y1, x2, y2)
        return ret

    def originalROI2LabelROI(
        self, roi: Tuple[int, int, int, int]
    ) -> Tuple[float, float, float, float]:
        """Convert ROI on the original image to ROI on the label."""
        # convert to roi of scaled pixmap
        x1, y1, x2, y2 = roi
        pixmap_size = self.pixmap().size()
        w, h = pixmap_size.width(), pixmap_size.height()
        original_size = self._original_pixmap.size()
        W, H = original_size.width(), original_size.height()
        if w == 0 or h == 0:
            px1, py1, px2, py2 = (0, 0, 0, 0)
        else:
            px1, py1, px2, py2 = (
                int(x1 / W * w),
                int(y1 / H * h),
                int(x2 / W * w),
                int(y2 / H * h),
            )
        # convert to roi of label
        lx1, ly1 = coords_pixmap2label(
            (px1, py1), pixmap_size, self.size(), self.alignment()
        )
        lx2, ly2 = coords_pixmap2label(
            (px2, py2), pixmap_size, self.size(), self.alignment()
        )
        return (lx1, ly1, lx2, ly2)

    def roiModels(self) -> List[ROIModel]:
        """Models to store ROI data."""
        return self._roi_models

    def addROIModel(self, model: ROIModel):
        """Append *model* to :meth:`roiModels`."""
        model.roiChanged.connect(self.update)
        self._roi_models.append(model)
        self.roiModelsChanged.emit(self.roiModels())
        self.update()

    def insertROIModel(self, index: int, model: ROIModel):
        """Insert *model* to :meth:`roiModels`."""
        model.roiChanged.connect(self.update)
        self._roi_models.insert(index, model)
        self.roiModelsChanged.emit(self.roiModels())
        self.update()

    def removeROIModel(self, model: ROIModel):
        """Remove *model* from :meth:`roiModels` without destroying."""
        model.roiChanged.disconnect(self.update)
        self._roi_models.remove(model)
        self.roiModelsChanged.emit(self.roiModels())
        self.update()

    def paintEvent(self, event: QPaintEvent):
        """
        If new ROI is being drawn, display it. Else, draw all ROIs from
        :meth:`roiModels`.
        """
        super().paintEvent(event)
        qp = QPainter(self)

        if not self._drawing:
            original_size = self._original_pixmap.size()
            W, H = original_size.width(), original_size.height()
            for model in self.roiModels():
                x1, y1, x2, y2 = model.roi()
                if x2 is None:
                    x2 = W
                if y2 is None:
                    y2 = H
                x1, y1, x2, y2 = [
                    int(i) for i in self.originalROI2LabelROI((x1, y1, x2, y2))
                ]
                br = QBrush(QColor(255, 0, 0, 50))
                qp.setBrush(br)
                qp.drawRect(QRect(QPoint(x1, y1), QPoint(x2, y2)))
        else:
            x1, y1, x2, y2 = [int(i) for i in self.originalROI2LabelROI(self._temp_roi)]
            for model in self.roiModels():
                br = QBrush(QColor(255, 0, 0, 50))
                qp.setBrush(br)
                qp.drawRect(QRect(QPoint(x1, y1), QPoint(x2, y2)))

    def mousePressEvent(self, event: QMouseEvent):
        """Start the drawing mode."""
        super().mousePressEvent(event)
        self._drawing = True
        pos = event.position()
        x, y = int(pos.x()), int(pos.y())
        roi = self.labelROI2OriginalROI((x, y, x, y))
        self._temp_roi = tuple(map(int, roi))
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Update ROI candidate."""
        super().mouseMoveEvent(event)
        if self._drawing:
            x1, y1, _, _ = self.originalROI2LabelROI(self._temp_roi)
            pos = event.position()
            x2, y2 = int(pos.x()), int(pos.y())
            roi = self.labelROI2OriginalROI((x1, y1, x2, y2))
            self._temp_roi = tuple(map(int, roi))
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        Terminate the drawing mode and update the drawn ROI to models
        in :meth:`roiModels`. ROI values are sorted that x1 is smaller
        than x2 and y1 is smaller than y2.
        """
        super().mouseReleaseEvent(event)
        x1, y1, _, _ = self.originalROI2LabelROI(self._temp_roi)
        pos = event.position()
        x2, y2 = int(pos.x()), int(pos.y())
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        roi = self.labelROI2OriginalROI((x1, y1, x2, y2))
        self._temp_roi = tuple(map(int, roi))

        for model in self.roiModels():
            model.setROI(*self._temp_roi)
        self._drawing = False
        self.update()


def coords_label2pixmap(
    p: Tuple[float, float], lsize: QSize, psize: QSize, alignment: Qt.Alignment
) -> Tuple[float, float]:
    """
    Convert the coordinates in ``QLabel`` to the coordinates in
    ``QPixmap``, which is in the label.

    Parameters
    ==========

    p
        Coordinates of a point in ``QLabel``

    lsize
        Size of ``QLabel``

    psize
        Size of ``QPixmap``

    alignment
        Alignment of ``QLabel``. Currently, only ``Qt.AlignCenter`` is
        supported.

    Examples
    ========

    >>> from PySide6.QtCore import Qt, QSize
    >>> from dipcoatimage.finitedepth_gui.display import coords_label2pixmap
    >>> lsize = QSize(100, 100)
    >>> psize = QSize(50, 50)
    >>> p = (60.0, 60.0)
    >>> coords_label2pixmap(p, lsize, psize, Qt.AlignCenter)
    (35.0, 35.0)

    """
    for hflag in [Qt.AlignLeft, Qt.AlignRight, Qt.AlignHCenter]:
        if int(alignment & hflag) != 0:  # type: ignore[operator]
            break
    else:
        raise NotImplementedError("Unsupported horizontal alignment")
    for vflag in [Qt.AlignTop, Qt.AlignBottom, Qt.AlignVCenter]:
        if int(alignment & vflag) != 0:  # type: ignore[operator]
            break
    else:
        raise NotImplementedError("Unsupported vertical alignment")

    W, H = lsize.width(), lsize.height()
    w, h = psize.width(), psize.height()
    x, y = p

    if hflag == Qt.AlignLeft:
        dx = 0.0
    elif hflag == Qt.AlignRight:
        dx = float(W - w)
    elif hflag == Qt.AlignHCenter:
        dx = (W - w) / 2

    if vflag == Qt.AlignTop:
        dy = 0.0
    elif vflag == Qt.AlignBottom:
        dy = float(H - h)
    elif vflag == Qt.AlignVCenter:
        dy = (H - h) / 2

    return (x - dx, y - dy)


def coords_pixmap2label(
    p: Tuple[int, int], psize: QSize, lsize: QSize, alignment: Qt.Alignment
) -> Tuple[float, float]:
    """
    Convert the coordinates in ``QPixmap`` to the coordinates in
    ``QLabel``, which contains the pixmap.

    Parameters
    ==========

    p
        Coordinates of a point in ``QPixmap``

    psize
        Size of ``QPixmap``

    lsize
        Size of ``QLabel``

    alignment
        Alignment of ``QLabel``. Currently, only ``Qt.AlignCenter`` is
        supported.

    Examples
    ========

    >>> from PySide6.QtCore import Qt, QSize
    >>> from dipcoatimage.finitedepth_gui.display import coords_pixmap2label
    >>> psize = QSize(50, 50)
    >>> lsize = QSize(100, 100)
    >>> p = (35.0, 35.0)
    >>> coords_pixmap2label(p, psize, lsize, Qt.AlignCenter)
    (60.0, 60.0)

    """
    for hflag in [Qt.AlignLeft, Qt.AlignRight, Qt.AlignHCenter]:
        if int(alignment & hflag) != 0:  # type: ignore[operator]
            break
    else:
        raise NotImplementedError("Unsupported horizontal alignment")
    for vflag in [Qt.AlignTop, Qt.AlignBottom, Qt.AlignVCenter]:
        if int(alignment & vflag) != 0:  # type: ignore[operator]
            break
    else:
        raise NotImplementedError("Unsupported vertical alignment")

    w, h = psize.width(), psize.height()
    W, H = lsize.width(), lsize.height()
    x, y = p

    if hflag == Qt.AlignLeft:
        dx = 0.0
    elif hflag == Qt.AlignRight:
        dx = float(W - w)
    elif hflag == Qt.AlignHCenter:
        dx = (W - w) / 2

    if vflag == Qt.AlignTop:
        dy = 0.0
    elif vflag == Qt.AlignBottom:
        dy = float(H - h)
    elif vflag == Qt.AlignVCenter:
        dy = (H - h) / 2

    return (x + dx, y + dy)


class DisplayWidgetToolBar(QToolBar):
    """Toolbar to controll the overall display."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._visualizeActionGroup = QActionGroup(self)
        self._visualizeAction = QAction("Toggle visualization")
        self._fastVisualizeAction = QAction("Toggle fast visualization")

        self.visualizeActionGroup().setExclusionPolicy(
            QActionGroup.ExclusionPolicy.ExclusiveOptional
        )
        self.visualizeActionGroup().addAction(self.visualizeAction())
        self.visualizeActionGroup().addAction(self.fastVisualizeAction())
        self.visualizeAction().setCheckable(True)
        self.visualizeAction().setChecked(True)
        self.addAction(self.visualizeAction())
        self.fastVisualizeAction().setCheckable(True)
        self.fastVisualizeAction().setChecked(False)
        self.addAction(self.fastVisualizeAction())

        self.initUI()

    def initUI(self):
        visIcon = QIcon()
        visIcon.addFile(get_icons_path("visualize.svg"), QSize(24, 24))
        self.visualizeAction().setIcon(visIcon)

        fastVisIcon = QIcon()
        fastVisIcon.addFile(get_icons_path("fastvisualize.svg"), QSize(24, 24))
        self.fastVisualizeAction().setIcon(fastVisIcon)

    def visualizeActionGroup(self) -> QActionGroup:
        return self._visualizeActionGroup

    def visualizeAction(self) -> QAction:
        """Action to toggle visualization mode."""
        return self._visualizeAction

    def fastVisualizeAction(self) -> QAction:
        """Action to toggle fast visualization mode."""
        return self._fastVisualizeAction


def get_icons_path(*paths: str) -> str:
    """
    Get the absolute path to the directory where the icon files are
    stored.

    Parameters
    ==========

    paths
        Subpaths under ``dipcoatimage/finitedepth_gui/icons/`` directory.

    Returns
    =======

    path
        Absolute path to the icon depending on the user's system.

    """
    module_path = os.path.abspath(dipcoatimage.finitedepth_gui.__file__)
    module_path = os.path.split(module_path)[0]
    sample_dir = os.path.join(module_path, "icons")
    sample_dir = os.path.normpath(sample_dir)
    sample_dir = os.path.normcase(sample_dir)

    path = os.path.join(sample_dir, *paths)
    return path


class ROIVideoWidget(NDArrayVideoPlayerWidget):
    def __init__(self, parent=None):
        self._roiVideoLabel = NDArrayROILabel()
        super().__init__(parent)

    def videoLabel(self):
        return self._roiVideoLabel


class ROICameraWidget(NDArrayCameraWidget):
    def __init__(self, parent=None):
        self._roiVideoLabel = NDArrayROILabel()
        super().__init__(parent)

    def videoLabel(self):
        return self._roiVideoLabel


class MainDisplayWindow(QMainWindow):
    """Main window which includes various display widgets."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._display_widget = QStackedWidget()
        self._img_display = NDArrayROILabel()
        self._vid_display = ROIVideoWidget()
        self._camera_display = ROICameraWidget()
        self._display_toolbar = DisplayWidgetToolBar()

        self.imageDisplayWidget().setAlignment(Qt.AlignCenter)
        self.videoDisplayWidget().setVideoPlayer(PreviewableNDArrayVideoPlayer(self))
        self.videoDisplayWidget().setArrayProcessor(ExperimentArrayProcessor())
        self.addToolBar(self.displayToolBar())

        self.initUI()

    def initUI(self):
        self.setCentralWidget(self.displayStackWidget())
        self.centralWidget().addWidget(self.imageDisplayWidget())
        self.centralWidget().addWidget(self.videoDisplayWidget())
        self.centralWidget().addWidget(self.cameraDisplayWidget())

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

    def visualizeActionGroup(self) -> QActionGroup:
        return self.displayToolBar().visualizeActionGroup()

    def visualizeAction(self) -> QAction:
        """Action to toggle visualization mode."""
        return self.displayToolBar().visualizeAction()

    def fastVisualizeAction(self) -> QAction:
        """Action to toggle fast visualization mode."""
        return self.displayToolBar().fastVisualizeAction()

    def exposedDisplayWidget(self) -> QWidget:
        """Return the display widget exposed to central area."""
        return self.displayStackWidget().currentWidget()

    def exposeDisplayWidget(self, widget: QWidget):
        """Expose the display widget to central area."""
        self.displayStackWidget().setCurrentWidget(widget)

    def currentDisplayingLabel(self):
        """Return the displaying label in current widget."""
        widget = self.exposedDisplayWidget()
        if isinstance(widget, NDArrayVideoPlayerWidget):
            label = widget.videoLabel()
        elif isinstance(widget, NDArrayCameraWidget):
            label = widget.videoLabel()
        elif isinstance(widget, QLabel):
            label = widget
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
