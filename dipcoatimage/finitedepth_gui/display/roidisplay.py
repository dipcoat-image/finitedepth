from araviq6 import NDArrayLabel
from PySide6.QtCore import Signal, QSize, QRect, QPoint, Qt
from PySide6.QtGui import QPaintEvent, QMouseEvent, QPainter, QBrush, QColor
from dipcoatimage.finitedepth_gui.roimodel import ROIModel
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from typing import Union, Tuple, List, Optional


__all__ = [
    "NDArrayROILabel",
    "NDArrayROILabel_V2",
    "coords_label2pixmap",
    "coords_pixmap2label",
]


Number = Union[int, float]
ROI = Tuple[Number, Number, Number, Number]


class NDArrayROILabel(NDArrayLabel):
    """
    ``araviq6.NDArrayLabel`` which multiple ROIs can be set.

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
        self, roi: Tuple[Number, Number, Number, Number]
    ) -> Tuple[Number, Number, Number, Number]:
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
            ret = (0, 0, 0, 0)
        else:
            x1 = max(px1 / w * W, 0)
            y1 = max(py1 / h * H, 0)
            x2 = min(px2 / w * W, W)
            y2 = min(py2 / h * H, H)
            ret = (x1, y1, x2, y2)  # type: ignore[assignment]
        return ret

    def originalROI2LabelROI(
        self, roi: Tuple[Number, Number, Number, Number]
    ) -> Tuple[Number, Number, Number, Number]:
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
            px1, py1, px2, py2 = (x1 / W * w, y1 / H * h, x2 / W * w, y2 / H * h)
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
                x1, y1, x2, y2 = map(int, self.originalROI2LabelROI((x1, y1, x2, y2)))
                br = QBrush(QColor(255, 0, 0, 50))
                qp.setBrush(br)
                qp.drawRect(QRect(QPoint(x1, y1), QPoint(x2, y2)))
        else:
            x1, y1, x2, y2 = map(int, self.originalROI2LabelROI(self._temp_roi))
            for model in self.roiModels():
                br = QBrush(QColor(255, 0, 0, 50))
                qp.setBrush(br)
                qp.drawRect(QRect(QPoint(x1, y1), QPoint(x2, y2)))

    def mousePressEvent(self, event: QMouseEvent):
        """Start the drawing mode."""
        super().mousePressEvent(event)
        self._drawing = True
        pos = event.position()
        x, y = pos.x(), pos.y()
        roi = self.labelROI2OriginalROI((x, y, x, y))
        self._temp_roi = roi
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Update ROI candidate."""
        super().mouseMoveEvent(event)
        if self._drawing:
            x1, y1, _, _ = self.originalROI2LabelROI(self._temp_roi)
            pos = event.position()
            x2, y2 = pos.x(), pos.y()
            roi = self.labelROI2OriginalROI((x1, y1, x2, y2))
            self._temp_roi = roi
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
        x2, y2 = pos.x(), pos.y()
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        roi = self.labelROI2OriginalROI((x1, y1, x2, y2))
        self._temp_roi = roi

        for model in self.roiModels():
            model.setROI(*map(int, self._temp_roi))
        self._drawing = False
        self.update()


class NDArrayROILabel_V2(NDArrayLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None

    def model(self) -> Optional[ExperimentDataModel]:
        return self._model

    def setModel(self, model: Optional[ExperimentDataModel]):
        self._model = model

    def _labelROI2OriginalROI(self, roi: ROI) -> ROI:
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
            ret = (0, 0, 0, 0)
        else:
            x1 = max(px1 / w * W, 0)
            y1 = max(py1 / h * H, 0)
            x2 = min(px2 / w * W, W)
            y2 = min(py2 / h * H, H)
            ret = (x1, y1, x2, y2)  # type: ignore[assignment]
        return ret

    def _originalROI2LabelROI(self, roi: ROI) -> ROI:
        # convert to roi of scaled pixmap
        x1, y1, x2, y2 = roi
        pixmap_size = self.pixmap().size()
        w, h = pixmap_size.width(), pixmap_size.height()
        original_size = self._original_pixmap.size()
        W, H = original_size.width(), original_size.height()
        if w == 0 or h == 0:
            px1, py1, px2, py2 = (0, 0, 0, 0)
        else:
            px1, py1, px2, py2 = (x1 / W * w, y1 / H * h, x2 / W * w, y2 / H * h)
        # convert to roi of label
        lx1, ly1 = coords_pixmap2label(
            (px1, py1), pixmap_size, self.size(), self.alignment()
        )
        lx2, ly2 = coords_pixmap2label(
            (px2, py2), pixmap_size, self.size(), self.alignment()
        )
        return (lx1, ly1, lx2, ly2)


def coords_label2pixmap(
    p: Tuple[Number, Number], lsize: QSize, psize: QSize, alignment: Qt.Alignment
) -> Tuple[Number, Number]:
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
        dx: Number = 0
    elif hflag == Qt.AlignRight:
        dx = W - w
    elif hflag == Qt.AlignHCenter:
        dx = (W - w) / 2

    if vflag == Qt.AlignTop:
        dy: Number = 0
    elif vflag == Qt.AlignBottom:
        dy = H - h
    elif vflag == Qt.AlignVCenter:
        dy = (H - h) / 2

    return (x - dx, y - dy)


def coords_pixmap2label(
    p: Tuple[Number, Number], psize: QSize, lsize: QSize, alignment: Qt.Alignment
) -> Tuple[Number, Number]:
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
        dx: Number = 0
    elif hflag == Qt.AlignRight:
        dx = W - w
    elif hflag == Qt.AlignHCenter:
        dx = (W - w) / 2

    if vflag == Qt.AlignTop:
        dy: Number = 0
    elif vflag == Qt.AlignBottom:
        dy = H - h
    elif vflag == Qt.AlignVCenter:
        dy = (H - h) / 2

    return (x + dx, y + dy)
