from araviq6 import NDArrayLabel
from PySide6.QtCore import Slot, QSize, QRect, QPoint, Qt, QModelIndex
from PySide6.QtGui import QPainter, QBrush, QColor
from dipcoatimage.finitedepth_gui.core import ROIDrawMode
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel, IndexRole
from typing import Union, Tuple


__all__ = [
    "NDArrayROILabel",
    "coords_label2pixmap",
    "coords_pixmap2label",
]


Number = Union[int, float]
ROI = Tuple[Number, Number, Number, Number]


class NDArrayROILabel(NDArrayLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._currentModelIndex = QModelIndex()
        self._roiDrawMode = ROIDrawMode.NONE
        self._drawing = False
        self._drawnROI = (-1, -1, -1, -1)

    def setActivatedIndex(self, index: QModelIndex):
        self._currentModelIndex = index

    def roiDrawMode(self) -> ROIDrawMode:
        return self._roiDrawMode

    @Slot(ROIDrawMode)
    def setROIDrawMode(self, flag: ROIDrawMode):
        self._roiDrawMode = flag
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        qp = QPainter(self)
        if not self._drawing:
            index = self._currentModelIndex
            if not index.isValid():
                return
            model = index.model()
            if not isinstance(model, ExperimentDataModel):
                return
            refArgsIdx = model.getIndexFor(IndexRole.REFARGS, index)
            drawMode = self._roiDrawMode
            if drawMode == ROIDrawMode.TEMPLATE:
                roiIdx = model.getIndexFor(IndexRole.REF_TEMPLATEROI, refArgsIdx)
            elif drawMode == ROIDrawMode.SUBSTRATE:
                roiIdx = model.getIndexFor(IndexRole.REF_SUBSTRATEROI, refArgsIdx)
            else:
                return
            roi = roiIdx.data(model.Role_ROI)

            originalSize = self._original_pixmap.size()
            W, H = originalSize.width(), originalSize.height()
            x1, y1, x2, y2 = roi
            if x2 is None:
                x2 = W
            if y2 is None:
                y2 = H
            x1, y1, x2, y2 = map(int, self._originalROI2LabelROI((x1, y1, x2, y2)))
            br = QBrush(QColor(255, 0, 0, 50))
            qp.setBrush(br)
            qp.drawRect(QRect(QPoint(x1, y1), QPoint(x2, y2)))
        else:
            x1, y1, x2, y2 = map(int, self._originalROI2LabelROI(self._drawnROI))
            br = QBrush(QColor(255, 0, 0, 50))
            qp.setBrush(br)
            qp.drawRect(QRect(QPoint(x1, y1), QPoint(x2, y2)))

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self._drawing = True
        pos = event.position()
        x, y = pos.x(), pos.y()
        roi = self._labelROI2OriginalROI((x, y, x, y))
        self._drawnROI = roi
        self.update()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._drawing:
            x1, y1, _, _ = self._originalROI2LabelROI(self._drawnROI)
            pos = event.position()
            x2, y2 = pos.x(), pos.y()
            roi = self._labelROI2OriginalROI((x1, y1, x2, y2))
            self._drawnROI = roi
            self.update()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._drawing = False

        x1, y1, _, _ = self._originalROI2LabelROI(self._drawnROI)
        pos = event.position()
        x2, y2 = pos.x(), pos.y()
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        roi = self._labelROI2OriginalROI((x1, y1, x2, y2))
        self._drawnROI = roi

        index = self._currentModelIndex
        if not index.isValid():
            return
        model = index.model()
        if not isinstance(model, ExperimentDataModel):
            return
        refArgsIdx = model.getIndexFor(IndexRole.REFARGS, index)
        drawMode = self._roiDrawMode
        if drawMode == ROIDrawMode.TEMPLATE:
            roiIdx = model.getIndexFor(IndexRole.REF_TEMPLATEROI, refArgsIdx)
        elif drawMode == ROIDrawMode.SUBSTRATE:
            roiIdx = model.getIndexFor(IndexRole.REF_SUBSTRATEROI, refArgsIdx)
        else:
            return
        model.setData(roiIdx, tuple(map(int, self._drawnROI)), model.Role_ROI)

        self.update()

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
    p: Tuple[Number, Number], lsize: QSize, psize: QSize, alignment: Qt.AlignmentFlag
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
    for hflag in [
        Qt.AlignmentFlag.AlignLeft,
        Qt.AlignmentFlag.AlignRight,
        Qt.AlignmentFlag.AlignHCenter,
    ]:
        if int(alignment & hflag) != 0:  # type: ignore[operator]
            break
    else:
        raise NotImplementedError("Unsupported horizontal alignment")
    for vflag in [
        Qt.AlignmentFlag.AlignTop,
        Qt.AlignmentFlag.AlignBottom,
        Qt.AlignmentFlag.AlignVCenter,
    ]:
        if int(alignment & vflag) != 0:  # type: ignore[operator]
            break
    else:
        raise NotImplementedError("Unsupported vertical alignment")

    W, H = lsize.width(), lsize.height()
    w, h = psize.width(), psize.height()
    x, y = p

    if hflag == Qt.AlignmentFlag.AlignLeft:
        dx: Number = 0
    elif hflag == Qt.AlignmentFlag.AlignRight:
        dx = W - w
    elif hflag == Qt.AlignmentFlag.AlignHCenter:
        dx = (W - w) / 2

    if vflag == Qt.AlignmentFlag.AlignTop:
        dy: Number = 0
    elif vflag == Qt.AlignmentFlag.AlignBottom:
        dy = H - h
    elif vflag == Qt.AlignmentFlag.AlignVCenter:
        dy = (H - h) / 2

    return (x - dx, y - dy)


def coords_pixmap2label(
    p: Tuple[Number, Number], psize: QSize, lsize: QSize, alignment: Qt.AlignmentFlag
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
    for hflag in [
        Qt.AlignmentFlag.AlignLeft,
        Qt.AlignmentFlag.AlignRight,
        Qt.AlignmentFlag.AlignHCenter,
    ]:
        if int(alignment & hflag) != 0:  # type: ignore[operator]
            break
    else:
        raise NotImplementedError("Unsupported horizontal alignment")
    for vflag in [
        Qt.AlignmentFlag.AlignTop,
        Qt.AlignmentFlag.AlignBottom,
        Qt.AlignmentFlag.AlignVCenter,
    ]:
        if int(alignment & vflag) != 0:  # type: ignore[operator]
            break
    else:
        raise NotImplementedError("Unsupported vertical alignment")

    w, h = psize.width(), psize.height()
    W, H = lsize.width(), lsize.height()
    x, y = p

    if hflag == Qt.AlignmentFlag.AlignLeft:
        dx: Number = 0
    elif hflag == Qt.AlignmentFlag.AlignRight:
        dx = W - w
    elif hflag == Qt.AlignmentFlag.AlignHCenter:
        dx = (W - w) / 2

    if vflag == Qt.AlignmentFlag.AlignTop:
        dy: Number = 0
    elif vflag == Qt.AlignmentFlag.AlignBottom:
        dy = H - h
    elif vflag == Qt.AlignmentFlag.AlignVCenter:
        dy = (H - h) / 2

    return (x + dx, y + dy)
