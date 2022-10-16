"""
ROI view
========

V2 for roimodel.py
"""

from dipcoatimage.finitedepth.util import IntROI, OptionalROI
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QSpinBox, QWidget, QHBoxLayout

__all__ = [
    "ROISpinBox",
    "ROIView",
]


class ROISpinBox(QSpinBox):
    def stepBy(self, steps):
        super().stepBy(steps)
        self.editingFinished.emit()


class ROIView(QWidget):

    editingFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._x1_spinbox = ROISpinBox()
        self._y1_spinbox = ROISpinBox()
        self._x2_spinbox = ROISpinBox()
        self._y2_spinbox = ROISpinBox()

        self._x1_spinbox.setMinimum(0)
        self._y1_spinbox.setMinimum(0)
        self._x2_spinbox.setMinimum(0)
        self._y2_spinbox.setMinimum(0)
        self.setROIMaximum(0, 0)
        self._x1_spinbox.editingFinished.connect(self.editingFinished)
        self._y1_spinbox.editingFinished.connect(self.editingFinished)
        self._x2_spinbox.editingFinished.connect(self.editingFinished)
        self._y2_spinbox.editingFinished.connect(self.editingFinished)

        self._x1_spinbox.setPrefix("x1 : ")
        self._y1_spinbox.setPrefix("y1 : ")
        self._x2_spinbox.setPrefix("x2 : ")
        self._y2_spinbox.setPrefix("y2 : ")

        layout = QHBoxLayout()
        layout.addWidget(self._x1_spinbox)
        layout.addWidget(self._y1_spinbox)
        layout.addWidget(self._x2_spinbox)
        layout.addWidget(self._y2_spinbox)
        self.setLayout(layout)

    def clear(self):
        self.setROIMaximum(0, 0)

    def setROIMaximum(self, w: int, h: int):
        self._x1_spinbox.setMaximum(w)
        self._y1_spinbox.setMaximum(h)
        self._x2_spinbox.setMaximum(w)
        self._y2_spinbox.setMaximum(h)

    def roi(self) -> IntROI:
        x1 = self._x1_spinbox.value()
        y1 = self._y1_spinbox.value()
        x2 = self._x2_spinbox.value()
        y2 = self._y2_spinbox.value()
        return (x1, y1, x2, y2)

    def setROI(self, roi: OptionalROI):
        W, H = self._x1_spinbox.maximum(), self._y1_spinbox.maximum()
        x1, y1, x2, y2 = roi
        if x2 is None:
            x2 = W
        if y2 is None:
            y2 = H
        self._x1_spinbox.setValue(x1)
        self._y1_spinbox.setValue(y1)
        self._x2_spinbox.setValue(x2)
        self._y2_spinbox.setValue(y2)
