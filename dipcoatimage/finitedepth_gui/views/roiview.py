"""
ROI view
========

V2 for roimodel.py
"""

from PySide6.QtWidgets import QWidget, QSpinBox, QHBoxLayout

__all__ = [
    "ROIView",
]


class ROIView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._x1_spinbox = QSpinBox()
        self._y1_spinbox = QSpinBox()
        self._x2_spinbox = QSpinBox()
        self._y2_spinbox = QSpinBox()

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
