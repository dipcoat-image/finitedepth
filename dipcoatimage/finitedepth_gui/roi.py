"""
ROI controller
==============

This module provides dedicated model and control to modify ROI data.

"""

from dipcoatimage.finitedepth.util import OptionalROI, IntROI
from PySide6.QtCore import Signal, Slot, QObject, QSignalBlocker
from PySide6.QtWidgets import QWidget, QSpinBox, QHBoxLayout
from typing import Tuple, Optional


__all__ = ["ROIModel", "ROIWidget"]


class ROIModel(QObject):
    """
    Data model to represent a single ROI data.

    :meth:`roi` returns current ROI value, which can be set by :meth:`setROI`.
    Setting the ROI always emits :attr:`roiChanged` signal.

    """

    roiChanged = Signal(int, int, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._roi = (0, 0, None, None)

    def roi(self) -> OptionalROI:
        """
        ROI value in ``(x1, y1, x2, y2)``.

        ``x2`` and ``y2`` can be :obj:`None`, which should be interpreted as
        maximum value possible.
        """
        return self._roi

    def setROI(self, x1: int, y1: int, x2: Optional[int], y2: Optional[int]):
        """Set :meth:`roi` and emit to :attr:`roiChanged`."""
        self._roi = (x1, y1, x2, y2)
        self.roiChanged.emit(*self.roi())


class ROIWidget(QWidget):
    """
    Widget to display and to control a single ROI data.

    .. rubric:: Model

    :meth:`roiModel` returns current model which stores the ROI data. Current
    model can be changed by :meth:`setROIModel`. Any change of ROI data from the
    model is updated to the widget.

    .. rubric:: View

    ROI data are displayed as integers in spin boxes. Minimum values of spin
    boxes are zero. Maximum values can be set by :meth:`setROIMaximum`, which
    emits :attr:`roiMaximumChanged` if the new value differs from the old value.

    Whenever the model data data are changed, they are updated to the spin boxes.
    Whenever the displayed data are updated, :attr:`roiChanged` emits new data.

    :attr:`roiChanged` signal does not require the change of model data. For
    example, if the ``x2`` data is :obj:`None` and maximum width is 10, widget
    displays 0 for ``x2`` value. If maximum width is changed to 20, displayed
    value changes to 20 and :attr:`roiChanged` is emitted, but ``x2`` data is
    still :obj:`None`.

    .. rubric:: Control

    When any spin box value is changed, ROI data in :meth:`roiModel` is updated.
    ROI value can be controlled by :meth:`setROIValue` as well.

    Examples
    ========

    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.roi import ROIWidget
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     widget = ROIWidget()
    ...     widget.setROIMaximum(10, 10)
    ...     widget.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    roiChanged = Signal(int, int, int, int)
    roiMaximumChanged = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._x1_spinbox = QSpinBox()
        self._y1_spinbox = QSpinBox()
        self._x2_spinbox = QSpinBox()
        self._y2_spinbox = QSpinBox()
        self._roi_model = ROIModel()
        self._roiMaximum = (0, 0)

        self.x1SpinBox().setMinimum(0)
        self.y1SpinBox().setMinimum(0)
        self.x2SpinBox().setMinimum(0)
        self.y2SpinBox().setMinimum(0)
        self.x1SpinBox().setMaximum(0)
        self.y1SpinBox().setMaximum(0)
        self.x2SpinBox().setMaximum(0)
        self.y2SpinBox().setMaximum(0)
        self.x1SpinBox().valueChanged.connect(self.onX1Change)
        self.y1SpinBox().valueChanged.connect(self.onY1Change)
        self.x2SpinBox().valueChanged.connect(self.onX2Change)
        self.y2SpinBox().valueChanged.connect(self.onY2Change)
        self.roiModel().roiChanged.connect(self.onROIChange)
        self.onROIChange(*self.roiModel().roi())

        self.initUI()

    def initUI(self):
        self.x1SpinBox().setPrefix("x1 : ")
        self.y1SpinBox().setPrefix("y1 : ")
        self.x2SpinBox().setPrefix("x2 : ")
        self.y2SpinBox().setPrefix("y2 : ")

        layout = QHBoxLayout()
        layout.addWidget(self.x1SpinBox())
        layout.addWidget(self.y1SpinBox())
        layout.addWidget(self.x2SpinBox())
        layout.addWidget(self.y2SpinBox())
        self.setLayout(layout)

    def x1SpinBox(self) -> QSpinBox:
        """Spin box to display ``x1`` in ``(x1, y1, x2, y2)``."""
        return self._x1_spinbox

    def y1SpinBox(self) -> QSpinBox:
        """Spin box to display ``y1`` in ``(x1, y1, x2, y2)``."""
        return self._y1_spinbox

    def x2SpinBox(self) -> QSpinBox:
        """Spin box to display ``x2`` in ``(x1, y1, x2, y2)``."""
        return self._x2_spinbox

    def y2SpinBox(self) -> QSpinBox:
        """Spin box to display ``y2`` in ``(x1, y1, x2, y2)``."""
        return self._y2_spinbox

    def roiModel(self) -> ROIModel:
        """Model to store ROI data."""
        return self._roi_model

    def setROIModel(self, model: ROIModel):
        """Set new model, connect signal and update view."""
        self.roiModel().roiChanged.disconnect(self.onROIChange)
        self._roi_model = model
        self.roiModel().roiChanged.connect(self.onROIChange)
        with QSignalBlocker(self):
            # do not emit signals since model data are not changed
            self.onROIChange(*self.roiModel().roi())

    def roiMaximum(self) -> Tuple[int, int]:
        return self._roiMaximum

    @Slot(int, int)
    def setROIMaximum(self, w: int, h: int):
        """
        If the value differs from old value, set as maximum values of spin boxes
        and emit to :attr:`roiMaximumChanged`. Also, update the view with new
        maximum value and emit :attr:`roiChanged`.
        """
        if (w, h) != self.roiMaximum():
            self._roiMaximum = (w, h)
            self.x1SpinBox().setMaximum(w)
            self.y1SpinBox().setMaximum(h)
            self.x2SpinBox().setMaximum(w)
            self.y2SpinBox().setMaximum(h)
            self.roiMaximumChanged.emit(w, h)
            self.onROIChange(*self.roiModel().roi())

    def displayedROI(self) -> IntROI:
        """Return ROI value displayed on spin boxes."""
        x1 = self.x1SpinBox().value()
        y1 = self.y1SpinBox().value()
        x2 = self.x2SpinBox().value()
        y2 = self.y2SpinBox().value()
        return (x1, y1, x2, y2)

    @Slot(int)
    def onX1Change(self, x1: int):
        """Set x1 value from control to model."""
        _, y1, x2, y2 = self.displayedROI()
        self.setROI(x1, y1, x2, y2)

    @Slot(int)
    def onY1Change(self, y1: int):
        """Set y1 value from control to model."""
        x1, _, x2, y2 = self.displayedROI()
        x1 = self.x1SpinBox().value()
        x2 = self.x2SpinBox().value()
        y2 = self.y2SpinBox().value()
        self.setROI(x1, y1, x2, y2)

    @Slot(int)
    def onX2Change(self, x2: int):
        """Set x2 value from control to model."""
        x1, y1, _, y2 = self.displayedROI()
        self.setROI(x1, y1, x2, y2)

    @Slot(int)
    def onY2Change(self, y2: int):
        """Set y2 value from control to model."""
        x1, y1, x2, _ = self.displayedROI()
        self.setROI(x1, y1, x2, y2)

    def setROI(self, x1: int, y1: int, x2: Optional[int], y2: Optional[int]):
        """Set values to model."""
        self.roiModel().setROI(x1, y1, x2, y2)

    @Slot(int, int, object, object)
    def onROIChange(self, x1: int, y1: int, x2: Optional[int], y2: Optional[int]):
        """
        Set ROI values from model to view, and emit :attr:`roiChanged`.
        """
        W, H = self.roiMaximum()
        if x2 is None:
            x2 = W
        if y2 is None:
            y2 = H
        with QSignalBlocker(self.x1SpinBox()):
            self.x1SpinBox().setValue(x1)
        with QSignalBlocker(self.y1SpinBox()):
            self.y1SpinBox().setValue(y1)
        with QSignalBlocker(self.x2SpinBox()):
            self.x2SpinBox().setValue(x2)
        with QSignalBlocker(self.y2SpinBox()):
            self.y2SpinBox().setValue(y2)
        self.roiChanged.emit(x1, y1, x2, y2)
