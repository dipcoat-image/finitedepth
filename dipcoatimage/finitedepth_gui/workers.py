"""
Workers
=======

This module provides workers to construct objects from the parameters passed to
control widgets.

"""

from dipcoatimage.finitedepth import SubstrateReferenceBase
from dipcoatimage.finitedepth.util import OptionalROI, DataclassProtocol
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Signal
from typing import Optional, Type
from .controlwidgets import ReferenceWidgetData


__all__ = [
    "ReferenceWorker",
]


class ReferenceWorker(QObject):
    """
    Worker to build the concreate instance of :class:`SubstrateReferenceBase`
    to visualize substrate reference image.

    Data for reference image object are :meth:`referenceType`, :meth:`image`,
    :meth:`templateROI`, :meth:`substrateROI`, :meth:`parameters` and
    :meth:`drawOptions`.
    Data can be updated by :meth:`setReferenceWidgetData`.

    :meth:`updateReference` constructs the reference object with data.
    Resulting object can be acquired from :meth:`reference`, or from
    :attr:`referenceChanged` signal after calling :meth:`emitReference`.

    Visualization result can be directly acquired from :meth:`visualizedImage`,
    or calling :meth:`emitImage` and listening to :attr:`visualizedImageChanged`.

    """

    referenceChanged = Signal(object)
    visualizedImageChanged = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initArgs()
        self._visualize_mode = True

    def initArgs(self):
        self._type = None
        self._img = None
        self._temproi = (0, 0, None, None)
        self._substroi = (0, 0, None, None)
        self._params = None
        self._draw_opts = None

        self._reference = None

    def referenceType(self) -> Optional[Type[SubstrateReferenceBase]]:
        """
        :class:`SubstrateReferenceBase` or its subclass type to construct
        :meth:`reference`. ``None`` indicates invalid value.
        """
        return self._type

    def image(self) -> npt.NDArray[np.uint8]:
        """
        *image* argument of :meth:`referenceType` to construct :meth:`reference`.
        """
        img = self._img
        if img is None:
            img = np.empty((0, 0))
        return img

    def setImage(self, img: Optional[npt.NDArray[np.uint8]]):
        """
        Directly update :meth:`image` with *img*.

        See Also
        ========

        setReferenceWidgetData
        """
        self._img = img

    def templateROI(self) -> OptionalROI:
        """
        *templateROI* argument of :meth:`referenceType` to construct
        :meth:`reference`.
        """
        return self._temproi

    def substrateROI(self) -> OptionalROI:
        """
        *substrateROI* argument of :meth:`referenceType` to construct
        :meth:`reference`.
        """
        return self._substroi

    def parameters(self) -> Optional[DataclassProtocol]:
        """
        *parameters* argument of :meth:`referenceType` to construct
        :meth:`reference`. ``None`` indicates invalid value.
        """
        return self._params

    def drawOptions(self) -> Optional[DataclassProtocol]:
        """
        *draw_options* argument of :meth:`referenceType` to construct
        :meth:`reference`. ``None`` indicates invalid value.
        """
        return self._draw_opts

    def setReferenceWidgetData(self, data: ReferenceWidgetData):
        """Update reference object data with *data*."""
        reftype = data.type
        if not (
            isinstance(reftype, type) and issubclass(reftype, SubstrateReferenceBase)
        ):
            reftype = None
        self._type = reftype

        self._img = data.image
        self._temproi = data.templateROI
        self._substroi = data.substrateROI

        params = data.parameters
        if reftype is None:
            params = None
        elif isinstance(params, reftype.Parameters):
            pass
        else:
            try:
                params = reftype.Parameters()
            except TypeError:
                params = None
        self._params = params

        drawopt = data.draw_options
        if reftype is None:
            drawopt = None
        elif isinstance(drawopt, reftype.DrawOptions):
            pass
        else:
            try:
                drawopt = reftype.DrawOptions()
            except TypeError:
                drawopt = None
        self._draw_opts = drawopt

    def updateReference(self):
        """Update :meth:`reference` and emit to :attr:`referenceChanged`."""
        ref = None
        default_invalid_args = [
            self.referenceType(),
            self.parameters(),
            self.drawOptions(),
        ]
        if all(x is not None for x in default_invalid_args) and self.image().size > 0:
            ref = self.referenceType()(
                self.image(),
                self.templateROI(),
                self.substrateROI(),
                parameters=self.parameters(),
                draw_options=self.drawOptions(),
            )
            if not ref.valid():
                ref = None
        self._reference = ref

    def reference(self) -> Optional[SubstrateReferenceBase]:
        """
        Concrete instance of :class:`SubstrateReferenceBase` constructed by
        substrate reference data. ``None`` indicates invalid value.

        Run :meth:`updateReference` to update this value.
        """
        return self._reference

    def emitReference(self):
        """Emit the result of :meth:`reference` to :attr:`referenceChanged`."""
        self.referenceChanged.emit(self.reference())

    def visualizationMode(self) -> bool:
        """If False, analysis result is never visualized."""
        return self._visualize_mode

    def setVisualizationMode(self, mode: bool):
        """Update :meth:`visualizationMode` with *mode*."""
        self._visualize_mode = mode

    def visualizedImage(self) -> npt.NDArray[np.uint8]:
        """
        Return visualization result of :meth:`reference`.

        If :meth:`reference` is invalid or :meth:`visualizationMode` is False,
        directly return :meth:`image`.
        """
        ref = self.reference()
        if ref is not None and self.visualizationMode():
            image = ref.draw()
        else:
            image = self.image()
        return image

    def emitImage(self):
        """
        Emit the result of :meth:`visualizedImage` to
        :attr:`visualizedImageChanged` signal.

        If visualization raises error, directly emit :meth:`image`.
        """
        self.visualizedImageChanged.emit(self.visualizedImage())

    def clear(self):
        """Initialize reference object data and :meth:`reference`."""
        self.initArgs()
