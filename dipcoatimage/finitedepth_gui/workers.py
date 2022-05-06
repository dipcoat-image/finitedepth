"""
Workers
=======

This module provides workers to construct objects from the parameters passed to
control widgets.

"""

from dipcoatimage.finitedepth import SubstrateReferenceBase, SubstrateBase
from dipcoatimage.finitedepth.util import OptionalROI, DataclassProtocol
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Signal
from typing import Optional, Type
from .controlwidgets import ReferenceWidgetData, SubstrateWidgetData


__all__ = [
    "ReferenceWorker",
    "SubstrateWorker",
]


class ReferenceWorker(QObject):
    """
    Worker to build the concreate instance of :class:`SubstrateReferenceBase`
    and to visualize it.

    Data for reference object are:
        1. :meth:`referenceType`
        2. :meth:`image`
        3. :meth:`templateROI`
        4. :meth:`substrateROI`
        5. :meth:`paramters`
        6. :meth:`drawOptions`
    All data can be updated by :meth:`setReferenceWidgetData`.
    Exceptionally, :meth:`image` can be directly updated by :meth:`setImage`.

    :meth:`updateReference` constructs the reference object with current data.
    Resulting object can be acquired from :meth:`reference`, or calling
    :meth:`emitReference` and listening to :attr:`referenceChanged` signal.

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
        Type object to construct :meth:`reference`.
        ``None`` indicates invalid value.
        """
        return self._type

    def image(self) -> npt.NDArray[np.uint8]:
        """*image* for :meth:`referenceType` to construct :meth:`reference`."""
        img = self._img
        if img is None:
            img = np.empty((0, 0, 0), dtype=np.uint8)
        return img

    def setImage(self, img: Optional[npt.NDArray[np.uint8]]):
        """
        Directly update :meth:`image` with *img*.

        See Also
        ========

        setReferenceWidgetData
            Update the image along with other data.
        """
        self._img = img

    def templateROI(self) -> OptionalROI:
        """
        *templateROI* for :meth:`referenceType` to construct :meth:`reference`.
        """
        return self._temproi

    def substrateROI(self) -> OptionalROI:
        """
        *substrateROI* for :meth:`referenceType` to construct :meth:`reference`.
        """
        return self._substroi

    def parameters(self) -> Optional[DataclassProtocol]:
        """
        *parameters* for :meth:`referenceType` to construct :meth:`reference`.
        ``None`` indicates invalid value.
        """
        return self._params

    def drawOptions(self) -> Optional[DataclassProtocol]:
        """
        *draw_options* for :meth:`referenceType` to construct :meth:`reference`.
        ``None`` indicates invalid value.
        """
        return self._draw_opts

    def setReferenceWidgetData(self, data: ReferenceWidgetData):
        """
        Update following values with *data*.

        1. :meth:`referenceType`
        2. :meth:`image`
        3. :meth:`templateROI`
        4. :meth:`substrateROI`
        5. :meth:`paramters`
        6. :meth:`drawOptions`
        """
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
        Concrete instance of :class:`SubstrateReferenceBase`.
        ``None`` indicates invalid value.

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


class SubstrateWorker(QObject):
    """
    Worker to build the concreate instance of :class:`SubstrateBase` and to
    visualize it.

    Data for substrate object are:
        1. :meth:`substrateType`
        2. :meth:`reference`
        3. :meth:`paramters`
        4. :meth:`drawOptions`
    All data, except :meth:`reference` which is updated by :meth:`setReference`,
    are updated by :meth:`setSubstrateWidgetData`.

    :meth:`updateSubstrate` constructs the substrate object with current data.
    Resulting object can be acquired from :meth:`substrate`, or calling
    :meth:`emitSubstrate` and listening to :attr:`substrateChanged` signal.

    Visualization result can be directly acquired from :meth:`visualizedImage`,
    or calling :meth:`emitImage` and listening to :attr:`visualizedImageChanged`.

    """

    substrateChanged = Signal(object)
    visualizedImageChanged = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initArgs()
        self._visualize_mode = True

    def initArgs(self):
        self._type = None
        self._ref = None
        self._params = None
        self._draw_opts = None

        self._substrate = None

    def substrateType(self) -> Optional[Type[SubstrateBase]]:
        """
        Type object to construct :meth:`substrate`.
        ``None`` indicates invalid value.
        """
        return self._type

    def reference(self) -> Optional[SubstrateReferenceBase]:
        """
        Substrate reference instance to construct :meth:`substrate`.
        ``None`` indicates invalid value.
        """
        return self._ref

    def parameters(self):
        """
        *parameters* for :meth:`substrateType` to construct :meth:`substrate`.
        ``None`` indicates invalid value.
        """
        return self._params

    def drawOptions(self):
        """
        *draw_options* for :meth:`substrateType` to construct :meth:`substrate`.
        ``None`` indicates invalid value.
        """
        return self._draw_opts

    def setSubstrateWidgetData(self, data: SubstrateWidgetData):
        """
        Update following values with *data*.

        1. :meth:`substrateType`
        2. :meth:`paramters`
        3. :meth:`drawOptions`
        """
        substtype = data.type
        if not (isinstance(substtype, type) and issubclass(substtype, SubstrateBase)):
            substtype = None
        self._type = substtype

        params = data.parameters
        if substtype is None:
            params = None
        elif isinstance(params, substtype.Parameters):
            pass
        else:
            try:
                params = substtype.Parameters()
            except TypeError:
                params = None
        self._params = params

        drawopt = data.draw_options
        if substtype is None:
            drawopt = None
        elif isinstance(drawopt, substtype.DrawOptions):
            pass
        else:
            try:
                drawopt = substtype.DrawOptions()
            except TypeError:
                drawopt = None
        self._draw_opts = drawopt

    def setReference(self, ref: Optional[SubstrateReferenceBase]):
        """Update :meth:`reference` with *ref*."""
        self._ref = ref

    def updateSubstrate(self):
        """Update :meth:`substrate` and emit to :attr:`substrateChanged`."""
        subst = None
        default_invalid_args = [
            self.substrateType(),
            self.reference(),
            self.parameters(),
            self.drawOptions(),
        ]
        if all(x is not None for x in default_invalid_args):
            subst = self.substrateType()(
                self.reference(),
                parameters=self.parameters(),
                draw_options=self.drawOptions(),
            )
            if not subst.valid():
                subst = None
        self._substrate = subst

    def substrate(self) -> Optional[SubstrateBase]:
        """
        Concrete instance of :class:`SubstrateBase`.
        ``None`` indicates invalid value.

        Run :meth:`updateSubstrate` to update this value.
        """
        return self._substrate

    def emitSubstrate(self):
        """Emit the result of :meth:`substrate` to :attr:`substrateChanged`."""
        self.substrateChanged.emit(self.substrate())

    def visualizationMode(self) -> bool:
        """If False, analysis result is never visualized."""
        return self._visualize_mode

    def setVisualizationMode(self, mode: bool):
        """Update :meth:`visualizationMode` with *mode*."""
        self._visualize_mode = mode

    def visualizedImage(self) -> npt.NDArray[np.uint8]:
        """
        Return visualization result of :meth:`substrate`.

        If :meth:`substrate` is invalid or :meth:`visualizationMode` is False,
        directly return :meth:`image`.
        """
        subst = self.substrate()
        ref = self.reference()
        if subst is not None:
            if self.visualizationMode():
                image = subst.draw()
            else:
                image = subst.image
        elif ref is not None:
            image = ref.substrate_image()
        else:
            image = np.empty((0, 0, 0), dtype=np.uint8)
        return image

    def emitImage(self):
        """
        Emit the result of :meth:`visualizedImage` to
        :attr:`visualizedImageChanged` signal.

        If visualization raises error, directly emit :meth:`image`.
        """
        self.visualizedImageChanged.emit(self.visualizedImage())

    def clear(self):
        """Initialize substrate object data and :meth:`substrate`."""
        self.initArgs()
