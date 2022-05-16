"""
Workers
=======

This module provides workers to construct objects from the parameters passed to
control widgets.

"""

import cv2  # type: ignore
from dipcoatimage.finitedepth import (
    SubstrateReferenceBase,
    SubstrateBase,
    CoatingLayerBase,
    ExperimentBase,
)
from dipcoatimage.finitedepth.util import OptionalROI, DataclassProtocol
import enum
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Slot, Signal
from typing import Optional, Type, Generator
from .controlwidgets import (
    ReferenceWidgetData,
    SubstrateWidgetData,
    CoatingLayerWidgetData,
    ExperimentWidgetData,
)


__all__ = [
    "ReferenceWorker",
    "SubstrateWorker",
    "ExperimentVisualizationMode",
    "ExperimentWorker",
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

    :meth:`image` is updated by :meth:`setImage`. Other data are updated by
    :meth:`setReferenceWidgetData`.

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

    @Slot(object)
    def setImage(self, img: Optional[npt.NDArray[np.uint8]]):
        """Update :meth:`image` with *img*."""
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

    def parameters(self) -> Optional[DataclassProtocol]:
        """
        *parameters* for :meth:`substrateType` to construct :meth:`substrate`.
        ``None`` indicates invalid value.
        """
        return self._params

    def drawOptions(self) -> Optional[DataclassProtocol]:
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
                image = subst.image()
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


class ExperimentVisualizationMode(enum.IntEnum):
    """
    Option for :class:`ExperimentWorker` to determine how the image is shown.

    Attributes
    ==========

    OFF
        Do not visualize.

    FULL
        Full visualization using coating layer decoration.

    FAST
        Fast visualization without coating layer decoration. Only
        substrate removal is performed.

    """

    OFF = 0
    FULL = 1
    FAST = 2


class ExperimentWorker(QObject):
    """
    Worker to build the concreate instance of :class:`ExperimentBase` and to
    visualize it.

    Data for experiment object are:

    1. :meth:`experimentType`
    2. :meth:`substrate`
    3. :meth:`coatingLayerType`
    4. :meth:`coatingLayerParameters`
    5. :meth:`coatingLayerDrawOptions`
    6. :meth:`coatingLayerDecoOptions`
    7. :meth:`parameters`

    1st and 7th data are updated by :meth:`setExperimentWidgetData`. 2nd datum is
    updated by :meth:`setSubstrate`. 3rd, 4th, 5th and 6th data are updated by
    :meth:`setCoatingLayerWidgetData`.

    :meth:`updateExperiment` constructs the experiment object with data.
    Resulting object can be acquired by :meth:`experiment`, or calling
    :meth:`emitExperiment` and listening to :attr:`experimentChanged` signal.

    To visualize the layer shape image, pass it to :meth:`setImage`
    first. Visualization result can be directly acquired from
    :meth:`visualizedImage`, or calling :meth:`emitImage` and listening
    to :attr:`visualizedImageChanged` signal.

    """

    experimentChanged = Signal(object)
    visualizedImageChanged = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initArgs()
        self._visualize_mode = ExperimentVisualizationMode.FULL

    def initArgs(self):
        self._type = None
        self._subst = None
        self._layer_type = None
        self._layer_params = None
        self._layer_drawopts = None
        self._layer_decoopts = None
        self._params = None

        self._expt = None
        self._layer_generator = None

        self._img = None

    def experimentType(self) -> Optional[Type[ExperimentBase]]:
        """
        Type object to construct :meth:`experiment`. ``None`` indicates invalid
        value.
        """
        return self._type

    def substrate(self) -> Optional[SubstrateBase]:
        """
        Substrate instance to construct :meth:`experiment`. ``None`` indicates
        invalid value.
        """
        return self._subst

    def coatingLayerType(self) -> Optional[Type[CoatingLayerBase]]:
        """
        Coating layer type object to construct :meth:`experiment`. ``None``
        indicates invalid value.
        """
        return self._layer_type

    def coatingLayerParameters(self) -> Optional[DataclassProtocol]:
        """
        *parameters* for :meth:`coatingLayerType` to construct
        :meth:`experiment`. ``None`` indicates invalid value.
        """
        return self._layer_params

    def drawOptions(self) -> Optional[DataclassProtocol]:
        """
        *draw_options* for :meth:`coatingLayerType` to construct
        :meth:`experiment`. ``None`` indicates invalid value.
        """
        return self._layer_drawopts

    def decoOptions(self) -> Optional[DataclassProtocol]:
        """
        *deco_options* for :meth:`coatingLayerType` to construct
        :meth:`experiment`. ``None`` indicates invalid value.
        """
        return self._layer_decoopts

    def parameters(self) -> Optional[DataclassProtocol]:
        """
        *parameters* for :meth:`experimentType` to construct :meth:`experiment`.
        ``None`` indicates invalid value.
        """
        return self._params

    def setExperimentWidgetData(self, data: ExperimentWidgetData):
        """
        Update following values with *data*.

        1. :meth:`experimentType`
        2. :meth:`experimentParameters`
        """
        expttype = data.type
        if not (isinstance(expttype, type) and issubclass(expttype, ExperimentBase)):
            expttype = None
        self._type = expttype

        params = data.parameters
        if expttype is None:
            params = None
        elif isinstance(params, expttype.Parameters):
            pass
        else:
            try:
                params = expttype.Parameters()
            except TypeError:
                params = None
        self._params = params

    def setSubstrate(self, subst: Optional[SubstrateBase]):
        """Update :meth:`substrate` with *subst*."""
        self._subst = subst

    def setCoatingLayerWidgetData(self, data: CoatingLayerWidgetData):
        """
        Update following values with *data*.

        1. :meth:`coatingLayerType`
        2. :meth:`coatingLayerParameters`
        3. :meth:`coatingLayerDrawOptions`
        4. :meth:`coatingLayerDecoOptions`
        """
        coattype = data.type

        if not (isinstance(coattype, type) and issubclass(coattype, CoatingLayerBase)):
            coattype = None
        self._layer_type = coattype

        coat_params = data.parameters
        if coattype is None:
            coat_params = None
        elif isinstance(coat_params, coattype.Parameters):
            pass
        else:
            try:
                coat_params = coattype.Parameters()
            except TypeError:
                coat_params = None
        self._layer_params = coat_params

        coat_drawopts = data.draw_options
        if coattype is None:
            coat_drawopts = None
        elif isinstance(coat_drawopts, coattype.DrawOptions):
            pass
        else:
            try:
                coat_drawopts = coattype.DrawOptions()
            except TypeError:
                coat_drawopts = None
        self._layer_drawopts = coat_drawopts

    def updateExperiment(self):
        """Update :meth:`experiment` and :meth:`layerGenerator`."""
        expt = None

        default_invalid_args = [
            self.experimentType(),
            self.substrate(),
            self.coatingLayerType(),
            self.coatingLayerParameters(),
            self.coatingLayerDrawOptions(),
            self.coatingLayerDecoOptions(),
            self.parameters(),
        ]
        if all(x is not None for x in default_invalid_args):
            expt = self.experimentType()(
                self.substrate(),
                self.coatingLayerType(),
                self.coatingLayerParameters(),
                self.coatingLayerDrawOptions(),
                self.coatingLayerDecoOptions(),
                parameters=self.parameters(),
            )
            if not expt.valid():
                expt = None
        self._expt = expt
        self.updateLayerGenerator()

    def experiment(self) -> Optional[ExperimentBase]:
        """
        Concrete instance of :class:`ExperimentBase` constructed by experiment
        object data. ``None`` indicates invalid value.

        Run :meth:`updateExperiment` to update this value.
        """
        return self._expt

    @Slot()
    def updateLayerGenerator(self):
        expt = self.experiment()
        if expt is not None:
            self._layer_generator = expt.layer_generator()
            next(self.layerGenerator())
        else:
            self._layer_generator = None

    def layerGenerator(
        self,
    ) -> Generator[CoatingLayerBase, npt.NDArray[np.uint8], None]:
        """
        :meth:`Experiment.layer_generator` from :meth:`experiment`.
        ``None`` indicates invalid value.

        Run :meth:`updateLayerGenerator` to update this value.
        """
        return self._layer_generator

    def visualizationMode(self) -> ExperimentVisualizationMode:
        """If False, analysis result is never visualized."""
        return self._visualize_mode

    def setVisualizationMode(self, mode: ExperimentVisualizationMode):
        """
        Update :meth:`visualizationMode` with *mode*.
        Also, update :meth:`layerGenerator` if *mode* is True.
        """
        self._visualize_mode = mode
        if mode:
            self.updateLayerGenerator()

    def image(self) -> npt.NDArray[np.uint8]:
        """Layer shape image to be visualized."""
        img = self._img
        if img is None:
            img = np.empty((0, 0, 0))
        return img

    def setImage(self, img: Optional[npt.NDArray[np.uint8]]):
        """Update :meth:`image` with *img*."""
        self._img = img

    def visualizedImage(self) -> npt.NDArray[np.uint8]:
        """
        Return visualization result of :meth:`image` analyzed by
        :meth:`experiment`.

        If possible, :meth:`ExperimentBase.layer_generator` is used to construct
        the visualized image. This implies that consecutively visualized images
        are consecutive in real world as well. If completely unrelated images
        should be passed, run :meth:`updateLayerGenerator` first.

        If :meth:`visualizationMode` is ``FULL``, visualization is done by
        constructing analysis objects. If it is ``FAST``, image is visualized by
        :meth:`fastVisualize`.

        If parameters are invalid or :meth:`visualizationMode` is ``OFF``,
        directly return :meth:`image`.

        """
        ls_gen = self.layerGenerator()
        vismode = self.visualizationMode()
        image = self.image()
        if ls_gen is not None and image.size > 0:
            if vismode == ExperimentVisualizationMode.FULL:
                ls = ls_gen.send(image)
                while ls is None:  # previous construction not finished
                    ls = ls_gen.send(image)
                image = ls.draw()
            elif vismode == ExperimentVisualizationMode.FAST:
                image = self.fastVisualize()
            else:
                pass
        return image

    def fastVisualize(self) -> npt.NDArray[np.uint8]:
        """
        Remove substrate from :meth:`image` without constructing analysis
        objects.

        This visualization lacks the versatility of full visualization, but it is
        considerably faster for quick overview.
        """
        substrate = self.substrate()
        if substrate is None:
            return self.image()

        x0, y0, x1, y1 = substrate.reference.templateROI
        template = substrate.reference.image[y0:y1, x0:x1]
        res = cv2.matchTemplate(self.image(), template, cv2.TM_CCOEFF)
        _, _, _, (tx, ty) = cv2.minMaxLoc(res)
        dx, dy = substrate.reference.temp2subst()
        x0, y0 = (tx + dx, ty + dy)
        subst_h, subst_w = substrate.image().shape[:2]
        img_h, img_w = self.image().shape[:2]
        x1, y1 = (x0 + subst_w, y0 + subst_h)

        _, bin_img = cv2.threshold(
            self.image(), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        bin_img_cropped = bin_img[
            max(y0, 0) : min(y1, img_h), max(x0, 0) : min(x1, img_w)
        ]
        subst_cropped = substrate.image()[
            max(-y0, 0) : min(img_h - y0, subst_h),
            max(-x0, 0) : min(img_w - x0, subst_w),
        ]
        _, bin_subst_cropped = cv2.threshold(
            subst_cropped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        xor = cv2.bitwise_xor(bin_img_cropped, bin_subst_cropped)
        nxor = cv2.bitwise_not(xor)
        bin_img[max(y0, 0) : min(y1, img_h), max(x0, 0) : min(x1, img_w)] = nxor
        return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)

    def emitImage(self):
        """
        Emit the result of :meth:`visualizedImage` to
        :attr:`visualizedImageChanged` signal. If visualization raises error,
        directly emit :meth:`image`.
        """
        self.visualizedImageChanged.emit(self.visualizedImage())

    def clear(self):
        """
        Initialize experiment object data, :meth:`experiment`,
        :meth:`image` and :meth:`paths`.
        """
        self.initArgs()
