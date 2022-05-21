import cv2  # type: ignore
from dipcoatimage.finitedepth import (
    SubstrateBase,
    CoatingLayerBase,
    ExperimentBase,
)
from dipcoatimage.finitedepth.util import DataclassProtocol
from dipcoatimage.finitedepth_gui.core import (
    StructuredCoatingLayerArgs,
    StructuredExperimentArgs,
    VisualizationMode,
)
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Slot
from typing import Optional, Type
from .base import WorkerBase


__all__ = [
    "ExperimentWorker",
]


class ExperimentWorker(WorkerBase):
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

    :meth:`substrate` is updated by :meth:`setSubstrate`, and other data are
    updated by :meth:`setStructuredCoatingLayerArgs`. and
    :meth:`setStructuredExperimentArgs`

    :meth:`updateExperiment` constructs the experiment object with data.
    Resulting object can be acquired by :meth:`experiment`.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._type = None
        self._subst = None
        self._layer_type = None
        self._layer_params = None
        self._layer_drawopts = None
        self._layer_decoopts = None
        self._params = None

        self._expt = None
        self._layer_generator = None

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

    def coatingLayerDrawOptions(self) -> Optional[DataclassProtocol]:
        """
        *draw_options* for :meth:`coatingLayerType` to construct
        :meth:`experiment`. ``None`` indicates invalid value.
        """
        return self._layer_drawopts

    def coatingLayerDecoOptions(self) -> Optional[DataclassProtocol]:
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

    @Slot(object)
    def setSubstrate(self, subst: Optional[SubstrateBase]):
        """
        Set :meth:`substrate` with *subst*.

        This does not update :meth:`experiment`. Run :meth:`updateExperiment`
        manually.
        """
        self._subst = subst

    def setStructuredCoatingLayerArgs(self, data: StructuredCoatingLayerArgs):
        """
        Set following values with *data*.

        1. :meth:`coatingLayerType`
        2. :meth:`coatingLayerParameters`
        3. :meth:`coatingLayerDrawOptions`
        4. :meth:`coatingLayerDecoOptions`

        This does not update :meth:`experiment`. Run :meth:`updateExperiment`
        manually.
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

        coat_decoopts = data.deco_options
        if coattype is None:
            coat_decoopts = None
        elif isinstance(coat_decoopts, coattype.DecoOptions):
            pass
        else:
            try:
                coat_decoopts = coattype.DecoOptions()
            except TypeError:
                coat_decoopts = None
        self._layer_decoopts = coat_decoopts

    def setStructuredExperimentArgs(self, data: StructuredExperimentArgs):
        """
        Set following values with *data*.

        1. :meth:`experimentType`
        2. :meth:`experimentParameters`

        This does not update :meth:`experiment`. Run :meth:`updateExperiment`
        manually.
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

    def updateExperiment(self):
        """Update :meth:`experiment`."""
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

    def experiment(self) -> Optional[ExperimentBase]:
        """
        Concrete instance of :class:`ExperimentBase` constructed by experiment
        object data. ``None`` indicates invalid value.

        Run :meth:`updateExperiment` to update this value.
        """
        return self._expt

    def visualizeImage(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Return visualization result of *img* analyzed by :meth:`experiment`.

        """
        expt = self.experiment()
        vismode = self.visualizationMode()
        if expt is not None and img.size > 0:
            if vismode == VisualizationMode.FULL:
                layer = expt.construct_coatinglayer(img)
                img = layer.draw()
            elif vismode == VisualizationMode.FAST:
                img = self.fastVisualize(img)
            else:
                pass
        return img

    def fastVisualize(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Remove substrate from *img* without constructing analysis objects.

        This visualization lacks the versatility of full visualization, but it is
        considerably faster for quick overview.
        """
        substrate = self.substrate()
        if substrate is None:
            return img

        x0, y0, x1, y1 = substrate.reference.templateROI
        template = substrate.reference.image[y0:y1, x0:x1]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
        _, _, _, (tx, ty) = cv2.minMaxLoc(res)
        dx, dy = substrate.reference.temp2subst()
        x0, y0 = (tx + dx, ty + dy)
        subst_h, subst_w = substrate.image().shape[:2]
        img_h, img_w = img.shape[:2]
        x1, y1 = (x0 + subst_w, y0 + subst_h)

        _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
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
