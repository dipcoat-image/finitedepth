from dipcoatimage.finitedepth import SubstrateReferenceBase
from dipcoatimage.finitedepth.util import OptionalROI, DataclassProtocol
from dipcoatimage.finitedepth_gui.core import StructuredReferenceArgs
import numpy as np
import numpy.typing as npt
from typing import Optional, Type
from .base import WorkerBase


__all__ = [
    "ReferenceWorker",
]


class ReferenceWorker(WorkerBase):
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
    :meth:`setStructuredReferenceArgs`.

    :meth:`updateReference` constructs the reference object with current data.
    Resulting object can be acquired from :meth:`reference`.

    Visualization result can be directly acquired from :meth:`visualizedImage`.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
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
        Set :meth:`image` with *img*.

        This does not update :meth:`reference`. Run :meth:`updateReference`
        manually.
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

    def setStructuredReferenceArgs(self, data: StructuredReferenceArgs):
        """
        Set following values with *data*.

        1. :meth:`referenceType`
        2. :meth:`image`
        3. :meth:`templateROI`
        4. :meth:`substrateROI`
        5. :meth:`paramters`
        6. :meth:`drawOptions`

        This does not update :meth:`reference`. Run :meth:`updateReference`
        manually.
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
        """Update :meth:`reference`."""
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

    def clear(self):
        self._type = None
        self._img = None
        self._temproi = (0, 0, None, None)
        self._substroi = (0, 0, None, None)
        self._params = None
        self._draw_opts = None
        self.updateReference()
