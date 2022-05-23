from dipcoatimage.finitedepth import SubstrateReferenceBase, SubstrateBase
from dipcoatimage.finitedepth.util import DataclassProtocol
from dipcoatimage.finitedepth_gui.core import StructuredSubstrateArgs
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Slot
from typing import Optional, Type
from .base import WorkerBase


__all__ = [
    "SubstrateWorker",
]


class SubstrateWorker(WorkerBase):
    """
    Worker to build the concreate instance of :class:`SubstrateBase` and to
    visualize it.

    Data for substrate object are:

    1. :meth:`substrateType`
    2. :meth:`reference`
    3. :meth:`paramters`
    4. :meth:`drawOptions`

    :meth:`reference` is updated by :meth:`setReference`, and other data are
    updated by :meth:`setStructuredSubstrateArgs`.

    :meth:`updateSubstrate` constructs the substrate object with current data.
    Resulting object can be acquired from :meth:`substrate`.

    Visualization result can be directly acquired from :meth:`visualizedImage`.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
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

    def setStructuredSubstrateArgs(self, data: StructuredSubstrateArgs):
        """
        Set following values with *data*.

        1. :meth:`substrateType`
        2. :meth:`paramters`
        3. :meth:`drawOptions`

        This does not update :meth:`substrate`. Run :meth:`updateSubstrate`
        manually.
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

    @Slot(object)
    def setReference(self, ref: Optional[SubstrateReferenceBase]):
        """
        Set :meth:`reference` with *ref*.

        This does not update :meth:`substrate`. Run :meth:`updateSubstrate`
        manually.
        """
        self._ref = ref

    def updateSubstrate(self):
        """Update :meth:`substrate`."""
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

    def clear(self):
        self._type = None
        self._ref = None
        self._params = None
        self._draw_opts = None
        self.updateSubstrate()
