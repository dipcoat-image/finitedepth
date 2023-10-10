"""Coating layer factory.

This module defines abstract class :class:`ExperimentBase` and its
implementation, :class:`Experiment`.
"""

import abc
import dataclasses
from typing import TYPE_CHECKING, Generic, Optional, Tuple, Type, TypeVar

import cv2
import numpy as np
import numpy.typing as npt

from .coatinglayer import CoatingLayerBase
from .substrate import SubstrateBase

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "LayerTypeVar",
    "ParamTypeVar",
    "ExperimentBase",
    "ExptParam",
    "Experiment",
]


LayerTypeVar = TypeVar("LayerTypeVar", bound=CoatingLayerBase)
"""Type variable for the coating layer type of :class:`ExperimentBase`."""
ParamTypeVar = TypeVar("ParamTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`ExperimentBase.ParamType`."""


class ExperimentBase(abc.ABC, Generic[LayerTypeVar, ParamTypeVar]):
    """Abstract base class for experiment instance.

    Experiment is an act of acquiring useful data from coated substrate images.
    Experiment instance is a coating layer factory which achieves this by constructing
    coating layer instances, which provide the layer data. Sequential construction of
    coating layer instances can be done by :meth:`coatinglayer`.

    Concrete subclass must assign dataclasses types to class attribute :attr:`ParamType`
    which defines type of :attr:`parameters`.

    Arguments:
        parameters: Coating layer construction parameters.
            If passed, must be an instance of :attr:`ParamType`.
            If not passed, attempts to construct :attr:`ParamType`
            instance without argument.
    """

    ParamType: Type[ParamTypeVar]
    """Type of :attr:`parameters.`

    This class attribute is defined but not set in :class:`ExperimentBase`.
    Concrete subclass must assign this attribute with frozen dataclass type.
    """

    def __init__(self, *, parameters: Optional[ParamTypeVar] = None):
        """Initialize the instance.

        *parameters* must be instance of :attr:`ParamType` or :obj:`None`.
        If :obj:`None`, a :attr:`ParamType` is attempted to be constructed.
        """
        if parameters is None:
            self._parameters = self.ParamType()
        else:
            if not isinstance(parameters, self.ParamType):
                raise TypeError(f"{parameters} is not instance of {self.ParamType}")
            self._parameters = parameters

    @property
    def parameters(self) -> ParamTypeVar:
        """Coating layer construction parameters.

        This property returns a frozen dataclass instance.
        Its type is :attr:`ParamType`.

        Note:
            This dataclass must be frozen to ensure reproducible results.
        """
        return self._parameters

    @abc.abstractmethod
    def verify(self):
        """Sanity check before coating layer construction.

        This method checks :attr:`parameters` and raises error if anything is wrong.
        """

    @abc.abstractmethod
    def coatinglayer(
        self,
        image: npt.NDArray[np.uint8],
        substrate: SubstrateBase,
        *,
        layer_type: Type[LayerTypeVar],
        layer_parameters: Optional["DataclassInstance"] = None,
        layer_drawoptions: Optional["DataclassInstance"] = None,
        layer_decooptions: Optional["DataclassInstance"] = None,
    ) -> LayerTypeVar:
        """Construct coating layer instance.

        Concrete class should implement this method. Especially, sequential construction
        can use data from the previous instance to affect the next instance.
        If required, :meth:`parameters` can be used to controll consecutive creation.

        A possible implementation of this method is to read the location of substrate
        from external source and pass it to *tempmatch* argument of the coating layer.
        """


@dataclasses.dataclass(frozen=True)
class ExptParam:
    """Coating layer construction parameters for :class:`Experiment` instance.

    Arguments:
        window: Restricts the possible location of template to boost speed.
            Negative value means no restriction in corresponding axis.
    """

    window: Tuple[int, int] = (-1, -1)


class Experiment(ExperimentBase[CoatingLayerBase, ExptParam]):
    """Experiment class with adjustable template matching window.

    Sequential construction of coating layer instances is boosted by restricting the
    template matching window.

    Arguments:
        parameters (ExptParam, optional)
    """

    ParamType = ExptParam
    """Assigned with :class:`ExptParam`."""

    def verify(self):
        """Implement :meth:`ExperimentBase.verify`."""
        pass

    def coatinglayer(
        self,
        image,
        substrate,
        layer_type,
        layer_parameters=None,
        layer_drawoptions=None,
        layer_decooptions=None,
    ):
        """Implement :meth:`ExperimentBase.coatinglayer`.

        Using the previous template location and :attr:`parameters`, this method
        performs fast template matching with reduced window and pass the result to
        the new coating layer instance.
        """
        prev = getattr(self, "_prev", None)
        x0, y0, x1, y1 = substrate.reference.templateROI
        window = self.parameters.window
        if not prev:
            target_image = image
            X0, Y0 = 0, 0
        else:
            X, Y = prev
            w0, h0 = window
            w1, h1 = x1 - x0, y1 - y0
            H, W = image.shape[:2]

            if w0 < 0:
                X0, X1 = 0, None
            else:
                X0, X1 = max(X - w0, 0), min(X + w1 + w0, W)
            if h0 < 0:
                Y0, Y1 = 0, None
            else:
                Y0, Y1 = max(Y - h0, 0), min(Y + h1 + h0, H)
            target_image = image[Y0:Y1, X0:X1]

        template = substrate.reference.image[y0:y1, x0:x1]
        res = cv2.matchTemplate(target_image, template, cv2.TM_SQDIFF_NORMED)
        score, _, (x, y), _ = cv2.minMaxLoc(res)
        loc = (X0 + x, Y0 + y)
        ret = layer_type(
            image,
            substrate,
            parameters=layer_parameters,
            draw_options=layer_drawoptions,
            deco_options=layer_decooptions,
            tempmatch=(loc, score),
        )
        self._prev = loc
        return ret
