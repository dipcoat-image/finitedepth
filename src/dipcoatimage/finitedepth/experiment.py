"""Coating layer factory."""

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
    "ExperimentError",
    "ExperimentBase",
    "Experiment",
]


CoatingLayerType = TypeVar("CoatingLayerType", bound=CoatingLayerBase)
ParametersType = TypeVar("ParametersType", bound="DataclassInstance")


class ExperimentError(Exception):
    """Base class for error from `ExperimentBase`."""

    pass


class ExperimentBase(abc.ABC, Generic[ParametersType]):
    """Abstract base class for coating layer factory.

    Experiment is an act of transforming incoming coated substrate images to
    coating layer data by processing them agains the bare substrate.
    `ExperimentBase` provides structured way to define transformation of
    a series of images.

    .. rubric:: Constructor

    Constructor signature must not be modified because high-level API use factory
    to generate experiment instances. Additional parameters can be introduced
    by definig class attribute :attr:`Parameters``.

    .. rubric:: Parameters

    Concrete class must have :attr:`Parameters` which returns dataclass type.
    Its instance is passed to the constructor at instance initialization, and can
    be accessed by :attr:`parameters`.

    .. rubric:: Sanity check

    Validity of the parameters can be checked by :meth:`verify`.

    .. rubric:: Coating layer construction

    :meth:`coatinglayer` method is responsible for transforming each coated
    substrate image into a coating layer instance. Finding the location of the
    substrate can be implemented in this method. Standard implementation use
    template matching, but another possible way is to use physically measured
    data (e.g., actuator log).
    """

    __slots__ = ("_parameters",)

    Parameters: Type[ParametersType]

    def __init__(self, *, parameters: Optional[ParametersType] = None):
        """Initialize the instance."""
        if parameters is None:
            self._parameters = self.Parameters()
        else:
            self._parameters = dataclasses.replace(parameters)

    @property
    def parameters(self) -> ParametersType:
        """Coating layer construction parameters."""
        return self._parameters

    @abc.abstractmethod
    def verify(self):
        """Check to detect error and raise before analysis."""

    @abc.abstractmethod
    def coatinglayer(
        self,
        image: npt.NDArray[np.uint8],
        substrate: SubstrateBase,
        *,
        layer_type: Type[CoatingLayerType],
        layer_parameters: Optional["DataclassInstance"] = None,
        layer_drawoptions: Optional["DataclassInstance"] = None,
        layer_decooptions: Optional["DataclassInstance"] = None,
    ) -> CoatingLayerBase:
        """Create coating layer.

        Implementation may define custom way to create new instance. For example,
        substrate location in previous image can be stored to boost template
        matching of incoming images. If required, :meth:`parameters` can be
        used to controll consecutive creation.
        """


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Additional parameters for `Experiment` instance.

    Attributes
    ----------
    window : tuple
        Restricts the possible location of template to boost speed.
        Negative value means no restriction in corresponding axis.
    """

    window: Tuple[int, int] = (-1, -1)


class Experiment(ExperimentBase[Parameters]):
    """Experiment class with adjustable template matching window.

    Specifying the window can significantly boost the evaluation.
    """

    __slots__ = ("_prev",)

    Parameters = Parameters

    def verify(self):
        """Check error."""
        pass

    def coatinglayer(
        self,
        image,
        substrate,
        *,
        layer_type,
        layer_parameters=None,
        layer_drawoptions=None,
        layer_decooptions=None,
    ):
        """Create coating layer.

        If *window* parameter has positive axis, template matching is boosted.
        """
        prev = getattr(self, "_prev", None)
        window = self.parameters.window
        if not prev:
            x0, y0, x1, y1 = substrate.reference.templateROI
        else:
            x0, y0, x1, y1 = substrate.reference.templateROI
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
            image = image[Y0:Y1, X0:X1]

        template = substrate.reference.image[y0:y1, x0:x1]
        res = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
        score, _, loc, _ = cv2.minMaxLoc(res)
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
