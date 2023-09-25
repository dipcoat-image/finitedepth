"""
Experiment
==========

:mod:`dipcoatimage.finitedepth.experiment` provides factory to construct
coating layer objects.

Base class
----------

.. autoclass:: ExperimentError
   :members:

.. autoclass:: ExperimentBase
   :members:

Implementation
--------------

.. autoclass:: Experiment
   :members:

"""

import abc
import dataclasses
import numpy as np
import numpy.typing as npt
from .substrate import SubstrateBase
from .coatinglayer import CoatingLayerBase
from .experiment_param import Parameters
from typing import TypeVar, Generic, Type, Optional, TYPE_CHECKING

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
    """Base class for error from :class:`ExperimentBase`."""

    pass


class ExperimentBase(abc.ABC, Generic[ParametersType]):
    """
    Abstract base class for coating layer factory.

    Experiment is an act of transforming incoming coated substrate images to
    coating layer data by processing them agains the bare substrate.
    :class:`ExperimentBase` provides structured way to define transformation of
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

    Validity of the parameters can be checked by :meth:`verify` or :meth:`valid`.
    Their result can be implemented by defining :meth:`examine`.

    .. rubric:: Coating layer construction

    :meth:`coatinglayer` method is responsible for transforming each coated
    substrate image into a coating layer instance.

    """

    __slots__ = ("_parameters",)

    Parameters: Type[ParametersType]

    def __init__(self, *, parameters: Optional[ParametersType] = None):
        if parameters is None:
            self._parameters = self.Parameters()
        else:
            self._parameters = dataclasses.replace(parameters)

    @property
    def parameters(self) -> ParametersType:
        return self._parameters

    @abc.abstractmethod
    def examine(self) -> Optional[ExperimentError]:
        """
        Check the sanity of parameters.

        If the instance is invalid, return error instance.
        Else, return :obj:`None`.
        """

    def verify(self):
        """
        Verify if all parameters are suitably set by raising error on failure.

        To implement sanity check for concrete class, define :meth:`examine`.
        """
        err = self.examine()
        if err is not None:
            raise err

    def valid(self) -> bool:
        """
        Verify if all parameters are suitably set by returning boolean value.

        To implement sanity check for concrete class, define :meth:`examine`.
        """
        return self.examine() is None

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
        """
        Factory method to create coating layer.

        Implementation may define custom way to create new instance. For example,
        substrate location in previous image can be stored to boost template
        matching of incoming images. If required, :meth:`parameters` can be
        used to controll consecutive creation.
        """


class Experiment(ExperimentBase[Parameters]):
    """
    Experiment with template matching optimization.
    """

    __slots__ = ("_prev",)

    Parameters = Parameters

    def examine(self) -> None:
        return None

    def object_function(
        self, T: npt.NDArray[np.uint8], I: npt.NDArray[np.uint8], x: int, y: int
    ) -> float:
        h, w = T.shape[:2]
        I_crop = I[y: y + h, x: x + w]
        num = int(np.sum((T - I_crop)**2))
        denom = np.sqrt(int(np.sum(T**2)) * int(np.sum(I_crop**2)))
        return num / denom

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
        prev = getattr(self, "_prev", None)
        if not prev:
            ret = layer_type(
                image,
                substrate,
                parameters=layer_parameters,
                draw_options=layer_drawoptions,
                deco_options=layer_decooptions,
            )
            loc, _ = ret.tempmatch
        else:
            # TODO: implement optimization
            ret = layer_type(
                image,
                substrate,
                parameters=layer_parameters,
                draw_options=layer_drawoptions,
                deco_options=layer_decooptions,
            )
            loc, _ = ret.tempmatch
        if self.parameters.fast:
            self._prev = loc
        return ret
