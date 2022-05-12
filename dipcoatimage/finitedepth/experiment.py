"""
Experiment
==========

:mod:`dipcoatimage.finitedepth.experiment` provides factory to construct coating
layer objects.

A finite-depth dip coating experiment consists of:

* Bare substrate image
* Coated substrate images

With each coated substrate image, experiment class constructs coating layer
instance using common bare substrate instance.

Base class
----------

.. autoclass:: ExperimentError
   :members:

.. autoclass:: ExperimentBase
   :members:

Implementation
--------------

.. autoclass:: ExperimentParameters
   :members:

.. autoclass:: Experiment
   :members:

"""

import abc
import dataclasses
import numpy as np
import numpy.typing as npt
from typing import TypeVar, Generic, Type, Optional, Generator

from .substrate import SubstrateBase
from .coatinglayer import CoatingLayerBase
from .util import DataclassProtocol


__all__ = [
    "ExperimentError",
    "ExperimentBase",
    "ExperimentParameters",
    "Experiment",
]


CoatingLayerType = TypeVar("CoatingLayerType", bound=CoatingLayerBase)
ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)


class ExperimentError(Exception):
    """Base class for error from :class:`ExperimentBase`."""

    pass


class ExperimentBase(abc.ABC, Generic[CoatingLayerType, ParametersType]):
    """
    Abstract base class for finite-depth dip coating experiment.

    Experiment consists of a bare substrate image and coated substrate images.
    Experiment class is a factory which constructs :class:`CoatingLayerBase`
    instances from the data.

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

    Coating layer is constructed by :meth:`construct_coatinglayer`. When
    analyzing consecutive images, coating layer parameters may need to be
    different for each instance. To support this, image number and previous
    instance can be passed. Subclass may override this method to apply different
    parameters.

    :meth:`layer_generator` returns a generator which receives coated substrate
    image and automatically calls :meth:`construct_coatinglayer` with image
    number and previous instance.

    Parameters
    ==========

    substrate
        Substrate instance.

    layer_type
        Concrete subclass of :class:`CoatingLayerBase`.

    layer_parameters, layer_drawoptions, layer_decooptions
        *parameters*, *draw_options*, and *deco_options* arguments for
        *layer_type*.

    parameters
        Additional parameters.

    """

    Parameters: Type[ParametersType]

    def __init__(
        self,
        substrate: SubstrateBase,
        layer_type: Type[CoatingLayerType],
        layer_parameters: Optional[DataclassProtocol] = None,
        layer_drawoptions: Optional[DataclassProtocol] = None,
        layer_decooptions: Optional[DataclassProtocol] = None,
        *,
        parameters: Optional[ParametersType] = None,
    ):
        self.substrate = substrate
        self.layer_type = layer_type

        if layer_parameters is None:
            self.layer_parameters = self.layer_type.Parameters()
        else:
            self.layer_parameters = dataclasses.replace(layer_parameters)

        if layer_drawoptions is None:
            self.layer_drawoptions = self.layer_type.DrawOptions()
        else:
            self.layer_drawoptions = dataclasses.replace(layer_drawoptions)

        if layer_decooptions is None:
            self.layer_decooptions = self.layer_type.DecoOptions()
        else:
            self.layer_decooptions = dataclasses.replace(layer_decooptions)

        if parameters is None:
            self.parameters = self.Parameters()
        else:
            self.parameters = dataclasses.replace(parameters)

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
        err = self.examine()
        ret = True
        if err is not None:
            ret = False
        return ret

    def construct_coatinglayer(
        self,
        image: npt.NDArray[np.uint8],
        i: int = 0,
        prev: Optional[CoatingLayerBase] = None,
    ) -> CoatingLayerBase:
        """
        Construct instance of :attr:`layer_type` with *image* and attributes.

        *i* and *prev* are passed to allow passing different parameters for
        each coating layer instance. Subclass may override this method modify
        the parameters.

        Parameters
        ==========

        image
            *image* argument for coating layer class.

        i
            Frame number for *img*.

        prev
            Previous coating layer instance.

        """
        ret = self.layer_type(
            image,
            self.substrate,
            parameters=self.layer_parameters,
            draw_options=self.layer_drawoptions,
            deco_options=self.layer_decooptions,
        )
        return ret

    def layer_generator(
        self,
    ) -> Generator[CoatingLayerBase, npt.NDArray[np.uint8], None]:
        """
        Generator which receives coated substrate image to yield instance of
        :attr:`layer_type`.

        As new image is sent, coating layer instance is created using
        :meth:`construct_coatinglayer` with the number of images passed so far
        and previous coating layer instance.

        """
        i = 0
        prev = None
        while True:
            img = yield  # type: ignore
            layer = self.construct_coatinglayer(img, i, prev)
            yield layer
            prev = layer
            i += 1


@dataclasses.dataclass(frozen=True)
class ExperimentParameters:
    """Additional parameters for :class:`Experiment` instance."""

    pass


class Experiment(ExperimentBase[CoatingLayerBase, ExperimentParameters]):
    """
    Simplest experiment class with no parameter.

    """

    Parameters = ExperimentParameters

    def examine(self) -> None:
        return None
