"""
Experiment
==========

:mod:`dipcoatimage.finitedepth.experiment` helps analyze the full dip coating
experiment with finite-depth substrate.

Single experiment consists of:

    * Bare substrate image file
    * Coated substrate image file(s), or video file

This module provides factory to handle consecutive images of coated substrate
with single image of bare substrate. Also, it serializes the parameters to
analyze the images and to save the result.

----------------
Analysis factory
----------------

Base class
----------

.. autoclass:: ExperimentError
   :members:

.. autoclass:: ExperimentBase
   :members:

Implementation
--------------

------------------
Data serialization
------------------

.. autoclass:: ImportArgs
   :members:

.. autoclass:: ReferenceArgs
   :members:

.. autoclass:: SubstrateArgs
   :members:

.. autoclass:: CoatingLayerArgs
   :members:

"""

import abc
import cv2  # type: ignore
import dataclasses
import numpy as np
import numpy.typing as npt
from typing import TypeVar, Generic, Type, Optional, Generator

from .reference import SubstrateReferenceBase
from .substrate import SubstrateBase
from .coatinglayer import CoatingLayerBase
from .util import DataclassProtocol, import_variable, data_converter, OptionalROI


__all__ = [
    "ExperimentError",
    "ExperimentBase",
    "ImportArgs",
    "ReferenceArgs",
    "SubstrateArgs",
    "CoatingLayerArgs",
]


ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)


class ExperimentError(Exception):
    """Base class for error from :class:`ExperimentBase`."""

    pass


class ExperimentBase(abc.ABC, Generic[ParametersType]):
    """
    Abstract base class for finite-depth dip coating experiment.

    Dip coating experiment consists of reference image, and coated substrate
    image(s) or video. Experiment class is a factory which constructs
    :class:`CoatingLayerBase` instances from the data.

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
        layer_type: Type[CoatingLayerBase],
        layer_parameters: Optional[DataclassProtocol] = None,
        layer_drawoptions: Optional[DataclassProtocol] = None,
        layer_decooptions: Optional[DataclassProtocol] = None,
        *,
        parameters: Optional[ParametersType] = None,
    ):
        self.substrate = substrate
        self.layer_type = layer_type
        self.layer_parameters = layer_parameters
        self.layer_drawoptions = layer_drawoptions
        self.layer_decooptions = layer_decooptions

        if parameters is None:
            self._parameters = self.Parameters()
        else:
            self._parameters = dataclasses.replace(parameters)

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
        i: int,
        image: npt.NDArray[np.uint8],
        prev: Optional[CoatingLayerBase] = None,
    ) -> CoatingLayerBase:
        """
        Construct instance of :attr:`layer_type` with *image* and attributes.

        *i* and *prev* are passed to allow passing different parameters for
        each coating layer instance. Subclass may override this method modify
        the parameters.

        Parameters
        ==========

        i
            Frame number for *img*.

        image
            *image* argument for coating layer class.

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
        and previous instance.

        Sending non-consecutive images may result incorrect result.

        """
        i = 0
        prev_ls = None
        while True:
            img = yield  # type: ignore
            ls = self.construct_coatinglayer(i, img, prev_ls)
            yield ls
            prev_ls = ls
            i += 1


@dataclasses.dataclass
class ImportArgs:
    """Arguments to import the variable from module."""

    name: str = ""
    module: str = "dipcoatimage.finitedepth"


@dataclasses.dataclass
class ReferenceArgs:
    """
    Data for the concrete instance of :class:`SubstrateReferenceBase`.

    Parameters
    ==========

    type
        Information to import reference class.
        Class name defaults to ``SubstrateReference``.

    path
        Path to the reference image file.

    templateROI, substrateROI, parameters, draw_options
        Arguments for :class:`SubstrateReferenceBase` instance.

    Examples
    ========

    .. plot::
       :include-source:

       >>> from dipcoatimage.finitedepth import get_samples_path
       >>> from dipcoatimage.finitedepth.experiment import ReferenceArgs
       >>> from dipcoatimage.finitedepth.util import cwd
       >>> refargs = ReferenceArgs(path="ref1.png",
       ...                         templateROI=(200, 100, 1200, 500),
       ...                         substrateROI=(300, 50, 1100, 600))
       >>> with cwd(get_samples_path()):
       ...     ref = refargs.as_reference()
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    """

    type: ImportArgs = dataclasses.field(default_factory=ImportArgs)
    path: str = ""
    templateROI: OptionalROI = (0, 0, None, None)
    substrateROI: OptionalROI = (0, 0, None, None)
    parameters: dict = dataclasses.field(default_factory=dict)
    draw_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.type.name:
            self.type.name = "SubstrateReference"

    def as_reference(self) -> SubstrateReferenceBase:
        """Construct the substrate reference instance."""
        name = self.type.name
        module = self.type.module
        refcls = import_variable(name, module)
        if not (
            isinstance(refcls, type) and issubclass(refcls, SubstrateReferenceBase)
        ):
            raise TypeError(f"{refcls} is not substrate reference class.")

        img = cv2.imread(self.path)
        if img is None:
            raise TypeError(f"Invalid path: {self.path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        params = data_converter.structure(
            self.parameters, refcls.Parameters  # type: ignore
        )
        drawopts = data_converter.structure(
            self.draw_options, refcls.DrawOptions  # type: ignore
        )

        ref = refcls(  # type: ignore
            img,
            self.templateROI,
            self.substrateROI,
            parameters=params,
            draw_options=drawopts,
        )
        return ref


@dataclasses.dataclass
class SubstrateArgs:
    """
    Data for the concrete instance of :class:`SubstrateBase`.

    Parameters
    ==========

    type
        Information to import substrate class.
        Class name defaults to ``Substrate``.

    parameters, draw_options
        Arguments for :class:`SubstrateBase` instance.

    Examples
    ========

    Construct substrate reference instance first.

    .. plot::
       :include-source:
       :context: reset

       >>> from dipcoatimage.finitedepth import get_samples_path
       >>> from dipcoatimage.finitedepth.experiment import ReferenceArgs
       >>> from dipcoatimage.finitedepth.util import cwd
       >>> refargs = ReferenceArgs(path="ref1.png",
       ...                         templateROI=(200, 100, 1200, 500),
       ...                         substrateROI=(300, 50, 1100, 600))
       >>> with cwd(get_samples_path()):
       ...     ref = refargs.as_reference()

    Construct substrate instance from the data.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import data_converter
       >>> from dipcoatimage.finitedepth.experiment import SubstrateArgs
       >>> params = {"Canny": {"threshold1": 50, "threshold2": 150},
       ...           "HoughLines": {"rho": 1, "theta": 0.01, "threshold": 100}}
       >>> arg = dict(type={"name": "RectSubstrate"}, parameters=params)
       >>> substargs = data_converter.structure(arg, SubstrateArgs)
       >>> subst = substargs.as_substrate(ref)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    """

    type: ImportArgs = dataclasses.field(default_factory=ImportArgs)
    parameters: dict = dataclasses.field(default_factory=dict)
    draw_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.type.name:
            self.type.name = "Substrate"

    def as_substrate(self, ref: SubstrateReferenceBase) -> SubstrateBase:
        """
        Construct the substrate instance.

        Parameters
        ==========

        img
            Substrate reference instance.

        """
        name = self.type.name
        module = self.type.module
        substcls = import_variable(name, module)
        if not (isinstance(substcls, type) and issubclass(substcls, SubstrateBase)):
            raise TypeError(f"{substcls} is not substrate class.")

        params = data_converter.structure(
            self.parameters, substcls.Parameters  # type: ignore
        )
        drawopts = data_converter.structure(
            self.draw_options, substcls.DrawOptions  # type: ignore
        )
        subst = substcls(  # type: ignore
            ref,
            parameters=params,
            draw_options=drawopts,
        )
        return subst


@dataclasses.dataclass
class CoatingLayerArgs:
    """
    Data for the concrete instance of :class:`CoatingLayerBase`.

    Parameters
    ==========

    type
        Information to import substrate class.
        Class name defaults to ``LayerArea``.

    parameters, draw_options, deco_options
        Arguments for :class:`CoatingLayerBase` instance.

    Examples
    ========

    Construct substrate reference instance and substrate instance first.

    .. plot::
       :include-source:
       :context: reset

       >>> from dipcoatimage.finitedepth import get_samples_path, data_converter
       >>> from dipcoatimage.finitedepth.experiment import (ReferenceArgs,
       ...     SubstrateArgs)
       >>> from dipcoatimage.finitedepth.util import cwd
       >>> refargs = ReferenceArgs(path="ref1.png",
       ...                         templateROI=(200, 100, 1200, 500),
       ...                         substrateROI=(300, 50, 1100, 600))
       >>> with cwd(get_samples_path()):
       ...     ref = refargs.as_reference()
       >>> params = {"Canny": {"threshold1": 50, "threshold2": 150},
       ...           "HoughLines": {"rho": 1, "theta": 0.01, "threshold": 100}}
       >>> arg = dict(type={"name": "RectSubstrate"}, parameters=params)
       >>> substargs = data_converter.structure(arg, SubstrateArgs)
       >>> subst = substargs.as_substrate(ref)

    Construct coating layer instance from the data.

    .. plot::
       :include-source:
       :context: close-figs

       >>> import cv2
       >>> from dipcoatimage.finitedepth.experiment import CoatingLayerArgs
       >>> arg = dict(type={"name": "RectLayerArea"})
       >>> layerargs = data_converter.structure(arg, CoatingLayerArgs)
       >>> img_path = get_samples_path("coat1.png")
       >>> layer = layerargs.as_coatinglayer(cv2.imread(img_path), subst)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(layer.draw()) #doctest: +SKIP

    """

    type: ImportArgs = dataclasses.field(default_factory=ImportArgs)
    parameters: dict = dataclasses.field(default_factory=dict)
    draw_options: dict = dataclasses.field(default_factory=dict)
    deco_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.type.name:
            self.type.name = "LayerArea"

    def as_coatinglayer(
        self, img: npt.NDArray[np.uint8], subst: SubstrateBase
    ) -> CoatingLayerBase:
        """
        Construct the coating layer instance.

        Parameters
        ==========

        img
            Coated substrate image. May be grayscale or RGB.

        subst
            Substrate instance.

        """
        name = self.type.name
        module = self.type.module
        layercls = import_variable(name, module)
        if not (isinstance(layercls, type) and issubclass(layercls, CoatingLayerBase)):
            raise TypeError(f"{layercls} is not coating layer class.")

        params = data_converter.structure(
            self.parameters, layercls.Parameters  # type: ignore
        )
        drawopts = data_converter.structure(
            self.draw_options, layercls.DrawOptions  # type: ignore
        )
        decoopts = data_converter.structure(
            self.deco_options, layercls.DecoOptions  # type: ignore
        )
        layer = layercls(  # type: ignore
            img, subst, parameters=params, draw_options=drawopts, deco_options=decoopts
        )
        return layer
