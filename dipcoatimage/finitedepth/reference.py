"""
Reference image
===============

:mod:`dipcoatimage.finitedepth.reference` provides class for reference image of
finite-depth substrate. It is used to analyze coated substrate image.

"""


import abc
import dataclasses
import numpy as np
import numpy.typing as npt
from typing import TypeVar, Type, Optional, cast, Tuple
from .util import DataclassProtocol, OptionalROI, IntROI


__all__ = ["SubstrateReferenceBase"]


ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)


class SubstrateReferenceBase(abc.ABC):
    """
    Abstract base class for substrate reference image.

    Image and ROIs
    --------------

    :attr:`image` is the picture of bare substrate before coating. Two ROIs,
    :attr:`templateROI` and :attr:`substrateROI`, are defined. Template ROI
    encloses the region which is common in both bare substrate image and coated
    substrate image. Substrate ROI encloses the entire bare substrate, narrowing
    down the target.

    Constructor
    -----------

    Constructor signature must not be modified because high-level API use factory
    to generate reference instances. Concrete classes can introduce additional
    parameters by defining dataclass and assigning to class attribute
    :attr:`Parameters` and :attr:`DrawOptions`.

    Sanity check
    ------------

    Validity of the parameters can be checked by :meth:`verify` or :meth:`valid`.

    Parameters
    ==========

    image
        Reference image.

    templateROI, substrateROI
        Slice indices in ``(x0, y0, x1, y1)`` for the template and the substrate.

    parameters
        Additional parameters. Instance of :attr:`Parameters`, or ``None``.

    draw_options
        Drawing options. Instance of :attr:`DrawOptions`, or ``None``.

    Notes
    =====

    Some properties and methods can be cached for performance. Thus :attr:`image`
    is not writable to prevent mutation. Also, concrete class must assign frozen
    dataclass type to :attr:`Parameters`. However, :attr:`DrawOptions` need not
    be frozen since visualization does not affect the identity of instance.
    """

    __slots__ = (
        "_image",
        "_templateROI",
        "_substrateROI",
        "_parameters",
        "_draw_options",
    )

    @property
    @abc.abstractmethod
    def Parameters(self) -> Type:
        """Frozen dataclass type for additional parameters in concrete class."""

    @property
    @abc.abstractmethod
    def DrawOptions(self) -> Type:
        """Dataclass type for drawing options in concrete class."""

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        templateROI: OptionalROI = (0, 0, None, None),
        substrateROI: OptionalROI = (0, 0, None, None),
        parameters: Optional[ParametersType] = None,
        *,
        draw_options: Optional[DrawOptionsType] = None
    ):
        super().__init__()
        self._image = image
        self._image = image
        self._image.setflags(write=False)

        h, w = image.shape[:2]
        full_roi = (0, 0, w, h)
        max_vars = (w, h, w, h)

        temp_roi = list(templateROI)
        for i, var in enumerate(templateROI):
            if var is None:
                temp_roi[i] = full_roi[i]
            elif var < 0:
                temp_roi[i] = max_vars[i] + var
        self._templateROI = cast(IntROI, tuple(temp_roi))

        subst_roi = list(substrateROI)
        for i, var in enumerate(substrateROI):
            if var is None:
                subst_roi[i] = full_roi[i]
            elif var < 0:
                subst_roi[i] = max_vars[i] + var
        self._substrateROI = cast(IntROI, tuple(subst_roi))

        if parameters is None:
            self._parameters = self.Parameters()
        else:
            self._parameters = dataclasses.replace(parameters)

        if draw_options is None:
            self._draw_options = self.DrawOptions()
        else:
            self._draw_options = dataclasses.replace(draw_options)

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """Reference image passed to constructor."""
        return self._image

    @property
    def templateROI(self) -> IntROI:
        """Slice indices in ``(x0, y0, x1, y1)`` for :attr:`template_image`."""
        return self._templateROI

    @property
    def substrateROI(self) -> IntROI:
        """Slice indices in ``(x0, y0, x1, y1)`` for :attr:`substrate_image`."""
        return self._substrateROI

    @property
    def parameters(self) -> ParametersType:
        """Additional parameters for the instance."""
        return self._parameters

    @property
    def draw_options(self) -> DrawOptionsType:
        """Options to visualize the reference image."""
        return self._draw_options

    @draw_options.setter
    def draw_options(self, options: DrawOptionsType):
        self._draw_options = options

    @property
    def template_image(self) -> npt.NDArray[np.uint8]:
        """Template image to locate the substrate in coated substrate image."""
        x0, y0, x1, y1 = self.templateROI
        return self.image[y0:y1, x0:x1]

    @property
    def substrate_image(self) -> npt.NDArray[np.uint8]:
        """Image focusing on bare substrate."""
        x0, y0, x1, y1 = self.substrateROI
        return self.image[y0:y1, x0:x1]

    @property
    def temp2subst(self) -> Tuple[int, int]:
        """Vector from :attr:`template_image` to :attr:`substrate_image`."""
        x0, y0 = self.templateROI[:2]
        x1, y1 = self.substrateROI[:2]
        return (x1 - x0, y1 - y0)

    @abc.abstractmethod
    def examine(self) -> Optional[Exception]:
        """
        Check the sanity of parameters.

        Returns
        =======

        Error instance if the instance is invalid, else ``None``.
        """

    def verify(self):
        """
        Verify if all parameters are suitably set by raising error on failure.

        To implement sanity check for concrete subclass, define :meth:`examine`.
        """
        err = self.examine()
        if err is not None:
            raise err

    def valid(self) -> bool:
        """
        Verify if all parameters are suitably set by returning boolean value.

        To implement sanity check for concrete subclass, define :meth:`examine`.
        """
        err = self.examine()
        ret = True
        if err is not None:
            ret = False
        return ret
