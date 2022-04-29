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
from typing import TypeVar, Type, Optional, cast
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
    substrate image. Substrate ROI encloses the entire bare substrate.

    Parameters
    ==========

    image
        Reference image.

    templateROI, substrateROI
        Slice indices in ``(x0, y0, x1, y1)`` for the template and the substrate.

    parameters
        Class-specific additional parameters.

    draw_options
        Options to draw the reference image.

    Notes
    =====

    Some properties and methods can be cached for performance. To
    prevent mutation, :attr:`image` should not be writable and
    :attr:`parameters` must be frozen.

    This does not hold to :attr:`draw_options` since visualization does
    not affect the identity of instance.
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
        """Dataclass type for class-specific additional parameters."""

    @property
    @abc.abstractmethod
    def DrawOptions(self) -> Type:
        """Dataclass type for drawing options."""

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
        if not image.size > 0:
            raise ReferenceError("Empty reference image")

        self._image = image
        self._image = image
        self._image.setflags(write=False)

        h, w = image.shape[:2]
        full_roi = (0, 0, w, h)
        max_vars = (w, h, w, h)

        temp_roi = []
        for passed_val, full_val, max_val in zip(templateROI, full_roi, max_vars):
            if passed_val is None:
                val = full_val
            elif passed_val < 0:
                val = passed_val + max_val
            else:
                val = passed_val
            temp_roi.append(val)
        self._templateROI = cast(IntROI, tuple(temp_roi))

        subst_roi = []
        for passed_val, full_val, max_val in zip(substrateROI, full_roi, max_vars):
            if passed_val is None:
                val = full_val
            elif passed_val < 0:
                val = passed_val + max_val
            else:
                val = passed_val
            subst_roi.append(val)
        self._substrateROI = cast(IntROI, tuple(subst_roi))

        if parameters is None:
            self._parameters = self.Parameters()
        else:
            self._parameters = dataclasses.replace(parameters)  # type: ignore

        if draw_options is None:
            self._draw_options = self.DrawOptions()
        else:
            self._draw_options = dataclasses.replace(draw_options)  # type: ignore

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """Reference image."""
        return self._image

    @property
    def templateROI(self) -> IntROI:
        """
        Slice indices in ``(x0, y0, x1, y1)`` for :attr:`template_image`.

        """
        return self._templateROI

    @property
    def substrateROI(self) -> IntROI:
        """
        Slice indices in ``(x0, y0, x1, y1)`` for :attr:`substrate_image`.

        """
        return self._substrateROI

    @property
    def parameters(self) -> ParametersType:
        """Additional parameters for the instance."""
        return self._parameters  # type: ignore

    @property
    def draw_options(self) -> DrawOptionsType:
        """Options to visualize the reference image."""
        return self._draw_options  # type: ignore

    @draw_options.setter
    def draw_options(self, options: DrawOptionsType):
        self._draw_options = options  # type: ignore
