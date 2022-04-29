"""
Reference image
===============

:mod:`dipcoatimage.finitedepth.reference` provides class for reference image of
finite-depth substrate. It is used to analyze coated substrate image.

"""


import abc
import cv2  # type: ignore
import dataclasses
import numpy as np
import numpy.typing as npt
from typing import TypeVar, Generic, Type, Optional, cast, Tuple
from .util import DataclassProtocol, OptionalROI, IntROI


__all__ = [
    "SubstrateReferenceBase",
    "SubstrateReferenceParameters",
    "SubstrateReferenceDrawOptions",
    "SubstrateReference",
]


ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)


class SubstrateReferenceBase(abc.ABC, Generic[ParametersType, DrawOptionsType]):
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
    parameters by defining dataclasses and assigning the classes to class
    attribute :attr:`Parameters` and :attr:`DrawOptions`. Additional parameters
    must be instance of these classes and can be passed to the constructor.

    Sanity check
    ------------

    Validity of the parameters can be checked by :meth:`verify` or :meth:`valid`.

    Visualization
    -------------

    :meth:`draw` defines the visualization logic for concrete class.

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
    def Parameters(self) -> Type[ParametersType]:
        """Frozen dataclass type for additional parameters in concrete class."""

    @property
    @abc.abstractmethod
    def DrawOptions(self) -> Type[DrawOptionsType]:
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
        """
        Additional parameters for concrete class. Instance of :attr:`Parameters`.
        """
        return self._parameters

    @property
    def draw_options(self) -> DrawOptionsType:
        """Options to visualize the image. Instance of :attr:`DrawOptions`."""
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

    @abc.abstractmethod
    def draw(self) -> npt.NDArray[np.uint8]:
        """Decorate and return the reference image."""


@dataclasses.dataclass(frozen=True)
class SubstrateReferenceParameters:
    """Additional parameters for :class:`SubstrateReference` instance."""

    pass


@dataclasses.dataclass
class SubstrateReferenceDrawOptions:
    """
    Drawing options for :class:`SubstrateReference`.

    Parameters
    ==========

    draw_templateROI
        Flag whether to draw template ROI box.

    templateROI_color
        Color for template ROI box. Ignored if *draw_templateROI* is false.

    templateROI_thickness
        Thickness for template ROI box. Ignored if *draw_templateROI* is false.

    draw_substrateROI
        Flag whether to draw substrate ROI box.

    substrateROI_color
        Color for substrate ROI box. Ignored if *draw_substrateROI* is false.

    substrateROI_thickness
        Thickness for substrate ROI box. Ignored if *draw_substrateROI* is false.

    """

    draw_templateROI: bool = True
    templateROI_color: Tuple[int, int, int] = (0, 255, 0)
    templateROI_thickness: int = 1

    draw_substrateROI: bool = True
    substrateROI_color: Tuple[int, int, int] = (255, 0, 0)
    substrateROI_thickness: int = 1


class SubstrateReference(
    SubstrateReferenceBase[SubstrateReferenceParameters, SubstrateReferenceDrawOptions]
):
    Parameters = SubstrateReferenceParameters
    DrawOptions = SubstrateReferenceDrawOptions

    def examine(self) -> Optional[Exception]:
        return None

    def draw(self) -> npt.NDArray[np.uint8]:
        ret = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        if self.draw_options.draw_substrateROI:
            x0, y0, x1, y1 = self.substrateROI
            color = self.draw_options.substrateROI_color
            thickness = self.draw_options.substrateROI_thickness
            cv2.rectangle(ret, (x0, y0), (x1, y1), color, thickness)

        if self.draw_options.draw_templateROI:
            x0, y0, x1, y1 = self.templateROI
            color = self.draw_options.templateROI_color
            thickness = self.draw_options.templateROI_thickness
            cv2.rectangle(ret, (x0, y0), (x1, y1), color, thickness)
        return ret
