"""
Reference image
===============

:mod:`dipcoatimage.finitedepth.reference` provides reference image class of
finite-depth dip coating process.

Base class
----------

.. autoclass:: SubstrateReferenceBase
   :members:

.. autoclass:: SubstrateReferenceParameters
   :members:

.. autoclass:: SubstrateReferenceDrawOptions
   :members:

Default implementation
----------------------

.. autoclass:: SubstrateReference
   :members:

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
    Abstract base class for substrate reference.

    .. plot::

       >>> import cv2
       >>> from dipcoatimage.finitedepth import get_samples_path
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(cv2.imread(get_samples_path('ref1.png'))) #doctest: +SKIP

    .. rubric:: Image and ROIs

    Substrate reference class wraps the data to locate the substrate in coated
    substrate image. It consists of the reference image and two ROIs.

    Reference image, which can be accessed by :attr:`image`, is the picture of
    bare substrate taken before coating. Two ROIs, :attr:`templateROI` and
    :attr:`substrateROI`, are defined. Template ROI encloses the region which is
    common in both bare substrate image and coated substrate image. Substrate ROI
    encloses the bare substrate region, narrowing down the target.

    .. rubric:: Constructor

    Constructor signature must not be modified because high-level API use factory
    to generate reference instances. Additional parameters can be introduced by
    definig class attribute :attr:`Parameters` and :attr:`DrawOptions`.

    .. rubric:: Parameters and DrawOptions

    Concrete class must have :attr:`Parameters` and :attr:`DrawOptions` which
    return dataclass types. Their instances are passed to the constructor at
    instance initialization, and can be accessed by :attr:`parameters` and
    :attr:`draw_options`.

    :attr:`Parameter` must be frozen to ensure immtability for caching. However,
    :attr:`DrawOptions` need not be frozen since visualization does not affect
    the identity of instance. Therefore methods affected by draw options must
    not be cached.

    .. rubric:: Sanity check

    Validity of the parameters can be checked by :meth:`verify` or :meth:`valid`.
    Their result can be implemented by defining :meth:`examine`.

    .. rubric:: Visualization

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

    """

    __slots__ = (
        "_image",
        "_templateROI",
        "_substrateROI",
        "_parameters",
        "_draw_options",
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]

    def __init_subclass__(cls) -> None:
        params = getattr(cls, "Parameters", None)
        if params is None:
            raise TypeError(f"{cls} has no attribute 'Parameters'.")
        elif not (isinstance(params, type) and dataclasses.is_dataclass(params)):
            raise TypeError(f"{params} is not dataclass type.")
        elif not params.__dataclass_params__.frozen:  # type: ignore
            raise TypeError(f"{params} is not frozen.")

        drawopts = getattr(cls, "DrawOptions", None)
        if drawopts is None:
            raise TypeError(f"{cls} has no attribute 'DrawOptions'.")
        elif not (isinstance(drawopts, type) and dataclasses.is_dataclass(drawopts)):
            raise TypeError(f"{drawopts} is not dataclass type.")
        return super().__init_subclass__()

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        templateROI: OptionalROI = (0, 0, None, None),
        substrateROI: OptionalROI = (0, 0, None, None),
        parameters: Optional[ParametersType] = None,
        *,
        draw_options: Optional[DrawOptionsType] = None,
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
        """
        Reference image passed to constructor.

        This array is not writable to enable caching which requires immutability.
        """
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
        Additional parameters for concrete class.

        Instance of :attr:`Parameters`, which must be a frozen dataclass.
        """
        return self._parameters

    @property
    def draw_options(self) -> DrawOptionsType:
        """
        Options to visualize the image.

        Instance of :attr:`DrawOptions` dataclass.
        """
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

        If the instance is invalid, return error instance. Else, return ``None``.
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
        """Decorate and return the reference image as RGB format."""


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


class SubstrateReference(SubstrateReferenceBase):
    """
    Substrate reference class with RGB image.

    Examples
    ========

    Construct with RGB image, and visualize with :meth:`draw`.

    .. plot::
       :include-source:

       >>> import cv2
       >>> from dipcoatimage.finitedepth import (SubstrateReference,
       ...     get_samples_path)
       >>> ref_path = get_samples_path('ref1.png')
       >>> img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
       >>> tempROI = (200, 50, 1200, 200)
       >>> substROI = (400, 100, 1000, 500)
       >>> ref = SubstrateReference(img, tempROI, substROI)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    """

    Parameters = SubstrateReferenceParameters
    DrawOptions = SubstrateReferenceDrawOptions

    def examine(self) -> None:
        return None

    def draw(self) -> npt.NDArray[np.uint8]:
        ret = self.image.copy()

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
