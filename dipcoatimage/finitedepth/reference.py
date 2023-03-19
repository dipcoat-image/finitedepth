"""
Reference image
===============

:mod:`dipcoatimage.finitedepth.reference` provides reference image class of
finite-depth dip coating process.

Base class
----------

.. autoclass:: SubstrateReferenceError
   :members:

.. autoclass:: SubstrateReferenceBase
   :members:

Implementation
--------------

.. autoclass:: SubstrateReferenceParameters
   :members:

.. autoclass:: SubstrateReferenceDrawOptions
   :members:

.. autoclass:: SubstrateReference
   :members:

"""


import abc
import cv2  # type: ignore
import dataclasses
import numpy as np
import numpy.typing as npt
from .util import (
    binarize,
    colorize,
    DataclassProtocol,
    OptionalROI,
    IntROI,
    sanitize_ROI,
    BinaryImageDrawMode,
    FeatureDrawingOptions,
    Color,
)
from typing import TypeVar, Generic, Type, Optional

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias


__all__ = [
    "SubstrateReferenceError",
    "SubstrateReferenceBase",
    "SubstrateReferenceParameters",
    "SubstrateReferenceDrawOptions",
    "SubstrateReference",
]


class SubstrateReferenceError(Exception):
    """Base class for error from :class:`SubstrateReferenceBase`."""

    pass


ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)


class SubstrateReferenceBase(abc.ABC, Generic[ParametersType, DrawOptionsType]):
    """
    Abstract base class for substrate reference.

    .. plot::

       >>> import cv2
       >>> from dipcoatimage.finitedepth import get_samples_path
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> img = cv2.imread(get_samples_path("ref1.png"))
       >>> plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #doctest: +SKIP

    .. rubric:: Image and ROIs

    Substrate reference class wraps the data to locate the substrate in coated
    substrate image. It consists of the reference image and two ROIs.

    Reference image, which can be accessed by :attr:`image`, is the picture
    of bare substrate taken before coating. Two ROIs, :attr:`templateROI` and
    :attr:`substrateROI`, are defined. Template ROI encloses the region which is
    common in both bare substrate image and coated substrate image. Substrate ROI
    encloses the bare substrate region, narrowing down the target.

    .. rubric:: Binary image

    Binarization is important for reference image. :meth:`binary_image` is the
    default implementation which relies on Otsu's thresholding. Subclass may
    redefine this method.

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
    Modifying :attr:`draw_options` changes the visualization result.

    Parameters
    ==========

    image
        Reference image. May be grayscale or RGB.

    templateROI, substrateROI
        Slice indices in ``(x0, y0, x1, y1)`` for the template and the substrate.

    parameters
        Additional parameters.

    draw_options
        Drawing options.

    """

    __slots__ = (
        "_image",
        "_templateROI",
        "_substrateROI",
        "_parameters",
        "_draw_options",
        "_binary_image",
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]

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
        self._image.setflags(write=False)

        h, w = image.shape[:2]
        self._templateROI = sanitize_ROI(templateROI, h, w)
        self._substrateROI = sanitize_ROI(substrateROI, h, w)

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

        This array is not writable to be immutable for caching.
        """
        return self._image

    @property
    def templateROI(self) -> IntROI:
        """Slice indices in ``(x0, y0, x1, y1)`` for template region."""
        return self._templateROI

    @property
    def substrateROI(self) -> IntROI:
        """Slice indices in ``(x0, y0, x1, y1)`` for substrate region."""
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

    def binary_image(self) -> npt.NDArray[np.uint8]:
        """
        Binarized :attr:`image` using Otsu's thresholding.

        Notes
        =====

        This method is cached. Do not modify its result.

        """
        if not hasattr(self, "_binary_image"):
            self._binary_image = binarize(self.image)
        return self._binary_image

    def substrate_image(self) -> npt.NDArray[np.uint8]:
        """:meth:`image` cropped by :meth:`substrateROI`."""
        x0, y0, x1, y1 = self.substrateROI
        return self.image[y0:y1, x0:x1]

    def temp2subst(self) -> npt.NDArray[np.int32]:
        """
        Vector from upper left point of template region to upper left point of
        substrate region.
        """
        x0, y0 = self.templateROI[:2]
        x1, y1 = self.substrateROI[:2]
        return np.array([x1 - x0, y1 - y0])

    @abc.abstractmethod
    def examine(self) -> Optional[SubstrateReferenceError]:
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

    @abc.abstractmethod
    def draw(self) -> npt.NDArray[np.uint8]:
        """Decorate and return the reference image as RGB format."""


@dataclasses.dataclass(frozen=True)
class SubstrateReferenceParameters:
    """Additional parameters for :class:`SubstrateReference` instance."""

    pass


@dataclasses.dataclass
class SubstrateReferenceDrawOptions:
    """Drawing options for :class:`SubstrateReference`."""

    draw_mode: BinaryImageDrawMode = BinaryImageDrawMode.BINARY
    templateROI: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(0, 255, 0), thickness=1
    )
    substrateROI: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(255, 0, 0), thickness=1
    )


class SubstrateReference(
    SubstrateReferenceBase[SubstrateReferenceParameters, SubstrateReferenceDrawOptions]
):
    """
    Substrate reference class with customizable binarization.

    Examples
    ========

    Construct with RGB image, and visualize with :meth:`draw`.

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import (SubstrateReference,
       ...     get_samples_path)
       >>> ref_path = get_samples_path("ref1.png")
       >>> img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
       >>> tempROI = (200, 50, 1200, 200)
       >>> substROI = (400, 175, 1000, 500)
       >>> ref = SubstrateReference(img, tempROI, substROI)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    Visualization can be controlled by modifying :attr:`draw_options`.

    .. plot::
       :include-source:
       :context: close-figs

       >>> ref.draw_options.substrateROI.color.blue = 255
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    """

    Parameters = SubstrateReferenceParameters
    DrawOptions = SubstrateReferenceDrawOptions

    DrawMode: TypeAlias = BinaryImageDrawMode

    def examine(self) -> None:
        return None

    def draw(self) -> npt.NDArray[np.uint8]:
        draw_mode = self.draw_options.draw_mode
        if draw_mode == self.DrawMode.ORIGINAL:
            image = self.image
        elif draw_mode == self.DrawMode.BINARY:
            image = self.binary_image()
        else:
            raise TypeError("Unrecognized draw mode: %s" % draw_mode)
        ret = colorize(image)

        substROI_opts = self.draw_options.substrateROI
        if substROI_opts.thickness > 0:
            x0, y0, x1, y1 = self.substrateROI
            color = dataclasses.astuple(substROI_opts.color)
            thickness = substROI_opts.thickness
            cv2.rectangle(ret, (x0, y0), (x1, y1), color, thickness)

        tempROI_opts = self.draw_options.templateROI
        if tempROI_opts.thickness > 0:
            x0, y0, x1, y1 = self.templateROI
            color = dataclasses.astuple(tempROI_opts.color)
            thickness = tempROI_opts.thickness
            cv2.rectangle(ret, (x0, y0), (x1, y1), color, thickness)
        return ret
