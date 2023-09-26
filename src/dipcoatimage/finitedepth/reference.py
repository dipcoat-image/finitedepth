"""
Reference image
===============

:mod:`dipcoatimage.finitedepth.reference` provides reference image class of
finite-depth dip coating process.

Base class
----------

.. autoclass:: ReferenceError
   :members:

.. autoclass:: ReferenceBase
   :members:

Implementation
--------------

.. autoclass:: Reference
   :members:

"""


import abc
import cv2
import dataclasses
import numpy as np
import numpy.typing as npt
from .reference_param import Parameters, DrawOptions, Data
from typing import TypeVar, Tuple, Optional, Generic, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "ReferenceError",
    "ReferenceBase",
    "Reference",
    "OptionalROI",
    "IntROI",
    "sanitize_ROI",
]


class ReferenceError(Exception):
    """Base class for error from :class:`ReferenceBase`."""

    pass


ParametersType = TypeVar("ParametersType", bound="DataclassInstance")
OptionalROI = Tuple[int, int, Optional[int], Optional[int]]
IntROI = Tuple[int, int, int, int]
DrawOptionsType = TypeVar("DrawOptionsType", bound="DataclassInstance")
DataType = TypeVar("DataType", bound="DataclassInstance")


class ReferenceBase(abc.ABC, Generic[ParametersType, DrawOptionsType, DataType]):
    """
    Abstract base class for substrate reference.

    .. plot::

       >>> import cv2
       >>> from dipcoatimage.finitedepth import get_data_path
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> img = cv2.imread(get_data_path("ref1.png"))
       >>> plt.imshow(img) #doctest: +SKIP

    .. rubric:: Image and ROIs

    Substrate reference class wraps the data to locate the substrate in coated
    substrate image. It consists of the reference image and two ROIs.

    Reference image, which can be accessed by :attr:`image`, is the picture
    of bare substrate taken before coating. Two ROIs, :attr:`templateROI` and
    :attr:`substrateROI`, are defined. Template ROI encloses the region which is
    common in both bare substrate image and coated substrate image. Substrate ROI
    encloses the bare substrate region, narrowing down the target.

    Input image should be binary.

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

    Validity of the parameters can be checked by :meth:`verify`.

    .. rubric:: Visualization

    :meth:`draw` defines the visualization logic for concrete class.
    Modifying :attr:`draw_options` changes the visualization result.

    .. rubric:: Analysis

    Concrete class must have :attr:`Data` which returns dataclass type and
    implement :meth:`analyze_reference` which returns data tuple compatible with
    :attr:`Data`.
    :meth:`analyze` is the API for analysis result.

    Parameters
    ==========

    image
        Reference image. Must be binary.

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
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    Data: Type[DataType]

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
        Reference image. Must be binary.

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
        return np.array([x1 - x0, y1 - y0], dtype=np.int32)

    @abc.abstractmethod
    def verify(self):
        """Check to detect error and raise before analysis."""

    @abc.abstractmethod
    def draw(self) -> npt.NDArray[np.uint8]:
        """Decorate and return the reference image as RGB format."""

    @abc.abstractmethod
    def analyze_reference(self) -> Tuple:
        """Analyze the reference image and return the data in tuple."""

    def analyze(self) -> DataType:
        """
        Return the result of :meth:`analyze_reference` as dataclass instance.
        """
        return self.Data(*self.analyze_reference())


class Reference(ReferenceBase[Parameters, DrawOptions, Data]):
    """
    Substrate reference class with customizable binarization.

    Examples
    ========

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import Reference, get_data_path
       >>> gray = cv2.imread(get_data_path("ref1.png"), cv2.IMREAD_GRAYSCALE)
       >>> _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
       >>> tempROI = (200, 50, 1200, 200)
       >>> substROI = (400, 175, 1000, 500)
       >>> ref = Reference(img, tempROI, substROI)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    Visualization can be controlled by modifying :attr:`draw_options`.

    .. plot::
       :include-source:
       :context: close-figs

       >>> ref.draw_options.substrateROI.color = (0, 255, 255)
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    """

    Parameters = Parameters
    DrawOptions = DrawOptions
    Data = Data

    def verify(self):
        pass

    def draw(self) -> npt.NDArray[np.uint8]:
        ret = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        substROI_opts = self.draw_options.substrateROI
        if substROI_opts.linewidth > 0:
            x0, y0, x1, y1 = self.substrateROI
            color = substROI_opts.color
            linewidth = substROI_opts.linewidth
            cv2.rectangle(ret, (x0, y0), (x1, y1), color, linewidth)

        tempROI_opts = self.draw_options.templateROI
        if tempROI_opts.linewidth > 0:
            x0, y0, x1, y1 = self.templateROI
            color = tempROI_opts.color
            linewidth = tempROI_opts.linewidth
            cv2.rectangle(ret, (x0, y0), (x1, y1), color, linewidth)
        return ret

    def analyze_reference(self) -> Tuple[()]:
        return ()


def sanitize_ROI(roi: OptionalROI, h: int, w: int) -> IntROI:
    full_roi = (0, 0, w, h)
    max_vars = (w, h, w, h)

    ret = list(roi)
    for i, var in enumerate(roi):
        if var is None:
            ret[i] = full_roi[i]
        elif var < 0:
            ret[i] = max_vars[i] + var
    return tuple(ret)  # type: ignore[return-value]
