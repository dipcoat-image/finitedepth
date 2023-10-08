"""Manage reference image and ROIs.

This module defines abstract class :class:`ReferenceBase` and its
implementation, :class:`Reference`.
"""


import abc
import dataclasses
from typing import TYPE_CHECKING, Generic, Optional, Tuple, Type, TypeVar

import cv2
import numpy as np
import numpy.typing as npt

from .parameters import LineOptions

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "ParamTypeVar",
    "DrawOptTypeVar",
    "DataTypeVar",
    "DynamicROI",
    "StaticROI",
    "ReferenceBase",
    "Parameters",
    "DrawOptions",
    "Data",
    "Reference",
    "sanitize_ROI",
]


ParamTypeVar = TypeVar("ParamTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`ReferenceBase.ParamType`."""
DrawOptTypeVar = TypeVar("DrawOptTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`ReferenceBase.DrawOptType`."""
DataTypeVar = TypeVar("DataTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`ReferenceBase.DataType`."""

DynamicROI = Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]
"""Type annotation for ROI whose upper limits can be dynamically determined.

This is a tuple of ``(x0, y0, x1, y1)``, where items can be integer or
:obj:`None` by Python's slicing convention.
"""
StaticROI = Tuple[int, int, int, int]
"""Type annotation for ROI whose items are all static.

This is a tuple of ``(x0, y0, x1, y1)``, where every item is
nonnegative :class:`int`.
"""


class ReferenceBase(abc.ABC, Generic[ParamTypeVar, DrawOptTypeVar, DataTypeVar]):
    """Abstract base class for reference instance.

    Reference instance stores a reference image, which is a binary image of
    uncoated substrate. It also contains ROIs for template region and
    substrate region in the image.

    Reference instance can visualize its data and analyze the reference image
    by the following methods:

    * :meth:`verify`: Sanity check before the analysis.
    * :meth:`draw`: Returns visualized result.
    * :meth:`analyze`: Returns analysis result.

    Concrete subclass must assign dataclasses types to the
    following class attributes:

    * :attr:`ParamType`: Type of :attr:`parameters`.
    * :attr:`DrawOptType`: Type of :attr:`draw_options`.
    * :attr:`DataType`: Type of :meth:`analyze`.

    Arguments:
        image: Binary reference image.
        templateROI: ROI for template region.
        substrateROI: ROI for substrate region.
        parameters: Analysis parameters.
            If passed, must be an instance of :attr:`ParamType`.
            If not passed, attempts to construct :attr:`ParamType`
            instance without argument.
        draw_options: Visualization options.
            If passed, must be an instance of :attr:`DrawOptType`.
            If not passed, attempts to construct :attr:`DrawOptType`
            instance without argument.
    """

    ParamType: Type[ParamTypeVar]
    """Type of :attr:`parameters.`

    This class attribute is defined but not set in :class:`ReferenceBase`.
    Concrete subclass must assign this attribute with frozen dataclass type.
    """
    DrawOptType: Type[DrawOptTypeVar]
    """Type of :attr:`draw_options.`

    This class attribute is defined but not set in :class:`ReferenceBase`.
    Concrete subclass must assign this attribute with dataclass type.
    """
    DataType: Type[DataTypeVar]
    """Type of return value of :attr:`analyze.`

    This class attribute is defined but not set in :class:`ReferenceBase`.
    Concrete subclass must assign this attribute with dataclass type.
    """

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        templateROI: DynamicROI = (0, 0, None, None),
        substrateROI: DynamicROI = (0, 0, None, None),
        parameters: Optional[ParamTypeVar] = None,
        *,
        draw_options: Optional[DrawOptTypeVar] = None,
    ):
        """Initialize the instance.

        - *image* is set to be immutable.
        - *templateROI* and *substrateROI* are converted using :func:`sanitize_ROI`.
        - *parameters* must be instance of :attr:`ParamType` or :obj:`None`.
          If :obj:`None`, a :attr:`ParamType` is attempted to be constructed.
        - *draw_options* must be instance of :attr:`DrawOptType` or :obj:`None`.
          If :obj:`None`, a :attr:`DrawOptType` is attempted to be constructed.
          If :attr:`DrawOptType`, the values are copied.
        """
        super().__init__()
        self._image = image
        self._image.setflags(write=False)

        h, w = image.shape[:2]
        self._templateROI = sanitize_ROI(templateROI, h, w)
        self._substrateROI = sanitize_ROI(substrateROI, h, w)

        if parameters is None:
            self._parameters = self.ParamType()
        else:
            if not isinstance(parameters, self.ParamType):
                raise TypeError(f"{parameters} is not instance of {self.ParamType}")
            self._parameters = parameters

        if draw_options is None:
            self._draw_options = self.DrawOptType()
        else:
            if not isinstance(draw_options, self.DrawOptType):
                raise TypeError(f"{draw_options} is not instance of {self.DrawOptType}")
            self._draw_options = dataclasses.replace(draw_options)

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """Binary reference image.

        Note:
            This array is immutable to allow caching.
        """
        return self._image

    @property
    def templateROI(self) -> StaticROI:
        """ROI for template region."""
        return self._templateROI

    @property
    def substrateROI(self) -> StaticROI:
        """ROI for substrate region."""
        return self._substrateROI

    @property
    def parameters(self) -> ParamTypeVar:
        """Analysis parameters.

        This property returns a frozen dataclass instance.
        Its type is :attr:`ParamType`.

        Note:
            This dataclass must be frozen to allow caching.
        """
        return self._parameters

    @property
    def draw_options(self) -> DrawOptTypeVar:
        """Visualization options.

        This property returns a mutable dataclass instance.
        Its type is :attr:`DrawOptType`.
        """
        return self._draw_options

    @draw_options.setter
    def draw_options(self, options: DrawOptTypeVar):
        self._draw_options = options

    @abc.abstractmethod
    def verify(self):
        """Sanity check before analysis.

        This method checks every intermediate step for analysis
        and raises error if anything is wrong. Passing this check
        should guarantee that :meth:`draw` and :meth:`analyze`
        returns without exception.
        """

    @abc.abstractmethod
    def draw(self) -> npt.NDArray[np.uint8]:
        """Return visualization result in RGB format.

        This method must always return without error. If visualization
        cannot be done, it should at least return original image.
        """

    @abc.abstractmethod
    def analyze(self) -> DataTypeVar:
        """Return analysis data of the reference image.

        This method returns analysis result as a dataclass instance
        whose type is :attr:`DataType`. If analysis is impossible,
        error may be raised.
        """


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Analysis parameters for :class:`Reference`.

    This is an empty dataclass.
    """

    pass


@dataclasses.dataclass
class DrawOptions:
    """Visualization options for :class:`Reference`.

    Arguments:
        templateROI: Options to visualize template ROI.
        substrateROI: Options to visualize substrate ROI.
    """

    templateROI: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(0, 255, 0), linewidth=1)
    )
    substrateROI: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(255, 0, 0), linewidth=1)
    )


@dataclasses.dataclass
class Data:
    """Analysis data for :class:`Reference`.

    This is an empty dataclass.
    """

    pass


class Reference(ReferenceBase[Parameters, DrawOptions, Data]):
    """Basic implementation of :class:`ReferenceBase`.

    Arguments:
        image
        templateROI
        substrateROI
        parameters (Parameters, optional)
        draw_options (DrawOptions, optional)

    Examples:
        .. plot::
            :include-source:
            :context: reset

            >>> import cv2
            >>> from dipcoatimage.finitedepth import Reference, get_data_path
            >>> gray = cv2.imread(get_data_path("ref1.png"), cv2.IMREAD_GRAYSCALE)
            >>> _, im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            >>> tempROI = (200, 50, 1200, 200)
            >>> substROI = (400, 175, 1000, 500)
            >>> ref = Reference(im, tempROI, substROI)
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(ref.draw()) #doctest: +SKIP

        Visualization can be controlled by modifying :attr:`draw_options`.

        .. plot::
            :include-source:
            :context: close-figs

            >>> ref.draw_options.substrateROI.linewidth = 3
            >>> plt.imshow(ref.draw()) #doctest: +SKIP
    """

    ParamType = Parameters
    """Assigned with :class:`Parameters`."""
    DrawOptType = DrawOptions
    """Assigned with :class:`DrawOptions`."""
    DataType = Data
    """Assigned with :class:`Data`."""

    def verify(self):
        """Implements :meth:`ReferenceBase.verify`."""
        pass

    def draw(self) -> npt.NDArray[np.uint8]:
        """Implements :meth:`ReferenceBase.draw`."""
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

    def analyze(self):
        """Implements :meth:`ReferenceBase.analyze`."""
        return self.DataType()


def sanitize_ROI(roi: DynamicROI, h: int, w: int) -> StaticROI:
    """Convert dynamic ROI to static ROI.

    Arguments:
        roi: Tuple in ``(x0, y0, x1, y1)``.
            Items can be integer or :obj:`None` by Python's
            slicing convention.
        h, w: Height and width of the image.

    Returns:
        Tuple in ``(x0, y0, x1, y1)``. Values are converted to
        positive integers.
    """
    full_roi = (0, 0, w, h)
    max_vars = (w, h, w, h)

    ret = list(roi)
    for i, var in enumerate(roi):
        if var is None:
            ret[i] = full_roi[i]
        elif var < 0:
            ret[i] = max_vars[i] + var
    return tuple(ret)  # type: ignore[return-value]
