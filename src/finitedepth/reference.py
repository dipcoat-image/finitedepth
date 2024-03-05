"""Manage reference image and ROIs."""

import abc
import dataclasses
from typing import TYPE_CHECKING, Generic, TypeVar

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

__all__ = [
    "DataTypeVar",
    "DynamicROI",
    "StaticROI",
    "sanitize_ROI",
    "ReferenceBase",
    "ReferenceData",
    "Reference",
]


DataTypeVar = TypeVar("DataTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`ReferenceBase.DataType`."""


DynamicROI = tuple[int | None, int | None, int | None, int | None]
"""Type annotation for ROI whose upper limits can be dynamically determined.

This is a tuple of ``(x0, y0, x1, y1)``, where items can be integer or
:obj:`None` by Python's slicing convention.
"""

StaticROI = tuple[int, int, int, int]
"""Type annotation for ROI whose items are all static.

This is a tuple of ``(x0, y0, x1, y1)``, where every item is
nonnegative :class:`int`.
"""


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


class ReferenceBase(abc.ABC, Generic[DataTypeVar]):
    """Abstract base class for reference object.

    Reference object stores reference image, which is a binary image of uncoated
    substrate. It also contains ROIs for template region and substrate region in
    the reference image.

    External API can use the following members to get analysis results of
    concrete subclasses.

    * :attr:`DataType`: Dataclass type for the analysis result.
    * :meth:`analyze`: :attr:`DataType` instance containing analysis result.
    * :meth:`draw`: Visualized result.

    Arguments:
        image: Binary reference image.
    """

    DataType: type[DataTypeVar]
    """Return type of :attr:`analyze`.

    Concrete subclass must assign this attribute with dataclass type.
    """

    def __init__(self, image: npt.NDArray[np.uint8]):
        """Initialize the instance.

        *image* is set to be not writable.
        """
        self._image = image
        self._image.setflags(write=False)

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """Binary reference image.

        For immutability, this image is not writable.
        """
        return self._image

    @property
    @abc.abstractmethod
    def templateROI(self) -> StaticROI:
        """ROI for template image."""

    @property
    @abc.abstractmethod
    def substrateROI(self) -> StaticROI:
        """ROI for substrate image."""

    @abc.abstractmethod
    def analyze(self) -> DataTypeVar:
        """Return analysis result as dataclass.

        Return type must be :attr:`DataType`.
        """

    def draw(
        self,
        templateColor: tuple[int, int, int] = (255, 0, 0),
        templateThickness: int = 1,
        substrateColor: tuple[int, int, int] = (0, 255, 0),
        substrateThickness: int = 1,
    ) -> npt.NDArray[np.uint8]:
        """Return visualization result in RGB format.

        Arguments:
            templateColor: Template ROI box color for :func:`cv2.rectangle`.
            templateThickness: Template ROI box line width for :func:`cv2.rectangle`.
            substrateColor: Substrate ROI box color for :func:`cv2.rectangle`.
            substrateThickness: Substrate ROI box line width for :func:`cv2.rectangle`.
        """
        ret = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        if templateThickness > 0:
            x0, y0, x1, y1 = self.templateROI
            cv2.rectangle(ret, (x0, y0), (x1, y1), templateColor, templateThickness)

        if substrateThickness > 0:
            x0, y0, x1, y1 = self.substrateROI
            cv2.rectangle(ret, (x0, y0), (x1, y1), substrateColor, substrateThickness)

        return ret  # type: ignore[return-value]


@dataclasses.dataclass
class ReferenceData:
    """Analysis data for :class:`Reference`."""

    pass


class Reference(ReferenceBase[ReferenceData]):
    """Reference image with ROIs specified.

    Arguments:
        image: Binary reference image.
        templateROI: ROI for template image.
        substrateROI: ROI for substrate image.

    Examples:
        .. plot::
            :include-source:
            :context: reset

            >>> import cv2
            >>> from finitedepth import get_sample_path, Reference
            >>> img = cv2.imread(get_sample_path("ref.png"), cv2.IMREAD_GRAYSCALE)
            >>> _, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            >>> ref = Reference(bin, (10, 10, 1250, 200), (100, 100, 1200, 500))
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(ref.draw()) #doctest: +SKIP
    """

    DataType = ReferenceData
    """Return :obj:`ReferenceData`."""

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        templateROI: DynamicROI = (0, 0, None, None),
        substrateROI: DynamicROI = (0, 0, None, None),
    ):
        """Initialize the instance.

        *templateROI* and *substrateROI* are converted to :obj:`StaticROI` using
        :func:`sanitize_ROI`.
        """
        super().__init__(image)

        h, w = image.shape[:2]
        self._templateROI = sanitize_ROI(templateROI, h, w)
        self._substrateROI = sanitize_ROI(substrateROI, h, w)

    @property
    def templateROI(self) -> StaticROI:
        """ROI for template image."""
        return self._templateROI

    def analyze(self):
        """Return empty :class:`ReferenceData`."""
        return self.DataType()

    @property
    def substrateROI(self) -> StaticROI:
        """ROI for substrate image."""
        return self._substrateROI
