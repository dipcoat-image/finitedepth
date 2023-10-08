"""Analyze substrate geometry.

This module defines abstract class :class:`SubstrateBase` and its
implementation, :class:`Substrate`.
"""

import abc
import dataclasses
from typing import TYPE_CHECKING, Generic, Optional, Tuple, Type, TypeVar

import cv2
import numpy as np
import numpy.typing as npt

from .cache import attrcache
from .reference import ReferenceBase

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "RefTypeVar",
    "ParamTypeVar",
    "DrawOptTypeVar",
    "DataTypeVar",
    "SubstrateBase",
    "Parameters",
    "DrawOptions",
    "Data",
    "Substrate",
]


RefTypeVar = TypeVar("RefTypeVar", bound=ReferenceBase)
"""Type variable for the reference type of :class:`SubstrateBase`."""
ParamTypeVar = TypeVar("ParamTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`SubstrateBase.ParamType`."""
DrawOptTypeVar = TypeVar("DrawOptTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`SubstrateBase.DrawOptType`."""
DataTypeVar = TypeVar("DataTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`SubstrateBase.DataType`."""


class SubstrateBase(
    abc.ABC, Generic[RefTypeVar, ParamTypeVar, DrawOptTypeVar, DataTypeVar]
):
    """Abstract base class for substrate instance.

    Substrate instance stores a substrate image, which is a binary image of
    substrate region from reference instance. The role of substrate instance
    is to analyze the shape of the bare substrate.

    Substrate instance can visualize its data and analyze the substrate image
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
        reference: Reference instance.
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

    This class attribute is defined but not set in :class:`SubstrateBase`.
    Concrete subclass must assign this attribute with frozen dataclass type.
    """
    DrawOptType: Type[DrawOptTypeVar]
    """Type of :attr:`draw_options.`

    This class attribute is defined but not set in :class:`SubstrateBase`.
    Concrete subclass must assign this attribute with dataclass type.
    """
    DataType: Type[DataTypeVar]
    """Type of return value of :attr:`analyze.`

    This class attribute is defined but not set in :class:`SubstrateBase`.
    Concrete subclass must assign this attribute with dataclass type.
    """

    def __init__(
        self,
        reference: RefTypeVar,
        parameters: Optional[ParamTypeVar] = None,
        *,
        draw_options: Optional[DrawOptTypeVar] = None,
    ):
        """Initialize the instance.

        - *reference* is not type-checked in runtime.
        - *parameters* must be instance of :attr:`ParamType` or :obj:`None`.
          If :obj:`None`, a :attr:`ParamType` is attempted to be constructed.
        - *draw_options* must be instance of :attr:`DrawOptType` or :obj:`None`.
          If :obj:`None`, a :attr:`DrawOptType` is attempted to be constructed.
          If :attr:`DrawOptType`, the values are copied.
        """
        super().__init__()
        # Do not type check reference (can be protocol)
        self._ref = reference

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
    def reference(self) -> RefTypeVar:
        """Reference instance which defines the substrate image."""
        return self._ref

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

    def image(self) -> npt.NDArray[np.uint8]:
        """Substrate image from :meth:`reference`."""
        # not property since it's not directly from the argument
        x0, y0, x1, y1 = self.reference.substrateROI
        return self.reference.image[y0:y1, x0:x1]

    @abc.abstractmethod
    def region_points(self) -> npt.NDArray[np.int32]:
        """Points in `[x, y]` in each discrete substrate region.

        Returns
        -------
        ndarray
            `(N, 2)`-shaped array, where `N` is the number of discrete substrate
            regions.

        Notes
        -----
        These points are used to locate and distinguish each substrate region
        from foreground pixels. Every image analysis implementation is based on
        these points.

        Subclass should implement this method to return robust points. The
        implementation must be simple and non-dynamic. Returning just the center
        point of the image is a good example; selection of the substrate region
        from the reference image must be done to obey this rule (e.g., select
        s.t. the center point falls on the substrate region)
        """

    @attrcache("_regions")
    def regions(self) -> npt.NDArray[np.int8]:
        """Return image labelled by each discrete substrate regions.

        Returns
        -------
        ndarray
            Labelled image. `i`-th region in :meth:`region_points` is labelled
            with `i`. `-1` denotes background.

        Notes
        -----
        Maximum number of regions is 128.

        See Also
        --------
        region_points
        """
        ret = np.full(self.image().shape[:2], -1, dtype=np.int8)
        _, labels = cv2.connectedComponents(cv2.bitwise_not(self.image()))
        for i, pt in enumerate(self.region_points()):
            ret[labels == labels[pt[1], pt[0]]] = i
        return ret

    def contours(
        self, region: int
    ) -> Tuple[Tuple[npt.NDArray[np.int32], ...], npt.NDArray[np.int32]]:
        """Find contours of a substrate region identified by *region*.

        Parameters
        ----------
        region : int
            Label in :meth:`regions`.

        Returns
        -------
        tuple
            Tuple of the result of :func:`cv2.findContours`.

        Notes
        -----
        Contours are dense, i.e., no approximation is made.

        See Also
        --------
        regions
        """
        if not hasattr(self, "_contours"):
            contours = []
            for i in range(len(self.region_points())):
                reg = (self.regions() == i) * np.uint8(255)
                cnt = cv2.findContours(reg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                contours.append(cnt)
            self._contours = tuple(contours)
        return self._contours[region]  # type: ignore[return-value]

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
    """Analysis parameters for :class:`Substrate`.

    This is an empty dataclass.
    """

    pass


@dataclasses.dataclass
class DrawOptions:
    """Visualization options for :class:`Reference`.

    This is an empty dataclass.
    """

    pass


@dataclasses.dataclass
class Data:
    """Analysis data for :class:`Substrate`.

    This is an empty dataclass.
    """

    pass


class Substrate(SubstrateBase[ReferenceBase, Parameters, DrawOptions, Data]):
    """Basic implementation of :class:`SubstrateBase`.

    Arguments:
        reference
        parameters (Parameters, optional)
        draw_options (DrawOptions, optional)

    Examples:
        Construct substrate reference instance first.

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

        Construct `Substrate` instance from reference instance.

        .. plot::
            :include-source:
            :context: close-figs

            >>> from dipcoatimage.finitedepth import Substrate
            >>> subst = Substrate(ref)
            >>> plt.imshow(subst.draw()) #doctest: +SKIP
    """

    ParamType = Parameters
    """Assigned with :class:`Parameters`."""
    DrawOptType = DrawOptions
    """Assigned with :class:`DrawOptions`."""
    DataType = Data
    """Assigned with :class:`Data`."""

    def region_points(self) -> npt.NDArray[np.int32]:
        """Return a point in substrate region."""
        w = self.image().shape[1]
        return np.array([[w / 2, 0]], dtype=np.int32)

    def verify(self):
        """Implements :meth:`ReferenceBase.verify`."""
        pass

    def draw(self) -> npt.NDArray[np.uint8]:
        """Implements :meth:`ReferenceBase.draw`."""
        return cv2.cvtColor(self.image(), cv2.COLOR_GRAY2RGB)

    def analyze(self):
        """Implements :meth:`ReferenceBase.analyze`."""
        return self.DataType()
