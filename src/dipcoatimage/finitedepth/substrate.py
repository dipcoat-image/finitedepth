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
    "SubstParam",
    "SubstDrawOpt",
    "SubstData",
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
    bare substrate acquired from reference instance. The role of substrate
    instance is to analyze the shape of the bare substrate.

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
        """Coordinates of points in each discrete substrate region.

        Substrate image can have multiple discrete substrate regions.
        Subclass should implement this method to return coordinates of points,
        one for each region.

        Returns:
            `(N, 2)`-shaped array, where `N` is the number of discrete substrate
            regions. Column should be the coordinates of points in ``(x, y)``.

        Note:
            These points are used to distinguish substrate regions from other
            foreground pixels, and give indices to each region.

            As higher-level methods are expected to rely on this method,
            implementation must be simple and non-dynamic.
            See :meth:`Substrate.region_points` for example.
        """

    @attrcache("_regions")
    def regions(self) -> npt.NDArray[np.int8]:
        """Labelled image of discrete substrate regions.

        Substrate regions are determined as connected component including
        a point in :meth:`region_points`.

        Returns:
            Labelled image. Value of ``i`` represents ``i``-th region in
            :meth:`region_points. ``-1`` represents background.

        Note:
            Maximum number of regions is 128.
        """
        ret = np.full(self.image().shape[:2], -1, dtype=np.int8)
        _, labels = cv2.connectedComponents(cv2.bitwise_not(self.image()))
        for i, pt in enumerate(self.region_points()):
            ret[labels == labels[pt[1], pt[0]]] = i
        return ret

    def contours(
        self, region: int
    ) -> Tuple[Tuple[npt.NDArray[np.int32], ...], npt.NDArray[np.int32]]:
        """Find contours of a discrete substrate region.

        Arguments:
            region: Label of the discrete region from :meth:`regions`.

        Returns:
            Tuple of the result of :func:`cv2.findContours`.

        Note:
            Contours are dense, i.e., no approximation is made.
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

        This method must always return without error. If visualization cannot be done,
        it should at least return original image.
        """

    @abc.abstractmethod
    def analyze(self) -> DataTypeVar:
        """Return analysis data of the reference image.

        This method returns analysis result as a dataclass instance
        whose type is :attr:`DataType`. If analysis is impossible,
        error may be raised.
        """


@dataclasses.dataclass(frozen=True)
class SubstParam:
    """Analysis parameters for :class:`Substrate`.

    This is an empty dataclass.
    """

    pass


@dataclasses.dataclass
class SubstDrawOpt:
    """Visualization options for :class:`Reference`.

    This is an empty dataclass.
    """

    pass


@dataclasses.dataclass
class SubstData:
    """Analysis data for :class:`Substrate`.

    This is an empty dataclass.
    """

    pass


class Substrate(SubstrateBase[ReferenceBase, SubstParam, SubstDrawOpt, SubstData]):
    """Basic implementation of :class:`SubstrateBase`.

    Arguments:
        reference
        parameters (SubstParam, optional)
        draw_options (SubstDrawOpt, optional)

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

    ParamType = SubstParam
    """Assigned with :class:`SubstParam`."""
    DrawOptType = SubstDrawOpt
    """Assigned with :class:`SubstDrawOpt`."""
    DataType = SubstData
    """Assigned with :class:`SubstData`."""

    def region_points(self) -> npt.NDArray[np.int32]:
        """Implements :meth:`SubstrateBase.region_points`.

        This method returns an upper center point of the substrate image.
        Substrate ROI in reference image must be selected so that
        this point falls into substrate region.

        Examples:
            .. plot::
                :include-source:
                :context: reset

                >>> import cv2
                >>> from dipcoatimage.finitedepth import get_data_path, Reference
                >>> ref_path = get_data_path("ref1.png")
                >>> gray = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                >>> _, im = cv2.threshold(
                ...     gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                ... )
                >>> tempROI = (200, 50, 1200, 200)
                >>> substROI = (400, 175, 1000, 500)
                >>> ref = Reference(im, tempROI, substROI)
                >>> from dipcoatimage.finitedepth import Substrate
                >>> subst = Substrate(ref)
                >>> import matplotlib.pyplot as plt #doctest: +SKIP
                >>> plt.imshow(subst.draw())  #doctest: +SKIP
                >>> plt.plot(*subst.region_points().T, "o")  #doctest: +SKIP
        """
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
