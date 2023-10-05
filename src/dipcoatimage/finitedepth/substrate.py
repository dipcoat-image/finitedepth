"""Bare substrate."""

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
    "SubstrateError",
    "SubstrateBase",
    "Substrate",
]


class SubstrateError(Exception):
    """Base class for error from `SubstrateBase`."""

    pass


ReferenceType = TypeVar("ReferenceType", bound=ReferenceBase)
ParametersType = TypeVar("ParametersType", bound="DataclassInstance")
DrawOptionsType = TypeVar("DrawOptionsType", bound="DataclassInstance")
DataType = TypeVar("DataType", bound="DataclassInstance")


class SubstrateBase(
    abc.ABC, Generic[ReferenceType, ParametersType, DrawOptionsType, DataType]
):
    """Abstract base class for substrate.

    Substrate class recognizes the geometry of substrate image from
    `.ReferenceBase`.

    .. rubric:: Constructor

    Constructor signature must not be modified because high-level API use factory
    to generate substrate instances. Additional parameters can be introduced by
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
    implement :meth:`analyze_substrate` which returns data tuple compatible with
    :attr:`Data`.
    :meth:`analyze` is the API for analysis result.
    """

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    Data: Type[DataType]

    def __init__(
        self,
        reference: ReferenceType,
        parameters: Optional[ParametersType] = None,
        *,
        draw_options: Optional[DrawOptionsType] = None,
    ):
        """Initialize the instance."""
        super().__init__()
        self._ref = reference

        if parameters is None:
            self._parameters = self.Parameters()
        else:
            if not isinstance(parameters, self.Parameters):
                raise TypeError(f"{parameters} is not instance of {self.Parameters}")
            self._parameters = dataclasses.replace(parameters)

        if draw_options is None:
            self._draw_options = self.DrawOptions()
        else:
            if not isinstance(draw_options, self.DrawOptions):
                raise TypeError(f"{draw_options} is not instance of {self.DrawOptions}")
            self._draw_options = dataclasses.replace(draw_options)

    @property
    def reference(self) -> ReferenceType:
        """Substrate reference instance passed to constructor."""
        return self._ref

    @property
    def parameters(self) -> ParametersType:
        """Additional parameters for concrete class.

        Instance of :attr:`Parameters`, which must be a frozen dataclass.
        """
        return self._parameters

    @property
    def draw_options(self) -> DrawOptionsType:
        """Options to visualize the image.

        Instance of :attr:`DrawOptions` dataclass.
        """
        return self._draw_options

    @draw_options.setter
    def draw_options(self, options: DrawOptionsType):
        self._draw_options = options

    def image(self) -> npt.NDArray[np.uint8]:
        """Substrate image from :meth:`reference`."""
        # not property since it's not directly from the argument
        return self.reference.substrate_image()

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
        """Check to detect error and raise before analysis."""

    @abc.abstractmethod
    def draw(self) -> npt.NDArray[np.uint8]:
        """Decorate and return the substrate image as RGB format."""

    @abc.abstractmethod
    def analyze(self) -> DataType:
        """Analyze the substrate image and return the data.

        May raise error if the instance is not valid.
        """


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Additional parameters for `Substrate` instance."""

    pass


@dataclasses.dataclass
class DrawOptions:
    """Drawing options for `Substrate`."""

    pass


@dataclasses.dataclass
class Data:
    """Analysis data for `Substrate`."""

    pass


class Substrate(SubstrateBase[ReferenceBase, Parameters, DrawOptions, Data]):
    """Simplest substrate class with no geometric information.

    Examples
    --------
    Construct substrate reference instance first.

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

    Construct `Substrate` instance from reference instance.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import Substrate
       >>> subst = Substrate(ref)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP
    """

    Parameters = Parameters
    DrawOptions = DrawOptions
    Data = Data

    def region_points(self) -> npt.NDArray[np.int32]:
        """Return a point in substrate region."""
        w = self.image().shape[1]
        return np.array([[w / 2, 0]], dtype=np.int32)

    def verify(self):
        """Check error."""
        pass

    def draw(self) -> npt.NDArray[np.uint8]:
        """Return visualized image."""
        return cv2.cvtColor(self.image(), cv2.COLOR_GRAY2RGB)

    def analyze(self):
        """Return analysis data."""
        return self.Data()
