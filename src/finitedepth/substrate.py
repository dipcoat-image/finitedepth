"""Analyze substrate geometry."""

import abc

import cv2
import numpy as np
import numpy.typing as npt

from .cache import attrcache
from .reference import ReferenceBase

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

__all__ = [
    "SubstrateBase",
    "Substrate",
    "RectSubstrate",
]


class SubstrateBase(abc.ABC):
    """Abstract base class for substrate instance.

    Substrate instance stores a substrate image, which is a binary image of
    bare substrate acquired from reference instance. The role of substrate
    instance is to analyze the shape of the bare substrate. :meth:`draw` returns
    visualized result.
    """

    @property
    @abc.abstractmethod
    def reference(self) -> ReferenceBase:
        """Reference instance which contains the substrate image."""

    def image(self) -> npt.NDArray[np.uint8]:
        """Substrate image from :meth:`reference`."""
        x0, y0, x1, y1 = self.reference.substrateROI
        return self.reference.image[y0:y1, x0:x1]

    @abc.abstractmethod
    def region_points(self) -> npt.NDArray[np.int32]:
        """Coordinates of points representing each substrate region.

        Substrate image can have multiple disconnected substrate regions.
        Concrete classes should implement this method to return coordinates of
        points representing each region.

        Returns:
            `(N, 2)`-shaped array, where `N` is the number of substrate regions.
            Column should be the coordinates of points in ``(x, y)``.

        Note:
            These points are used to distinguish substrate regions from other
            foreground pixels, and give indices to each region.

            As higher-level methods are expected to rely on this method,
            it is best to keep this method simple and independent.
        """

    @attrcache("_regions")
    def regions(self) -> npt.NDArray[np.int8]:
        """Labelled image of substrate regions.

        Substrate regions are determined as connected component including
        a point in :meth:`region_points`.

        Returns:
            Labelled image. Value of ``i`` represents ``i``-th region in
            :meth:`region_points`. ``-1`` represents background.

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
    ) -> tuple[tuple[npt.NDArray[np.int32], ...], npt.NDArray[np.int32]]:
        """Find contours of a substrate region.

        Arguments:
            region: Label of the region from :meth:`regions`.

        Returns:
            Tuple of the result of :func:`cv2.findContours`.

        Note:
            Contours are dense, i.e., no approximation is made.
        """
        reg = (self.regions() == region) * np.uint8(255)
        return cv2.findContours(reg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    @abc.abstractmethod
    def draw(self, *args, **kwargs) -> npt.NDArray[np.uint8]:
        """Return visualization result."""


class Substrate(SubstrateBase):
    """Basic implementation of substrate without any geometric specification.

    Examples:
        .. plot::
            :include-source:
            :context: reset

            >>> import cv2
            >>> from finitedepth import get_sample_path, Reference, Substrate
            >>> img = cv2.imread(get_sample_path("ref.png"), cv2.IMREAD_GRAYSCALE)
            >>> _, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            >>> ref = Reference(bin, (10, 10, 1250, 200), (100, 100, 1200, 500))
            >>> subst = Substrate(ref)
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(subst.draw()) #doctest: +SKIP
    """

    def __init__(self, reference: ReferenceBase):
        """Initialize the instance.

        Arguments:
            reference: Reference instance which contains the substrate image.
        """
        self._ref = reference

    @classmethod
    def from_dict(cls, reference: ReferenceBase, d: dict) -> Self:
        """Construct an instance from *reference* and a dictionary *d*.

        The dictionary must not have any item.
        """
        return cls(reference, **d)

    @property
    def reference(self) -> ReferenceBase:
        """Reference instance which contains the substrate image."""
        return self._ref

    def region_points(self) -> npt.NDArray[np.int32]:
        """Implement :meth:`SubstrateBase.region_points`.

        This method returns an upper center point of the substrate image.
        Substrate ROI in reference image must be selected so that
        this point falls into substrate region.
        """
        return np.array([[self.image().shape[1] / 2, 0]], dtype=np.int32)

    def draw(self) -> npt.NDArray[np.uint8]:
        """Implement :meth:`ReferenceBase.draw`."""
        ret = cv2.cvtColor(self.image(), cv2.COLOR_GRAY2RGB)
        return ret  # type: ignore[return-value]


class RectSubstrate:
    """Rectangular substrate."""

    @classmethod
    def from_dict(cls, d): ...
