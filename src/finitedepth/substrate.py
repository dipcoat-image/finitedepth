"""Analyze substrate geometry."""

import abc
import dataclasses
from typing import TYPE_CHECKING, Generic, TypeVar

import cv2
import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore
from scipy.ndimage import gaussian_filter1d  # type: ignore
from scipy.signal import find_peaks, peak_prominences  # type: ignore

from .cache import attrcache
from .reference import ReferenceBase

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

__all__ = [
    "RefTypeVar",
    "DataTypeVar",
    "SubstrateBase",
    "SubstrateData",
    "Substrate",
    "PolySubstrateBase",
    "houghline_accum",
    "RectSubstData",
    "RectSubstrate",
]


RefTypeVar = TypeVar("RefTypeVar", bound=ReferenceBase)
"""Type variable for the reference type of :class:`SubstrateBase`."""
DataTypeVar = TypeVar("DataTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`SubstrateBase.DataType`."""


class SubstrateBase(abc.ABC, Generic[RefTypeVar, DataTypeVar]):
    """Abstract base class for substrate object.

    Substrate object stores substrate image, which is a binary image of bare substrate
    acquired from :class:`ReferenceBase` object. The role of substrate object is to
    analyze the shape of the bare substrate.

    External API can use the following members to get analysis results of
    concrete subclasses.

    * :attr:`DataType`: Dataclass type for the analysis result.
    * :meth:`analyze`: :attr:`DataType` instance containing analysis result.
    * :meth:`draw`: Visualized result.

    Arguments:
        reference: Reference instance which contains the substrate image.
    """

    DataType: type[DataTypeVar]
    """Return type of :attr:`analyze`.

    Concrete subclass must assign this attribute with dataclass type.
    """

    def __init__(self, reference: RefTypeVar):
        """Initialize the instance."""
        self._reference = reference

    @property
    def reference(self) -> RefTypeVar:
        """Reference instance which contains the substrate image."""
        return self._reference

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
            Labelled image.
            Value of ``i`` represents ``i``-th region in :meth:`region_points`.
            ``-1`` represents background.

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
    def analyze(self) -> DataTypeVar:
        """Return analysis result as dataclass.

        Return type must be :attr:`DataType`.
        """

    @abc.abstractmethod
    def draw(self, *args, **kwargs) -> npt.NDArray[np.uint8]:
        """Return visualization result."""


@dataclasses.dataclass
class SubstrateData:
    """Analysis data for :class:`Substrate`."""

    pass


class Substrate(SubstrateBase[ReferenceBase, SubstrateData]):
    """Basic implementation of substrate without any geometric specification.

    Arguments:
        reference: Reference instance which contains the substrate image.

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

    DataType = SubstrateData
    """Return :obj:`SubstrateData`."""

    def region_points(self) -> npt.NDArray[np.int32]:
        """Return an upper center point of the substrate image.

        Substrate ROI in reference image must be selected so that this point falls into
        substrate region.
        """
        return np.array([[self.image().shape[1] / 2, 0]], dtype=np.int32)

    def analyze(self):
        """Return empty :class:`SubstrateData`."""
        return self.DataType()

    def draw(self) -> npt.NDArray[np.uint8]:
        """Return :meth:`image` in RGB."""
        ret = cv2.cvtColor(self.image(), cv2.COLOR_GRAY2RGB)
        return ret  # type: ignore[return-value]


class PolySubstrateBase(SubstrateBase[RefTypeVar, DataTypeVar]):
    """Abstract base class for substrate whose cross section is a simple polygon.

    A simple polygon does not have intersection nor hole [#poly]_. Smooth corners are
    allowed.

    Arguments:
        reference: Reference instance which contains the substrate image.

    Note:
        Substrate image should *not* have:
            * Multiple substrates in one image
            * Multiple contours (e.g. substrate with holes)

    .. [#poly] https://en.wikipedia.org/wiki/Simple_polygon
    """

    @property
    @abc.abstractmethod
    def sigma(self) -> float:
        """Standard deviation of gaussian filter to smooth the noise in contour."""

    @property
    @abc.abstractmethod
    def hough_parameters(self) -> tuple[float, float, int]:
        r"""Parameters for Hough line transformation.

        Returns:
            Tuple of numbers.

            * rho: Resolution of :math:`\rho`.
            * theta: Resolution of :math:`\theta`.
            * step: Step size to jump the points for better performance.
        """

    @abc.abstractmethod
    def n(self) -> int:
        """Number of polygon vertices."""

    def region_points(self) -> npt.NDArray[np.int32]:
        """Return an upper center point of the substrate image.

        Substrate ROI in reference image must be selected so that this point falls into
        substrate region.
        """
        return np.array([[self.image().shape[1] / 2, 0]], dtype=np.int32)

    def contour(self) -> npt.NDArray[np.int32]:
        """Return the polygon contour."""
        (cnt,), _ = self.contours(0)
        return cnt

    def vertices(self) -> npt.NDArray[np.int32]:
        """Find :meth:`n` vertices of the polygon.

        A vertex is a point where two sides of a polygon meet [#vertex-geom]_.
        When the sides are curves, the vertices are defined as local extrema of
        curvature [#vertex-curve]_.

        The vertices are found by smoothing :meth:`contour` with :attr:`sigma` and
        finding the local extrema of curvature [#contour-curvature]_.

        Returns:
            Indices of the vertex points in :meth:`contour`.

        .. [#vertex-geom] https://en.wikipedia.org/wiki/Vertex_(geometry)
        .. [#vertex-curve] https://en.wikipedia.org/wiki/Vertex_(curve)
        .. [#contour-curvature] https://stackoverflow.com/q/32629806
        """
        cnt = self.contour().astype(np.float64)

        # 1. Calculate curvatures
        f = gaussian_filter1d(cnt, self.sigma, axis=0, order=0, mode="wrap")
        f_dt = np.gradient(f, axis=0)
        f_dt2 = np.gradient(f_dt, axis=0)
        K = np.abs(np.cross(f_dt, f_dt2)) / np.linalg.norm(f_dt, axis=-1) ** 3

        # 2. Repeat the array (periodic)
        L = len(K)
        (K_rpt,) = np.concatenate([K[-L // 2 :], K, K[: L // 2]], axis=0).T

        # 3. Find peak
        peaks = find_peaks(K_rpt)[0].astype(np.int32)
        (idxs,) = np.where((L // 2 <= peaks) & (peaks < 3 * L // 2))
        peaks = peaks[idxs]
        prom, _, _ = peak_prominences(K_rpt, peaks)
        prom_peaks = peaks[np.argsort(prom)[-self.n() :]]
        return np.sort((prom_peaks - (L // 2)) % L)

    def sides(self) -> tuple[npt.NDArray[np.int32], ...]:
        """Find :meth:`n` sides of the polygon.

        The sides are found by slicing :meth:`contour` by :meth:`vertices`.

        Returns:
            Tuple of array containing points on each side of the polygon contour.
            The arrays are sorted so that the side containing the first point of the
            contour comes first.

        Note:
            Sides can be noisy and curved. Use :meth:`sidelines` to get linear models.

            The term "side" is used instead of "edge" to avoid confusion from
            other image processing methods (e.g. Canny edge detection).
        """
        cnt = self.contour()
        vert = self.vertices()
        sides = np.split(cnt, vert)
        dists = np.linalg.norm(cnt[vert] - cnt[0], axis=-1)
        shift = vert[np.argmin(dists)]
        L = len(cnt)
        sides = np.split(np.roll(cnt, shift, axis=0), np.sort((vert - shift) % L))[1:]
        return tuple(sides)

    @attrcache("_sidelines")
    def sidelines(self) -> npt.NDArray[np.float32]:
        r"""Find :meth:`n` sidelines of the polygon.

        Sideline is the line that contains one side of the polygon [#sideline]_.
        The sidelines are found by performing Hough line transformation on :meth:`sides`
        with :attr:`hough_parameters`.

        Returns:
            Vector of line parameters in :math:`(\rho, \theta)`.
            :math:`\rho` is the distance from the coordinate origin.
            :math:`\theta` is the angle of normal vector from the origin
            to the line.

        Note:
            Range of angle is
            :math:`\theta \in (-\frac{3 \pi}{2}, \frac{\pi}{2}]`.
            Arctangent direction can be acquired by
            :math:`\theta + \frac{\pi}{2}`.

        .. [#sideline] https://en.wikipedia.org/wiki/Extended_side
        """
        # Do not find the line from smoothed contour. Noise is removed anyway
        # without smoothing by Hough transformation. In fact, smoothing
        # propagates the outlier error to nearby data.
        RHO_RES, THETA_RES, STEP = self.hough_parameters
        lines = []
        # Directly use Hough transformation to find lines
        for side in self.sides():
            tan = np.diff(side, axis=0)
            atan = np.arctan2(tan[..., 1], tan[..., 0])  # -pi < atan <= pi
            theta = atan - np.pi / 2
            tmin, tmax = theta.min(), theta.max()
            if tmin < tmax:
                theta_rng = np.arange(tmin, tmax, THETA_RES, dtype=np.float32)
            else:
                theta_rng = np.array([tmin], dtype=np.float32)

            # Interpolate & perform hough transformation.
            c = side[::STEP]
            rho = c[..., 0] * np.cos(theta_rng) + c[..., 1] * np.sin(theta_rng)
            rho_digit = (rho / RHO_RES).astype(np.int32)

            _, (rho, theta_idx) = houghline_accum(rho_digit)
            lines.append([[rho * RHO_RES, theta_rng[theta_idx]]])

        return np.array(lines, dtype=np.float32)

    def sideline_intersections(self) -> npt.NDArray[np.float32]:
        """Find intersections of :meth:`sidelines`."""
        ((r1, t1),) = self.sidelines().transpose(1, 2, 0)
        ((r2, t2),) = np.roll(self.sidelines(), 1, axis=0).transpose(1, 2, 0)
        mat = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]]).transpose(
            2, 0, 1
        )
        vec = np.array([[r1], [r2]]).transpose(2, 0, 1)
        (ret,) = (np.linalg.inv(mat) @ vec).transpose(2, 0, 1)
        return ret


@njit(cache=True)
def houghline_accum(
    rho_array: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], tuple[float, int]]:
    """Perform hough line accumulation.

    Arguments:
        rho_array: Array containing rho and theta values for every points.
            The shape must be ``(P, T)``, where ``P`` is the number of points
            and ``T`` is the numbers of digitized theta intervals.

    Returns:
        Tuple of accumulation matrix and detected ``(rho, theta_idx)`` value.
    """
    rho_min = np.min(rho_array)
    n_rho = np.max(rho_array) - rho_min + 1
    n_pts, n_theta = rho_array.shape
    accum = np.zeros((n_rho, n_theta), dtype=np.int32)

    maxloc = (0, 0)
    for i in range(n_pts):
        for j in range(n_theta):
            r = rho_array[i, j] - rho_min
            accum[r, j] += 1
            if accum[r, j] > accum[maxloc]:
                maxloc = (r, j)

    rho_theta = (float(maxloc[0] + rho_min), int(maxloc[1]))
    return accum, rho_theta


@dataclasses.dataclass
class RectSubstData:
    """Analysis data for :class:`RectSubstrate`.

    Arguments:
        Width: Width of the rectangular cross section in pixels.
    """

    Width: np.float32


class RectSubstrate(PolySubstrateBase[ReferenceBase, RectSubstData]):
    """Substrate having rectangular cross section.

    Arguments:
        reference: Reference instance which contains the substrate image.
        sigma: Standard deviation of gaussian filter to smooth the noise in contour.
        rho_thres, theta_thres, hough_step: Hough line transformation parameters.

    Examples:
        .. plot::
            :include-source:
            :context: reset

            >>> import cv2
            >>> from finitedepth import get_sample_path, Reference, RectSubstrate
            >>> img = cv2.imread(get_sample_path("ref.png"), cv2.IMREAD_GRAYSCALE)
            >>> _, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            >>> ref = Reference(bin, (10, 10, 1250, 200), (100, 100, 1200, 500))
            >>> subst = RectSubstrate(ref, 3.0, 1.0, 0.01)
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(subst.draw()) #doctest: +SKIP
    """

    DataType = RectSubstData
    """Return :obj:`RectSubstData`."""

    def __init__(
        self,
        reference: ReferenceBase,
        sigma: float,
        rho_thres: float,
        theta_thres: float,
        hough_step: int = 1,
    ):
        """Initialize the instance by passed arguments."""
        super().__init__(reference)
        self._sigma = sigma
        self._hough_param = (rho_thres, theta_thres, hough_step)

    @property
    def sigma(self) -> float:
        """Sigma value passed to the constructor."""
        return self._sigma

    @property
    def hough_parameters(self) -> tuple[float, float, int]:
        """Hough line transformation parameters passed to the constructor."""
        return self._hough_param

    def n(self) -> int:
        """Number of vertices of rectangle, which is 4."""
        return 4

    def analyze(self):
        """Return :class:`RectSubstData`."""
        _, B, C, _ = self.sideline_intersections()
        return self.DataType(np.linalg.norm(B - C))

    def draw(
        self,
        mode: str = "image",
        vertice_color: tuple[int, int, int] = (0, 255, 0),
        vertice_thickness: int = 1,
        vertice_markerSize: int = 20,
        sideline_color: tuple[int, int, int] = (0, 0, 255),
        sideline_thickness: int = 1,
    ) -> npt.NDArray[np.uint8]:
        """Draw substrate image and show vertices and sidelines.

        Arguments:
            mode ({`'image', 'contour'`}): Draw mode.
                `'image'` draws :meth:`image`, while `'contour'` draws :meth:`contour`.
            vertice_color: Vertice marker color for :func:`cv2.drawMarker`.
            vertice_thickness: Vertice marker thickness for :func:`cv2.drawMarker`.
            vertice_markerSize: Vertice marker size for :func:`cv2.drawMarker`.
            sideline_color: Sideline color for :func:`cv2.line`.
            sideline_thickness: Sideline thickness for :func:`cv2.line`.
        """
        if mode == "image":
            image = self.image()
        elif mode == "contour":
            h, w = self.image().shape[:2]
            image = np.full((h, w), 255, np.uint8)
            cv2.drawContours(image, self.contour(), -1, 0, 1)  # type: ignore
        else:
            raise TypeError("Unrecognized draw mode: %s" % mode)
        ret = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if vertice_thickness > 0:
            for (pt,) in self.contour()[self.vertices()]:
                cv2.drawMarker(
                    ret,
                    pt,
                    color=vertice_color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=vertice_markerSize,
                    thickness=vertice_thickness,
                )

        if sideline_thickness > 0:
            try:
                tl, bl, br, tr = self.sideline_intersections().astype(np.int32)

                cv2.line(ret, tl, tr, sideline_color, sideline_thickness)
                cv2.line(ret, tr, br, sideline_color, sideline_thickness)
                cv2.line(ret, br, bl, sideline_color, sideline_thickness)
                cv2.line(ret, bl, tl, sideline_color, sideline_thickness)
            except ValueError:
                pass

        return ret  # type: ignore[return-value]
