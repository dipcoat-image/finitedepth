"""
Polygonal Substrate
===================

:mod:`dipcoatimage.finitedepth.polysubstrate` provides abstract base class for
substrate with polygonal cross section shape.

"""
import cv2  # type: ignore
import dataclasses
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d  # type: ignore
from scipy.signal import find_peaks, peak_prominences  # type: ignore
from typing import TypeVar, List, Tuple, Optional, Type
from .substrate import SubstrateError, SubstrateBase
from .util import DataclassProtocol


__all__ = [
    "PolySubstrateError",
    "HoughLinesParameters",
    "PolySubstrateParameters",
    "PolySubstrateBase",
]


class PolySubstrateError(SubstrateError):
    """Base class for the errors from polygonal substrate class."""

    pass


@dataclasses.dataclass(frozen=True)
class HoughLinesParameters:
    """Parameters for :func:`cv2.HoughLines`."""

    rho: float
    theta: float
    threshold: int
    srn: float = 0.0
    stn: float = 0.0
    min_theta: float = 0.0
    max_theta: float = np.pi


@dataclasses.dataclass(frozen=True)
class PolySubstrateParameters:
    """
    Parameters for the polygonal substrate class to detect its sides.
    """

    HoughLines: HoughLinesParameters
    GaussianSigma: int = 3
    Theta: float = 0.1


ParametersType = TypeVar("ParametersType", bound=PolySubstrateParameters)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)


class PolySubstrateBase(SubstrateBase[ParametersType, DrawOptionsType]):
    """
    Abstract base class for substrate whose cross section is simple polygon.

    :class:`PolySubstrateBase` provides method to detect the sides of the
    polygonal substrate. The corners of the substrate can be smooth.

    Concrete class must define :attr:`SidesNum` class attribute, which is the
    number of the sides of polygon.
    """

    __slots__ = (
        "_lines",
        "_vertex_points",
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    SidesNum: int

    def contour(self) -> npt.NDArray[np.int32]:
        (cnt,), _ = cv2.findContours(
            cv2.bitwise_not(self.binary_image()),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        return cnt

    def split_contour(self) -> List[npt.NDArray[np.int32]]:
        contour = self.contour()

        # 1. Get theta values of the tangent curve and take smooth derivative.
        # This allows us to find sides even from jittery polygons.
        # Since the contour is periodic and edge information might be lost,
        # we repeat theta in both direction.
        tan = np.gradient(contour, axis=0)
        theta = np.arctan2(tan[..., 1], tan[..., 0])
        theta2 = np.concatenate(
            [
                theta[-(len(theta) // 2) :],
                theta,
                theta[: (len(theta) // 2)],
            ],
            axis=0,
        )
        sigma = self.parameters.GaussianSigma
        theta2_smooth = gaussian_filter1d(theta2, sigma, axis=0)

        # 2. Find peak. Each peak shows the point where side changes. This allows
        # us to discern individual sides lying on same line.
        # Since we repeated theta, we find (2*self.SidesNum) peaks and discard
        # the duplicates.
        theta2_grad = np.gradient(theta2_smooth, axis=0)
        theta2_abs = np.abs(theta2_grad)[..., 0]
        peaks, _ = find_peaks(theta2_abs)
        prom, _, _ = peak_prominences(theta2_abs, peaks)
        k = 2 * self.SidesNum
        prom_peaks = peaks[np.sort(np.argsort(prom)[-k:])]
        (idxs,) = np.where(
            (len(theta) // 2 <= prom_peaks) & (prom_peaks < 3 * len(theta) // 2)
        )
        corner_peaks = np.sort(prom_peaks[idxs][: self.SidesNum]) - len(theta) // 2

        # 3. Digitize smoothed line and get votes to determine main theta.
        # We must take vote from disgitized smoothed line, not from raw theta
        # in order to be robust from jittery noises.
        theta_smooth = theta2_smooth[len(theta) // 2 : -len(theta) // 2]
        # roll s.t. no residual section at the beginning
        SHIFT = corner_peaks[0]
        theta_smooth = np.roll(theta_smooth, -SHIFT, axis=0)
        corner_peaks = corner_peaks - SHIFT

        THETA_STEP = self.parameters.Theta
        indices = []
        for region in np.split(theta_smooth, corner_peaks[1:], axis=0):
            digitized = (region / THETA_STEP).astype(np.int32) * THETA_STEP
            val, count = np.unique(digitized, return_counts=True)
            main_theta = val[np.argmax(count)]
            # XXX: give more offset
            idxs, _ = np.nonzero(digitized == main_theta)
            indices.append([idxs[0], idxs[-1]])

        base_indices = SHIFT + corner_peaks[..., np.newaxis]
        split_indices = np.sort((base_indices + np.array(indices)) % len(theta), axis=0)
        sections = np.split(contour, split_indices.flatten())
        return sections

    def edge(self) -> npt.NDArray[np.bool_]:
        """
        Return the substrate edge as boolean array.

        The edge locations are acquired from :meth:`contour`.
        """
        h, w = self.image().shape[:2]
        ret = np.zeros((h, w), bool)
        ((x, y),) = self.contour().transpose(1, 2, 0)
        ret[y, x] = True
        return ret

    def edge_hull(self) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
        hull = np.flip(cv2.convexHull(self.contour()), axis=0)
        # TODO: get more points by interpolating to `hull`
        tangent = np.gradient(hull, axis=0)
        # TODO: perform edge tangent flow to get smoother curve
        return hull, tangent

    def lines(self) -> npt.NDArray[np.float32]:
        """
        Get :func:`cv2.HoughLines` result from the binary image of *self*.

        This method first acquires the edge image from :meth:`edge`, and
        apply Hough line transformation with the parameters defined in
        :attr:`parameters`.

        If no line can be found, an empty array is returned.

        Notes
        =====

        This property is cached. Do not mutate the result.

        """
        if not hasattr(self, "_lines"):
            # TODO: find way to directly get lines from contour, not edge image
            hparams = dataclasses.asdict(self.parameters.HoughLines)
            lines = cv2.HoughLines(self.edge().astype(np.uint8), **hparams)
            if lines is None:
                lines = np.empty((0, 1, 2), dtype=np.float32)
            self._lines = lines
        return self._lines

    def sides(self) -> npt.NDArray[np.float32]:
        """
        Return the sides of the substrate polygon.

        Sides are sorted along the contour.
        """
        lines = self.lines()[: self.SidesNum]
        # find the closest line for each point
        ((r, theta),) = lines.transpose(1, 2, 0)
        A = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        Ap = np.repeat(self.contour(), self.SidesNum, axis=1) - A
        AB = np.column_stack([np.sin(theta), -np.cos(theta)])
        t = np.sum(Ap * AB, axis=-1)
        AC = np.repeat(t[..., np.newaxis], 2, axis=-1) * AB
        dists = np.linalg.norm(Ap - AC, axis=-1)
        point_labels = np.argmin(dists, axis=-1)
        # sort the lines along the contour
        line_order = []
        for i in range(len(lines)):
            (pos,) = np.where(point_labels == i)
            line_order.append(np.mean(pos))
        return lines[np.argsort(line_order)]

    def vertex_points(self):
        if not hasattr(self, "_vertex_points"):
            sides = self.sides()
            ((r1, t1),) = sides.transpose(1, 2, 0)
            ((r2, t2),) = np.roll(sides, 1, axis=0).transpose(1, 2, 0)
            mat = np.array(
                [[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]]
            ).transpose(2, 0, 1)
            vec = np.array([[r1], [r2]]).transpose(2, 0, 1)
            sol_exists = np.linalg.det(mat) != 0
            (self._vertex_points,) = (
                np.linalg.inv(mat[np.where(sol_exists)]) @ vec[np.where(sol_exists)]
            ).transpose(2, 0, 1)
        return self._vertex_points

    def examine(self) -> Optional[PolySubstrateError]:
        l_num = len(self.lines())
        N = self.SidesNum
        if l_num < N:
            return PolySubstrateError(
                f"Insufficient lines from HoughLines (needs >= {N}, got {l_num})"
            )

        return None
