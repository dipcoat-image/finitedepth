"""
Polygonal Substrate
===================

:mod:`dipcoatimage.finitedepth.polysubstrate` provides abstract base class for
substrate with polygonal cross section.

"""
import cv2  # type: ignore
import dataclasses
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d  # type: ignore
from scipy.signal import find_peaks, peak_prominences  # type: ignore
from typing import TypeVar, Optional, Type
from .substrate import SubstrateError, SubstrateBase
from .util import DataclassProtocol


__all__ = [
    "PolySubstrateError",
    "PolySubstrateParameters",
    "PolySubstrateBase",
]


class PolySubstrateError(SubstrateError):
    """Base class for the errors from :class:`PolySubstrate`."""

    pass


@dataclasses.dataclass(frozen=True)
class PolySubstrateParameters:
    """
    Parameters for :class:`PolySubstrate`.

    Parameters
    ----------
    GaussianSigma: positive float
        Standard deviation for Gaussian kernel. Used to smooth the signal for
        finding corners and edges.
    Theta: positive float
        Angle resolution to detect the polygon sides.

    """

    GaussianSigma: float = 3.0
    Theta: float = 0.01


ParametersType = TypeVar("ParametersType", bound=PolySubstrateParameters)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)


class PolySubstrateBase(SubstrateBase[ParametersType, DrawOptionsType]):
    """
    Abstract base class for substrate whose cross section is simple polygon.

    :class:`PolySubstrateBase` provides method to detect the sides of the
    polygonal substrate. Smooth corners are allowed.

    Concrete class must define :attr:`SidesNum` class attribute, which is the
    number of the sides of polygon.
    """

    __slots__ = (
        "_contour",
        "_corners",
        "_sides",
        "_vertex_points",
        "_regions",
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    SidesNum: int

    def contour(self) -> npt.NDArray[np.int32]:
        """
        Return the contour of the substrate.

        Only one contour must exist in the image, i.e. no discontius point.
        """
        if hasattr(self, "_contour"):
            return self._contour  # type: ignore[has-type]

        (cnt,), _ = cv2.findContours(
            cv2.bitwise_not(self.binary_image()),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        self._contour = cnt.astype(np.int32)
        return self._contour

    def corners(self) -> npt.NDArray[np.int64]:
        """
        Return the indices of the corner points on contour.

        This method returns the indices of the local maxima of the changes in the
        direction of the tangent vector. To get the entire corner region of
        smooth polygon, split the contour with indices from :meth:`regions`.

        See Also
        --------
        sides
            Detects linear sides.
        vertex_points
            Return coordinates of intersections of sides.

        """
        if hasattr(self, "_corners"):
            return self._corners  # type: ignore[has-type]

        # 1. Get theta values of the tangent curve and take smooth derivative.
        # This allows us to find corners even from jittery polygons.
        # Since the contour is periodic we repeat theta in both direction to
        # ensure smoothing is uniformly done and boundary peaks can be found.
        L = len(self.contour())
        contour2 = np.concatenate(
            [
                self.contour()[-(L // 2) :],
                self.contour(),
                self.contour()[: (L // 2)],
            ],
            axis=0,
        )  # DON'T replace it with gaussian mode="wrap" (peak finding will fail)
        tan2 = np.diff(contour2, axis=0)
        theta2 = np.arctan2(tan2[..., 1], tan2[..., 0])
        grad = gaussian_filter1d(theta2, self.parameters.GaussianSigma, axis=0, order=1)

        # 2. Find peak. Each peak shows the point where side changes. This allows
        # us to discern individual sides lying on same line.
        # Since we repeated theta, we select the peaks in desired region.
        theta2_abs = np.abs(grad)[..., 0]
        peaks2, _ = find_peaks(theta2_abs)
        (idxs,) = np.where((L // 2 <= peaks2) & (peaks2 < 3 * L // 2))
        peaks = peaks2[idxs]
        prom, _, _ = peak_prominences(theta2_abs, peaks)
        if len(prom) < self.SidesNum:
            msg = (
                "Insufficient number of corners"
                f" (needs {self.SidesNum}, detected {len(prom)})"
            )
            raise PolySubstrateError(msg)
        prom_peaks = peaks[np.sort(np.argsort(prom)[-self.SidesNum :])]
        corners = np.sort(prom_peaks) - (L // 2)

        # 3. Compensate index-by-one error, which is probably from np.diff().
        # This error makes perfectly sharp corner incorrectly located by -1.
        self._corners = np.sort((corners + 1) % L)
        return self._corners

    def sides(self) -> npt.NDArray[np.float32]:
        r"""
        Find the linear sides of the polygon.

        Returns
        -------
        lines
            Line parameters in $(\rho, \theta)$, where $\rho$ is the distance
            from the coordinate origin and $\theta \in [0, \pi]$ is the angle of
            normal vector from the origin to the line.
        """
        if hasattr(self, "_sides"):
            return self._sides  # type: ignore[has-type]

        tan = np.diff(
            np.concatenate([self.contour(), self.contour()[:1]], axis=0),
            axis=0,
        )
        theta = np.arctan2(tan[..., 1], tan[..., 0])
        corners = self.corners()

        # roll s.t. no section is divided by the boundary
        SHIFT = corners[0]
        corners = corners - SHIFT
        theta_roll = np.roll(theta, -SHIFT, axis=0)
        cnt_roll = np.roll(self.contour(), -SHIFT, axis=0)

        THETA_STEP = self.parameters.Theta
        lines = []
        for t, c in zip(
            np.split(theta_roll, corners[1:], axis=0),
            np.split(cnt_roll, corners[1:], axis=0),
        ):
            smooth_t = gaussian_filter1d(t, self.parameters.GaussianSigma, axis=0)
            digitized = (smooth_t / THETA_STEP).astype(int) * THETA_STEP
            val, count = np.unique(digitized, return_counts=True)
            t0 = val[np.argmax(count)]
            idxs, _ = np.nonzero(digitized == t0)
            center = np.mean(c[idxs], axis=0)
            (rho,) = center[..., 0] * np.cos(t0) + center[..., 1] * np.sin(t0)
            lines.append([[rho, t0]])

        self._sides = np.array(lines, dtype=np.float32)
        return self._sides

    def vertex_points(self) -> npt.NDArray[np.float32]:
        """
        Return the coordinates of intersections of polygon sides.

        If the polygon has smooth corner, the vertex points are different from
        the corner points on the contour.

        See Also
        --------
        corners
            Return indices of the corner points on contour.

        """
        if hasattr(self, "_vertex_points"):
            return self._vertex_points  # type: ignore[has-type]

        ((r1, t1),) = self.sides().transpose(1, 2, 0)
        ((r2, t2),) = np.roll(self.sides(), 1, axis=0).transpose(1, 2, 0)
        mat = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]]).transpose(
            2, 0, 1
        )
        vec = np.array([[r1], [r2]]).transpose(2, 0, 1)

        (self._vertex_points,) = (np.linalg.inv(mat) @ vec).transpose(2, 0, 1)
        return self._vertex_points

    def regions(self) -> npt.NDArray[np.int64]:
        """
        Return the indices of the contour points which fit best to linear sides.
        """
        if hasattr(self, "_regions"):
            return self._regions  # type: ignore[has-type]

        # XXX: implement fitting to bezier curve
        L = len(self.contour())
        tan = np.diff(
            np.concatenate([self.contour(), self.contour()[:1]], axis=0),
            axis=0,
        )
        theta = np.arctan2(tan[..., 1], tan[..., 0])
        corners = self.corners()

        # roll s.t. no section is divided by the boundary
        SHIFT = corners[0]
        theta_roll = np.roll(theta, -SHIFT, axis=0)

        THETA_STEP = self.parameters.Theta
        indices = []
        for t in np.split(theta_roll, corners[1:], axis=0):
            smooth_t = gaussian_filter1d(t, self.parameters.GaussianSigma, axis=0)
            digitized = (smooth_t / THETA_STEP).astype(int) * THETA_STEP
            val, count = np.unique(digitized, return_counts=True)
            main_theta = val[np.argmax(count)]
            # XXX: may need to detect lines by the points-line distances, not by
            # theta angles.
            idxs, _ = np.nonzero(digitized == main_theta)
            indices.append([idxs[0], idxs[-1]])

        base_indices = SHIFT + corners[..., np.newaxis]
        sides_indices = (base_indices + np.array(indices, dtype=np.int64)) % L
        sortidx = np.argsort(sides_indices, axis=0)
        sides_indices = sides_indices[sortidx[..., 0]]

        # Compensate index-by-one error, which is probably from np.diff().
        # This error makes perfectly sharp corner incorrectly located by -1.
        self._regions = sides_indices + 1
        return self._regions

    def examine(self) -> Optional[PolySubstrateError]:
        try:
            self.corners()
        except PolySubstrateError as err:
            return err
        return None
