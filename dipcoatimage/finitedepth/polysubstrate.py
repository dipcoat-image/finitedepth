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


ROTATION_MATRIX = np.array([[0, 1], [-1, 0]])


@dataclasses.dataclass(frozen=True)
class PolySubstrateParameters:
    """
    Parameters for :class:`PolySubstrate`.

    Parameters
    ----------
    Sigma: positive float
        Standard deviation for Gaussian kernel. Used to smooth the signal for
        finding corners and edges.
    Rho: positive float
        Radian resolution for Hough transformation to detect the polygon sides.
    Theta: positive float
        Angle resolution for Hough transformation to detect the polygon sides.

    """

    Sigma: float
    Rho: float
    Theta: float


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

    def corners(self) -> npt.NDArray[np.int32]:
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
        grad = gaussian_filter1d(theta2, self.parameters.Sigma, axis=0, order=1)

        # 2. Find peak. Each peak shows the point where side changes. This allows
        # us to discern individual sides lying on same line.
        # Since we repeated theta, we select the peaks in desired region.
        theta2_abs = np.abs(grad)[..., 0]
        peaks2, _ = find_peaks(theta2_abs)
        peaks2 = peaks2.astype(np.int32)
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
        Find linear sides of the polygon using Hough line transformation.

        Returns
        -------
        lines
            Vector of line parameters in $(\rho, \theta)$. $\rho$ is the distance
            from the coordinate origin. $\theta$ is the angle of normal vector
            from the origin to the line.

        Notes
        -----
        The ranges of parameters are $\rho \in (-\infty, \infty)$ and
        $\theta \in (-\frac{3 \pi}{2}, \frac{\pi}{2}]$. Arctan direction of the
        side vector can be acquired by $\theta + \frac{\pi}{2}$.
        """
        if hasattr(self, "_sides"):
            return self._sides  # type: ignore[has-type]

        SHIFT = self.corners()[0]
        corners = self.corners() - SHIFT
        cnt_roll = np.roll(self.contour(), -SHIFT, axis=0)

        RHO_RES = self.parameters.Rho
        THETA_RES = self.parameters.Theta
        lines = []
        # Directly use Hough transformation to find lines
        for c in np.split(cnt_roll, corners[1:], axis=0):
            tan = np.diff(c, axis=0)
            atan = np.arctan2(tan[..., 1], tan[..., 0])  # -pi < atan <= pi
            theta = atan - np.pi/2
            tmin, tmax = theta.min(), theta.max()

            if tmin < tmax:
                theta_rng = np.arange(tmin, tmax, THETA_RES, dtype=np.float32)
            else:
                theta_rng = np.array([tmin], dtype=np.float32)
            rho = c[..., 0] * np.cos(theta_rng) + c[..., 1] * np.sin(theta_rng)
            rho_digit = (rho / RHO_RES).astype(np.int32)

            rho_min = rho_digit.min()
            theta_idxs = np.arange(len(theta_rng))

            # encode 2d array into 1d array for faster accumulation
            # TODO: try Cython to make faster. (iterate over 2D rows, not encode)
            idxs = (rho_digit - rho_min) * len(theta_idxs) + theta_idxs
            val, counts = np.unique(idxs, return_counts=True)
            max_idx = val[np.argmax(counts)]

            r0_idx = max_idx // len(theta_idxs)
            r0 = (r0_idx + rho_min) * RHO_RES
            t0_idx = max_idx % len(theta_idxs)
            t0 = theta_rng[t0_idx]
            lines.append([[r0, t0]])

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

    def regions(self) -> npt.NDArray[np.int32]:
        """
        Return the indices of the contour points which fit best to linear sides.
        """
        if hasattr(self, "_regions"):
            return self._regions  # type: ignore[has-type]

        # XXX: implement corner region detection
        ret = np.column_stack(
            [
                self.corners(),
                np.append(self.corners()[1:], len(self.contour())),
            ]
        )
        self._regions = ret
        return self._regions

    def normal(self):
        """
        Return unit normal vectors on each point of contour.

        Uses polygon model.
        """
        # XXX: Current model assumes sharp polygon.
        # After smooth corner region detection is done, enhance this!
        SHIFT = self.corners()[0]
        corners = self.corners() - SHIFT
        reps = np.diff(np.append(corners, len(self.contour())))

        _, thetas = self.sides().transpose(2, 0, 1)
        n = np.stack([-np.cos(thetas), -np.sin(thetas)]).transpose(1, 2, 0)

        return np.roll(np.repeat(n, reps, axis=0), SHIFT, axis=0)

    def examine(self) -> Optional[PolySubstrateError]:
        try:
            self.corners()
        except PolySubstrateError as err:
            return err
        return None
