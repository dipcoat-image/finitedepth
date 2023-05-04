"""
Polygonal Substrate
===================

:mod:`dipcoatimage.finitedepth.polysubstrate` provides abstract base class for
substrate with polygonal cross section.

"""
import dataclasses
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d  # type: ignore
from scipy.signal import find_peaks, peak_prominences  # type: ignore
from typing import TypeVar, Optional, Type, Tuple
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
        Standard deviation of Gaussian kernel to smooth the signal for
        vertex finding.
    Rho: positive float
        Radian resolution for Hough transformation to detect the sidelines.
    Theta: positive float
        Angle resolution for Hough transformation to detect the sidelines.

    """

    Sigma: float
    Rho: float
    Theta: float


ParametersType = TypeVar("ParametersType", bound=PolySubstrateParameters)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)


class PolySubstrateBase(SubstrateBase[ParametersType, DrawOptionsType]):
    """
    Abstract base class for substrates whose cross section is a simple
    polygon[1]_.

    :class:`PolySubstrateBase` provides method to detect the sides of the
    a polygonal substrate. The sides are expected to be mostly linear. Smooth
    corners are allowed.

    Concrete class must define :attr:`SidesNum` class attribute, which is the
    number of the sides of the polygon.

    The following substrate images are not supported:
    - Multiple substrates in one image
    - Multiple contours (e.g. substrate with holes)

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Simple_polygon
    """

    __slots__ = (
        "_vertices",
        "_sidelines",
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    SidesNum: int

    def nestled_points(self) -> npt.NDArray[np.int32]:
        # XXX: Need better way to find center...
        w = self.image().shape[1]
        return np.array([[w / 2, 0]], dtype=np.int32)

    def contour(self) -> npt.NDArray[np.int32]:
        """Return the contour of the substrate."""
        (((cnt,), _),) = self.contours()
        return cnt

    def vertices(self) -> npt.NDArray[np.int32]:
        """
        Find the polygon vertices.

        Returns
        -------
        ndarray
            Indices of the vertex points in :meth:`contour`.

        Notes
        -----
        A vertex is a point where two or more sides of a polygon meet[1]_.
        The sides can be curves, where the vertices can be defined as local
        extrema of curvature[2]_.

        This method finds the vertices by locating a certain number (defined by
        :attr:`SidesNum`) of curvature extrema.

        See Also
        --------
        sides
            Points of each side split by vertices.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Vertex_(geometry)
        .. [2] https://en.wikipedia.org/wiki/Vertex_(curve)

        """
        if not hasattr(self, "_vertices"):
            # 1. Calculate the change of direction instead of curvature because
            # it's faster and still gives accurate result.
            # Since the contour is periodic we repeat the data in both directions
            # to ensure boundary peaks are be found.
            L = len(self.contour())
            contour2 = np.concatenate(
                [
                    self.contour()[-(L // 2) :],
                    self.contour(),
                    self.contour()[: (L // 2)],
                ],
                axis=0,
            )
            tan2 = np.diff(contour2, axis=0)
            theta2 = np.arctan2(tan2[..., 1], tan2[..., 0])
            # Take smooth derivative to deal with jittery noises.
            grad = gaussian_filter1d(theta2, self.parameters.Sigma, axis=0, order=1)

            # 2. Find peak. Each peak shows the point where side changes.
            # This allows us to discern individual sides lying on same line.
            # Since we repeated theta, we select the peaks in desired region.
            theta2_abs = np.abs(grad)[..., 0]
            peaks2, _ = find_peaks(theta2_abs)
            peaks2 = peaks2.astype(np.int32)
            (idxs,) = np.where((L // 2 <= peaks2) & (peaks2 < 3 * L // 2))
            peaks = peaks2[idxs]
            prom, _, _ = peak_prominences(theta2_abs, peaks)
            if len(prom) < self.SidesNum:
                msg = (
                    "Insufficient number of vertices"
                    f" (needs {self.SidesNum}, detected {len(prom)})"
                )
                raise PolySubstrateError(msg)
            prom_peaks = peaks[np.sort(np.argsort(prom)[-self.SidesNum :])]
            vertices = np.sort(prom_peaks) - (L // 2)

            # 3. Compensate index-by-one error, which is probably from np.diff().
            # This error makes perfectly sharp corner incorrectly located by -1.
            self._vertices = np.sort((vertices + 1) % L)
        return self._vertices

    def sides(self) -> Tuple[npt.NDArray[np.int32], ...]:
        """
        Return the points of each side of the polygon.

        Returns
        -------
        tuple_of_points: tuple

        Notes
        -----
        Side can be curved and can have noises. Use :meth:`sidelines` to get the
        linear models for each side.

        The term "side" is used instead of "edge" to avoid confusion from other
        image processing methods (e.g. Canny edge detection).

        See Also
        --------
        sidelines
            Linear model of sides.
        """
        N = self.vertices()[0]
        return tuple(np.split(np.roll(self.contour(), -N, axis=0), self.vertices() - N))

    def sidelines(self) -> npt.NDArray[np.float32]:
        r"""
        Find linear model of polygon sides.

        Sides of the polygon can be curves and can have noises. This method finds
        straight sidelines[1]_ using Hough line transformation.

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

        See Also
        --------
        sideline_intersections
            Return coordinates of intersections of sidelines.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Extended_sides
        """
        if not hasattr(self, "_sidelines"):
            SHIFT = self.vertices()[0]
            vertices = self.vertices() - SHIFT
            cnt_roll = np.roll(self.contour(), -SHIFT, axis=0)

            RHO_RES = self.parameters.Rho
            THETA_RES = self.parameters.Theta
            lines = []
            # Directly use Hough transformation to find lines
            for c in np.split(cnt_roll, vertices[1:], axis=0):
                tan = np.diff(c, axis=0)
                atan = np.arctan2(tan[..., 1], tan[..., 0])  # -pi < atan <= pi
                theta = atan - np.pi / 2
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
                # TODO: try numba to make faster. (iterate over 2D rows, not encode)
                idxs = (rho_digit - rho_min) * len(theta_idxs) + theta_idxs
                val, counts = np.unique(idxs, return_counts=True)
                max_idx = val[np.argmax(counts)]

                r0_idx = max_idx // len(theta_idxs)
                r0 = (r0_idx + rho_min) * RHO_RES
                t0_idx = max_idx % len(theta_idxs)
                t0 = theta_rng[t0_idx]
                lines.append([[r0, t0]])

            self._sidelines = np.array(lines, dtype=np.float32)
        return self._sidelines

    def sideline_intersections(self) -> npt.NDArray[np.float32]:
        """
        Return the coordinates of intersections of polygon sidelines.

        If the polygon has smooth corner, the vertex points are different from
        the corner points on the contour.

        See Also
        --------
        vertices
            Return indices of the corner points on contour.

        """
        ((r1, t1),) = self.sidelines().transpose(1, 2, 0)
        ((r2, t2),) = np.roll(self.sidelines(), 1, axis=0).transpose(1, 2, 0)
        mat = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]]).transpose(
            2, 0, 1
        )
        vec = np.array([[r1], [r2]]).transpose(2, 0, 1)

        (ret,) = (np.linalg.inv(mat) @ vec).transpose(2, 0, 1)
        return ret

    def normal(self):
        """
        Return unit normal vectors on each point of contour.

        Uses polygon model.
        """
        # XXX: Current model assumes sharp polygon.
        # After smooth corner region detection is done, enhance this!
        SHIFT = self.vertices()[0]
        vertices = self.vertices() - SHIFT
        reps = np.diff(np.append(vertices, len(self.contour())))

        _, thetas = self.sidelines().transpose(2, 0, 1)
        n = np.stack([-np.cos(thetas), -np.sin(thetas)]).transpose(1, 2, 0)

        return np.roll(np.repeat(n, reps, axis=0), SHIFT, axis=0)

    def examine(self) -> Optional[PolySubstrateError]:
        try:
            self.vertices()
        except PolySubstrateError as err:
            return err
        return None
