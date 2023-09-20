"""
Polygonal Substrate
===================

:mod:`dipcoatimage.finitedepth.polysubstrate` provides abstract base class for
substrate with polygonal cross section.

"""
import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore
from scipy.ndimage import gaussian_filter1d  # type: ignore
from scipy.signal import find_peaks, peak_prominences  # type: ignore
from typing import TypeVar, Optional, Type, Tuple
from .substrate import SubstrateError, SubstrateBase
from .polysubstrate_param import Parameters
from .util import DataclassProtocol


__all__ = [
    "PolySubstrateError",
    "PolySubstrateBase",
    "houghline_accum",
]


class PolySubstrateError(SubstrateError):
    """Base class for the errors from :class:`PolySubstrate`."""

    pass


ROTATION_MATRIX = np.array([[0, 1], [-1, 0]])


ParametersType = TypeVar("ParametersType", bound=Parameters)
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

    def region_points(self) -> npt.NDArray[np.int32]:
        w = self.image().shape[1]
        return np.array([[w / 2, 0]], dtype=np.int32)

    def contour(self) -> npt.NDArray[np.int32]:
        """Return the polygon contour."""
        (cnt,), _ = self.contours(0)
        return cnt

    def vertices(self) -> npt.NDArray[np.int32]:
        """
        Find the polygon vertices from dense contour.

        Returns
        -------
        ndarray
            Indices of the vertex points in :meth:`contour`.

        Notes
        -----
        A vertex is a point where two or more sides of a polygon meet[1]_.
        The sides can be curves, where the vertices can be defined as local
        extrema of curvature[2]_. This method finds the vertices by locating a
        certain number (defined by d:attr:`SidesNum`) of the extrema.

        Gaussian filter was used to reduce noise from contour[3]_. Sigma value
        is determined from :meth:`parameters`.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Vertex_(geometry)
        .. [2] https://en.wikipedia.org/wiki/Vertex_(curve)
        .. [3] https://stackoverflow.com/q/32629806

        """
        cnt = self.contour().astype(np.float64)

        # 1. Calculate curvatures
        s = self.parameters.Sigma
        f = gaussian_filter1d(cnt, s, axis=0, order=0, mode="wrap")
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
        prom_peaks = peaks[np.argsort(prom)[-self.SidesNum :]]
        return np.sort((prom_peaks - (L // 2)) % L)

    def sides(self) -> Tuple[npt.NDArray[np.int32], ...]:
        """
        Return the points of each side of the polygon.

        Returns
        -------
        tuple_of_points: tuple

        Notes
        -----
        Sides are sorted so that the one containing the first point of the
        contour comes first.

        Side can be curved and can have noises. Use :meth:`sidelines` to get the
        linear models for each side.

        The term "side" is used instead of "edge" to avoid confusion from other
        image processing methods (e.g. Canny edge detection).

        See Also
        --------
        sidelines
            Linear model of sides.
        """
        cnt = self.contour()
        vert = self.vertices()
        sides = np.split(cnt, vert)
        dists = np.linalg.norm(cnt[vert] - cnt[0], axis=-1)
        shift = vert[np.argmin(dists)]
        L = len(cnt)
        sides = np.split(np.roll(cnt, shift, axis=0), np.sort((vert - shift) % L))[1:]
        return tuple(sides)

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
        .. [1] https://en.wikipedia.org/wiki/Extended_side
        """
        if not hasattr(self, "_sidelines"):
            # Do not find the line from smoothed contour. Noise is removed anyway
            # without smoothing by Hough transformation. In fact, smoothing
            # propagates the outlier error to nearby data.
            RHO_RES = self.parameters.Rho
            THETA_RES = self.parameters.Theta
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
                STEP = 10  # Sample points for performance. TODO: use parameter
                c = side[::STEP]
                rho = c[..., 0] * np.cos(theta_rng) + c[..., 1] * np.sin(theta_rng)
                rho_digit = (rho / RHO_RES).astype(np.int32)

                _, (rho, theta_idx) = houghline_accum(rho_digit)
                lines.append([[rho * RHO_RES, theta_rng[theta_idx]]])

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

    def examine(self) -> Optional[PolySubstrateError]:
        try:
            self.vertices()
        except PolySubstrateError as err:
            return err
        return None


@njit(cache=True)
def houghline_accum(
    rho_array: npt.NDArray[np.int32],
) -> Tuple[npt.NDArray[np.int32], Tuple[float, int]]:
    """
    Performs hough line accumulation.

    Parameters
    ----------
    rho_array: ndarray
        Array which contains rho and theta values for every points.
        The shape must be `(P, T)`, where `P` is the number of points and `T` is
        the length of digitized theta ranges.
        If an element at index `(p, t)` has value `r`, it indicates that a line:
        * Passing `p`-th point
        * Angle is `t`-th element in digitized theta range.
        * Distance from the origin is `r`.

    Returns
    -------
    accum: ndarray
        Accumulation matrix.
    rho_theta: tuple
        `(rho, theta_idx)` value for the detected line.

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
