"""Analyze polygonal substrate.

This module defines :class:`PolySubstrateBase`, which is an abstract base class
for substrate with polygonal shape.
"""
import dataclasses
from typing import Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore
from numpy.linalg import LinAlgError
from scipy.ndimage import gaussian_filter1d  # type: ignore
from scipy.signal import find_peaks, peak_prominences  # type: ignore

from .cache import attrcache
from .substrate import DataTypeVar, DrawOptTypeVar, RefTypeVar, SubstrateBase

__all__ = [
    "PolySubstParam",
    "ParamTypeVar",
    "PolySubstrateBase",
    "houghline_accum",
]


@dataclasses.dataclass(frozen=True)
class PolySubstParam:
    """Analysis parameters for :class:`PolySubstrate`.

    Arguments:
        Sigma: Standard deviation of Gaussian kernel to smooth the noise.
        Rho: Radian resolution for Hough transformation to detect the sidelines.
        Theta: Angle resolution for Hough transformation to detect the sidelines.
        Step: Sampling rate of points for Hough transformation.
            Larger step makes evaluation faster.
    """

    Sigma: float
    Rho: float
    Theta: float
    Step: int = 1


ROTATION_MATRIX = np.array([[0, 1], [-1, 0]])


ParamTypeVar = TypeVar("ParamTypeVar", bound=PolySubstParam)
"""Type variable for :attr:`PolySubstrate.ParamType`."""


class PolySubstrateBase(
    SubstrateBase[RefTypeVar, ParamTypeVar, DrawOptTypeVar, DataTypeVar]
):
    """Abstract base class for substrate whose cross section is a simple polygon.

    A simple polygon does not have intersection nor hole [#simple-polygon]_.
    Smooth corners are allowed.

    The following substrate images are not supported:

    * Multiple substrates in one image
    * Multiple contours (e.g. substrate with holes)

    Concrete subclass must assign the following class attributes:

    * :attr:`SidesNum`: Number of sides of the polygon.
        For example, this should be ``4`` if cross section of the substrate is
        tetragon.

    Arguments:
        image
        templateROI
        substrateROI
        parameters (PolySubstParam, optional)
        draw_options

    References:
        .. [#simple-polygon] https://en.wikipedia.org/wiki/Simple_polygon
    """

    SidesNum: int
    """Number of sides of polygon.

    This class attribute is defined but not set in :class:`PolySubstrateBase`.
    Concrete subclass must assign this attribute with integer.
    """

    def region_points(self) -> npt.NDArray[np.int32]:
        """Implement :meth:`SubstrateBase.region_points`.

        This method returns an upper center point of the substrate image. Substrate ROI
        in reference image must be selected so that this point falls into substrate
        region.
        """
        w = self.image().shape[1]
        return np.array([[w / 2, 0]], dtype=np.int32)

    def contour(self) -> npt.NDArray[np.int32]:
        """Return the polygon contour."""
        (cnt,), _ = self.contours(0)
        return cnt

    def vertices(self) -> npt.NDArray[np.int32]:
        """Find vertices of the polygon from its contour.

        A vertex is a point where two sides of a polygon meet[#vertex-geom]_.
        The sides can be curves, where the vertices can be defined as local
        extrema of curvature [#vertex-curve]_.

        Returns:
            Indices of the vertex points in :meth:`contour`.
            The number of the vertices is defined by :attr:`SidesNum`.

        Note:
            Contour is Gaussian-filtered to reduce noise[#so-curvature]_.
            Sigma value of the kernel is determined from :attr:`parameters`.

        References:
            .. [#vertex-geom] https://en.wikipedia.org/wiki/Vertex_(geometry)
            .. [#vertex-curve] https://en.wikipedia.org/wiki/Vertex_(curve)
            .. [#so-curvature] https://stackoverflow.com/q/32629806
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
        """Find each side of the polygon.

        Side can be curved and can have noises. Use :meth:`sidelines` to get the
        linear models for each side.

        Returns:
            Tuple of array. Each array contains points on each side of the
            polygon contour.

        Note:
            Result is sorted so that the side containing the first point of the
            contour comes first.

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
        r"""Find linear model of polygon sides.

        This method finds straight sidelines [#extended-side]_ using
        Hough line transformation. Radian and angle resolutions are determined
        by :attr:`parameters`.

        Returns:
            Vector of line parameters in :math:`(\rho, \theta)`.
            :math:`\rho` is the distance from the coordinate origin.
            :math:`\theta` is the angle of normal vector from the origin to the line.

        Note:
            Range of angle is
            :math:`\theta \in (-\frac{3 \pi}{2}, \frac{\pi}{2}]`.
            Arctangent direction can be acquired by
            :math:`\theta + \frac{\pi}{2}`.

        References:
            .. [#extended-side] https://en.wikipedia.org/wiki/Extended_side
        """
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
            c = side[:: self.parameters.Step]
            rho = c[..., 0] * np.cos(theta_rng) + c[..., 1] * np.sin(theta_rng)
            rho_digit = (rho / RHO_RES).astype(np.int32)

            _, (rho, theta_idx) = houghline_accum(rho_digit)
            lines.append([[rho * RHO_RES, theta_rng[theta_idx]]])

        return np.array(lines, dtype=np.float32)

    def sideline_intersections(self) -> npt.NDArray[np.float32]:
        """Return the intersections of :meth:`sidelines`.

        Note:
            If the polygon has smooth corner, the vertex points are different
            from the corner points on the contour.
        """
        ((r1, t1),) = self.sidelines().transpose(1, 2, 0)
        ((r2, t2),) = np.roll(self.sidelines(), 1, axis=0).transpose(1, 2, 0)
        mat = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]]).transpose(
            2, 0, 1
        )
        vec = np.array([[r1], [r2]]).transpose(2, 0, 1)
        (ret,) = (np.linalg.inv(mat) @ vec).transpose(2, 0, 1)
        return ret

    def verify(self):
        """Implement :meth:`SubstrateBase.verify`.

        Check if :meth:`sideline_intersections` returns without error.
        """
        try:
            self.sideline_intersections()
        except LinAlgError:
            raise ValueError("Cannot find sideline intersections.")


@njit(cache=True)
def houghline_accum(
    rho_array: npt.NDArray[np.int32],
) -> Tuple[npt.NDArray[np.int32], Tuple[float, int]]:
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
