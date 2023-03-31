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
from typing import TypeVar, Tuple, Optional, Type
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
    Theta: float = 0.01


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
        "_sides",
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

    def sides(self) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Return the contour indices and polar parameters for linear sides."""
        if hasattr(self, "_sides"):
            return self._sides  # type: ignore[has-type]

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
        # Since we repeated theta, we select the peaks in desired region.
        theta2_grad = np.gradient(theta2_smooth, axis=0)
        theta2_abs = np.abs(theta2_grad)[..., 0]
        peaks2, _ = find_peaks(theta2_abs)
        (idxs,) = np.where((len(theta) // 2 <= peaks2) & (peaks2 < 3 * len(theta) // 2))
        peaks = peaks2[idxs]
        prom, _, _ = peak_prominences(theta2_abs, peaks)
        if len(prom) < self.SidesNum:
            msg = (
                "Insufficient number of sides"
                f" (needs {self.SidesNum}, detected {len(prom)})"
            )
            raise PolySubstrateError(msg)
        prom_peaks = peaks[np.sort(np.argsort(prom)[-self.SidesNum :])]
        corner_peaks = np.sort(prom_peaks) - (len(theta) // 2)

        # 3. Digitize smoothed line and get votes to determine main theta.
        # We must take vote from disgitized smoothed line, not from raw theta
        # in order to be robust from jittery noises.
        # We smooth each line separately to minimize the effect on shap corner.
        SHIFT = corner_peaks[0]
        # roll s.t. no residual section at the beginning
        theta_roll = np.roll(theta, -SHIFT, axis=0)
        corner_peaks = corner_peaks - SHIFT

        THETA_STEP = self.parameters.Theta
        indices = []
        thetas = []
        for region in np.split(theta_roll, corner_peaks[1:], axis=0):
            region_smooth = gaussian_filter1d(region, sigma, axis=0)
            digitized = (region_smooth / THETA_STEP).astype(int) * THETA_STEP
            val, count = np.unique(digitized, return_counts=True)
            main_theta = val[np.argmax(count)]
            idxs, _ = np.nonzero(digitized == main_theta)
            indices.append([idxs[0], idxs[-1]])
            thetas.append(main_theta)

        base_indices = SHIFT + corner_peaks[..., np.newaxis]
        split_indices = (base_indices + np.array(indices, dtype=np.int64)) % len(theta)
        sortidx = np.argsort(split_indices, axis=0)
        split_indices = split_indices[sortidx[..., 0]]

        # convert slope theta to polar angle (just as HoughLines parameter)
        thetas_array = (np.array(thetas) - np.pi / 2) % np.pi
        angles = thetas_array[sortidx[..., 0]][..., np.newaxis]
        lines = [contour[i0:i1] for (i0, i1) in split_indices]
        line_centers = np.array([np.mean(line, axis=0) for line in lines])

        x_cos = line_centers[..., 0] * np.cos(angles)
        y_sin = line_centers[..., 1] * np.sin(angles)
        r = x_cos + y_sin
        line_params = np.stack([r, angles]).transpose(1, 2, 0).astype(np.float32)

        self._sides = split_indices, line_params
        return self._sides

    def vertex_points(self):
        if hasattr(self, "_vertex_points"):
            return self._vertex_points  # type: ignore[has-type]

        _, sides = self.sides()
        ((r1, t1),) = sides.transpose(1, 2, 0)
        ((r2, t2),) = np.roll(sides, 1, axis=0).transpose(1, 2, 0)
        mat = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]]).transpose(
            2, 0, 1
        )
        vec = np.array([[r1], [r2]]).transpose(2, 0, 1)
        sol_exists = np.linalg.det(mat) != 0

        (self._vertex_points,) = (
            np.linalg.inv(mat[np.where(sol_exists)]) @ vec[np.where(sol_exists)]
        ).transpose(2, 0, 1)
        return self._vertex_points

    def examine(self) -> Optional[PolySubstrateError]:
        try:
            self.sides()
        except PolySubstrateError as err:
            return err
        return None
