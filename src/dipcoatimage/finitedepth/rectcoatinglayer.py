"""Coating layer over rectangular substrate."""

import dataclasses
import enum
from typing import TYPE_CHECKING, Tuple, Type, TypeVar

import cv2
import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore
from scipy.optimize import root  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore

from .cache import attrcache
from .coatinglayer import (
    CoatingLayerBase,
    CoatingLayerError,
    SubtractionMode,
    images_XOR,
)
from .parameters import LineOptions, PatchOptions
from .rectsubstrate import RectSubstrate

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "RectCoatingLayerBase",
    "RectLayerShape",
    "equidistant_interpolate",
    "polyline_parallel_area",
    "acm",
    "owp",
]


ParametersType = TypeVar("ParametersType", bound="DataclassInstance")
DrawOptionsType = TypeVar("DrawOptionsType", bound="DataclassInstance")
DecoOptionsType = TypeVar("DecoOptionsType", bound="DataclassInstance")
DataType = TypeVar("DataType", bound="DataclassInstance")


ROTATION_MATRIX = np.array([[0, 1], [-1, 0]])


class RectCoatingLayerBase(
    CoatingLayerBase[
        RectSubstrate, ParametersType, DrawOptionsType, DecoOptionsType, DataType
    ]
):
    """Abstract base class for coating layer over rectangular substrate."""

    __slots__ = (
        "_interfaces",
        "_contour",
        "_surface_indices",
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    DecoOptions: Type[DecoOptionsType]
    Data: Type[DataType]

    @attrcache("_layer_contours")
    def layer_contours(self) -> Tuple[npt.NDArray[np.int32], ...]:
        """Return contours of :meth:`extract_layer`."""
        layer_cnts, _ = cv2.findContours(
            self.extract_layer().astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        return tuple(layer_cnts)

    @attrcache("_interfaces")
    def interfaces(self) -> Tuple[npt.NDArray[np.int64], ...]:
        """Find indices of solid-liquid interfaces on :meth:`SubstrateBase.contour`.

        Returns
        -------
        tuple
            Tuple of arrays.
            - Each array represents layer contours.
            - Array contains indices of the interface intervals of layer contour.

        Notes
        -----
        A substrate can be covered by multiple blobs of coating layer, and a
        single blob can make multiple contacts to a substrate. Each array in a
        tuple represents each blob. The shape of the array is `(N, 2)`, where
        `N` is the number of interface intervals.

        Each interval describes continuous patch on the substrate contour covered
        by the layer. To acquire the interface points, slice the substrate
        contour with the indices.
        """
        subst_cnt = self.substrate.contour() + self.substrate_point()
        ret = []
        for layer_cnt in self.layer_contours():
            H, W = self.image.shape[:2]
            lcnt_img = np.zeros((H, W), dtype=np.uint8)
            lcnt_img[layer_cnt[..., 1], layer_cnt[..., 0]] = 255
            dilated_lcnt = cv2.dilate(lcnt_img, np.ones((3, 3))).astype(bool)

            x, y = subst_cnt.transpose(2, 0, 1)
            mask = dilated_lcnt[np.clip(y, 0, H - 1), np.clip(x, 0, W - 1)]

            # Find indices of continuous True blocks
            idxs = np.where(
                np.diff(np.concatenate(([False], mask[:, 0], [False]))) == 1
            )[0].reshape(-1, 2)
            ret.append(idxs)
        return tuple(ret)

    @attrcache("_contour")
    def contour(self) -> npt.NDArray[np.int32]:
        """Contour of the entire coated substrate."""
        (cnt,), _ = cv2.findContours(
            self.coated_substrate().astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        return cnt

    def surface(self) -> npt.NDArray[np.int32]:
        """Return the surface of the entire coated region.

        Returns
        -------
        ndarray
            Points in :meth:`contour` which comprises the coating layer surface.
            Surface is continuous, i.e., if multiple discrete blobs of layer
            exist, the surface includes the points on the exposed substrate
            between them.

        See Also
        --------
        contour
        """
        if not self.interfaces():
            return np.empty((0, 1, 2), np.int32)

        if not hasattr(self, "_surface_indices"):
            (i0, i1) = np.sort(np.concatenate(self.interfaces()).flatten())[[0, -1]]
            subst_cnt = self.substrate.contour() + self.substrate_point()
            endpoints = subst_cnt[[i0, i1]]

            vec = self.contour() - endpoints.transpose(1, 0, 2)
            self._surface_indices = np.argmin(np.linalg.norm(vec, axis=-1), axis=0)
        (I0, I1) = self._surface_indices
        return self.contour()[I0 : I1 + 1]

    def capbridge_broken(self) -> bool:
        """Check if capillary bridge is ruptured."""
        p0 = self.substrate_point()
        _, bl, br, _ = self.substrate.contour()[self.substrate.vertices()]
        (B,) = p0 + bl
        (C,) = p0 + br
        top = np.max([B[1], C[1]])
        bot = self.image.shape[0]
        if top > bot:
            # substrate is located outside of the frame
            return False
        left = B[0]
        right = C[0]
        roi_binimg = self.image[top:bot, left:right]
        return bool(np.any(np.all(roi_binimg, axis=1)))


class DistanceMeasure(enum.Enum):
    """Distance measures to compute the curve similarity.

    Members
    -------
    DTW
        Dynamic time warping.
    SDTW
        Squared dynamic time warping.
    """

    DTW = "DTW"
    SDTW = "SDTW"


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Analysis parameters for `RectLayerShape` instance.

    Attributes
    ----------
    KernelSize : tuple
        Size of the kernel for morphological operation to remove noises.
    ReconstructRadius : int
        Connected components outside of this radius from bottom corners of the
        substrate are regarded as image artifacts.
    RoughnessMeasure : DistanceMeasure
        Measure to compute layer roughness.
    """

    KernelSize: Tuple[int, int]
    ReconstructRadius: int
    RoughnessMeasure: DistanceMeasure


class PaintMode(enum.Enum):
    """Option to determine how the coating layer image is painted.

    Members
    -------
    ORIGINAL
        Show the original image.
    EMPTY
        Show empty image. Only the layer will be drawn.
    """

    ORIGINAL = "ORIGINAL"
    EMPTY = "EMPTY"


@dataclasses.dataclass
class DrawOptions:
    """Drawing options for `RectLayerShape` instance.

    Attributes
    ----------
    paint : PaintMode
    subtraction : SubtractionMode
    """

    paint: PaintMode = PaintMode.ORIGINAL
    subtraction: SubtractionMode = SubtractionMode.NONE


@dataclasses.dataclass
class LinesOptions:
    """Parameters to draw lines in the image.

    Attributes
    ----------
    color : tuple
        Color of the lines in RGB
    linewidth : int
        Width of the line.
        Zero value is the flag to not draw the line.
    step : int
        Steps to jump the lines. `1` draws every line.
    """

    color: Tuple[int, int, int] = (0, 0, 0)
    linewidth: int = 1
    step: int = 1


@dataclasses.dataclass
class DecoOptions:
    """Options to show the analysis result on `RectLayerShape`.

    Attributes
    ----------
    layer : PatchOptions
    contact_line, thickness, uniform_layer : LineOptions
    conformality, roughness : LinesOptions
    """

    layer: PatchOptions = dataclasses.field(
        default_factory=lambda: PatchOptions(
            fill=True,
            edgecolor=(0, 0, 255),
            facecolor=(255, 255, 255),
            linewidth=1,
        )
    )
    contact_line: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(0, 0, 255), linewidth=1)
    )
    thickness: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(0, 0, 255), linewidth=1)
    )
    uniform_layer: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(255, 0, 0), linewidth=1)
    )
    conformality: LinesOptions = dataclasses.field(
        default_factory=lambda: LinesOptions(color=(0, 255, 0), linewidth=1, step=10)
    )
    roughness: LinesOptions = dataclasses.field(
        default_factory=lambda: LinesOptions(color=(255, 0, 0), linewidth=1, step=10)
    )


@dataclasses.dataclass
class Data:
    """Analysis data for `RectLayerShape` instance.

    - LayerLength_{Left, Right}: Distance between the bottom sideline of the
      substrate and the upper limit of the coating layer.
    - Conformality: Conformality of the coating layer.
    - AverageThickness: Average thickness of the coating layer.
    - Roughness: Roughness of the coating layer.
    - MaxThickness_{Left, Bottom, Right}: Number of the pixels for the maximum
      thickness on each region.

    The following data are the metadata for the analysis.

    - MatchError: Template matching error between 0 to 1. 0 means perfect match.
    """

    LayerLength_Left: np.float64
    LayerLength_Right: np.float64

    Conformality: float
    AverageThickness: np.float64
    Roughness: float

    MaxThickness_Left: np.float64
    MaxThickness_Bottom: np.float64
    MaxThickness_Right: np.float64

    MatchError: float


class RectLayerShape(
    RectCoatingLayerBase[
        Parameters,
        DrawOptions,
        DecoOptions,
        Data,
    ]
):
    """Coating layer over rectangular substrate.

    Examples
    --------
    Construct substrate reference class first.

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import Reference, get_data_path
       >>> gray = cv2.imread(get_data_path("ref3.png"), cv2.IMREAD_GRAYSCALE)
       >>> _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
       >>> tempROI = (13, 10, 1246, 200)
       >>> substROI = (100, 100, 1200, 500)
       >>> ref = Reference(img, tempROI, substROI)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    Construct the parameters and substrate instance from reference instance.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import RectSubstrate
       >>> param = RectSubstrate.Parameters(Sigma=3.0, Rho=1.0, Theta=0.01)
       >>> subst = RectSubstrate(ref, param)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    Construct `RectLayerShape` from substrate class. :meth:`analyze`
    returns the number of pixels in coating area region.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import RectLayerShape
       >>> gray = cv2.imread(get_data_path("coat3.png"), cv2.IMREAD_GRAYSCALE)
       >>> _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
       >>> param = RectLayerShape.Parameters(
       ...     KernelSize=(1, 1),
       ...     ReconstructRadius=50,
       ...     RoughnessMeasure=RectLayerShape.DistanceMeasure.SDTW,
       ... )
       >>> coat = RectLayerShape(img, subst, param)
       >>> plt.imshow(coat.draw()) #doctest: +SKIP
    """

    __slots__ = (
        "_uniform_layer",
        "_conformality",
        "_roughness",
        "_max_thickness",
    )

    Parameters = Parameters
    DrawOptions = DrawOptions
    DecoOptions = DecoOptions
    Data = Data

    DistanceMeasure = DistanceMeasure
    PaintMode = PaintMode
    SubtractionMode = SubtractionMode

    def verify(self):
        """Check error."""
        ksize = self.parameters.KernelSize
        if not all(i == 0 or i % 2 == 1 for i in ksize):
            raise CoatingLayerError("Kernel size must be odd.")
        if not self.capbridge_broken():
            raise CoatingLayerError("Capillary bridge is not broken.")

    @attrcache("_extracted_layer")
    def extract_layer(self) -> npt.NDArray[np.bool_]:
        """Extract coating layer region."""
        # Perform opening to remove error pixels. We named the parameter as
        # "closing" because the coating layer is black in original image, but
        # in fact we do opening since the layer is True in extracted layer.
        ksize = self.parameters.KernelSize
        if any(i == 0 for i in ksize):
            img = super().extract_layer().astype(np.uint8) * 255
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
            img = cv2.morphologyEx(
                super().extract_layer().astype(np.uint8) * 255,
                cv2.MORPH_OPEN,
                kernel,
            )

        # closed image may still have error pixels, and at least we have to
        # remove the errors that are disconnected to the layer.
        # we identify the layer pixels as the connected components that are
        # close to the lower vertices.
        vicinity_mask = np.zeros(img.shape, np.uint8)
        p0 = self.substrate_point()
        _, bl, br, _ = self.substrate.contour()[self.substrate.vertices()]
        (B,) = p0 + bl
        (C,) = p0 + br
        R = self.parameters.ReconstructRadius
        cv2.circle(vicinity_mask, B.astype(np.int32), R, 1, -1)
        cv2.circle(vicinity_mask, C.astype(np.int32), R, 1, -1)
        n = np.dot((C - B) / np.linalg.norm((C - B)), ROTATION_MATRIX)
        pts = np.stack([B, B + R * n, C + R * n, C]).astype(np.int32)
        cv2.fillPoly(vicinity_mask, [pts], 1)
        _, labels = cv2.connectedComponents(img)
        layer_comps = np.unique(labels[np.where(vicinity_mask.astype(bool))])
        layer_mask = np.isin(labels, layer_comps[layer_comps != 0])

        return layer_mask

    @attrcache("_uniform_layer")
    def uniform_layer(self) -> Tuple[np.float64, npt.NDArray[np.float64]]:
        """Return thickness and points for uniform layer."""
        if not self.interfaces():
            return (np.float64(0), np.empty((0, 1, 2), np.float64))

        (i0, i1) = np.sort(np.concatenate(self.interfaces()).flatten())[[0, -1]]
        subst_cnt = self.substrate.contour() + self.substrate_point()
        covered_subst = subst_cnt[i0:i1]
        # Acquiring parallel curve from contour is difficult because of noise.
        # Noise induces small bumps which are greatly amplified in parallel curve.
        # Smoothing is not an answer since it cannot 100% remove the bumps.
        # Instead, points must be fitted to a model.
        # Here, we simply use convex hull as a model.
        covered_hull = np.flip(cv2.convexHull(covered_subst), axis=0)

        S = np.count_nonzero(self.extract_layer())
        (t,) = root(lambda x: polyline_parallel_area(covered_hull, x) - S, 0).x
        t = np.float64(t)

        normal = np.dot(np.gradient(covered_hull, axis=0), ROTATION_MATRIX)
        n = normal / np.linalg.norm(normal, axis=-1)[..., np.newaxis]
        ul_sparse = covered_hull + t * n

        ul_len = np.ceil(cv2.arcLength(ul_sparse.astype(np.float32), closed=False))
        ul = equidistant_interpolate(ul_sparse, int(ul_len))

        return (t, ul)

    @attrcache("_conformality")
    def conformality(self) -> Tuple[float, npt.NDArray[np.int32]]:
        """Conformality of the coating layer and its optimal path."""
        if not self.interfaces():
            return (np.nan, np.empty((0, 2), dtype=np.int32))

        (i0, i1) = np.sort(np.concatenate(self.interfaces()).flatten())[[0, -1]]
        subst_cnt = self.substrate.contour() + self.substrate_point()
        intf = subst_cnt[i0:i1]

        surf = self.surface()

        dist = cdist(np.squeeze(surf, axis=1), np.squeeze(intf, axis=1))
        mat = acm(dist)
        path = owp(mat)
        d = dist[path[:, 0], path[:, 1]]
        d_avrg = mat[-1, -1] / len(path)
        C = 1 - np.sum(np.abs(d - d_avrg)) / mat[-1, -1]
        return (float(C), path)

    @attrcache("_roughness")
    def roughness(self) -> Tuple[float, npt.NDArray[np.int32]]:
        """Roughness of the coating layer and its optimal path."""
        surf = self.surface()
        _, ul = self.uniform_layer()

        if surf.size == 0 or ul.size == 0:
            return (np.nan, np.empty((0, 2), dtype=np.int32))

        measure = self.parameters.RoughnessMeasure
        if measure == DistanceMeasure.DTW:
            dist = cdist(np.squeeze(surf, axis=1), np.squeeze(ul, axis=1))
            mat = acm(dist)
            path = owp(mat)
            roughness = mat[-1, -1] / len(path)
        elif measure == DistanceMeasure.SDTW:
            dist = cdist(np.squeeze(surf, axis=1), np.squeeze(ul, axis=1))
            mat = acm(dist**2)
            path = owp(mat)
            roughness = np.sqrt(mat[-1, -1] / len(path))
        else:
            raise TypeError(f"Unknown measure: {measure}")
        return (float(roughness), path)

    @attrcache("_max_thickness")
    def max_thickness(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Maximum thickness on each side (left, bottom, right) and their points."""
        corners = self.substrate.sideline_intersections() + self.substrate_point()
        surface = self.surface()
        thicknesses, points = [], []
        for A, B in zip(corners[:-1], corners[1:]):
            AB = B - A
            mask = np.cross(AB, surface - A) >= 0
            pts = surface[mask]
            if pts.size == 0:
                thicknesses.append(np.float64(0))
                points.append(np.array([[-1, -1], [-1, -1]], np.float64))
            else:
                Ap = pts - A
                Proj = A + AB * (np.dot(Ap, AB) / np.dot(AB, AB))[..., np.newaxis]
                dist = np.linalg.norm(Proj - pts, axis=-1)
                max_idx = np.argmax(dist)
                thicknesses.append(dist[max_idx])
                points.append(np.stack([pts[max_idx], Proj[max_idx]]))
        return (np.array(thicknesses), np.array(points))

    def draw(self) -> npt.NDArray[np.uint8]:
        """Return visualized image."""
        paint = self.draw_options.paint
        if paint == self.PaintMode.ORIGINAL:
            image = self.image
        elif paint == self.PaintMode.EMPTY:
            image = np.full(self.image.shape, 255, dtype=np.uint8)
        else:
            raise TypeError("Unrecognized paint mode: %s" % paint)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        subtraction = self.draw_options.subtraction
        if subtraction in [
            self.SubtractionMode.TEMPLATE,
            self.SubtractionMode.FULL,
        ]:
            x0, y0, x1, y1 = self.substrate.reference.templateROI
            tempImg = self.substrate.reference.image[y0:y1, x0:x1]
            h, w = tempImg.shape[:2]
            (X0, Y0), _ = self.tempmatch
            binImg = self.image[Y0 : Y0 + h, X0 : X0 + w]
            mask = images_XOR(~binImg.astype(bool), ~tempImg.astype(bool))
            image[Y0 : Y0 + h, X0 : X0 + w][~mask] = 255
        if subtraction in [
            self.SubtractionMode.SUBSTRATE,
            self.SubtractionMode.FULL,
        ]:
            x0, y0, x1, y1 = self.substrate.reference.substrateROI
            substImg = self.substrate.reference.image[y0:y1, x0:x1]
            h, w = substImg.shape[:2]
            X0, Y0 = self.substrate_point()
            binImg = self.image[Y0 : Y0 + h, X0 : X0 + w]
            mask = images_XOR(~binImg.astype(bool), ~substImg.astype(bool))
            image[Y0 : Y0 + h, X0 : X0 + w][~mask] = 255

        layer_opts = self.deco_options.layer
        if layer_opts.fill:
            image[self.extract_layer()] = layer_opts.facecolor
        if layer_opts.linewidth > 0:
            cv2.drawContours(
                image,
                self.layer_contours(),
                -1,
                layer_opts.edgecolor,
                layer_opts.linewidth,
            )

        contactline_opts = self.deco_options.contact_line
        if len(self.interfaces()) > 0 and contactline_opts.linewidth > 0:
            (i0, i1) = np.sort(np.concatenate(self.interfaces()).flatten())[[0, -1]]
            subst_cnt = self.substrate.contour() + self.substrate_point()
            (p0,), (p1,) = subst_cnt[[i0, i1]].astype(np.int32)
            cv2.line(
                image,
                p0,
                p1,
                contactline_opts.color,
                contactline_opts.linewidth,
            )

        if not self.capbridge_broken():
            return image

        thickness_opts = self.deco_options.thickness
        if thickness_opts.linewidth > 0:
            lines = []
            for dist, pts in zip(*self.max_thickness()):
                if dist == 0:
                    continue
                lines.append(pts.astype(np.int32))
            cv2.polylines(
                image,
                lines,
                isClosed=False,
                color=thickness_opts.color,
                thickness=thickness_opts.linewidth,
            )

        uniformlayer_opts = self.deco_options.uniform_layer
        if uniformlayer_opts.linewidth > 0:
            _, points = self.uniform_layer()
            cv2.polylines(
                image,
                [points.astype(np.int32)],
                isClosed=False,
                color=uniformlayer_opts.color,
                thickness=uniformlayer_opts.linewidth,
            )

        conformality_opts = self.deco_options.conformality
        if len(self.interfaces()) > 0 and conformality_opts.linewidth > 0:
            surf = self.surface()
            (i0, i1) = np.sort(np.concatenate(self.interfaces()).flatten())[[0, -1]]
            intf = (self.substrate.contour() + self.substrate_point())[i0:i1]
            _, path = self.conformality()
            path = path[:: conformality_opts.step]
            lines = np.concatenate([surf[path[..., 0]], intf[path[..., 1]]], axis=1)
            cv2.polylines(
                image,
                lines,
                isClosed=False,
                color=conformality_opts.color,
                thickness=conformality_opts.linewidth,
            )

        roughness_opts = self.deco_options.roughness
        if len(self.interfaces()) > 0 and roughness_opts.linewidth > 0:
            surf = self.surface()
            _, ul = self.uniform_layer()
            _, path = self.roughness()
            path = path[:: roughness_opts.step]
            lines = np.concatenate(
                [surf[path[..., 0]], ul[path[..., 1]]], axis=1
            ).astype(np.int32)
            cv2.polylines(
                image,
                lines,
                isClosed=False,
                color=roughness_opts.color,
                thickness=roughness_opts.linewidth,
            )

        return image

    def analyze(self):
        """Return analysis data."""
        _, B, C, _ = self.substrate.sideline_intersections() + self.substrate_point()

        if not self.interfaces():
            LEN_L = LEN_R = np.float64(0)
        else:
            (i0, i1) = np.sort(np.concatenate(self.interfaces()).flatten())[[0, -1]]
            subst_cnt = self.substrate.contour() + self.substrate_point()
            pts = subst_cnt[[i0, i1]]

            Bp = pts - B
            BC = C - B
            Proj = B + BC * (np.dot(Bp, BC) / np.dot(BC, BC))[..., np.newaxis]
            dists = np.linalg.norm(Proj - pts, axis=-1)
            (LEN_L,), (LEN_R,) = dists.astype(np.float64)

        C, _ = self.conformality()
        AVRGTHCK, _ = self.uniform_layer()
        ROUGH, _ = self.roughness()
        (THCK_L, THCK_B, THCK_R), _ = self.max_thickness()

        _, ERR = self.tempmatch

        return self.Data(
            LEN_L,
            LEN_R,
            C,
            AVRGTHCK,
            ROUGH,
            THCK_L,
            THCK_B,
            THCK_R,
            ERR,
        )


def equidistant_interpolate(points, n) -> npt.NDArray[np.float64]:
    """Interpolate *points* with *n* number of points with same distances.

    Parameters
    ----------
    points: ndarray
        Points that are interpolated.
        The shape must be `(N, 1, D)` where `N` is the number of points and `D`
        is the dimension.
    n: int
        Number of new points.

    Returns
    -------
    ndarray
        Interpolated points with same distances.
        If `N` is positive number, the shape is `(n, 1, D)`. If `N` is zero,
        the shape is `(n, 0, D)`.
    """
    # https://stackoverflow.com/a/19122075
    if points.size == 0:
        return np.empty((n, 0, points.shape[-1]), dtype=np.float64)
    vec = np.diff(points, axis=0)
    dist = np.linalg.norm(vec, axis=-1)
    u = np.insert(np.cumsum(dist), 0, 0)
    t = np.linspace(0, u[-1], n)
    ret = np.column_stack([np.interp(t, u, a) for a in np.squeeze(points, axis=1).T])
    return ret.reshape((n,) + points.shape[1:])


def polyline_parallel_area(line: npt.NDArray, t: float) -> np.float64:
    """Calculate the area formed by convex polyline [1]_ and its parallel curve [2]_.

    Parameters
    ----------
    line : ndarray
        Vertices of a polyline.
        The first dimension must be the number of vertices and the last dimension
        must be the dimension of the manifold.
    t : float
        Thickness between *line* and its parallel curve.

    Returns
    -------
    area : float

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Polygonal_chain
    .. [2] https://en.wikipedia.org/wiki/Parallel_curve
    """
    vec = np.diff(line, axis=0)
    d_l = np.linalg.norm(vec, axis=-1)
    d_theta = np.abs(np.diff(np.arctan2(vec[..., 1], vec[..., 0])))
    return np.float64(np.sum(d_l) * t + np.sum(d_theta) * (t**2) / 2)


@njit(cache=True)
def acm(cm: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute accumulated cost matrix from local cost matrix.

    Implements the algorithm from [1]_, with modification from [2]_.

    Parameters
    ----------
    cm: ndarray
        Local cost matrix.

    Returns
    -------
    ndarray
        Accumulated cost matrix.
        The element at `[-1, -1]` is the total sum along the optimal path.
        If *P* or *Q* is empty, return value is an empty array.

    References
    ----------
    .. [1] Senin, Pavel. "Dynamic time warping algorithm review." Information and
       Computer Science Department University of Hawaii at Manoa Honolulu,
       USA 855.1-23 (2008): 40.

    .. [2] https://pypi.org/project/similaritymeasures/

    See Also
    --------
    owp : Compute optimal warping path from the accumulated cost matrix.
    """
    p, q = cm.shape
    ret = np.zeros((p, q), dtype=np.float64)
    if p == 0 or q == 0:
        return ret

    ret[0, 0] = cm[0, 0]

    for i in range(1, p):
        ret[i, 0] = ret[i - 1, 0] + cm[i, 0]

    for j in range(1, q):
        ret[0, j] = ret[0, j - 1] + cm[0, j]

    for i in range(1, p):
        for j in range(1, q):
            ret[i, j] = min(ret[i - 1, j], ret[i, j - 1], ret[i - 1, j - 1]) + cm[i, j]

    return ret


@njit(cache=True)
def owp(acm: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    """Compute optimal warping path from accumulated cost matrix.

    Implements the algorithm from [1]_, with modification from [2]_.

    Parameters
    ----------
    acm: ndarray
        Accumulated cost matrix.

    Returns
    -------
    ndarray
        Indices for the two series to get the optimal warping path.

    References
    ----------
    .. [1] Senin, Pavel. "Dynamic time warping algorithm review." Information and
       Computer Science Department University of Hawaii at Manoa Honolulu,
       USA 855.1-23 (2008): 40.

    .. [2] https://pypi.org/project/similaritymeasures/

    See Also
    --------
    acm : Compute accumulated cost matrix.
    """
    p, q = acm.shape
    if p == 0 or q == 0:
        return np.empty((0, 2), dtype=np.int32)

    path = np.zeros((p + q - 1, 2), dtype=np.int32)
    path_len = np.int32(0)

    i, j = p - 1, q - 1
    path[path_len] = [i, j]
    path_len += 1

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            d = min(acm[i - 1, j], acm[i, j - 1], acm[i - 1, j - 1])
            if acm[i - 1, j] == d:
                i -= 1
            elif acm[i, j - 1] == d:
                j -= 1
            else:
                i -= 1
                j -= 1

        path[path_len] = [i, j]
        path_len += 1

    return path[-(len(path) - path_len + 1) :: -1, :]
