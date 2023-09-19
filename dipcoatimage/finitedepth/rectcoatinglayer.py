"""
:mod:`dipcoatimage.finitedepth.rectcoatinglayer` provides class to analyze
the coating layer over rectangular substrate.

Base class
==========

.. autoclass:: RectCoatingLayerBase
   :members:

Implementation
==============

.. autoclass:: RectLayerShapeParameters
   :members:

.. autoclass:: RectLayerShapeDrawOptions
   :members:

.. autoclass:: RectLayerShapeDecoOptions
   :members:

.. autoclass:: RectLayerShapeData
   :members:

.. autoclass:: RectLayerShape
   :members:

"""

import cv2  # type: ignore
import dataclasses
import enum
import numpy as np
import numpy.typing as npt
from scipy.optimize import root  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from typing import TypeVar, Type, Tuple, List, Optional
from .rectsubstrate import RectSubstrate
from .coatinglayer import (
    CoatingLayerError,
    CoatingLayerBase,
    BackgroundDrawMode,
    SubtractionDrawMode,
)
from .util import (
    images_XOR,
    DataclassProtocol,
    FeatureDrawingOptions,
    Color,
    colorize,
)
from .util.dtw import acm, owp
from .util.geometry import (
    split_polyline,
    line_polyline_intersections,
    project_on_polylines,
    polylines_external_points,
    closest_in_polylines,
    polylines_internal_points,
    polyline_parallel_area,
    equidistant_interpolate,
)

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias


__all__ = [
    "RectCoatingLayerBase",
    "MorphologyClosingParameters",
    "DistanceMeasure",
    "RectLayerShapeParameters",
    "RectLayerShapeDrawOptions",
    "RectLayerShapeDecoOptions",
    "RectLayerShapeData",
    "RectLayerShape",
]


ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)
DecoOptionsType = TypeVar("DecoOptionsType", bound=DataclassProtocol)
DataType = TypeVar("DataType", bound=DataclassProtocol)


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

    def interfaces(self) -> Tuple[npt.NDArray[np.int64], ...]:
        """
        Find indices of solid-liquid interfaces on :meth:`SubstrateBase.contour`.

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
        if not hasattr(self, "_interfaces"):
            subst_cnt = self.substrate.contour() + self.substrate_point()
            layer_cnts, _ = cv2.findContours(
                self.extract_layer().astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            ret = []
            for layer_cnt in layer_cnts:
                lcnt_img = np.zeros(self.image.shape[:2], dtype=np.uint8)
                lcnt_img[layer_cnt[..., 1], layer_cnt[..., 0]] = 255
                dilated_lcnt = cv2.dilate(lcnt_img, np.ones((3, 3))).astype(bool)

                x, y = subst_cnt.transpose(2, 0, 1)
                mask = dilated_lcnt[y, x]

                # Find indices of continuous True blocks
                idxs = np.where(
                    np.diff(np.concatenate(([False], mask[:, 0], [False]))) == 1
                )[0].reshape(-1, 2)
                ret.append(idxs)
            self._interfaces = tuple(ret)
        return self._interfaces

    def contour(self) -> npt.NDArray[np.int32]:
        """
        Contour of the entire coated substrate.
        """
        if not hasattr(self, "_contour"):
            (cnt,), _ = cv2.findContours(
                self.coated_substrate().astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            self._contour = cnt
        return self._contour

    def surface(self) -> npt.NDArray[np.int32]:
        """
        Return the surface of the entire coated region.

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
            indices, = self.interfaces()
            (i0, i1) = indices.flatten()[[0, -1]]
            subst_cnt = self.substrate.contour() + self.substrate_point()
            endpoints = subst_cnt[[i0, i1]]

            vec = self.contour() - endpoints.transpose(1, 0, 2)
            self._surface_indices = np.argmin(np.linalg.norm(vec, axis=-1), axis=0)
        (I0, I1) = self._surface_indices
        return self.contour()[I0: I1 + 1]

    def capbridge_broken(self) -> bool:
        p0 = self.substrate_point()
        _, bl, br, _ = self.substrate.contour()[self.substrate.vertices()]
        (B,) = p0 + bl
        (C,) = p0 + br
        top = np.max([B[1], C[1]])
        bot = self.binary_image().shape[0]
        if top > bot:
            # substrate is located outside of the frame
            return False
        left = B[0]
        right = C[0]
        roi_binimg = self.binary_image()[top:bot, left:right]
        return bool(np.any(np.all(roi_binimg, axis=1)))


@dataclasses.dataclass(frozen=True)
class MorphologyClosingParameters:
    """
    Parameter to perform Morphological closing operation.

    Kernel sizes MUST be odd lest the operation leaves residue pixels.
    """

    kernelSize: Tuple[int, int]
    anchor: Tuple[int, int] = (-1, -1)
    iterations: int = 1


class DistanceMeasure(enum.Enum):
    """
    Distance measures to compute the curve similarity.

    - DTW : Dynamic time warping
    - SDTW : Squared dynamic time warping
    """

    DTW = "DTW"
    SDTW = "SDTW"


@dataclasses.dataclass(frozen=True)
class RectLayerShapeParameters:
    """Analysis parameters for :class:`RectLayerShape` instance."""

    MorphologyClosing: MorphologyClosingParameters
    ReconstructRadius: int
    RoughnessMeasure: DistanceMeasure


@dataclasses.dataclass
class RectLayerShapeDrawOptions:
    """Drawing options for :class:`RectLayerShape` instance."""

    background: BackgroundDrawMode = BackgroundDrawMode.ORIGINAL
    subtract_mode: SubtractionDrawMode = SubtractionDrawMode.NONE


@dataclasses.dataclass
class RectLayerShapeDecoOptions:
    """Decorating options for :class:`RectLayerShape` instance."""

    layer: FeatureDrawingOptions = dataclasses.field(
        default_factory=FeatureDrawingOptions
    )
    contact_line: FeatureDrawingOptions = dataclasses.field(
        default_factory=lambda: FeatureDrawingOptions(color=Color(0, 255, 0))
    )
    thickness_lines: FeatureDrawingOptions = dataclasses.field(
        default_factory=lambda: FeatureDrawingOptions(color=Color(0, 0, 255))
    )
    uniform_layer: FeatureDrawingOptions = dataclasses.field(
        default_factory=lambda: FeatureDrawingOptions(
            color=Color(255, 0, 0), thickness=0
        )
    )
    roughness_pairs: FeatureDrawingOptions = dataclasses.field(
        default_factory=lambda: FeatureDrawingOptions(
            color=Color(0, 255, 255), thickness=0, drawevery=1
        )
    )


@dataclasses.dataclass
class RectLayerShapeData:
    """
    Analysis data for :class:`RectLayerShape` instance.

    - LayerLength_{Left, Right}: Distance between the bottom sideline of the
      substrate and the upper limit of the coating layer.
    - Conformality: Conformality of the coating layer.
    - AverageThickness: Average thickness of the coating layer.
    - Roughness: Roughness of the coating layer.
    - MaxThickness_{Left, Bottom, Right}: Number of the pixels for the maximum
      thickness on each region.

    The following data are the metadata for the analysis.

    - MatchError: Template matching error between 0 to 1. 0 means perfect match.
    - ChipWidth: Number of the pixels between lower vertices of the substrate.

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
    ChipWidth: np.float32


class RectLayerShape(
    RectCoatingLayerBase[
        RectLayerShapeParameters,
        RectLayerShapeDrawOptions,
        RectLayerShapeDecoOptions,
        RectLayerShapeData,
    ]
):
    """
    Class for analyzing the shape and thickness of the coating layer over
    rectangular substrate.

    Examples
    ========

    Construct substrate reference class first.

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import (SubstrateReference,
       ...     get_samples_path)
       >>> ref_path = get_samples_path("ref3.png")
       >>> img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
       >>> tempROI = (13, 10, 1246, 200)
       >>> substROI = (100, 100, 1200, 500)
       >>> ref = SubstrateReference(img, tempROI, substROI)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    Construct the parameters and substrate instance from reference instance.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import RectSubstrate, data_converter
       >>> param_val = dict(Sigma=3.0, Rho=1.0, Theta=0.01)
       >>> param = data_converter.structure(param_val, RectSubstrate.Parameters)
       >>> subst = RectSubstrate(ref, param)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    Construct :class:`RectLayerShape` from substrate class. :meth:`analyze`
    returns the number of pixels in coating area region.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import RectLayerShape
       >>> coat_path = get_samples_path("coat3.png")
       >>> coat_img = cv2.cvtColor(cv2.imread(coat_path), cv2.COLOR_BGR2RGB)
       >>> param_val = dict(
       ...     MorphologyClosing=dict(kernelSize=(1, 1)),
       ...     ReconstructRadius=50,
       ...     RoughnessMeasure="SDTW",
       ... )
       >>> param = data_converter.structure(param_val, RectLayerShape.Parameters)
       >>> coat = RectLayerShape(coat_img, subst, param)
       >>> plt.imshow(coat.draw()) #doctest: +SKIP

    """

    __slots__ = (
        "_uniform_layer",
        "_conformality",
        "_roughness",
    )

    Parameters = RectLayerShapeParameters
    DrawOptions = RectLayerShapeDrawOptions
    DecoOptions = RectLayerShapeDecoOptions
    Data = RectLayerShapeData

    BackgroundDrawMode: TypeAlias = BackgroundDrawMode
    SubtractionDrawMode: TypeAlias = SubtractionDrawMode

    def examine(self) -> Optional[CoatingLayerError]:
        ksize = self.parameters.MorphologyClosing.kernelSize
        if not all(i == 0 or i % 2 == 1 for i in ksize):
            return CoatingLayerError("Kernel size must be odd.")
        if not self.capbridge_broken():
            return CoatingLayerError("Capillary bridge is not broken.")
        return None

    def extract_layer(self) -> npt.NDArray[np.bool_]:
        if not hasattr(self, "_extracted_layer"):
            # Perform opening to remove error pixels. We named the parameter as
            # "closing" because the coating layer is black in original image, but
            # in fact we do opening since the layer is True in extracted layer.
            closingParams = self.parameters.MorphologyClosing
            if any(i == 0 for i in closingParams.kernelSize):
                img = super().extract_layer().astype(np.uint8) * 255
            else:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, closingParams.kernelSize
                )
                img = cv2.morphologyEx(
                    super().extract_layer().astype(np.uint8) * 255,
                    cv2.MORPH_OPEN,
                    kernel,
                    anchor=closingParams.anchor,
                    iterations=closingParams.iterations,
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

            self._extracted_layer = layer_mask
        return self._extracted_layer

    def uniform_layer(self) -> Tuple[np.float64, npt.NDArray[np.float64]]:
        """Return thickness and points for uniform layer."""
        if not self.interfaces():
            return (np.float64(0), np.empty((0, 1, 2), np.float64))

        if not hasattr(self, "_uniform_layer"):
            indices, = self.interfaces()
            (i0, i1) = indices.flatten()[[0, -1]]
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

            self._uniform_layer = (t, ul)
        return self._uniform_layer

    def conformality(self) -> Tuple[np.float64, npt.NDArray[np.int32]]:
        """Conformality of the coating layer and its optimal path."""
        if not hasattr(self, "_conformality"):
            indices, = self.interfaces()
            (i0, i1) = indices.flatten()[[0, -1]]
            subst_cnt = self.substrate.contour() + self.substrate_point()
            intf = subst_cnt[i0:i1]

            surf = self.surface()

            if surf.size == 0 or intf.size == 0:
                self._conformality = (np.nan, np.empty((0, 2), dtype=np.int32))
            else:
                dist = cdist(np.squeeze(surf, axis=1), np.squeeze(intf, axis=1))
                mat = acm(dist)
                path = owp(mat)
                d = dist[path[:, 0], path[:, 1]]
                d_avrg = mat[-1, -1] / len(path)
                C = 1 - np.sum(np.abs(d - d_avrg)) / mat[-1, -1]
                # pairs = np.stack([surf[path[..., 0]], intf[path[..., 1]]])
                self._conformality = (np.float64(C), path)
    
        return self._conformality

    def roughness(self) -> Tuple[np.float64, npt.NDArray[np.int32]]:
        """Roughness of the coating layer and its optimal path."""
        if not hasattr(self, "_roughness"):
            surf = self.surface()
            _, ul = self.uniform_layer()

            if surf.size == 0 or ul.size == 0:
                self._roughness = (np.nan, np.empty((0, 2), dtype=np.float64))
            else:
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
                # pairs = np.stack([surf[path[..., 0]], ul[path[..., 1]]])
                self._roughness = (np.float64(roughness), path)

        return self._roughness

    def surface_projections(self, side: str) -> npt.NDArray[np.float64]:
        """
        For *side*, return the relevant surface points and its projections to
        side line.

        Parameters
        ----------
        side: {"left", "bottom", "right"}

        Returns
        -------
        ndarray
            Array of the points and projections coordinates. Shape of the array
            is `(N, 2, D)`, where `N` is the number of points and `D` is the
            dimension of the point. On the second axis, 0-th index is the surface
            points and 1-th index is the projection points.
        """
        A, B, C, D = self.substrate.sideline_intersections() + self.substrate_point()
        if side == "left":
            P1, P2 = A, B
        elif side == "bottom":
            P1, P2 = B, C
        elif side == "right":
            P1, P2 = C, D
        else:
            return np.empty((0, 2, 2), dtype=np.float64)

        surf_idx = self.enclosing_surface()
        if surf_idx.size == 0:
            surface = np.empty((0, 1, 2), dtype=np.float64)
        else:
            layer_cnt = self.contour()
            _, surface, _ = split_polyline(surf_idx, layer_cnt.transpose(1, 0, 2))
            surface = surface.transpose(1, 0, 2)

        mask = np.cross(P2 - P1, surface - P1) >= 0
        pts = surface[mask][:, np.newaxis]
        lines = np.stack([P1, P2])[np.newaxis, ...]

        prj = project_on_polylines(pts, lines)
        prj_pts = np.squeeze(polylines_external_points(prj, lines), axis=2)
        return np.concatenate([pts, prj_pts], axis=1)

    def draw(self) -> npt.NDArray[np.uint8]:
        background = self.draw_options.background
        if background == self.BackgroundDrawMode.ORIGINAL:
            image = self.image.copy()
        elif background == self.BackgroundDrawMode.BINARY:
            image = self.binary_image().copy()
        elif background == self.BackgroundDrawMode.EMPTY:
            image = np.full(self.image.shape, 255, dtype=np.uint8)
        else:
            raise TypeError("Unrecognized background mode: %s" % background)
        image = colorize(image)

        subtract_mode = self.draw_options.subtract_mode
        if subtract_mode & self.SubtractionDrawMode.TEMPLATE:
            x0, y0, x1, y1 = self.substrate.reference.templateROI
            tempImg = self.substrate.reference.binary_image()[y0:y1, x0:x1]
            h, w = tempImg.shape[:2]
            _, (X0, Y0) = self.match_substrate()
            binImg = self.binary_image()[Y0 : Y0 + h, X0 : X0 + w]
            mask = images_XOR(~binImg.astype(bool), ~tempImg.astype(bool))
            image[Y0 : Y0 + h, X0 : X0 + w][~mask] = 255
        if subtract_mode & self.SubtractionDrawMode.SUBSTRATE:
            x0, y0, x1, y1 = self.substrate.reference.substrateROI
            substImg = self.substrate.reference.binary_image()[y0:y1, x0:x1]
            h, w = substImg.shape[:2]
            X0, Y0 = self.substrate_point()
            binImg = self.binary_image()[Y0 : Y0 + h, X0 : X0 + w]
            mask = images_XOR(~binImg.astype(bool), ~substImg.astype(bool))
            image[Y0 : Y0 + h, X0 : X0 + w][~mask] = 255

        layer_opts = self.deco_options.layer
        if layer_opts.thickness != 0:
            image[self.extract_layer()] = 255
            cv2.drawContours(
                image,
                self.layer_contours(),
                -1,
                dataclasses.astuple(layer_opts.color),
                layer_opts.thickness,
            )

        contactline_opts = self.deco_options.contact_line
        if contactline_opts.thickness > 0:
            contactline_points = self.interfaces_boundaries()
            if len(contactline_points) != 0:
                (p0,), (p1,) = contactline_points.astype(np.int32)
                cv2.line(
                    image,
                    p0,
                    p1,
                    dataclasses.astuple(contactline_opts.color),
                    contactline_opts.thickness,
                )

        thicknesslines_opts = self.deco_options.thickness_lines
        if thicknesslines_opts.thickness > 0:
            color = dataclasses.astuple(thicknesslines_opts.color)
            t = thicknesslines_opts.thickness
            for side in ["left", "bottom", "right"]:
                surf_proj = self.surface_projections(side)
                dists = np.linalg.norm(np.diff(surf_proj, axis=1), axis=-1)
                if dists.size == 0:
                    continue
                max_idxs, _ = np.nonzero(dists == np.max(dists))
                # split the max indices by continuous locations
                idx_groups = np.split(max_idxs, np.where(np.diff(max_idxs) != 1)[0] + 1)
                for idxs in idx_groups:
                    surf = np.mean(surf_proj[:, 0][idxs], axis=0).astype(np.int32)
                    proj = np.mean(surf_proj[:, 1][idxs], axis=0).astype(np.int32)
                    cv2.line(image, surf, proj, color, t)

        uniformlayer_opts = self.deco_options.uniform_layer
        if uniformlayer_opts.thickness > 0:
            _, points = self.uniform_layer()
            cv2.polylines(
                image,
                [points.astype(np.int32)],
                isClosed=False,
                color=dataclasses.astuple(uniformlayer_opts.color),
                thickness=uniformlayer_opts.thickness,
            )

        pair_opts = self.deco_options.roughness_pairs
        if pair_opts.thickness > 0:
            surf = self.surface()
            _, ul = self.uniform_layer()
            _, path = self.roughness()
            pairs = np.stack([surf[path[..., 0]], ul[path[..., 1]]]).astype(np.int32)
            for surf_pt, ul_pt in pairs.transpose(1, 0, 2, 3)[:: pair_opts.drawevery]:
                cv2.line(
                    image,
                    *surf_pt,
                    *ul_pt,
                    color=dataclasses.astuple(pair_opts.color),
                    thickness=pair_opts.thickness,
                )
            # always draw the last line
            if pairs.size > 0:
                cv2.line(
                    image,
                    *pairs[0, -1, ...],
                    *pairs[1, -1, ...],
                    color=dataclasses.astuple(pair_opts.color),
                    thickness=pair_opts.thickness,
                )

        return image

    def analyze_layer(self):
        _, B, C, _ = self.substrate.sideline_intersections() + self.substrate_point()

        contactline_points = self.interfaces_boundaries()
        if len(contactline_points) != 0:
            Bp = contactline_points - B
            BC = C - B
            t = np.dot(Bp, BC) / np.dot(BC, BC)
            dists = np.linalg.norm(Bp - np.tensordot(t, BC, axes=0), axis=-1)
            (LEN_L,), (LEN_R,) = dists.astype(np.float64)
        else:
            LEN_L = LEN_R = np.float64(0)

        C, _ = self.conformality()
        AVRGTHCK, _ = self.uniform_layer()
        ROUGH, _ = self.roughness()

        max_dists = []
        for side in ["left", "bottom", "right"]:
            surf_proj = self.surface_projections(side)
            dists = np.linalg.norm(np.diff(surf_proj, axis=1), axis=-1)
            if dists.size == 0:
                max_dists.append(np.float64(0))
            else:
                max_dists.append(dists.max())
        THCK_L, THCK_B, THCK_R = max_dists

        ERR, _ = self.match_substrate()
        CHIPWIDTH = np.linalg.norm(B - C)

        return (
            LEN_L,
            LEN_R,
            C,
            AVRGTHCK,
            ROUGH,
            THCK_L,
            THCK_B,
            THCK_R,
            ERR,
            CHIPWIDTH,
        )
