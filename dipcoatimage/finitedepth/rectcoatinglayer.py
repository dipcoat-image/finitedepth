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
from typing import TypeVar, Type, Tuple, Optional, List
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
    dfd,
    dfd_pair,
    sfd,
    sfd_path,
    ssfd,
    ssfd_path,
    Color,
    colorize,
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
    "find_projection",
]


ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)
DecoOptionsType = TypeVar("DecoOptionsType", bound=DataclassProtocol)
DataType = TypeVar("DataType", bound=DataclassProtocol)


class RectCoatingLayerBase(
    CoatingLayerBase[
        RectSubstrate, ParametersType, DrawOptionsType, DecoOptionsType, DataType
    ]
):
    """Abstract base class for coating layer over rectangular substrate."""

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    DecoOptions: Type[DecoOptionsType]
    Data: Type[DataType]

    def capbridge_broken(self) -> bool:
        p0 = self.substrate_point()
        _, p1, p2, _ = self.substrate.vertex_points()
        bl = (p0 + p1).astype(np.int32)
        br = (p0 + p2).astype(np.int32)
        top = np.max([bl[1], br[1]])
        bot = self.binary_image().shape[0]
        if top > bot:
            # substrate is located outside of the frame
            return False
        left = bl[0]
        right = br[0]
        roi_binimg = self.binary_image()[top:bot, left:right]
        return bool(np.any(np.all(roi_binimg, axis=1)))

    def enclosing_surface(self) -> npt.NDArray[np.int32]:
        """
        Return an open curve which covers the surfaces of every layer region.

        The result is a continuous curve over entire coating layer regions.
        Discontinuous layers are connected by the substrate surface, i.e. the
        domains in-between are regarded to be covered with zero-thickness layer.

        See Also
        ========

        layer_contours
            Contours for each discrete coating layer region.

        interfaces
            Substrate-liquid interfaces and gas-liquid interfaces for each
            discrete coating layer region.
        """
        interfaces = self.interface_points(1)
        if not interfaces:
            return np.empty((0, 1, 2), dtype=np.int32)
        p0, p1 = interfaces[0][-1], interfaces[-1][0]

        (cnt,), _ = cv2.findContours(
            self.coated_substrate().astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        idx0 = np.argmin(np.linalg.norm(cnt - p0, axis=-1))
        idx1 = np.argmin(np.linalg.norm(cnt - p1, axis=-1))
        return cnt[int(idx0) : int(idx1 + 1)]


@dataclasses.dataclass(frozen=True)
class MorphologyClosingParameters:
    kernelSize: Tuple[int, int]
    anchor: Tuple[int, int] = (-1, -1)
    iterations: int = 1


class DistanceMeasure(enum.Enum):
    """
    Distance measure used to define the curve similarity.

    - DFD : Discrete Fréchet Distance
    - SFD : Summed Fréchet Distance
    - SSFD : Summed Square Fréchet Distance
    """

    DFD = "DFD"
    SFD = "SFD"
    SSFD = "SSFD"


@dataclasses.dataclass(frozen=True)
class RectLayerShapeParameters:
    """Analysis parameters for :class:`RectLayerShape` instance."""

    MorphologyClosing: MorphologyClosingParameters
    ReconstructRadius: int
    RoughnessMeasure: DistanceMeasure
    RoughnessSamples: int


@dataclasses.dataclass
class RectLayerShapeDrawOptions:
    """Drawing options for :class:`RectLayerShape` instance."""

    background: BackgroundDrawMode = BackgroundDrawMode.ORIGINAL
    subtract_mode: SubtractionDrawMode = SubtractionDrawMode.NONE


@dataclasses.dataclass
class RectLayerShapeDecoOptions:
    """Decorating options for :class:`RectLayerShape` instance."""

    layer: FeatureDrawingOptions = FeatureDrawingOptions()
    contact_line: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(0, 255, 0),
    )
    thickness_lines: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(0, 0, 255),
    )
    uniform_layer: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(255, 0, 0), thickness=0
    )
    roughness_pairs: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(0, 255, 255), thickness=0, drawevery=1
    )


@dataclasses.dataclass
class RectLayerShapeData:
    """
    Analysis data for :class:`RectLayerShape` instance.

    - Area: Number of the pixels in the coating layer region.
    - LayerLength_{Left/Right}: Number of the pixels between the lower
      vertices of the substrate and the upper limit of the coating layer.
    - Thickness_{Left/Bottom/Right}: Number of the pixels for the maximum
      thickness on each region.
    - UniformThickness: Number of the pixels for the thickness of the fictitous
      uniform coating layer whose area is same as ``Area``.
    - Roughness: Roughness of the coating layer shape compared to the fictitous
      uniform coating layer. This value has a dimension which is the pixel number
      and can be normalized by dividing with ``UniformThickness``.

    The following data are the metadata for the analysis.

    - MatchError: Template matching error between 0 to 1. 0 means perfect match.
    - ChipWidth: Number of the pixels between lower vertices of the substrate.

    """

    Area: int

    LayerLength_Left: np.float64
    LayerLength_Right: np.float64

    Thickness_Left: np.float64
    Thickness_Bottom: np.float64
    Thickness_Right: np.float64

    UniformThickness: np.float64
    Roughness: np.float64

    MatchError: float
    ChipWidth: np.float32


ROTATION_MATRIX = np.array([[0, 1], [-1, 0]])


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
       >>> tempROI = (100, 50, 1200, 200)
       >>> substROI = (300, 100, 950, 600)
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
       ...     RoughnessMeasure="SSFD",
       ...     RoughnessSamples=100,
       ... )
       >>> param = data_converter.structure(param_val, RectLayerShape.Parameters)
       >>> coat = RectLayerShape(coat_img, subst, param)
       >>> plt.imshow(coat.draw()) #doctest: +SKIP

    """

    __slots__ = (
        "_layer_area",
        "_uniform_layer",
        "_roughness",
    )

    Parameters = RectLayerShapeParameters
    DrawOptions = RectLayerShapeDrawOptions
    DecoOptions = RectLayerShapeDecoOptions
    Data = RectLayerShapeData

    RoughnessMeasure: TypeAlias = DistanceMeasure
    BackgroundDrawMode: TypeAlias = BackgroundDrawMode
    SubtractionDrawMode: TypeAlias = SubtractionDrawMode

    def examine(self) -> Optional[CoatingLayerError]:
        if not self.capbridge_broken():
            return CoatingLayerError("Capillary bridge is not broken.")
        return None

    def extract_layer(self) -> npt.NDArray[np.bool_]:
        if not hasattr(self, "_extracted_layer"):
            # Perform opening to remove error pixels. We named the parameter as
            # "closing" because the coating layer is black in original image, but
            # in fact we do opening since the layer is True in extracted layer.
            closingParams = self.parameters.MorphologyClosing
            kernel = np.ones(closingParams.kernelSize)
            img_closed = cv2.morphologyEx(
                super().extract_layer().astype(np.uint8) * 255,
                cv2.MORPH_OPEN,
                kernel,
                anchor=closingParams.anchor,
                iterations=closingParams.iterations,
            )

            # closed image may still have error pixels. at least we have to
            # remove the errors that are disconnected to the layer.
            # we identify the layer pixels as the connected components that are
            # close to the bottom line.
            vicinity_mask = np.zeros(img_closed.shape, np.uint8)
            p0 = self.substrate_point()
            _, bl, br, _ = self.substrate.vertex_points().astype(np.int32)
            B = p0 + bl
            C = p0 + br
            R = self.parameters.ReconstructRadius
            cv2.circle(vicinity_mask, B, R, 1, -1)
            cv2.circle(vicinity_mask, C, R, 1, -1)
            n = np.dot((C - B) / np.linalg.norm((C - B)), ROTATION_MATRIX)
            pts = np.stack([B, B + R * n, C + R * n, C]).astype(np.int32)
            cv2.fillPoly(vicinity_mask, [pts], 1)
            _, labels = cv2.connectedComponents(img_closed)
            layer_comps = np.unique(labels[np.where(vicinity_mask.astype(bool))])
            layer_mask = np.isin(labels, layer_comps[layer_comps != 0])

            self._extracted_layer = layer_mask
        return self._extracted_layer

    def layer_area(self) -> int:
        """Return the number of pixels in coating layer region."""
        if not hasattr(self, "_layer_area"):
            self._layer_area = np.count_nonzero(self.extract_layer())
        return self._layer_area

    def surface_projections(
        self, side: str
    ) -> List[Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]]:
        """
        For the relevant surface points, find projection points to a side line.

        Parameters
        ----------
        side: {"left", "bottom", "right"}

        Returns
        -------
        list
            List of tuple of indices and points. Each tuple represents a layer
            region contour, sorted in the order along the substrate contour.
            Indices are the location of surface points in the arrays from
            :meth:`surface_points`.

        Notes
        -----
        Relevance of the surface points to side is determined by their position.

        - `"left"`: Left of the left side line.
        - `"bottom"`: Under the bottom side line.
        - `"right"`: Right of the right side line.
        """
        A, B, C, D = self.substrate.vertex_points() + self.substrate_point()
        if side == "left":
            P1, P2 = A, B
        elif side == "bottom":
            P1, P2 = B, C
        elif side == "right":
            P1, P2 = C, D
        else:
            return []

        projections = []
        surface = self.surface_points(1)
        for surf in surface:
            (mask,) = (np.cross(P2 - P1, surf - P1) >= 0).T
            (indices,) = np.nonzero(mask)
            proj = find_projection(surf[mask], P1, P2)
            projections.append((indices, proj))

        return projections

    def uniform_layer(self) -> Tuple[np.float64, npt.NDArray[np.float64]]:
        """
        Return thickness and points for uniform layer that satisfies
        :meth:`layer_area`.
        """
        if not hasattr(self, "_uniform_layer"):
            # get contact line points
            interfaces = self.interface_points(1)
            if not interfaces:
                layer = np.empty((0, 1, 2), dtype=np.float64)
                self._uniform_layer = (np.float64(0), layer)
                return self._uniform_layer
            p1, p2 = interfaces[0][-1], interfaces[-1][0]

            subst_point = self.substrate_point()
            hull, _ = self.substrate.edge_hull()
            (hull,) = (hull + subst_point).transpose(1, 0, 2)
            dh = np.diff(hull, axis=0)
            dh_dot_dh = np.sum(dh * dh, axis=-1)

            def find_projection(p):
                h_p = p - hull[:-1, ...]
                dh_scale_p = np.sum(h_p * dh, axis=-1) / dh_dot_dh
                p_mask = (0 <= dh_scale_p) & (dh_scale_p <= 1)
                if np.any(p_mask):
                    p_proj_origins = hull[:-1, ...][p_mask, ...]
                    p_proj_vecs = dh_scale_p[p_mask][..., np.newaxis] * dh[p_mask, ...]
                    p_proj_dists = np.linalg.norm(h_p[p_mask] - p_proj_vecs, axis=-1)
                    idx = np.argmin(p_proj_dists)
                    i = np.arange(hull[:-1, ...].shape[0])[p_mask][idx]
                    p_proj = (p_proj_origins + p_proj_vecs)[idx]
                else:
                    i = 0
                    p_proj = np.empty((0, 2), dtype=np.float64)
                return i + 1, p_proj

            (i1, proj1), (i2, proj2) = sorted(
                [find_projection(p1), find_projection(p2)], key=lambda x: x[0]
            )
            new_hull = hull[int(i1) : int(i2)]
            if not np.all(new_hull[0] == proj1):
                new_hull = np.insert(new_hull, 0, proj1, axis=0)
            nh_len = new_hull.shape[0]
            if not np.all(new_hull[nh_len - 1] == proj2):
                new_hull = np.insert(new_hull, nh_len, proj2, axis=0)
            t = np.arange(new_hull.shape[0])

            # find thickness
            dt = np.diff(t, append=t[-1] + (t[-1] - t[-2]))
            tangent = np.gradient(new_hull, t, axis=0)
            normal = np.dot(tangent, ROTATION_MATRIX)
            n = normal / np.linalg.norm(normal, axis=1)[..., np.newaxis]
            dndt = np.gradient(n, t, axis=0)

            S = self.layer_area()
            L0 = 10  # initial value
            L_NUM = 100  # interval number

            def findL(l0, l_num):
                l_pts = np.linspace(0, l0, l_num)
                dl = np.diff(l_pts, append=l_pts[-1] + (l_pts[-1] - l_pts[-2]))
                e_t = tangent[..., np.newaxis] + np.tensordot(dndt, l_pts, axes=0)
                G = (
                    np.sum(e_t * e_t, axis=1)
                    - np.tensordot(np.sum(n * dndt, axis=1), l_pts, axes=0) ** 2
                )
                dS = np.sqrt(G) * dt[..., np.newaxis] * dl
                S_i = np.cumsum(np.sum(dS, axis=0))
                k0 = (S_i[-1] - S_i[-2]) / dl[-1]
                newL = max(0, (S - S_i[-1]) / k0 + l0)
                if l_pts[-2] <= newL <= l_pts[-1]:
                    return newL
                return findL(newL, l_num)

            L = findL(L0, L_NUM)
            self._uniform_layer = (L, new_hull + L * n)

        return self._uniform_layer

    def roughness(self) -> Tuple[np.float64, npt.NDArray[np.float64]]:
        """Dimensional roughness value of the coating layer surface."""
        if not hasattr(self, "_roughness"):
            (surface,) = self.enclosing_surface().transpose(1, 0, 2)
            _, uniform_layer = self.uniform_layer()
            if surface.size == 0 or uniform_layer.size == 0:
                return (np.float64(np.nan), np.empty((0, 2, 2), dtype=np.float64))

            NUM_POINTS = self.parameters.RoughnessSamples

            def equidistant_interp(points):
                # https://stackoverflow.com/a/19122075
                vec = np.diff(points, axis=0)
                dist = np.linalg.norm(vec, axis=1)
                u = np.insert(np.cumsum(dist), 0, 0)
                t = np.linspace(0, u[-1], NUM_POINTS)
                y, x = points.T
                return np.stack([np.interp(t, u, y), np.interp(t, u, x)]).T

            P = equidistant_interp(surface).astype(np.float64)
            Q = equidistant_interp(uniform_layer).astype(np.float64)

            if self.parameters.RoughnessMeasure == self.RoughnessMeasure.DFD:
                ca = dfd(P, Q)
                path = dfd_pair(ca)
                roughness = ca[-1, -1] / len(path)
            elif self.parameters.RoughnessMeasure == self.RoughnessMeasure.SFD:
                ca = sfd(P, Q)
                path = sfd_path(ca)
                roughness = ca[-1, -1] / len(path)
            elif self.parameters.RoughnessMeasure == self.RoughnessMeasure.SSFD:
                ca = ssfd(P, Q)
                path = ssfd_path(ca)
                roughness = np.sqrt(ca[-1, -1] / len(path))
            else:
                raise TypeError(f"Unknown option: {self.parameters.RoughnessMeasure}")

            similarity_pairs = np.stack([P[path[..., 0]], Q[path[..., 1]]])
            self._roughness = (roughness, similarity_pairs.transpose(1, 0, 2))

        return self._roughness

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
            interfaces = self.interface_points(1)
            if interfaces:
                (p1,), (p2,) = interfaces[0][-1], interfaces[-1][0]
                cv2.line(
                    image,
                    p1,
                    p2,
                    dataclasses.astuple(contactline_opts.color),
                    contactline_opts.thickness,
                )

        thicknesslines_opts = self.deco_options.thickness_lines
        if thicknesslines_opts.thickness > 0:
            color = dataclasses.astuple(thicknesslines_opts.color)
            t = thicknesslines_opts.thickness
            for side in ["left", "bottom", "right"]:
                surf_pts, proj_pts = [], []
                for surf, (idx, proj) in zip(
                    self.surface_points(1), self.surface_projections(side)
                ):
                    surf_pts.append(surf[idx])
                    proj_pts.append(proj)
                if not surf_pts or not proj_pts:
                    continue
                surf = np.concatenate(surf_pts, axis=0)
                proj = np.concatenate(proj_pts, axis=0)
                if surf.size == 0 or proj.size == 0:
                    continue
                dists = np.linalg.norm(surf - proj, axis=-1)
                max_idxs, _ = np.nonzero(dists == np.max(dists))
                # split the max indices by continuous locations
                idx_groups = np.split(max_idxs, np.where(np.diff(max_idxs) != 1)[0] + 1)
                for idxs in idx_groups:
                    (p1,) = np.mean(surf[idxs], axis=0).astype(np.int32)
                    (p2,) = np.mean(proj[idxs], axis=0).astype(np.int32)
                    cv2.line(image, p1, p2, color, t)

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

        roughnesspair_opts = self.deco_options.roughness_pairs
        if roughnesspair_opts.thickness > 0:
            _, pairs = self.roughness()
            for pair in pairs[:: roughnesspair_opts.drawevery]:
                cv2.line(
                    image,
                    *pair.astype(np.int32),
                    color=dataclasses.astuple(roughnesspair_opts.color),
                    thickness=roughnesspair_opts.thickness,
                )
            # always draw the last line
            cv2.line(
                image,
                *pairs[-1].astype(np.int32),
                color=dataclasses.astuple(roughnesspair_opts.color),
                thickness=roughnesspair_opts.thickness,
            )

        return image

    def analyze_layer(
        self,
    ) -> Tuple[
        int,
        np.float64,
        np.float64,
        np.float64,
        np.float64,
        np.float64,
        np.float64,
        np.float64,
        float,
        np.float32,
    ]:
        AREA = self.layer_area()

        _, B, C, _ = self.substrate.vertex_points() + self.substrate_point()

        interfaces = self.interface_points(1)
        if interfaces:
            points = np.concatenate([interfaces[0][-1], interfaces[-1][0]])
            Bp = points - B
            BC = C - B
            t = np.dot(Bp, BC) / np.dot(BC, BC)
            dists = np.linalg.norm(Bp - np.tensordot(t, BC, axes=0), axis=1)
            LEN_L, LEN_R = dists.astype(np.float64)
        else:
            LEN_L = LEN_R = np.float64(0)

        max_dists = []
        for side in ["left", "bottom", "right"]:
            dists = []
            for surf, (idx, proj) in zip(
                self.surface_points(1), self.surface_projections(side)
            ):
                dists.append(np.linalg.norm((surf[idx] - proj), axis=-1))
            if not dists:
                max_d = np.float64(0)
            else:
                dists_concat = np.concatenate(dists, axis=0)
                if dists_concat.size == 0:
                    max_d = np.float64(0)
                else:
                    max_d = np.max(dists_concat)
            max_dists.append(max_d)
        THCK_L, THCK_B, THCK_R = max_dists

        THCK_U, _ = self.uniform_layer()
        ROUGH, _ = self.roughness()

        ERR, _ = self.match_substrate()
        CHIPWIDTH = np.linalg.norm(B - C)

        return (
            AREA,
            LEN_L,
            LEN_R,
            THCK_L,
            THCK_B,
            THCK_R,
            THCK_U,
            ROUGH,
            ERR,
            CHIPWIDTH,
        )


def find_projection(point, A, B):
    Ap = point - A
    AB = B - A
    t = np.dot(Ap, AB) / np.dot(AB, AB)
    A_Proj = np.tensordot(t, AB, axes=0)
    return A + A_Proj
