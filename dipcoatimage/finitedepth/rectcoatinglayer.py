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
from typing import TypeVar, Type, Tuple, Optional
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
from .util.geometry import project_on_polyline, polyline_points

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
    "polyline_parallel_area",
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

    __slots__ = ("_interfaces_boundaries",)

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    DecoOptions: Type[DecoOptionsType]
    Data: Type[DataType]

    def capbridge_broken(self) -> bool:
        p0 = self.substrate_point()
        _, p1, p2, _ = self.substrate.contour()[self.substrate.vertices()]
        (bl,) = (p0 + p1).astype(np.int32)
        (br,) = (p0 + p2).astype(np.int32)
        top = np.max([bl[1], br[1]])
        bot = self.binary_image().shape[0]
        if top > bot:
            # substrate is located outside of the frame
            return False
        left = bl[0]
        right = br[0]
        roi_binimg = self.binary_image()[top:bot, left:right]
        return bool(np.any(np.all(roi_binimg, axis=1)))

    def interfaces_boundaries(self) -> npt.NDArray[np.float64]:
        """
        Return two extremal points of the entire union of interface patches.
        """
        if not hasattr(self, "_interfaces_boundaries"):
            (cnt_interfaces,) = self.interfaces(0)
            if not cnt_interfaces:
                return np.empty((0, 1, 2), dtype=np.float64)
            interface_patches = np.concatenate(cnt_interfaces)
            if len(interface_patches) == 0:
                return np.empty((0, 1, 2), dtype=np.float64)
            starts, ends = interface_patches[..., 0], interface_patches[..., 1]
            t0, t1 = np.sort(starts)[0], np.sort(ends)[-1]

            (subst_cnt,), _ = self.substrate.contours()[0]
            subst_cnt = subst_cnt + self.substrate_point()  # DON'T USE += !!
            subst_cnt = np.concatenate([subst_cnt, subst_cnt[:1]])  # closed line
            vec = np.diff(subst_cnt, axis=0)

            t0_int = np.int32(t0)
            t0_dec = t0 - t0_int
            p0 = subst_cnt[t0_int] + vec[t0_int] * t0_dec

            t1_int = np.int32(t1)
            t1_dec = t1 - t1_int
            p1 = subst_cnt[t1_int] + vec[t1_int] * t1_dec

            self._interfaces_boundaries = np.stack([p0, p1])
        return self._interfaces_boundaries

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
        contactline_points = self.interfaces_boundaries()
        if len(contactline_points) == 0:
            return np.empty((0, 1, 2), dtype=np.int32)

        p0, p1 = contactline_points
        (cnt,), _ = cv2.findContours(
            self.coated_substrate().astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        idx0 = np.argmin(np.linalg.norm(cnt - p0[np.newaxis, ...], axis=-1))
        idx1 = np.argmin(np.linalg.norm(cnt - p1[np.newaxis, ...], axis=-1))
        return cnt[int(idx0) : int(idx1 + 1)]


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
    - MeanThickness_Global: Mean thickness in pixel number.
    - Roughness_Global: Roughness of the coating layer compared to the mean
      thickness.
    - MaxThickness_{Left/Bottom/Right}: Number of the pixels for the maximum
      thickness on each region.

    The following data are the metadata for the analysis.

    - MatchError: Template matching error between 0 to 1. 0 means perfect match.
    - ChipWidth: Number of the pixels between lower vertices of the substrate.

    """

    Area: int

    LayerLength_Left: np.float64
    LayerLength_Right: np.float64

    MeanThickness_Global: np.float64

    Roughness_Global: np.float64

    MaxThickness_Left: np.float64
    MaxThickness_Bottom: np.float64
    MaxThickness_Right: np.float64

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
            cv2.circle(vicinity_mask, B, R, 1, -1)
            cv2.circle(vicinity_mask, C, R, 1, -1)
            n = np.dot((C - B) / np.linalg.norm((C - B)), ROTATION_MATRIX)
            pts = np.stack([B, B + R * n, C + R * n, C]).astype(np.int32)
            cv2.fillPoly(vicinity_mask, [pts], 1)
            _, labels = cv2.connectedComponents(img)
            layer_comps = np.unique(labels[np.where(vicinity_mask.astype(bool))])
            layer_mask = np.isin(labels, layer_comps[layer_comps != 0])

            self._extracted_layer = layer_mask
        return self._extracted_layer

    def layer_area(self) -> int:
        """Return the number of pixels in coating layer region."""
        if not hasattr(self, "_layer_area"):
            self._layer_area = np.count_nonzero(self.extract_layer())
        return self._layer_area

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
        # TODO: filter out the projections not onto the interface
        A, B, C, D = self.substrate.sideline_intersections() + self.substrate_point()
        if side == "left":
            P1, P2 = A, B
        elif side == "bottom":
            P1, P2 = B, C
        elif side == "right":
            P1, P2 = C, D
        else:
            return np.empty((0, 2, 2), dtype=np.float64)

        (surface,) = self.enclosing_surface().transpose(1, 0, 2)
        mask = np.cross(P2 - P1, surface - P1) >= 0
        pts = surface[mask]
        proj = find_projection(pts, P1, P2)
        return np.stack([pts, proj]).transpose(1, 0, 2)

    def uniform_layer(self) -> Tuple[np.float64, npt.NDArray[np.float64]]:
        """
        Return thickness and points for uniform layer that satisfies
        :meth:`layer_area`.
        """
        if not hasattr(self, "_uniform_layer"):
            # get contact line points
            contact_points = self.interfaces_boundaries()
            if len(contact_points) == 0:
                layer = np.empty((0, 1, 2), dtype=np.float64)
                self._uniform_layer = (np.float64(0), layer)
                return self._uniform_layer

            hull = self.substrate.hull() + self.substrate_point()
            hull = np.concatenate([hull, hull[:1]])
            i0, i1 = project_on_polyline(contact_points, hull)
            idx = np.arange(int(i0 + 1), int(i1 + 1), dtype=float)
            if idx[0] > i0:
                idx = np.insert(idx, 0, i0)
            if idx[-1] < i1:
                idx = np.insert(idx, len(idx), i1)
            new_hull = polyline_points(idx, hull)

            S = self.layer_area()
            L0 = 10
            L = root(lambda t: polyline_parallel_area(new_hull, t) - S, L0).x

            tan = np.gradient(new_hull, axis=0)
            normal = np.dot(tan, ROTATION_MATRIX)
            n = normal / np.linalg.norm(normal, axis=-1)[..., np.newaxis]

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
            if pairs.size > 0:
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

        MEANTHCK_G, _ = self.uniform_layer()
        ROUGH_G, _ = self.roughness()

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
            AREA,
            LEN_L,
            LEN_R,
            MEANTHCK_G,
            ROUGH_G,
            THCK_L,
            THCK_B,
            THCK_R,
            ERR,
            CHIPWIDTH,
        )


def find_projection(point, A, B):
    Ap = point - A
    AB = B - A
    t = np.dot(Ap, AB) / np.dot(AB, AB)
    A_Proj = np.tensordot(t, AB, axes=0)
    return A + A_Proj


def polyline_parallel_area(P: npt.NDArray, t: float):
    vec = np.diff(P, axis=0)
    d_l = np.linalg.norm(vec, axis=-1)
    d_theta = np.abs(np.diff(np.arctan2(vec[..., 1], vec[..., 0])))
    return np.sum(d_l) * t + np.sum(d_theta) * (t**2) / 2
