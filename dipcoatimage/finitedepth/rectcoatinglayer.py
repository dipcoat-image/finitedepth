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
from .util.frechet import (
    dfd,
    dfd_pair,
    sfd,
    sfd_path,
    ssfd,
    ssfd_path,
)
from .util.geometry import (
    split_polyline,
    line_polyline_intersections,
    project_on_polylines,
    polylines_external_points,
    closest_in_polylines,
    polylines_internal_points,
    equidistant_interpolate,
    polyline_parallel_area,
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
    "uniform_layer",
    "roughness",
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
        "_contour",
        "_interfaces_boundaries",
        "_enclosing_surface",
        "_enclosing_interface",
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    DecoOptions: Type[DecoOptionsType]
    Data: Type[DataType]

    def capbridge_broken(self) -> bool:
        vert = polylines_internal_points(
            self.substrate.vertices(), self.substrate.contour().transpose(1, 0, 2)
        )
        _, (bl,), (br,), _ = (vert + self.substrate_point()).astype(np.int32)
        top = np.max([bl[1], br[1]])
        bot = self.binary_image().shape[0]
        if top > bot:
            # substrate is located outside of the frame
            return False
        left = bl[0]
        right = br[0]
        roi_binimg = self.binary_image()[top:bot, left:right]
        return bool(np.any(np.all(roi_binimg, axis=1)))

    def contour(self) -> npt.NDArray[np.int32]:
        if not hasattr(self, "_contour"):
            (cnt,), _ = cv2.findContours(
                self.coated_substrate().astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            self._contour = cnt
        return self._contour

    def interfaces_boundaries(self) -> npt.NDArray[np.float64]:
        """
        Return two extremal points of the entire union of interface patches.
        """
        # XXX: after implementing uniform layer for convex polyline and removing
        # contour() overriding of RectSubstrate, make this method return
        # parameters for boundaries in contour().
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
            pts = polylines_internal_points(
                np.stack([t0, t1]).reshape(-1, 1),
                subst_cnt.transpose(1, 0, 2),
            )

            self._interfaces_boundaries = pts
        return self._interfaces_boundaries

    def enclosing_interface(self) -> npt.NDArray[np.float64]:
        """
        Return parameters of substrate contour for open curve over solid-liquid
        interfaces of every coating layer region.

        Returns
        -------
        points: ndarray
            Parameters for the interval in substrate contour.

        See Also
        --------
        enclosing_surface : Open curve over liquid-gas surfaces.

        Notes
        -----
        To get the coordinates of polyline vertices, pass this result with
        substrate contour to :func:`split_polyline`.
        """
        if not hasattr(self, "_enclosing_interface"):
            contactline_points = self.interfaces_boundaries()
            if len(contactline_points) == 0:
                return np.empty((0, 1), dtype=np.int32)
            self._enclosing_interface = closest_in_polylines(
                contactline_points - self.substrate_point(),
                self.substrate.contour().transpose(1, 0, 2),
            )
        return self._enclosing_interface

    def enclosing_surface(self) -> npt.NDArray[np.float64]:
        """
        Return parameters of substrate contour for open curve over liquid-gas
        surfaces of every coating layer region.

        Returns
        -------
        points: ndarray
            Parameters for the interval in coated substrate contour.

        See Also
        --------
        enclosing_interface : Open curve over solid-liquid interfaces.

        Notes
        -----
        To get the coordinates of polyline vertices, pass this result with
        :meth:`contour` to :func:`split_polyline`.
        """
        if not hasattr(self, "_enclosing_surface"):
            contactline_points = self.interfaces_boundaries()
            if len(contactline_points) == 0:
                return np.empty((0, 1), dtype=np.int32)
            self._enclosing_surface = closest_in_polylines(
                contactline_points,
                self.contour().transpose(1, 0, 2),
            )
        return self._enclosing_surface

    def layer_regions(
        self,
    ) -> Tuple[List[npt.NDArray[np.float64]], List[npt.NDArray[np.float64]]]:
        """
        Return the interface points and the surface points for the layer on each
        region of the substrate, i.e. left, bottom, and right.

        Returns
        -------
        intf_reg, surf_reg: list of ndarray
            Coordinates of points. Each array represents a portion of
            :meth:`enclosing_interface` or :meth:`enclosing_surface`, split by
            the substrate surface regions.

        Notes
        -----
        The substrate region is split by the vertex points on the substrate
        surface. For each of two boundaries of the substrate region, normal
        vector is drawn and the closest intersection with the coaating layer
        surface is found. The coating layer region is split by this intersection.
        """
        intf_idx = self.enclosing_interface()
        surf_idx = self.enclosing_surface()
        if intf_idx.size == 0 or surf_idx.size == 0:
            intf_reg = [np.empty((0, 1, 2), dtype=np.float64) for _ in range(3)]
            surf_reg = [np.empty((0, 1, 2), dtype=np.float64) for _ in range(3)]
            return intf_reg, surf_reg
        subst_vert_idx = np.clip(self.substrate.vertices(), *intf_idx)
        subst_cnt = self.substrate.contour() + self.substrate_point()
        intf_reg = split_polyline(subst_vert_idx, subst_cnt.transpose(1, 0, 2))[1:-1]
        intf_reg = [reg.transpose(1, 0, 2) for reg in intf_reg]

        _, surf, _ = split_polyline(surf_idx, self.contour().transpose(1, 0, 2))

        surf_reg = []
        for reg in intf_reg:
            if not len(reg) > 1:
                surf_reg.append(np.empty((0, 1, 2), dtype=np.float64))
                continue
            boundary_dr = reg[[1, -1]] - reg[[0, -2]]
            if np.any(np.linalg.norm(boundary_dr, axis=-1) == 0):
                surf_reg.append(np.empty((0, 1, 2), dtype=np.float64))
                continue

            pt = reg[[0, -1]]
            n = np.dot(boundary_dr, ROTATION_MATRIX)
            surf_reg_idx = []
            for line in np.concatenate([pt, pt + n], axis=1):
                intrsct_idx = line_polyline_intersections(line[np.newaxis, ...], surf)
                intrsct_pts = polylines_internal_points(
                    intrsct_idx[..., np.newaxis], surf
                )
                vec = intrsct_pts - line[0]
                # Find intersection which is on external direction
                valid = np.dot(vec, line[1] - line[0]) >= 0
                if not np.any(valid):
                    # Intersection not found due to pixel accuracy error!
                    # Fallback to closest point.
                    line_start = line[0].reshape(1, 1, -1)
                    (idx,) = closest_in_polylines(line_start, surf).reshape(-1)
                    surf_reg_idx.append(idx)
                    continue
                dist = np.linalg.norm(
                    intrsct_pts[valid] - line[np.newaxis, :1, :], axis=-1
                )
                (idx,) = intrsct_idx.reshape(-1, 1, 1)[valid][np.argmin(dist)]
                surf_reg_idx.append(idx)

            _, sreg, _ = split_polyline(np.array(surf_reg_idx)[..., np.newaxis], surf)
            surf_reg.append(sreg.transpose(1, 0, 2))

        return intf_reg, surf_reg


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
    - AverageThickness_{Global, Left, Bottom, Right}: Average thickness of the
      coating layer.
    - Roughness_{Global, Left, Bottom, Right}: Roughness of the coating layer.
    - MaxThickness_{Left, Bottom, Right}: Number of the pixels for the maximum
      thickness on each region.

    The following data are the metadata for the analysis.

    - MatchError: Template matching error between 0 to 1. 0 means perfect match.
    - ChipWidth: Number of the pixels between lower vertices of the substrate.

    """

    LayerLength_Left: np.float64
    LayerLength_Right: np.float64

    AverageThickness_Global: np.float64
    Roughness_Global: float
    AverageThickness_Left: np.float64
    Roughness_Left: float
    AverageThickness_Bottom: np.float64
    Roughness_Bottom: float
    AverageThickness_Right: np.float64
    Roughness_Right: float

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
       ...     RoughnessMeasure="SSFD",
       ...     RoughnessSamples=100,
       ... )
       >>> param = data_converter.structure(param_val, RectLayerShape.Parameters)
       >>> coat = RectLayerShape(coat_img, subst, param)
       >>> plt.imshow(coat.draw()) #doctest: +SKIP

    """

    __slots__ = (
        "_uniform_layer",
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
            _, bl, br, _ = polylines_internal_points(
                self.substrate.vertices(), self.substrate.contour().transpose(1, 0, 2)
            )
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
        if not hasattr(self, "_uniform_layer"):
            intf_idx = self.enclosing_interface()
            if intf_idx.size == 0:
                intf = np.empty((0, 1, 2), dtype=np.float64)
            else:
                subst_cnt = self.substrate.contour() + self.substrate_point()
                _, intf, _ = split_polyline(intf_idx, subst_cnt.transpose(1, 0, 2))
                intf = intf.transpose(1, 0, 2)

            surf_idx = self.enclosing_surface()
            if surf_idx.size == 0:
                surf = np.empty((0, 1, 2), dtype=np.float64)
            else:
                layer_cnt = self.contour()
                _, surf, _ = split_polyline(surf_idx, layer_cnt.transpose(1, 0, 2))
                surf = surf.transpose(1, 0, 2)

            L, ul = uniform_layer(intf.astype(np.float32), surf.astype(np.float32))
            self._uniform_layer = (L, ul)
        return self._uniform_layer

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

    def roughness(self) -> Tuple[float, npt.NDArray[np.float64]]:
        """Dimensional roughness value of the coating layer surface."""
        if not hasattr(self, "_roughness"):
            surf_idx = self.enclosing_surface()
            if surf_idx.size == 0:
                surf = np.empty((0, 1, 2), dtype=np.float64)
            else:
                layer_cnt = self.contour()
                _, surf, _ = split_polyline(surf_idx, layer_cnt.transpose(1, 0, 2))
                surf = surf.transpose(1, 0, 2)

            _, ul = self.uniform_layer()

            if surf.size == 0 or ul.size == 0:
                return (np.nan, np.empty((2, 0, 1, 2), dtype=np.float64))

            surf = equidistant_interpolate(surf, self.parameters.RoughnessSamples)
            ul = equidistant_interpolate(ul, self.parameters.RoughnessSamples)
            self._roughness = roughness(surf, ul, self.parameters.RoughnessMeasure)

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

        pair_opts = self.deco_options.roughness_pairs
        if pair_opts.thickness > 0:
            _, pairs = self.roughness()
            pairs = pairs.astype(np.int32)
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

        AVRGTHCK_G, _ = self.uniform_layer()
        ROUGH_G, _ = self.roughness()

        (intf_L, intf_B, intf_R), (surf_L, surf_B, surf_R) = self.layer_regions()
        AVRGTHCK_L, ul_L = uniform_layer(
            intf_L.astype(np.float32),
            surf_L.astype(np.float32),
        )
        ROUGH_L, _ = roughness(
            equidistant_interpolate(surf_L, self.parameters.RoughnessSamples),
            equidistant_interpolate(ul_L, self.parameters.RoughnessSamples),
            self.parameters.RoughnessMeasure,
        )
        AVRGTHCK_B, ul_B = uniform_layer(
            intf_B.astype(np.float32),
            surf_B.astype(np.float32),
        )
        ROUGH_B, _ = roughness(
            equidistant_interpolate(surf_B, self.parameters.RoughnessSamples),
            equidistant_interpolate(ul_B, self.parameters.RoughnessSamples),
            self.parameters.RoughnessMeasure,
        )
        AVRGTHCK_R, ul_R = uniform_layer(
            intf_R.astype(np.float32),
            surf_R.astype(np.float32),
        )
        ROUGH_R, _ = roughness(
            equidistant_interpolate(surf_R, self.parameters.RoughnessSamples),
            equidistant_interpolate(ul_R, self.parameters.RoughnessSamples),
            self.parameters.RoughnessMeasure,
        )

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
            AVRGTHCK_G,
            ROUGH_G,
            AVRGTHCK_L,
            ROUGH_L,
            AVRGTHCK_B,
            ROUGH_B,
            AVRGTHCK_R,
            ROUGH_R,
            THCK_L,
            THCK_B,
            THCK_R,
            ERR,
            CHIPWIDTH,
        )


def uniform_layer(
    interface: npt.NDArray[np.float32], surface: npt.NDArray[np.float32]
) -> Tuple[np.float64, npt.NDArray[np.float64]]:
    """
    Return the information of uniform layer from the interface and the surface.

    Parameters
    ----------
    interface, surface: ndarray
        Coordinates of the polyline vertices for the solid-liquid interface and
        liquid-gas surface. The shape must be `(N, 1, D)`, where `N` is the
        number of vertices and `D` is the dimension.

    Returns
    -------
    L: float64
        Thickness of the uniform layer.
    uniform_layer: ndarray
        Coordinates polyline vertices for the uniform layer.

    Notes
    -----
    Uniform layer is defined as "what the layer would be if the liquid was
    uniformly distributed". Mathematically, it is a parallel line of the
    interface having same area to the layer.
    """
    if interface.size == 0 or surface.size == 0:
        return (np.float64(0), np.empty((0, 1, 2), dtype=np.float64))

    S = cv2.contourArea(np.concatenate([interface, np.flip(surface, axis=0)]))
    (L,) = root(lambda t: polyline_parallel_area(interface, t) - S, 0).x
    if L == 0:
        return (np.float64(0), np.empty((0, 1, 2), dtype=np.float64))

    normal = np.dot(np.gradient(interface, axis=0), ROTATION_MATRIX)
    n = normal / np.linalg.norm(normal, axis=-1)[..., np.newaxis]
    return (L, interface + L * n)


def roughness(
    surface: npt.NDArray, uniform_layer: npt.NDArray, measure: DistanceMeasure
) -> Tuple[float, npt.NDArray[np.float64]]:
    """
    Calculate the roughness of arbitrary curve.

    Parameters
    ----------
    surface, uniform_layer: ndarray
        Coordinates of the polyline vertices for the liquid-gas surface and the
        uniform layer that the surface is compared to. The shape must be
        `(N, 1, D)` where `N` is the number of vertices and `D` is the dimension.
    measure: DistanceMeasure
        Type of the measure of similarity between two curves.

    Returns
    -------
    roughness: float
        Roughness value of *surface*.
    pairs: ndarray
        Coordinates of Frechet pairs between *surface* and *uniform_layer*.
        The shape is `(2, P, 1, D)` where `P` is the number of pairs.
        The first axis represents the points on *surface* and *uniform_layer*,
        respectively.

    """
    if surface.size == 0 or uniform_layer.size == 0:
        return np.nan, np.empty((2, 0, 1, surface.shape[-1]), dtype=np.float64)

    if measure == DistanceMeasure.DFD:
        ca = dfd(np.squeeze(surface, axis=1), np.squeeze(uniform_layer, axis=1))
        path = dfd_pair(ca)
        roughness = ca[-1, -1]
    elif measure == DistanceMeasure.SFD:
        ca = sfd(np.squeeze(surface, axis=1), np.squeeze(uniform_layer, axis=1))
        path = sfd_path(ca)
        roughness = ca[-1, -1] / len(path)
    elif measure == DistanceMeasure.SSFD:
        ca = ssfd(np.squeeze(surface, axis=1), np.squeeze(uniform_layer, axis=1))
        path = ssfd_path(ca)
        roughness = np.sqrt(ca[-1, -1] / len(path))
    else:
        raise TypeError(f"Unknown measure: {measure}")

    pairs = np.stack([surface[path[..., 0]], uniform_layer[path[..., 1]]])
    return (float(roughness), pairs)
