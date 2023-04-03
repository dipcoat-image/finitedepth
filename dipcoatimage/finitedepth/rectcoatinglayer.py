"""
:mod:`dipcoatimage.finitedepth.rectcoatinglayer` provides class to analyze
the coating layer over rectangular substrate.

Base class
==========

.. autoclass:: LayerRegionFlag
   :members:

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
    Color,
    colorize,
)

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias


__all__ = [
    "LayerRegionFlag",
    "RectCoatingLayerBase",
    "MorphologyClosingParameters",
    "RectLayerShapeParameters",
    "RectLayerShapeDrawOptions",
    "RectLayerShapeDecoOptions",
    "RectLayerShapeData",
    "RectLayerShape",
    "get_extended_line",
]


ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)
DecoOptionsType = TypeVar("DecoOptionsType", bound=DataclassProtocol)
DataType = TypeVar("DataType", bound=DataclassProtocol)


class LayerRegionFlag(enum.IntFlag):
    """
    Label to classify the coating layer pixels by their regions.

    - BACKGROUND: Null value for pixels that are not the coating layer.
    - LAYER: Denotes that the pixel is in the coating layer.
    - LEFTHALF: Left-hand side w.r.t. the vertical center line.
    - LEFTWALL: Left-hand side w.r.t. the left-hand side substrate wall.
    - RIGHTWALL: Right-hand side w.r.t. the right-hand side substrate wall.
    - BOTTOM: Under the substrate bottom surface.

    """

    BACKGROUND = 0
    LAYER = 1
    LEFTHALF = 2
    LEFTWALL = 4
    RIGHTWALL = 8
    BOTTOM = 16


class RectCoatingLayerBase(
    CoatingLayerBase[
        RectSubstrate, ParametersType, DrawOptionsType, DecoOptionsType, DataType
    ]
):
    """
    Abstract base class for coating layer over rectangular substrate.

    :class:`RectCoatingLayerBase` is capable of classifying the coating layer
    pixels by their location relative to the substrate. To get the classification
    map, use :meth:`label_layer`.

    """

    __slots__ = ("_labelled_layer",)

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    DecoOptions: Type[DecoOptionsType]
    Data: Type[DataType]

    Region: TypeAlias = LayerRegionFlag

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

    def label_layer(self) -> npt.NDArray[np.uint8]:
        """
        Return the classification map of the pixels.

        Pixels are labelled with :class:`LayerRegionFlag` by their location
        relative to the substrate. The values can be combined to denote the pixel
        in the corner, i.e. ``LEFT | BOTTOM`` for the lower left region.

        """
        if not hasattr(self, "_labelled_layer"):
            mask = self.extract_layer()
            points = np.flip(np.stack(np.where(mask)), axis=0).T

            p0 = self.substrate_point()
            tl, bl, br, tr = self.substrate.vertex_points()
            A = p0 + tl
            B = p0 + bl
            C = p0 + br
            D = p0 + tr
            M1, M2 = (A + D) / 2, (B + C) / 2
            M1M2 = M2 - M1

            # cv2.fillPoly is only marginally faster than this (~0.5 ms) so
            # just use np.cross for the sake of code quality.
            lefthalf = np.cross(M1M2, points - M1) >= 0
            leftwall = np.cross(B - A, points - A) >= 0
            rightwall = np.cross(D - C, points - C) >= 0
            bottom = np.cross(C - B, points - B) >= 0

            h, w = self.image.shape[:2]
            ret = np.full((h, w), self.Region.BACKGROUND)

            _x, _y = points.T
            ret[_y, _x] |= self.Region.LAYER
            _x, _y = points[lefthalf].T
            ret[_y, _x] |= self.Region.LEFTHALF
            _x, _y = points[leftwall].T
            ret[_y, _x] |= self.Region.LEFTWALL
            _x, _y = points[rightwall].T
            ret[_y, _x] |= self.Region.RIGHTWALL
            _x, _y = points[bottom].T
            ret[_y, _x] |= self.Region.BOTTOM

            self._labelled_layer = ret

        return self._labelled_layer


@dataclasses.dataclass(frozen=True)
class MorphologyClosingParameters:
    kernelSize: Tuple[int, int]
    anchor: Tuple[int, int] = (-1, -1)
    iterations: int = 1


@dataclasses.dataclass(frozen=True)
class RectLayerShapeParameters:
    """Analysis parameters for :class:`RectLayerShape` instance."""

    MorphologyClosing: MorphologyClosingParameters
    ReconstructRadius: int


@dataclasses.dataclass
class RectLayerShapeDrawOptions:
    """Drawing options for :class:`RectLayerShape` instance."""

    background: BackgroundDrawMode = BackgroundDrawMode.ORIGINAL
    subtract_mode: SubtractionDrawMode = SubtractionDrawMode.NONE


@dataclasses.dataclass
class RectLayerShapeDecoOptions:
    """Decorating options for :class:`RectLayerShape` instance."""

    layer: FeatureDrawingOptions = FeatureDrawingOptions(thickness=1)
    contact_line: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(0, 255, 0), thickness=1
    )
    thickness_lines: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(0, 0, 255), thickness=1
    )
    uniform_layer: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(255, 0, 0), thickness=0
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
       >>> param_val = dict(Sigma=3.0, Theta=0.01)
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
       ...     ReconstructRadius=50
       ... )
       >>> param = data_converter.structure(param_val, RectLayerShape.Parameters)
       >>> coat = RectLayerShape(coat_img, subst, param)
       >>> plt.imshow(coat.draw()) #doctest: +SKIP

    """

    __slots__ = (
        "_layer_area",
        "_thickness_points",
        "_uniform_layer",
    )

    Parameters = RectLayerShapeParameters
    DrawOptions = RectLayerShapeDrawOptions
    DecoOptions = RectLayerShapeDecoOptions
    Data = RectLayerShapeData

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

    def thickness_points(self) -> npt.NDArray[np.float64]:
        # TODO: make as tuple of points
        if not hasattr(self, "_thickness_points"):
            # TODO: use surface and hausdorff distance (maybe scipy)
            contours = self.layer_contours()
            if not contours:
                cnt_points = np.empty((0, 1, 2), dtype=np.int32)
            else:
                cnt_points = np.concatenate(contours, axis=0)
            cnt_x, cnt_y = cnt_points.transpose(2, 0, 1)

            cnt_labels = self.label_layer()[cnt_y, cnt_x]
            on_layer = (cnt_labels & self.Region.LAYER).astype(bool)
            is_left = (cnt_labels & self.Region.LEFTWALL).astype(bool)
            is_bottom = (cnt_labels & self.Region.BOTTOM).astype(bool)
            is_right = (cnt_labels & self.Region.RIGHTWALL).astype(bool)

            p0 = self.substrate_point()
            tl, bl, br, tr = self.substrate.vertex_points()
            A = p0 + tl
            B = p0 + bl
            C = p0 + br
            D = p0 + tr

            def find_thickest(points, A, B):
                Ap = points - A
                AB = B - A
                t = np.dot(Ap, AB) / np.dot(AB, AB)
                AC = np.tensordot(t, AB, axes=0)
                dists = np.linalg.norm(Ap - AC, axis=1)
                mask = dists == np.max(dists)
                pts = np.stack(
                    [
                        np.mean(points[mask], axis=0),
                        np.mean((A + AC)[mask], axis=0),
                    ]
                )
                return pts

            cnt_left = cnt_points[on_layer & is_left & ~is_bottom]
            if cnt_left.size == 0:
                p = A / 2 + B / 2
                left_p = np.stack([p, p])
            else:
                left_p = find_thickest(cnt_left, A, B)

            cnt_bottom = cnt_points[on_layer & is_bottom]
            if cnt_bottom.size == 0:
                p = B / 2 + C / 2
                bottom_p = np.stack([p, p])
            else:
                bottom_p = find_thickest(cnt_bottom, B, C)

            cnt_right = cnt_points[on_layer & is_right & ~is_bottom]
            if cnt_right.size == 0:
                p = C / 2 + D / 2
                right_p = np.stack([p, p])
            else:
                right_p = find_thickest(cnt_right, C, D)

            self._thickness_points = np.stack([left_p, bottom_p, right_p])

        return self._thickness_points

    def uniform_layer(self) -> Tuple[np.float64, npt.NDArray[np.float64]]:
        """
        Return thickness and points for uniform layer that satisfies
        :meth:`layer_area`.
        """
        if not hasattr(self, "_uniform_layer"):
            # get contact line points
            sl_interfaces, _ = self.interfaces()
            if len(sl_interfaces) == 0:
                layer = np.empty((0, 1, 2), dtype=np.float64)
                self._uniform_layer = (np.float64(0), layer)
                return self._uniform_layer
            sl_points = np.concatenate(sl_interfaces)
            if len(sl_points) == 0:
                layer = np.empty((0, 1, 2), dtype=np.float64)
                self._uniform_layer = (np.float64(0), layer)
                return self._uniform_layer
            p1, p2 = sl_points[0], sl_points[-1]

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

    def roughness(self) -> np.float64:
        """Dimensional roughness value of the coating layer surface."""
        (surface,) = self.surface().transpose(1, 0, 2)
        _, uniform_layer = self.uniform_layer()
        if surface.size == 0 or uniform_layer.size == 0:
            return np.float64(np.nan)

        NUM_POINTS = 1000

        def equidistant_interp(points):
            # https://stackoverflow.com/a/19122075
            vec = np.diff(points, axis=0)
            dist = np.linalg.norm(vec, axis=1)
            u = np.insert(np.cumsum(dist), 0, 0)
            t = np.linspace(0, u[-1], NUM_POINTS)
            y, x = points.T
            return np.stack([np.interp(t, u, y), np.interp(t, u, x)]).T

        l_interp = equidistant_interp(surface)
        ul_interp = equidistant_interp(uniform_layer)
        deviation = np.linalg.norm(l_interp - ul_interp, axis=1)

        return np.sqrt(np.trapz(deviation**2) / deviation.shape[0])

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
            sl_interfaces, _ = self.interfaces()
            if len(sl_interfaces) != 0:
                sl_points = np.concatenate(sl_interfaces)
                if len(sl_points) != 0:
                    (p1,), (p2,) = sl_points[0], sl_points[-1]
                    cv2.line(
                        image,
                        p1,
                        p2,
                        dataclasses.astuple(contactline_opts.color),
                        contactline_opts.thickness,
                    )

        thicknesslines_opts = self.deco_options.thickness_lines
        if thicknesslines_opts.thickness > 0:
            points = self.thickness_points()
            for p1, p2 in points:
                cv2.line(
                    image,
                    p1.astype(np.int32),
                    p2.astype(np.int32),
                    dataclasses.astuple(thicknesslines_opts.color),
                    thicknesslines_opts.thickness,
                )

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
        sl_interfaces, _ = self.interfaces()
        if len(sl_interfaces) == 0:
            LEN_L = LEN_R = np.float64(0)
        else:
            sl_points = np.concatenate(sl_interfaces)
            if len(sl_points) == 0:
                LEN_L = LEN_R = np.float64(0)
            else:
                points = np.concatenate([sl_points[0], sl_points[-1]])
                Bp = points - B
                BC = C - B
                t = np.dot(Bp, BC) / np.dot(BC, BC)
                dists = np.linalg.norm(Bp - np.tensordot(t, BC, axes=0), axis=1)
                LEN_L, LEN_R = dists.astype(np.float64)

        tp_l, tp_b, tp_r = self.thickness_points()
        THCK_L = np.linalg.norm(np.diff(tp_l, axis=0))
        THCK_B = np.linalg.norm(np.diff(tp_b, axis=0))
        THCK_R = np.linalg.norm(np.diff(tp_r, axis=0))

        THCK_U, _ = self.uniform_layer()
        ROUGH = self.roughness()

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


def get_extended_line(
    frame_shape: Tuple[int, int], p1: npt.NDArray[np.int32], p2: npt.NDArray[np.int32]
) -> npt.NDArray[np.float64]:
    # TODO: make it more elegant with matrix determinant and sorta things
    h, w = frame_shape
    x1, y1 = p1
    dx, dy = p2 - p1

    points = []
    if dx != 0:
        points.append(np.array([0, dy / dx * (-x1) + y1], dtype=np.float64))
        points.append(np.array([w, dy / dx * (w - x1) + y1], dtype=np.float64))
    if dy != 0:
        points.append(np.array([dx / dy * (-y1) + x1, 0], dtype=np.float64))
        points.append(np.array([dx / dy * (h - y1) + x1, h], dtype=np.float64))
    return np.stack(points)
