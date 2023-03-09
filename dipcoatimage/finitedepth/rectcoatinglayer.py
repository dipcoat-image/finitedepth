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
from typing import TypeVar, Type, Tuple, List, Optional
from .rectsubstrate import RectSubstrate
from .coatinglayer import (
    images_XOR,
    CoatingLayerError,
    CoatingLayerBase,
)
from .util import (
    DataclassProtocol,
    BinaryImageDrawMode,
    MorphologyClosingParameters,
    SubstrateSubtractionMode,
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
    "RectLayerShapeParameters",
    "RectLayerShapeDrawOptions",
    "RectLayerShapeDecoOptions",
    "RectLayerShapeData",
    "RectLayerShape",
    "get_extended_line",
]


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


ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)
DecoOptionsType = TypeVar("DecoOptionsType", bound=DataclassProtocol)
DataType = TypeVar("DataType", bound=DataclassProtocol)


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
        x0, y0 = self.substrate_point()
        vertex_points = np.stack(
            [p for p in self.substrate.vertex_points() if p.size > 0]
        )
        v_y, v_x = vertex_points.transpose(2, 0, 1)

        top = y0 + np.max(v_y).astype(int)
        bot, _ = self.binary_image().shape
        if top > bot:
            # substrate is located outside of the frame
            return False

        left = x0 + np.min(v_x).astype(int)
        right = x0 + np.max(v_x).astype(int)

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
            mask = cv2.bitwise_not(self.extract_layer()).astype(bool)
            points = np.flip(np.stack(np.where(mask)), axis=0).T

            p0 = self.substrate_point()
            tl, bl, br, tr = self.substrate.vertex_points()
            A = p0 + tl
            B = p0 + bl
            C = p0 + br
            D = p0 + tr
            M1, M2 = (A + D) / 2, (B + C) / 2
            M1M2 = M2 - M1

            lefthalf = np.cross(M1M2, points - M1, axis=1) >= 0
            leftwall = np.cross(B - A, points - A, axis=1) >= 0
            rightwall = np.cross(D - C, points - C, axis=1) >= 0
            bottom = np.cross(C - B, points - B, axis=1) >= 0

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
class RectLayerShapeParameters:
    """Analysis parameters for :class:`RectLayerShape` instance."""

    MorphologyClosing: MorphologyClosingParameters
    ReconstructRadius: int


@dataclasses.dataclass
class RectLayerShapeDrawOptions:
    """Drawing options for :class:`RectLayerShape` instance."""

    draw_mode: BinaryImageDrawMode = BinaryImageDrawMode.BINARY
    subtract_mode: SubstrateSubtractionMode = SubstrateSubtractionMode.FULL


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

    - Score: Template matching score between 0 to 1. 0 means perfect match.
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

    Score: float
    ChipWidth: np.float32


ROTATION_MATRIX = np.array([[0, -1], [1, 0]])


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

       >>> from dipcoatimage.finitedepth import (HoughLinesParameters,
       ...     RectSubstrate)
       >>> hparams = HoughLinesParameters(1, 0.01, 100)
       >>> params = RectSubstrate.Parameters(hparams)
       >>> subst = RectSubstrate(ref, parameters=params)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    Construct :class:`RectLayerShape` from substrate class. :meth:`analyze`
    returns the number of pixels in coating area region.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import (RectLayerShape,
       ...     MorphologyClosingParameters)
       >>> coat_path = get_samples_path("coat3.png")
       >>> coat_img = cv2.cvtColor(cv2.imread(coat_path), cv2.COLOR_BGR2RGB)
       >>> mparams = MorphologyClosingParameters((1, 1))
       >>> params = RectLayerShape.Parameters(mparams, 50)
       >>> coat = RectLayerShape(coat_img, subst, params)
       >>> plt.imshow(coat.draw()) #doctest: +SKIP

    """

    __slots__ = (
        "_contactline_points",
        "_refined_layer",
        "_layer_contours",
        "_layer_area",
        "_uniform_layer",
        "_thickness_points",
    )

    Parameters = RectLayerShapeParameters
    DrawOptions = RectLayerShapeDrawOptions
    DecoOptions = RectLayerShapeDecoOptions
    Data = RectLayerShapeData

    DrawMode: TypeAlias = BinaryImageDrawMode
    SubtractMode: TypeAlias = SubstrateSubtractionMode

    def examine(self) -> Optional[CoatingLayerError]:
        if not self.capbridge_broken():
            return CoatingLayerError("Capillary bridge is not broken.")
        return None

    def contactline_points(self) -> npt.NDArray[np.int64]:
        """
        Get the coordinates of the contact line points of the layer.

        Return value as ``(left x, left y, right x, right y)``.
        """
        if not hasattr(self, "_contactline_points"):
            # perform closing to remove error pixels
            img = self.extract_layer()
            closingParams = self.parameters.MorphologyClosing
            kernel = np.ones(closingParams.kernelSize)
            img_closed = cv2.morphologyEx(
                img,
                cv2.MORPH_CLOSE,
                kernel,
                anchor=closingParams.anchor,
                iterations=closingParams.iterations,
            )

            # closed image may still have error pixels. at least we have to
            # remove the errors that are disconnected to the layer.
            # we identify the layer pixels as the connected components that are
            # close to the corners.
            p0 = self.substrate_point()
            _, bl, br, _ = self.substrate.vertex_points()
            B = p0 + bl
            C = p0 + br
            _, labels = cv2.connectedComponents(cv2.bitwise_not(img_closed))

            dist_thres = self.parameters.ReconstructRadius
            points = np.flip(np.stack(np.where(labels)), axis=0).T
            left_dist = np.linalg.norm(points - B, axis=1)
            right_dist = np.linalg.norm(points - C, axis=1)
            near_corner = (left_dist < dist_thres) | (right_dist < dist_thres)
            row, col = points[near_corner].T
            layer_comps = np.unique(labels[col, row])
            mask = np.zeros(labels.shape, dtype=bool)
            for i in layer_comps:
                mask[np.where(labels == i)] = True

            # get contact line points
            layer_label = self.label_layer().copy()
            layer_label[~mask] = self.Region.BACKGROUND
            left = (layer_label & self.Region.LEFTHALF).astype(bool)
            left_row, left_col = np.where(left)
            left_points = np.stack([left_col, left_row], axis=1)
            if left_points.size != 0:
                left_cp = left_points[np.argmin(left_points, axis=0)[1]]
            else:
                left_cp = B.astype(np.int64).flatten()
            right = (layer_label & ~left).astype(bool)
            right_row, right_col = np.where(right)
            right_points = np.stack([right_col, right_row], axis=1)
            if right_points.size != 0:
                right_cp = right_points[np.argmin(right_points, axis=0)[1]]
            else:
                right_cp = C.astype(np.int64).flatten()

            self._contactline_points = np.stack([left_cp, right_cp])

        return self._contactline_points

    def refine_layer(self) -> npt.NDArray[np.uint8]:
        """Get the refined coating layer image without error pixels."""
        if not hasattr(self, "_refined_layer"):
            layer_img = self.extract_layer().copy()
            h, w = layer_img.shape[:2]
            p1, p2 = self.contactline_points()
            ext_p1, ext_p2 = get_extended_line((h, w), p1, p2)
            pts = np.array([(0, 0), ext_p1, ext_p2, (w, 0)])
            # remove every pixels above the contact line
            cv2.fillPoly(layer_img, [pts], 255)  # faster than np.cross
            layer_img = cv2.bitwise_not(layer_img)

            self._refined_layer = layer_img

        return self._refined_layer

    def layer_contours(self) -> List[npt.NDArray[np.int32]]:
        if not hasattr(self, "_layer_contours"):
            contours, _ = cv2.findContours(
                self.refine_layer(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            self._layer_contours = list(contours)
        return self._layer_contours

    def thickness_points(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_thickness_points"):
            contours = self.layer_contours()
            if not contours:
                cnt_points = np.empty((0, 1, 2), dtype=np.int32)
            else:
                cnt_points = np.concatenate(contours, axis=0)
            cnt_y, cnt_x = cnt_points.transpose(2, 0, 1)

            cnt_labels = self.label_layer()[cnt_x, cnt_y]
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
                t = np.dot(Ap, AB.T) / np.dot(AB, AB.T)
                AC = t * AB
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
                left_p = np.concatenate([p, p], axis=0)
            else:
                left_p = find_thickest(cnt_left, A, B)

            cnt_bottom = cnt_points[on_layer & is_bottom]
            if cnt_bottom.size == 0:
                p = B / 2 + C / 2
                bottom_p = np.concatenate([p, p], axis=0)
            else:
                bottom_p = find_thickest(cnt_bottom, B, C)

            cnt_right = cnt_points[on_layer & is_right & ~is_bottom]
            if cnt_right.size == 0:
                p = C / 2 + D / 2
                right_p = np.concatenate([p, p], axis=0)
            else:
                right_p = find_thickest(cnt_right, C, D)

            self._thickness_points = np.stack([left_p, bottom_p, right_p])

        return self._thickness_points

    def layer_area(self) -> int:
        """Return the number of pixels in coating layer region."""
        if not hasattr(self, "_layer_area"):
            self._layer_area = np.count_nonzero(self.refine_layer())
        return self._layer_area

    def uniform_layer(self) -> Tuple[np.float64, npt.NDArray[np.float64]]:
        """
        Return thickness and points for uniform layer that satisfies
        :meth:`layer_area`.
        """
        if not hasattr(self, "_uniform_layer"):
            subst_point = self.substrate_point()
            hull, _ = self.substrate.edge_hull()
            hull = np.squeeze(hull + subst_point)

            # find projection points from contactline_points to hull and add to hull
            dh = np.diff(hull, axis=0)
            dh_dot_dh = np.sum(dh * dh, axis=-1)
            p1, p2 = self.contactline_points()

            def find_projection(p):
                h_p = p - hull[:-1, ...]
                dh_scale_p = np.sum(h_p * dh, axis=-1) / dh_dot_dh
                p_mask = (0 <= dh_scale_p) & (dh_scale_p <= 1)
                p_proj_origins = hull[:-1, ...][p_mask, ...]
                p_proj_vecs = dh_scale_p[p_mask][..., np.newaxis] * dh[p_mask, ...]
                p_proj_dists = np.linalg.norm(h_p[p_mask] - p_proj_vecs, axis=-1)
                if p_proj_dists.size != 0:
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

            if S != 0:
                L = findL(L0, L_NUM)
            else:
                L = np.float64(0)
            self._uniform_layer = (L, new_hull + L * n)

        return self._uniform_layer

    def roughness(self) -> np.float64:
        """Dimensional roughness value of the coating layer surface."""
        contours = self.layer_contours()
        if not contours:
            return np.float64(0)

        layer_points = np.concatenate(contours, axis=0)
        (layer,) = cv2.convexHull(layer_points).transpose(1, 0, 2)
        p, _ = self.contactline_points()
        i = np.argmin(np.linalg.norm(layer - p, axis=1)) + 1
        layer = np.roll(layer, -i, axis=0)

        L, uniform_layer = self.uniform_layer()

        NUM_POINTS = 1000

        def equidistant_interp(points):
            # https://stackoverflow.com/a/19122075
            vec = np.diff(points, axis=0)
            dist = np.linalg.norm(vec, axis=1)
            u = np.insert(np.cumsum(dist), 0, 0)
            t = np.linspace(0, u[-1], NUM_POINTS)
            y, x = points.T
            return np.stack([np.interp(t, u, y), np.interp(t, u, x)]).T

        l_interp = equidistant_interp(layer)
        ul_interp = equidistant_interp(uniform_layer)
        deviation = np.linalg.norm(l_interp - ul_interp, axis=1)

        return np.sqrt(np.trapz(deviation**2) / deviation.shape[0])

    def draw(self) -> npt.NDArray[np.uint8]:
        draw_mode = self.draw_options.draw_mode
        if draw_mode == self.DrawMode.ORIGINAL:
            image = self.image
        elif draw_mode == self.DrawMode.BINARY:
            image = self.binary_image()
        else:
            raise TypeError("Unrecognized draw mode: %s" % draw_mode)

        subtract_mode = self.draw_options.subtract_mode
        if subtract_mode == self.SubtractMode.NONE:
            pass
        elif subtract_mode == self.SubtractMode.TEMPLATE:
            x0, y0, x1, y1 = self.substrate.reference.templateROI
            tempImg = self.substrate.reference.binary_image()[y0:y1, x0:x1]
            _, (X0, Y0) = self.match_substrate()
            mask = images_XOR(
                ~self.binary_image().astype(bool), ~tempImg.astype(bool), (X0, Y0)
            )
            H, W = tempImg.shape
            X1, Y1 = X0 + W, Y0 + H
            image[Y0:Y1, X0:X1][mask] = 255
        elif subtract_mode == self.SubtractMode.SUBSTRATE:
            substImg = self.substrate.binary_image()
            x0, y0 = self.substrate_point()
            mask = images_XOR(
                ~self.binary_image().astype(bool), ~substImg.astype(bool), (x0, y0)
            )
            h, w = substImg.shape
            x1, y1 = x0 + w, y0 + h
            image[y0:y1, x0:x1][mask] = 255
        elif subtract_mode == self.SubtractMode.FULL:
            image = cv2.bitwise_not(self.refine_layer())
        else:
            raise TypeError("Unrecognized subtraction mode: %s" % subtract_mode)
        image = colorize(image)

        layer_opts = self.deco_options.layer
        if layer_opts.thickness != 0:
            image[self.refine_layer().astype(bool)] = (255, 255, 255)
            cv2.drawContours(
                image,
                self.layer_contours(),
                -1,
                dataclasses.astuple(layer_opts.color),
                layer_opts.thickness,
            )

        contactline_opts = self.deco_options.contact_line
        if contactline_opts.thickness > 0:
            p1, p2 = self.contactline_points()
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

        subst_p = self.substrate_point()
        _, bl, br, _ = self.substrate.vertex_points()
        bottomleft = subst_p + bl
        bottomright = subst_p + br
        p1, p2 = self.contactline_points()
        LEN_L = np.linalg.norm(bottomleft - p1)
        LEN_R = np.linalg.norm(bottomright - p2)

        tp_l, tp_b, tp_r = self.thickness_points()
        THCK_L = np.linalg.norm(np.diff(tp_l, axis=0))
        THCK_B = np.linalg.norm(np.diff(tp_b, axis=0))
        THCK_R = np.linalg.norm(np.diff(tp_r, axis=0))

        THCK_U, _ = self.uniform_layer()
        ROUGH = self.roughness()

        SCORE, _ = self.match_substrate()
        CHIPWIDTH = np.linalg.norm(bl - br)

        return (
            AREA,
            LEN_L,
            LEN_R,
            THCK_L,
            THCK_B,
            THCK_R,
            THCK_U,
            ROUGH,
            SCORE,
            CHIPWIDTH,
        )


def get_extended_line(
    frame_shape: Tuple[int, int], p1: npt.NDArray[np.int64], p2: npt.NDArray[np.int64]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # TODO: make it more elegant with matrix determinant and sorta things
    h, w = frame_shape
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and y1 == y2:
        raise ZeroDivisionError("Duplicate points: %s and %s" % (p1, p2))

    elif x1 != x2 and y1 != y2:
        candidates = (
            (int((x2 - x1) / (y2 - y1) * (0 - y1) + x1), 0),
            (int((x2 - x1) / (y2 - y1) * (h - y1) + x1), h),
            (0, int((y2 - y1) / (x2 - x1) * (0 - x1) + y1)),
            (w, int((y2 - y1) / (x2 - x1) * (w - x1) + y1)),
        )

        ret = []
        for x, y in set(candidates):
            if 0 <= x <= w and 0 <= y <= h:
                ret.append((x, y))
        ret.sort()

        ext_p1, ext_p2 = ret

    elif x1 == x2 and y1 != y2:
        ext_p1 = (x1, 0)
        ext_p2 = (x1, h)

    elif x1 != x2 and y1 == y2:
        ext_p1 = (0, y1)
        ext_p2 = (w, y1)

    return ext_p1, ext_p2
