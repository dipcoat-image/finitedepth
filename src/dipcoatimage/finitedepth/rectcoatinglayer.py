"""
:mod:`dipcoatimage.finitedepth.rectcoatinglayer` provides class to analyze
the coating layer over rectangular substrate.

Base class
==========

.. autoclass:: RectCoatingLayerBase
   :members:

Implementation
==============

.. autoclass:: RectLayerShape
   :members:

"""

import cv2
import numpy as np
import numpy.typing as npt
from scipy.optimize import root  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from .rectsubstrate import RectSubstrate
from .coatinglayer import CoatingLayerError, CoatingLayerBase, images_XOR
from .rectcoatinglayer_param import (
    DistanceMeasure,
    Parameters,
    DrawOptions,
    PaintMode,
    SubtractionMode,
    DecoOptions,
    Data,
)
from .util.imgprocess import colorize
from .util.dtw import acm, owp
from .util.geometry import polyline_parallel_area, equidistant_interpolate
from typing import TypeVar, Type, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "RectCoatingLayerBase",
    "RectLayerShape",
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

    def layer_contours(self) -> Tuple[npt.NDArray[np.int32], ...]:
        """
        Return contours of :meth:`extract_layer`.
        """
        if not hasattr(self, "_layer_contours"):
            layer_cnts, _ = cv2.findContours(
                self.extract_layer().astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            self._layer_contours = tuple(layer_cnts)
        return self._layer_contours

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
            (i0, i1) = np.sort(np.concatenate(self.interfaces()).flatten())[[0, -1]]
            subst_cnt = self.substrate.contour() + self.substrate_point()
            endpoints = subst_cnt[[i0, i1]]

            vec = self.contour() - endpoints.transpose(1, 0, 2)
            self._surface_indices = np.argmin(np.linalg.norm(vec, axis=-1), axis=0)
        (I0, I1) = self._surface_indices
        return self.contour()[I0 : I1 + 1]

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


class RectLayerShape(
    RectCoatingLayerBase[
        Parameters,
        DrawOptions,
        DecoOptions,
        Data,
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
       >>> from dipcoatimage.finitedepth import Reference, get_data_path
       >>> ref_path = get_data_path("ref3.png")
       >>> img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
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

    Construct :class:`RectLayerShape` from substrate class. :meth:`analyze`
    returns the number of pixels in coating area region.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import RectLayerShape
       >>> coat_path = get_data_path("coat3.png")
       >>> coat_img = cv2.cvtColor(cv2.imread(coat_path), cv2.COLOR_BGR2RGB)
       >>> param = RectLayerShape.Parameters(
       ...     KernelSize=(1, 1),
       ...     ReconstructRadius=50,
       ...     RoughnessMeasure=RectLayerShape.DistanceMeasure.SDTW,
       ... )
       >>> coat = RectLayerShape(coat_img, subst, param)
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

    def examine(self) -> Optional[CoatingLayerError]:
        ksize = self.parameters.KernelSize
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

            self._extracted_layer = layer_mask
        return self._extracted_layer

    def uniform_layer(self) -> Tuple[np.float64, npt.NDArray[np.float64]]:
        """Return thickness and points for uniform layer."""
        if not self.interfaces():
            return (np.float64(0), np.empty((0, 1, 2), np.float64))

        if not hasattr(self, "_uniform_layer"):
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

            self._uniform_layer = (t, ul)
        return self._uniform_layer

    def conformality(self) -> Tuple[float, npt.NDArray[np.int32]]:
        """Conformality of the coating layer and its optimal path."""
        if not self.interfaces():
            return (np.nan, np.empty((0, 2), dtype=np.int32))

        if not hasattr(self, "_conformality"):
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
            self._conformality = (float(C), path)

        return self._conformality

    def roughness(self) -> Tuple[float, npt.NDArray[np.int32]]:
        """Roughness of the coating layer and its optimal path."""
        surf = self.surface()
        _, ul = self.uniform_layer()

        if surf.size == 0 or ul.size == 0:
            return (np.nan, np.empty((0, 2), dtype=np.int32))

        if not hasattr(self, "_roughness"):
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
            self._roughness = (float(roughness), path)

        return self._roughness

    def max_thickness(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Return the maximum thickness on each side (left, bottom, right) and their
        points.
        """
        if not hasattr(self, "_max_thickness"):
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
            self._max_thickness = (np.array(thicknesses), np.array(points))
        return self._max_thickness

    def draw(self) -> npt.NDArray[np.uint8]:
        paint = self.draw_options.paint
        if paint == self.PaintMode.ORIGINAL:
            image = self.image.copy()
        elif paint == self.PaintMode.BINARY:
            image = self.binary_image().copy()
        elif paint == self.PaintMode.EMPTY:
            image = np.full(self.image.shape, 255, dtype=np.uint8)
        else:
            raise TypeError("Unrecognized paint mode: %s" % paint)
        image = colorize(image)

        subtraction = self.draw_options.subtraction
        if subtraction in [
            self.SubtractionMode.TEMPLATE,
            self.SubtractionMode.FULL,
        ]:
            x0, y0, x1, y1 = self.substrate.reference.templateROI
            tempImg = self.substrate.reference.binary_image()[y0:y1, x0:x1]
            h, w = tempImg.shape[:2]
            _, (X0, Y0) = self.match_substrate()
            binImg = self.binary_image()[Y0 : Y0 + h, X0 : X0 + w]
            mask = images_XOR(~binImg.astype(bool), ~tempImg.astype(bool))
            image[Y0 : Y0 + h, X0 : X0 + w][~mask] = 255
        if subtraction in [
            self.SubtractionMode.SUBSTRATE,
            self.SubtractionMode.FULL,
        ]:
            x0, y0, x1, y1 = self.substrate.reference.substrateROI
            substImg = self.substrate.reference.binary_image()[y0:y1, x0:x1]
            h, w = substImg.shape[:2]
            X0, Y0 = self.substrate_point()
            binImg = self.binary_image()[Y0 : Y0 + h, X0 : X0 + w]
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

    def analyze_layer(self):
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

        ERR, _ = self.match_substrate()

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
        )
