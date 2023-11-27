"""Analyze coating layer over rectangular substrate.

This module defines :class:`RectCoatingLayerBase`, which is an
abstract base class for coating layer over rectangular substrate.
Also, its implementation :class:`RectLayerShape` is defined, which quantifies
several measures for the layer shape.
"""
import abc
import dataclasses
import enum
from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore
from scipy.optimize import root  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from shapely import LineString, offset_curve  # type: ignore

from .cache import attrcache
from .coatinglayer import (
    CoatingLayerBase,
    DataTypeVar,
    DecoOptTypeVar,
    DrawOptTypeVar,
    ParamTypeVar,
    SubtractionMode,
    images_XOR,
)
from .parameters import LineOptions, PatchOptions
from .rectsubstrate import RectSubstrate

__all__ = [
    "RectCoatingLayerBase",
    "DistanceMeasure",
    "RectLayerShapeParam",
    "PaintMode",
    "RectLayerShapeDrawOpt",
    "LinesOptions",
    "RectLayerShapeDecoOpt",
    "RectLayerShapeData",
    "RectLayerShape",
    "equidistant_interpolate",
    "parallel_curve",
    "acm",
    "owp",
]


ROTATION_MATRIX = np.array([[0, 1], [-1, 0]])


class RectCoatingLayerBase(
    CoatingLayerBase[
        RectSubstrate, ParamTypeVar, DrawOptTypeVar, DecoOptTypeVar, DataTypeVar
    ]
):
    """Abstract base class for coating layer over :class:`RectSubstrate`."""

    @attrcache("_layer_contours")
    def layer_contours(self) -> Tuple[npt.NDArray[np.int32], ...]:
        """Find contours of coating layer region.

        This method finds external contours of :meth:`CoatingLayerBase.extract_layer`.
        Each contour encloses each discrete region of coating layer.
        """
        layer_cnts, _ = cv2.findContours(
            self.extract_layer().astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        return tuple(layer_cnts)

    @attrcache("_interfaces")
    def interfaces(self) -> Tuple[npt.NDArray[np.int64], ...]:
        """Find solid-liquid interfaces.

        This method returns indices for :meth:`SubstrateBase.contour` where
        solid-liquid interfaces start and stop.

        A substrate can have contact with multiple discrete coating layer regions,
        and a single coating layer region can have multiple contacts to the substrate.
        The interfaces are detected by points in :meth:`SubstrateBase.contour`
        adjacent to any of the point in :meth:`layer_contours`.

        Returns:
            Tuple of arrays.
                - ``i``-th array represents ``i``-th coating layer region in
                  :meth:`layer_contours`.
                - Shape of the array is ``(N, 2)``, where ``N`` is the number of
                  contacts the coating layer region makes. Each column represents
                  starting and ending indices for the interface interval in
                  substrate contour.

        Note:
            Each interval describes continuous patch on the substrate contour covered
            by the layer. To acquire the interface points, slice :attr:`substrate`'s
            :meth:`~SubstrateBase.contour` with the indices.
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
        """Contour of the entire coated substrate.

        This method finds external contour of :meth:`CoatingLayerBase.coated_substrate`.
        Only one contour must exist.
        """
        (cnt,), _ = cv2.findContours(
            self.coated_substrate().astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        return cnt

    @abc.abstractmethod
    def surface(self) -> Tuple[np.int64, np.int64]:
        """Find liquid-gas interface of the coating layer.

        This method returns indices for :meth:`contour` where liquid-gas interface
        starts and stops.

        Here, the term "coating layer" has **conceptual meaning**. The interval
        defined by this method can include solid-gas interface of the exposed substrate,
        which is considered as coating layer with zero thickness.

        Concrete class can implement this method either dynamically or statically,
        depending on the interest of analysis.
        For example, suppose a coating layer was applied with a length of 1 mm on
        substrate. If a desired coating length was 2 mm, then this is a wetting failure
        and indicates "bad" coating. In this case, this method should statically return
        the desired range. On the other hand, if the wetting length is expected to vary,
        this method can dynamically return the range for wetted region.

        Returns:
            Starting and ending indices for the surface interval in coated substrate
            contour.

        Note:
            To acquire the surface points, slice :meth:`contour` with the indices.
        """

    def capbridge_broken(self) -> bool:
        """Check if capillary bridge is ruptured.

        As substrate is withdrawn from fluid bath, capillary bridge forms between the
        coating layer and bulk fluid, and ruptures.

        An image patch beneath the substrate location is inspected. If any row is
        all-background, the capillary bridge is considered to be broken.
        If the substrate region extends beyond the frame, the substrate is considered
        to be still immersed in the bath and capillary bridge not broken.

        Note:
            This method cannot distinguish uncoated substrate and coated substrate.
        """
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

    .. rubric:: **Members**

    - DTW: Dynamic time warping.
    - SDTW: Squared dynamic time warping.
    """

    DTW = "DTW"
    SDTW = "SDTW"


@dataclasses.dataclass(frozen=True)
class RectLayerShapeParam:
    """Analysis parameters for :class:`RectLayerShape`.

    Arguments:
        KernelSize: Size of the kernel for morphological operation.
            Kernel size must be zero or positive odd number.
        ReconstructRadius: Radius of the "safe zone" for noise removal.
            Draws imagniary circles having this radius on bottom corners of the
            substrate. Connected components not passing these circles are
            regarded as image artifacts.
        RoughnessMeasure: Measure to compute layer roughness.
    """

    KernelSize: Tuple[int, int]
    ReconstructRadius: int
    RoughnessMeasure: DistanceMeasure

    def __post_init__(self):
        """Check value of :attr:`KernelSize`."""
        if not all(i == 0 or (i > 0 and i % 2 == 1) for i in self.KernelSize):
            raise ValueError("Kernel size must be zero or odd.")


class PaintMode(enum.Enum):
    """Option to determine how the coating layer image is painted.

    .. rubric:: **Members**

    ORIGINAL: Show the original image.
    EMPTY: Show empty image. Only the layer will be drawn.
    """

    ORIGINAL = "ORIGINAL"
    EMPTY = "EMPTY"


@dataclasses.dataclass
class RectLayerShapeDrawOpt:
    """Drawing options for :class:`RectLayerShape`."""

    paint: PaintMode = PaintMode.ORIGINAL
    subtraction: SubtractionMode = SubtractionMode.NONE


@dataclasses.dataclass
class LinesOptions:
    """Parameters to draw lines in the image.

    Arguments:
        color: Color of the lines in RGB
        linewidth: Width of the line.
            If ``0``, lines are not drawn.
        step: Steps to jump the lines.
            `1` draws every line.
    """

    color: Tuple[int, int, int] = (0, 0, 0)
    linewidth: int = 1
    step: int = 1


@dataclasses.dataclass
class RectLayerShapeDecoOpt:
    """Decorating options for :class:`RectLayerShape`.

    Arguments:
        layer: Determine how the coating layer is painted.
        contact_line: Determine how the contact line is drawn.
        thickness: Determine how the regional maximum thicknesses are drawn.
        uniform_layer: Determine how the imaginary uniform layer is drawn.
        conformality, roughness: Determine how the optimal pairs are drawn.
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
class RectLayerShapeData:
    """Analysis data for :class:`RectLayerShape`.

    Arguments:
        LayerLength_Left, LayerLength_Right: Length of the layer on each wall.
        Conformality: Conformality of the coating layer.
        AverageThickness: Average thickness of the coating layer.
        Roughness: Roughness of the coating layer.
        MaxThickness_Left, MaxThickness_Bottom, MaxThickness_Right: Regional maximum
            thicknesses.
        MatchError: Template matching error between ``0`` to ``1``.
            ``0`` means perfect match.
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
        RectLayerShapeParam,
        RectLayerShapeDrawOpt,
        RectLayerShapeDecoOpt,
        RectLayerShapeData,
    ]
):
    """Analyze shape of the coating layer over :class:`RectSubstrate`.

    #. No static range is used for liquid-gas interface (:meth:`surface`).
    #. Heighest wetting points are interpreted as contact line.
    #. Coating layer is segmented into left, bottom, and right region.
       Maximum thickness for each region is analyzed.
    #. An imaginary uniform coating layer with the same cross-sectional area as the
       actual coating layer is calculated.
    #. DTW-based layer conformality and surface roughness are acquired.

    Arguments:
        image
        substrate
        parameters (RectLayerShapeParam, optional)
        draw_options (RectLayerShapeDrawOpt, optional)
        deco_options (RectLayerShapeDecoOpt, optional)
        tempmatch

    Examples:
        Construct substrate instance first.

        .. plot::
            :include-source:
            :context: reset

            >>> import cv2
            >>> from dipcoatimage.finitedepth import Reference, get_data_path
            >>> gray = cv2.imread(get_data_path("ref3.png"), cv2.IMREAD_GRAYSCALE)
            >>> _, im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            >>> tempROI = (10, 10, 1250, 200)
            >>> substROI = (100, 100, 1200, 500)
            >>> ref = Reference(im, tempROI, substROI)
            >>> from dipcoatimage.finitedepth import RectSubstrate
            >>> param = RectSubstrate.ParamType(Sigma=3.0, Rho=1.0, Theta=0.01)
            >>> subst = RectSubstrate(ref, param)

        Construct coating layer instance from target image and the substrate instance.

        .. plot::
            :include-source:
            :context: close-figs

            >>> from dipcoatimage.finitedepth import RectLayerShape
            >>> gray = cv2.imread(get_data_path("coat3.png"), cv2.IMREAD_GRAYSCALE)
            >>> _, im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            >>> param = RectLayerShape.ParamType(
            ...     KernelSize=(1, 1),
            ...     ReconstructRadius=50,
            ...     RoughnessMeasure=RectLayerShape.DistanceMeasure.SDTW,
            ... )
            >>> coat = RectLayerShape(im, subst, param)
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(coat.draw()) #doctest: +SKIP

        :attr:`draw_options` controls visualization of the substrate region.

        .. plot::
            :include-source:
            :context: close-figs

            >>> coat.draw_options.paint = coat.PaintMode.EMPTY
            >>> plt.imshow(coat.draw()) #doctest: +SKIP

        :attr:`deco_options` controls visualization of the coating layer region.

        .. plot::
            :include-source:
            :context: close-figs

            >>> coat.deco_options.roughness.linewidth = 0
            >>> plt.imshow(coat.draw()) #doctest: +SKIP
    """

    ParamType = RectLayerShapeParam
    """Assigned with :class:`RectLayerShapeParam`."""
    DrawOptType = RectLayerShapeDrawOpt
    """Assigned with :class:`RectLayerShapeParam`."""
    DecoOptType = RectLayerShapeDecoOpt
    """Assigned with :class:`RectLayerShapeDecoOpt`."""
    DataType = RectLayerShapeData
    """Assigned with :class:`RectLayerShapeData`."""

    DistanceMeasure = DistanceMeasure
    """Assigned with :class:`DistanceMeasure`."""
    PaintMode = PaintMode
    """Assigned with :class:`PaintMode`."""
    SubtractionMode = SubtractionMode
    """Assigned with :class:`SubtractionMode`."""

    def verify(self):
        """Implement :meth:`CoatingLayerBase.verify`.

        Check if capillary bridge is broken.
        """
        if not self.capbridge_broken():
            raise ValueError("Capillary bridge is not broken.")

    @attrcache("_extracted_layer")
    def extract_layer(self) -> npt.NDArray[np.bool_]:
        """Extend :meth:`CoatingLayerBase.extract_layer`."""
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

    @attrcache("_surface")
    def surface(self) -> Tuple[np.int64, np.int64]:
        """Implement :meth:`RectCoatingLayerBase.surface`.

        The coating layer is dynamically defined, ranging between the heighest wetting
        points on substrate walls.
        """
        if not self.interfaces():
            return (np.int64(-1), np.int64(0))

        (i0, i1) = np.sort(np.concatenate(self.interfaces()).flatten())[[0, -1]]
        subst_cnt = self.substrate.contour() + self.substrate_point()
        endpoints = subst_cnt[[i0, i1]]

        vec = self.contour() - endpoints.transpose(1, 0, 2)
        (I0, I1) = np.argmin(np.linalg.norm(vec, axis=-1), axis=0)
        return (I0, I1 + 1)

    @attrcache("_uniform_layer")
    def uniform_layer(self) -> Tuple[np.float64, npt.NDArray[np.float64]]:
        """Return thickness and points for imaginary uniform layer.

        Uniform layer is a parallel curve of substrate surface which has the same
        cross-sectional area as the actual coating layer.

        Returns:
            Thickness and points of the uniform layer.

        References:
            * https://en.wikipedia.org/wiki/Parallel_curve

        Note:
            This method returns polyline vertices as the uniform layer.
        """
        if not self.interfaces():
            return (np.float64(0), np.empty((0, 1, 2), np.float64))

        (i0, i1) = np.sort(np.concatenate(self.interfaces()).flatten())[[0, -1]]
        subst_cnt = self.substrate.contour() + self.substrate_point()
        s = subst_cnt[i0:i1]

        A = np.count_nonzero(self.extract_layer())
        (t,) = root(
            lambda x: cv2.contourArea(
                np.concatenate([s, np.flip(parallel_curve(s, x[0]), axis=0)]).astype(
                    np.float32
                )
            )
            - A,
            [0],
        ).x
        return (t, parallel_curve(s, t))

    @attrcache("_conformality")
    def conformality(self) -> Tuple[float, npt.NDArray[np.int32]]:
        """DTW-based conformality of the coating layer.

        Returns:
            Conformality and optimal pairs between substrate surface and layer surface.
        """
        if not self.interfaces():
            return (np.nan, np.empty((0, 2), dtype=np.int32))

        (i0, i1) = np.sort(np.concatenate(self.interfaces()).flatten())[[0, -1]]
        subst_cnt = self.substrate.contour() + self.substrate_point()
        intf = subst_cnt[i0:i1]

        I0, I1 = self.surface()
        surf = self.contour()[I0:I1]

        dist = cdist(np.squeeze(surf, axis=1), np.squeeze(intf, axis=1))
        mat = acm(dist)
        path = owp(mat)
        d = dist[path[:, 0], path[:, 1]]
        d_avrg = mat[-1, -1] / len(path)
        C = 1 - np.sum(np.abs(d - d_avrg)) / mat[-1, -1]
        return (float(C), path)

    @attrcache("_roughness")
    def roughness(self) -> Tuple[float, npt.NDArray[np.int32]]:
        """DTW-based surface roughness of the coating layer.

        Returns:
            Roughness and optimal pairs between layer surface and uniform layer.
        """
        I0, I1 = self.surface()
        surf = self.contour()[I0:I1]
        _, ul = self.uniform_layer()

        if surf.size == 0 or ul.size == 0:
            return (np.nan, np.empty((0, 2), dtype=np.int32))

        measure = self.parameters.RoughnessMeasure
        if measure == DistanceMeasure.DTW:
            ul_len = np.ceil(cv2.arcLength(ul.astype(np.float32), closed=False))
            ul = equidistant_interpolate(ul, int(ul_len))
            dist = cdist(np.squeeze(surf, axis=1), np.squeeze(ul, axis=1))
            mat = acm(dist)
            path = owp(mat)
            roughness = mat[-1, -1] / len(path)
        elif measure == DistanceMeasure.SDTW:
            ul_len = np.ceil(cv2.arcLength(ul.astype(np.float32), closed=False))
            ul = equidistant_interpolate(ul, int(ul_len))
            dist = cdist(np.squeeze(surf, axis=1), np.squeeze(ul, axis=1))
            mat = acm(dist**2)
            path = owp(mat)
            roughness = np.sqrt(mat[-1, -1] / len(path))
        else:
            raise TypeError(f"Unknown measure: {measure}")
        return (float(roughness), path)

    @attrcache("_max_thickness")
    def max_thickness(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Regional maximum thicknesses.

        Coating layer is segmented using :meth:`RectSubstrate.sideline_intersections`.
        Points on layer surface and sideline for maximum distance in each region are
        found.

        Returns:
            Tuple of two arrays.
                - The first array contains maximum thickness values on left, bottom, and
                  right region.
                  Value of ``0`` indicates no coating layer on that region.
                - The second array contains points on layer surface and substrate lines
                  for the maximum thickness.
                  Shape of the array is ``(3, 2, 2)``; 1st axis indicates left, bottom
                  and right region, 2nd axis indicates layer surface and substrate line,
                  and 3rd axis indicates ``(x, y)`` coordinates.
        """
        corners = self.substrate.sideline_intersections() + self.substrate_point()
        I0, I1 = self.surface()
        surface = self.contour()[I0:I1]
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
        """Implement :meth:`SubstrateBase.draw`.

        #. Draw the substrate with by :class:`PaintMode`.
        #. Display the template matching result with :class:`SubtractionMode`.
        #. Draw coating layer and contact line.
        #. If capillary bridge is broken, draw regional maximum thicknesses,
           uniform layer, conformality pairs and roughness pairs.
        """
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
            I0, I1 = self.surface()
            surf = self.contour()[I0:I1]
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
            I0, I1 = self.surface()
            surf = self.contour()[I0:I1]
            _, ul = self.uniform_layer()
            ul_len = np.ceil(cv2.arcLength(ul.astype(np.float32), closed=False))
            ul = equidistant_interpolate(ul, int(ul_len))
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
        """Implement :meth:`CoatingLayerBase.analyze`."""
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

        return self.DataType(
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
    """Interpolate points with equidistant new points.

    Arguments:
        points: Points that are interpolated.
            The shape must be ``(N, 1, D)`` where ``N`` is the number of points
            and ``D`` is the dimension.
        n: Number of new points.

    Returns:
        Interpolated points.
        - If ``N`` is positive number, the shape is ``(n, 1, D)``.
        - If ``N`` is zero, the shape is ``(n, 0, D)``.
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


def parallel_curve(curve: npt.NDArray, dist: float) -> npt.NDArray:
    """Return parallel curve of *curve* with offset distance *dist*.

    Arguments:
        curve: Vertices of a polyline.
            The shape is ``(V, 1, D)``, where ``V`` is the number of vertices and
            ``D`` is the dimension.
        dist: offset distance of the parallel curve.

    Returns:
        Round-joint parallel curve of shape ``(V, 1, D)``.
    """
    if dist == 0:
        return curve
    ret = offset_curve(LineString(np.squeeze(curve, axis=1)), dist, join_style="round")
    return np.array(ret.coords)[:, np.newaxis]


@njit(cache=True)
def acm(cm: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute accumulated cost matrix from local cost matrix.

    Arguments:
        cm: Local cost matrix.

    Returns:
        Accumulated cost matrix.
            The element at `[-1, -1]` is the total sum along the optimal path.
            If *cm* is empty, return value is an empty array.

    References:
        * Senin, Pavel. "Dynamic time warping algorithm review."
          Information and Computer Science Department University of Hawaii at
          Manoa Honolulu, USA 855.1-23 (2008): 40.
        * https://pypi.org/project/similaritymeasures/
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

    Arguments:
        acm: Accumulated cost matrix.

    Returns:
        Indices for the two series to get the optimal warping path.

    References:
        * Senin, Pavel. "Dynamic time warping algorithm review."
          Information and Computer Science Department University of Hawaii at
          Manoa Honolulu, USA 855.1-23 (2008): 40.
        * https://pypi.org/project/similaritymeasures/
    """
    p, q = acm.shape
    if p == 0 or q == 0:
        return np.empty((0, 2), dtype=np.int32)

    path = np.zeros((p + q - 1, 2), dtype=np.int32)
    path_len = np.int32(0)

    i, j = p - 1, q - 1
    path[path_len] = [i, j]
    path_len += np.int32(1)

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
        path_len += np.int32(1)

    return path[-(len(path) - path_len + 1) :: -1, :]
