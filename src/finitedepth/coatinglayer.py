"""Analyze coating layer and similarity measures [#senin]_ [#sim]_.

.. [#senin] Senin, P. (2008).
.. [#sim] https://pypi.org/project/similaritymeasures/
"""

import abc
import dataclasses
from typing import TYPE_CHECKING, Generic, TypeVar

import cv2
import numpy as np
import numpy.typing as npt
from curvesimilarities import dtw_acm, dtw_owp  # type: ignore
from scipy.interpolate import splev, splprep  # type: ignore
from scipy.optimize import root  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from shapely import LineString, get_coordinates, offset_curve  # type: ignore

from .cache import attrcache
from .substrate import RectSubstrate, SubstrateBase

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

__all__ = [
    "CoatingLayerBase",
    "CoatingLayer",
    "RectLayerShape",
    "images_XOR",
    "images_ANDXOR",
    "sample_polyline",
    "parallel_curve",
]


SubstTypeVar = TypeVar("SubstTypeVar", bound=SubstrateBase)
"""Type variable for the substrate type of :class:`CoatingLayerBase`."""
DataTypeVar = TypeVar("DataTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`CoatingLayerBase.DataType`."""


class CoatingLayerBase(abc.ABC, Generic[SubstTypeVar, DataTypeVar]):
    """Abstract base class for coating layer object.

    Coating layer object stores :class:`SubstrateBase` object and target image, which is
    a binary image of coated substrate. The role of coating layer object is to acquire
    coating layer region by template matching and analyze its shape.

    External API can use the following members to get analysis results of
    concrete subclasses.

    * :attr:`DataType`: Dataclass type for the analysis result.
    * :meth:`analyze`: :attr:`DataType` instance containing analysis result.
    * :meth:`draw`: Visualized result.

    Arguments:
        image: Binary target image.
        substrate: Substrate instance storing binary reference image.
        tempmatch: Pre-computed template matching result.
            External constructor can pass this argument to force the template matching
            result. If not passed, :meth:`match_template` performs matching.
    """

    DataType: type[DataTypeVar]
    """Return type of :attr:`analyze`.

    Concrete subclass must assign this attribute with dataclass type.
    """

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        substrate: SubstTypeVar,
        *,
        tempmatch: tuple[tuple[int, ...], float] | None = None,
    ):
        """Initialize the instance.

        *image* is set to be immutable, and template matching is performed if
        *tempmatch* is ``None``.
        """
        self._image = image
        self._image.setflags(write=False)
        self._substrate = substrate

        if tempmatch is None:
            image = self.image
            x0, y0, x1, y1 = self.substrate.reference.templateROI
            template = self.substrate.reference.image[y0:y1, x0:x1]
            self._tempmatch = self.match_template(image, template)
        else:
            self._tempmatch = tempmatch

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """Binary target image.

        For immutability, this image is not writable.
        """
        return self._image

    @property
    def substrate(self) -> SubstTypeVar:
        """Substrate instance which contains substrate and reference information."""
        return self._substrate

    @property
    def tempmatch(self) -> tuple[tuple[int, ...], float]:
        """Template matching location and score."""
        return self._tempmatch

    def match_template(
        self, image: npt.NDArray[np.uint8], template: npt.NDArray[np.uint8]
    ) -> tuple[tuple[int, ...], float]:
        """Perform template matching between *image* and *template*.

        Template matching is performed using :func:`cv2.matchTemplate` with
        :obj:`cv2.TM_SQDIFF_NORMED`. Subclass may override this method to apply
        other algorithm.

        Arguments:
            image: Binary target image.
            template: Binary template image.
        """
        res = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
        score, _, loc, _ = cv2.minMaxLoc(res)
        return (tuple(loc), score)

    def substrate_point(self) -> npt.NDArray[np.int32]:
        """Upper left point of the substrate image in target image.

        Returns:
            Coordinates in ``(x, y)``.
        """
        temp_point, _ = self.tempmatch
        t = self.substrate.reference.templateROI[:2]
        s = self.substrate.reference.substrateROI[:2]
        return np.array(temp_point, dtype=np.int32) - t + s

    @attrcache("_coated_substrate")
    def coated_substrate(self) -> npt.NDArray[np.bool_]:
        """Coated substrate region.

        Returns:
            Target image without artifacts, e.g., bath surface.
        """
        _, img = cv2.connectedComponents(cv2.bitwise_not(self.image))
        x, y = (self.substrate_point() + self.substrate.region_points()).T
        return np.isin(img, img[y, x])

    @attrcache("_extracted_layer")
    def extract_layer(self) -> npt.NDArray[np.bool_]:
        """Coating layer region extracted from target image."""
        # remove the substrate
        x0, y0 = self.substrate_point()
        subst_mask = self.substrate.regions() >= 0
        ret = images_ANDXOR(self.coated_substrate(), subst_mask, (x0, y0))
        ret[:y0, :] = False
        return ret

    @abc.abstractmethod
    def valid(self) -> bool:
        """Return if the analysis can be performed as expected.

        Sometimes, the coating layer instance should be constructed but not analyzed at
        all. For example, the coating video may contains frames where the capillary
        bridge is not ruptured yet. This method allows analyzer to skip such instances.
        """

    @abc.abstractmethod
    def analyze(self) -> DataTypeVar:
        """Return analysis result as dataclass.

        Return type must be :attr:`DataType`.
        """

    @abc.abstractmethod
    def draw(self, *args, **kwargs) -> npt.NDArray[np.uint8]:
        """Return visualization result."""


@dataclasses.dataclass
class CoatingLayerData:
    """Analysis data for :class:`CoatingLayer`."""

    pass


class CoatingLayer(CoatingLayerBase[SubstrateBase, CoatingLayerData]):
    """Basic implementation of coating layer without any analysis.

    Arguments:
        image: Binary target image.
        substrate: Substrate instance.
        tempmatch: Pre-computed template matching result.

    Examples:
        Construct substrate instance first.

        .. plot::
            :include-source:
            :context: reset

            >>> import cv2
            >>> from finitedepth import get_sample_path, Reference, Substrate
            >>> img = cv2.imread(get_sample_path("ref.png"), cv2.IMREAD_GRAYSCALE)
            >>> _, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            >>> ref = Reference(bin, (10, 10, 1250, 200), (100, 100, 1200, 500))
            >>> subst = Substrate(ref)
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(subst.draw()) #doctest: +SKIP

        Then, construct coating layer instance.

        .. plot::
            :include-source:
            :context: close-figs

            >>> from finitedepth import CoatingLayer
            >>> img = cv2.imread(get_sample_path("coat.png"), cv2.IMREAD_GRAYSCALE)
            >>> _, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            >>> coat = CoatingLayer(bin, subst)
            >>> plt.imshow(coat.draw()) #doctest: +SKIP
    """

    DataType = CoatingLayerData
    """Return :obj:`CoatingLayerData`."""

    def valid(self) -> bool:
        """Return true, as analysis is not performed at all."""
        return True

    def analyze(self):
        """Return empty :class:`CoatingLayerData`."""
        return self.DataType()

    def draw(
        self,
        subtraction_mode: str = "none",
        layer_color: tuple[int, int, int] = (255, 0, 0),
        layer_thickness: int = -1,
    ) -> npt.NDArray[np.uint8]:
        """Subtract the template match result and paint the coating layer.

        Arguments:
            subtraction_mode ({`'none', 'template', 'substrate', 'full'`}): Subtraction
                mode. `'template'` and `'substrate'` removes overlapping template region
                and substrate region, respectively. `'full'` removes both.
            layer_color: Layer color for :func:`cv2.drawContours`.
            layer_thickness: Layer thickness for :func:`cv2.drawContours`.
        """
        image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        if subtraction_mode not in ["none", "template", "substrate", "full"]:
            raise TypeError("Unrecognized subtraction mode: %s" % subtraction_mode)
        if subtraction_mode in ["template", "full"]:
            x0, y0, x1, y1 = self.substrate.reference.templateROI
            tempImg = self.substrate.reference.image[y0:y1, x0:x1]
            h, w = tempImg.shape[:2]
            (X0, Y0), _ = self.tempmatch
            binImg = self.image[Y0 : Y0 + h, X0 : X0 + w]
            mask = images_XOR(~binImg.astype(bool), ~tempImg.astype(bool))
            image[Y0 : Y0 + h, X0 : X0 + w][~mask] = 255
        if subtraction_mode in ["substrate", "full"]:
            x0, y0, x1, y1 = self.substrate.reference.substrateROI
            substImg = self.substrate.reference.image[y0:y1, x0:x1]
            h, w = substImg.shape[:2]
            X0, Y0 = self.substrate_point()
            binImg = self.image[Y0 : Y0 + h, X0 : X0 + w]
            mask = images_XOR(~binImg.astype(bool), ~substImg.astype(bool))
            image[Y0 : Y0 + h, X0 : X0 + w][~mask] = 255

        cnts, _ = cv2.findContours(
            self.extract_layer().astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        cv2.drawContours(image, cnts, -1, layer_color, layer_thickness)

        return image  # type: ignore[return-value]


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


class RectLayerShape(CoatingLayerBase[RectSubstrate, RectLayerShapeData]):
    """Coating layer over rectangular substrate.

    Arguments:
        image: Binary target image.
        substrate: Substrate instance.
        opening_ksize: Kernel size for morphological operation.
            Elements must be zero or odd number.
        reconstruct_radius: Radius of the "safe zone" for noise removal.
            Imaginary circles with this radius are drawn on bottom corners of the
            substrate. Connected components not passing these circles are regarded as
            image artifacts.
        roughness_measure: Similarity measure to quantify roughness.

            `'DTW'`
                Dynamice time warping.
            `'SDTW'`
                Root mean square of dynamic time warping.
        tempmatch: Pre-computed template matching result.

    Examples:
        Construct substrate instance first.

        .. plot::
            :include-source:
            :context: reset

            >>> import cv2
            >>> from finitedepth import get_sample_path, Reference, RectSubstrate
            >>> img = cv2.imread(get_sample_path("ref.png"), cv2.IMREAD_GRAYSCALE)
            >>> _, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            >>> ref = Reference(bin, (10, 10, 1250, 200), (100, 100, 1200, 500))
            >>> subst = RectSubstrate(ref, 3.0, 1.0, 0.01)
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(subst.draw()) #doctest: +SKIP

        Then, construct coating layer instance.

        .. plot::
            :include-source:
            :context: close-figs

            >>> from finitedepth import RectLayerShape
            >>> img = cv2.imread(get_sample_path("coat.png"), cv2.IMREAD_GRAYSCALE)
            >>> _, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            >>> coat = RectLayerShape(bin, subst, (1, 1), 50, "DTW")
            >>> plt.imshow(
            ...     coat.draw(conformality_step=10, roughness_step=1)
            ... ) #doctest: +SKIP
    """

    DataType = RectLayerShapeData
    """Return :obj:`RectLayerShapeData`."""

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        substrate: RectSubstrate,
        opening_ksize: tuple[int, int],
        reconstruct_radius: int,
        roughness_measure: str,
        *,
        tempmatch: tuple[tuple[int, ...], float] | None = None,
    ):
        """Initialize the instance.

        Check whether kernel size are zero or odd and roughness measure is valid.
        """
        if not all(i == 0 or (i > 0 and i % 2 == 1) for i in opening_ksize):
            raise ValueError("Kernel size must be zero or odd.")
        if roughness_measure not in ["DTW", "SDTW"]:
            raise TypeError(f"Unknown roughness measure: {roughness_measure}")
        super().__init__(image, substrate, tempmatch=tempmatch)
        self._opening_ksize = opening_ksize
        self._reconstruct_radius = reconstruct_radius
        self._roughness_measure = roughness_measure

    @property
    def opening_ksize(self) -> tuple[int, int]:
        """Kernel size for morphological operation."""
        return self._opening_ksize

    @property
    def reconstruct_radius(self) -> int:
        """Radius of the "safe zone" for noise removal."""
        return self._reconstruct_radius

    @property
    def roughness_measure(self) -> str:
        """Similarity measure to quantify roughness."""
        return self._roughness_measure

    @attrcache("_layer_contours")
    def layer_contours(self) -> tuple[npt.NDArray[np.int32], ...]:
        """Find contours of coating layer region.

        This method finds external contours of :meth:`~.extract_layer`.
        Each contour encloses each discrete region of coating layer.
        """
        layer_cnts, _ = cv2.findContours(
            self.extract_layer().astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        return tuple(layer_cnts)

    @attrcache("_interfaces")
    def interfaces(self) -> tuple[npt.NDArray[np.int64], ...]:
        """Find solid-liquid interfaces.

        A substrate can have contact with multiple discrete coating layer regions,
        and a single coating layer region can have multiple contacts to the substrate.
        This method returns indices for :meth:`~finitedepth.PolySubstrateBase.contour`
        where solid-liquid interfaces start and stop.

        Returns:
            tuple of arrays. ``i``-th array represents ``i``-th coating layer region in
            :meth:`layer_contours`. Shape of the array is ``(N, 2)``, where ``N`` is the
            number of contacts the coating layer region makes. Each column represents
            starting and ending indices for the interface interval in substrate contour.

        Note:
            Each interval describes continuous patch on the substrate contour covered
            by the layer. To acquire the interface points, slice :attr:`substrate`'s
            :meth:`~finitedepth.PolySubstrateBase.contour` with the indices.
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

    def capbridge_broken(self) -> bool:
        """Check if capillary bridge is ruptured.

        As substrate is withdrawn from fluid bath, capillary bridge forms between the
        coating layer and bulk fluid and then ruptures.
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

    valid = capbridge_broken

    @attrcache("_extracted_layer")
    def extract_layer(self) -> npt.NDArray[np.bool_]:
        """Coating layer region extracted from target image.

        Error pixels are removed by performing morphological operation and removing the
        disconnected components that are far from the substrate.
        """
        # Perform opening to remove error pixels. We named the parameter as
        # "closing" because the coating layer is black in original image, but
        # in fact we do opening since the layer is True in extracted layer.
        ksize = self.opening_ksize
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
        R = self.reconstruct_radius
        cv2.circle(
            vicinity_mask, B.astype(np.int32), R, 1, -1
        )  # type: ignore[call-overload]
        cv2.circle(
            vicinity_mask, C.astype(np.int32), R, 1, -1
        )  # type: ignore[call-overload]
        n = np.dot((C - B) / np.linalg.norm((C - B)), np.array([[0, 1], [-1, 0]]))
        pts = np.stack([B, B + R * n, C + R * n, C]).astype(np.int32)
        cv2.fillPoly(vicinity_mask, [pts], 1)  # type: ignore[call-overload]
        _, labels = cv2.connectedComponents(img)
        layer_comps = np.unique(labels[np.where(vicinity_mask.astype(bool))])
        layer_mask = np.isin(labels, layer_comps[layer_comps != 0])

        return layer_mask

    @attrcache("_surface")
    def surface(self) -> tuple[np.int64, np.int64]:
        """Liquid-gas interface of the coating layer.

        Substrate surface exposed to air is considered to be covered by coating layer
        with zero thickness.

        Returns:
            Starting and ending indices for the surface interval in coated substrate
            contour.

        Note:
            To acquire the surface points, slice
            :meth:`~finitedepth.PolySubstrateBase.contour` with the indices.
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
    def uniform_layer(self) -> tuple[np.float64, npt.NDArray[np.float64]]:
        """Imaginary uniform layer.

        Uniform layer is a parallel curve [#parallel-curve]_ of substrate surface
        which has the same cross-sectional area as the actual coating layer.

        Returns:
            Thickness and polyline vertices of the uniform layer.

        .. [#parallel-curve] https://en.wikipedia.org/wiki/Parallel_curve
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
    def conformality(self) -> tuple[float, npt.NDArray[np.int32]]:
        """DTW-based conformality of the coating layer.

        Returns:
            Conformality between layer surface and substrate surface and its pair of
            points in curve space.
        """
        if not self.interfaces():
            return (np.nan, np.empty((0, 2), dtype=np.int32))

        (i0, i1) = np.sort(np.concatenate(self.interfaces()).flatten())[[0, -1]]
        subst_cnt = self.substrate.contour() + self.substrate_point()
        intf = subst_cnt[i0:i1]

        I0, I1 = self.surface()
        surf = self.contour()[I0:I1]

        dist = cdist(np.squeeze(surf, axis=1), np.squeeze(intf, axis=1))
        mat = dtw_acm(dist)
        path = dtw_owp(mat)
        d = dist[path[:, 0], path[:, 1]]
        d_avrg = mat[-1, -1] / len(path)
        C = 1 - np.sum(np.abs(d - d_avrg)) / mat[-1, -1]
        pairs = np.concatenate([surf[path[..., 0]], intf[path[..., 1]]], axis=1)
        return (float(C), pairs)

    @attrcache("_roughness")
    def roughness(self) -> tuple[float, npt.NDArray[np.float64]]:
        """Similarity-based surface roughness of the coating layer.

        Returns:
            Roughness between layer surface and uniform layer and its pair of points in
            curve space.
        """
        I0, I1 = self.surface()
        surf = self.contour()[I0:I1]
        _, ul = self.uniform_layer()

        if surf.size == 0 or ul.size == 0:
            return (np.nan, np.empty((0, 2), dtype=np.float64))

        ul_len = cv2.arcLength(ul.astype(np.float32), closed=False)
        if self.roughness_measure == "DTW":
            ul = sample_polyline(ul, int(np.ceil(ul_len)))
            dist = cdist(np.squeeze(surf, axis=1), np.squeeze(ul, axis=1))
            mat = dtw_acm(dist)
            path = dtw_owp(mat)
            roughness = mat[-1, -1] / len(path)
            pairs = np.concatenate([surf[path[..., 0]], ul[path[..., 1]]], axis=1)
        elif self.roughness_measure == "SDTW":
            ul = sample_polyline(ul, int(np.ceil(ul_len)))
            dist = cdist(np.squeeze(surf, axis=1), np.squeeze(ul, axis=1))
            mat = dtw_acm(dist**2)
            path = dtw_owp(mat)
            roughness = np.sqrt(mat[-1, -1] / len(path))
            pairs = np.concatenate([surf[path[..., 0]], ul[path[..., 1]]], axis=1)
        else:
            roughness = np.nan
            pairs = np.empty((0, 2), dtype=np.float64)
        return (float(roughness), pairs)

    @attrcache("_max_thickness")
    def max_thickness(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Regional maximum thicknesses.

        Coating layer is segmented using :meth:`RectSubstrate.sideline_intersections`.
        Points on layer surface and sideline for maximum distance in each region are
        found.

        Returns:
            tuple of two arrays. The first array contains maximum thickness values on
            left, bottom, and right region. Value of ``0`` indicates no coating layer on
            that region. The second array contains points on layer surface and substrate
            lines for the maximum thickness.
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

    def analyze(self):
        """Return :class:`RectLayerShapeData`."""
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

    def draw(
        self,
        background_mode: str = "image",
        subtraction_mode: str = "none",
        layer_color: tuple[int, int, int] = (255, 0, 0),
        layer_thickness: int = -1,
        contactline_color: tuple[int, int, int] = (0, 255, 0),
        contactline_thickness: int = 1,
        maxthickness_color: tuple[int, int, int] = (0, 255, 0),
        maxthickness_thickness: int = 1,
        uniformlayer_color: tuple[int, int, int] = (0, 0, 255),
        uniformlayer_thickness: int = 1,
        conformality_color: tuple[int, int, int] = (0, 0, 255),
        conformality_thickness: int = 1,
        conformality_step: int = 1,
        roughness_color: tuple[int, int, int] = (0, 0, 255),
        roughness_thickness: int = 1,
        roughness_step: int = 1,
    ) -> npt.NDArray[np.uint8]:
        """Visualize the analysis result.

        #. Draw the substrate with by :class:`PaintMode`.
        #. Display the template matching result with :class:`SubtractionMode`.
        #. Draw coating layer and contact line.
        #. If capillary bridge is broken, draw regional maximum thicknesses,
           uniform layer, conformality pairs and roughness pairs.

        Arguments:
            background_mode ({`'image', 'empty'`}): Determine how background is drawn.
                `'image'` draws original background image while `'empty'` draws on
                empty frame.
            subtraction_mode ({`'none', 'template', 'substrate', 'full'`}): Subtraction
                mode. `'template'` and `'substrate'` removes overlapping template region
                and substrate region, respectively. `'full'` removes both.
            layer_color: Layer contour's color for :func:`cv2.drawContours`.
            layer_thickness: Layer contour's thickness for :func:`cv2.drawContours`.
            contactline_color: Contact line's color for :func:`cv2.line`.
            contactline_thickness: Contact line's thickness for :func:`cv2.line`.
            maxthickness_color: Regional maximum thickness line's color for
                :func:`cv2.polylines`.
            maxthickness_thickness: Regional maximum thickness line's thickness for
                :func:`cv2.polylines`.
            uniformlayer_color: Imaginary uniform layer's color for
                :func:`cv2.polylines`.
            uniformlayer_thickness: Imaginary uniform layer's thickness for
                :func:`cv2.polylines`.
            conformality_color: Conformality pairs' color for :func:`cv2.polylines`.
            conformality_thickness: Conformality pairs' thickness for
                :func:`cv2.polylines`.
            conformality_step: Step size to skip conformality pairs.
            roughness_color: Roughness pairs' color for :func:`cv2.polylines`.
            roughness_thickness: Roughness pairs' thickness for :func:`cv2.polylines`.
            roughness_step: Step size to skip roughness pairs.
        """
        if background_mode == "image":
            image = self.image
        elif background_mode == "empty":
            image = np.full(self.image.shape, 255, dtype=np.uint8)
        else:
            raise TypeError("Unrecognized paint mode: %s" % background_mode)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # type: ignore[assignment]

        if subtraction_mode not in ["none", "template", "substrate", "full"]:
            raise TypeError("Unrecognized subtraction mode: %s" % subtraction_mode)
        if subtraction_mode in ["template", "full"]:
            x0, y0, x1, y1 = self.substrate.reference.templateROI
            tempImg = self.substrate.reference.image[y0:y1, x0:x1]
            h, w = tempImg.shape[:2]
            (X0, Y0), _ = self.tempmatch
            binImg = self.image[Y0 : Y0 + h, X0 : X0 + w]
            mask = images_XOR(~binImg.astype(bool), ~tempImg.astype(bool))
            image[Y0 : Y0 + h, X0 : X0 + w][~mask] = 255
        if subtraction_mode in ["substrate", "full"]:
            x0, y0, x1, y1 = self.substrate.reference.substrateROI
            substImg = self.substrate.reference.image[y0:y1, x0:x1]
            h, w = substImg.shape[:2]
            X0, Y0 = self.substrate_point()
            binImg = self.image[Y0 : Y0 + h, X0 : X0 + w]
            mask = images_XOR(~binImg.astype(bool), ~substImg.astype(bool))
            image[Y0 : Y0 + h, X0 : X0 + w][~mask] = 255

        cv2.drawContours(image, self.layer_contours(), -1, layer_color, layer_thickness)

        if len(self.interfaces()) > 0:
            (i0, i1) = np.sort(np.concatenate(self.interfaces()).flatten())[[0, -1]]
            subst_cnt = self.substrate.contour() + self.substrate_point()
            (p0,), (p1,) = subst_cnt[[i0, i1]].astype(np.int32)
            cv2.line(image, p0, p1, contactline_color, contactline_thickness)

        if not self.capbridge_broken():
            return image

        lines = []
        for dist, pts in zip(*self.max_thickness()):
            if dist == 0:
                continue
            lines.append(pts.astype(np.int32))
        cv2.polylines(
            image,
            lines,
            isClosed=False,
            color=maxthickness_color,
            thickness=maxthickness_thickness,
        )

        _, points = self.uniform_layer()
        cv2.polylines(
            image,
            [points.astype(np.int32)],
            isClosed=False,
            color=uniformlayer_color,
            thickness=uniformlayer_thickness,
        )

        if len(self.interfaces()) > 0:
            _, pairs = self.conformality()
            cv2.polylines(
                image,
                pairs[::conformality_step],
                isClosed=False,
                color=conformality_color,
                thickness=conformality_thickness,
            )

        if len(self.interfaces()) > 0:
            _, pairs = self.roughness()
            cv2.polylines(
                image,
                pairs.astype(np.int32)[::roughness_step],
                isClosed=False,
                color=roughness_color,
                thickness=roughness_thickness,
            )

        return image


def images_XOR(
    img1: npt.NDArray[np.bool_],
    img2: npt.NDArray[np.bool_],
    point: tuple[int, int] = (0, 0),
) -> npt.NDArray[np.bool_]:
    """Perform subtraction between two images by XOR operation.

    Arguments:
        img1: Image from which *img2* is subtracted.
        img2: Image patch which is subtracted from *img1*.
        point: Location in *img1* where *img2* is subtracted.
    """
    H, W = img1.shape
    h, w = img2.shape
    x0, y0 = point
    x1, y1 = x0 + w, y0 + h

    img1 = img1.copy()
    img1_crop = img1[max(y0, 0) : min(y1, H), max(x0, 0) : min(x1, W)]
    img2_crop = img2[max(-y0, 0) : min(H - y0, h), max(-x0, 0) : min(W - x0, w)]
    img1_crop ^= img2_crop
    return img1


def images_ANDXOR(
    img1: npt.NDArray[np.bool_],
    img2: npt.NDArray[np.bool_],
    point: tuple[int, int] = (0, 0),
) -> npt.NDArray[np.bool_]:
    """Perform subtraction between two images by AND and XOR operations.

    Arguments:
        img1: Image from which *img2* is subtracted.
        img2: Image patch which is subtracted from *img1*.
        point: Location in *img1* where *img2* is subtracted.
    """
    H, W = img1.shape
    h, w = img2.shape
    x0, y0 = point
    x1, y1 = x0 + w, y0 + h

    img1 = img1.copy()
    img1_crop = img1[max(y0, 0) : min(y1, H), max(x0, 0) : min(x1, W)]
    img2_crop = img2[max(-y0, 0) : min(H - y0, h), max(-x0, 0) : min(W - x0, w)]
    common = img1_crop & img2_crop
    img1_crop ^= common
    return img1


def sample_polyline(vertices: npt.NDArray, n: int) -> npt.NDArray[np.float64]:
    """Sample *n* points from a polyline which consists of *vertices*.

    Arguments:
        vertices
            An array whose shape is ``(N, 1, D)`` where ``N`` is the number of
            points and ``D`` is the dimension.
        n
            Number of new points to be sampled.

    Returns:
        pts
            Sampled points of shape ``(n, 1, D)``.
    """
    tck, _ = splprep(vertices.squeeze(axis=1).T, s=0)
    u = np.linspace(0, 1, n)
    pts = np.stack(splev(u, tck)).T[:, np.newaxis, ...]
    return pts


def parallel_curve(vertices: npt.NDArray, dist: float) -> npt.NDArray[np.float64]:
    """Parallel curve of a polyline of *vertices* with offset distance *dist*.

    Arguments:
        vertices
            An array whose shape is ``(N, 1, D)`` where ``N`` is the number of
            points and ``D`` is the dimension.
        dist: offset distance of the parallel curve.

    Returns:
        Round-joint parallel curve of shape ``(N, 1, D)``.
    """
    ret = offset_curve(
        LineString(np.squeeze(vertices, axis=1)), dist, join_style="round"
    )
    return get_coordinates(ret)[:, np.newaxis]
