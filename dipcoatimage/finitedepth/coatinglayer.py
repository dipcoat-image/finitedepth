"""
Coated Substrate Image
======================

-------------------
Basic coating layer
-------------------

:mod:`dipcoatimage.finitedepth.coatinglayer` provides class to analyze the
coating layer from coated substrate image.

Base class
----------

.. autoclass:: CoatingLayerError
   :members:

.. autoclass:: CoatingLayerBase
   :members:

Implementation
--------------

.. autoclass:: LayerAreaParameters
   :members:

.. autoclass:: BackgroundDrawMode
   :members:

.. autoclass:: SubtractionDrawMode
   :members:

.. autoclass:: LayerAreaDrawOptions
   :members:

.. autoclass:: LayerAreaDecoOptions
   :members:

.. autoclass:: LayerAreaData
   :members:

.. autoclass:: LayerArea
   :members:

----------------------------------------
Coating layer over rectangular substrate
----------------------------------------

.. automodule:: dipcoatimage.finitedepth.rectcoatinglayer

"""


import abc
import cv2  # type: ignore
import dataclasses
import enum
import numpy as np
import numpy.typing as npt
from .substrate import SubstrateBase
from .util import (
    match_template,
    images_XOR,
    images_ANDXOR,
    DataclassProtocol,
    FeatureDrawingOptions,
    Color,
    binarize,
    colorize,
)
from typing import TypeVar, Generic, Type, Optional, Tuple, List

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias


__all__ = [
    "CoatingLayerError",
    "CoatingLayerBase",
    "LayerAreaParameters",
    "BackgroundDrawMode",
    "SubtractionDrawMode",
    "LayerAreaDrawOptions",
    "LayerAreaDecoOptions",
    "LayerAreaData",
    "LayerArea",
    "find_polyline_projections",
]


SubstrateType = TypeVar("SubstrateType", bound=SubstrateBase)
ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)
DecoOptionsType = TypeVar("DecoOptionsType", bound=DataclassProtocol)
DataType = TypeVar("DataType", bound=DataclassProtocol)


class CoatingLayerError(Exception):
    """Base class for error from :class:`CoatingLayerBase`."""

    pass


class CoatingLayerBase(
    abc.ABC,
    Generic[SubstrateType, ParametersType, DrawOptionsType, DecoOptionsType, DataType],
):
    """
    Abstract base class for coating layer.

    Coating layer class extracts the coating layer region from coated substrate
    image and analyze it.

    .. rubric:: Constructor

    Constructor signature must not be modified because high-level API use factory
    to generate coating layer instances. Additional parameters can be introduced
    by definig class attribute :attr:`Parameters`, :attr:`DrawOptions` and
    :attr:`DecoOptions`.

    .. rubric:: Parameters, DrawOptions and DecoOptions

    Concrete class must have :attr:`Parameters`, :attr:`DrawOptions` and
    :attr:`DecoOptions` which return dataclass types. Their instances are passed
    to the constructor at instance initialization, and can be accessed by
    :attr:`parameters`, :attr:`draw_options` and :attr:`deco_options`.

    :attr:`Parameter` must be frozen to ensure immtability for caching. However,
    :attr:`DrawOptions` and :attr:`DecoOptions` need not be frozen since
    visualization does not affect the identity of instance. Therefore methods
    affected by draw options and deco options must not be cached.

    .. rubric:: Sanity check

    Validity of the parameters can be checked by :meth:`verify` or :meth:`valid`.
    Their result can be implemented by defining :meth:`examine`.

    .. rubric:: Visualization

    :meth:`draw` defines the visualization logic for concrete class using
    :attr:`draw_options` and :attr:`deco_options`. Modifying these attributes
    changes the visualization result.

    Two options are not strictly distinguished, but the intention is that draw
    option controls the overall behavior and deco option controls how the coating
    layer is painted.

    .. rubric:: Analysis

    Concrete class must have :attr:`Data` which returns dataclass type and
    implement :meth:`analyze_layer` which returns data tuple compatible with
    :attr:`Data`.
    :meth:`analyze` is the API for analysis result.

    Parameters
    ==========

    image
        Coated substrate image. May be grayscale or RGB.

    substrate
        Substrate instance.

    parameters
        Additional parameters.

    draw_options, deco_options
        Coated substrate drawing option and coating layer decorating option.

    """

    __slots__ = (
        "_image",
        "_substrate",
        "_parameters",
        "_draw_options",
        "_deco_options",
        "_binary_image",
        "_match_substrate",
        "_coated_substrate",
        "_extracted_layer",
        "_layer_contours",
        "_interfaces",
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    DecoOptions: Type[DecoOptionsType]
    Data: Type[DataType]

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        substrate: SubstrateType,
        parameters: Optional[ParametersType] = None,
        *,
        draw_options: Optional[DrawOptionsType] = None,
        deco_options: Optional[DecoOptionsType] = None,
    ):
        super().__init__()
        self._image = image
        self._image.setflags(write=False)

        self._substrate = substrate

        if parameters is None:
            self._parameters = self.Parameters()
        else:
            self._parameters = dataclasses.replace(parameters)

        if draw_options is None:
            self._draw_options = self.DrawOptions()
        else:
            self._draw_options = dataclasses.replace(draw_options)

        if deco_options is None:
            self._deco_options = self.DecoOptions()
        else:
            self._deco_options = dataclasses.replace(deco_options)

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """
        Coated substrate image passed to the constructor.

        This array is not writable to enable caching which requires immutability.
        """
        return self._image

    @property
    def substrate(self) -> SubstrateType:
        """Substrate instance passed to the constructor."""
        return self._substrate

    @property
    def parameters(self) -> ParametersType:
        """
        Additional parameters for concrete class.

        Instance of :attr:`Parameters`, which must be a frozen dataclass.
        """
        return self._parameters

    @property
    def draw_options(self) -> DrawOptionsType:
        """
        Options to visualize the coated substrate image.

        Instance of :attr:`DrawOptions` dataclass.
        """
        return self._draw_options

    @draw_options.setter
    def draw_options(self, options: DrawOptionsType):
        self._draw_options = options

    @property
    def deco_options(self) -> DecoOptionsType:
        """
        Options to decorate the coating layer region.

        Instance of :attr:`DecoOptions` dataclass.
        """
        return self._deco_options

    @deco_options.setter
    def deco_options(self, options: DecoOptionsType):
        self._deco_options = options

    def binary_image(self) -> npt.NDArray[np.uint8]:
        """
        Binarized :attr:`image` using Otsu's thresholding.

        Notes
        =====

        This method is cached. Do not modify its result.

        """
        if not hasattr(self, "_binary_image"):
            self._binary_image = binarize(self.image)
        return self._binary_image

    def match_substrate(self) -> Tuple[float, npt.NDArray[np.int32]]:
        """
        Return template matching score and point between the coating layer image
        and the reference template image.

        Matching is done by :func:`match_template`.

        Notes
        =====

        This method is cached. Do not modify its result.

        """
        if not hasattr(self, "_match_substrate"):
            image = self.binary_image()
            x0, y0, x1, y1 = self.substrate.reference.templateROI
            template = self.substrate.reference.binary_image()[y0:y1, x0:x1]
            score, point = match_template(image, template)
            self._match_substrate = score, np.array(point, dtype=np.int32)
        return self._match_substrate

    def substrate_point(self) -> npt.NDArray[np.int32]:
        """
        Upper left point in ``(x, y)`` where the substrate is located.

        """
        _, temp_point = self.match_substrate()
        temp2subst = self.substrate.reference.temp2subst()
        return temp_point + temp2subst

    def capbridge_broken(self) -> bool:
        """
        Determines if the capillary bridge is broken in :attr:`self.image`.
        """
        _, y = self.substrate_point()
        below_subst = self.binary_image()[y:]
        # if any row is all-white, capillary bridge is broken
        row_white = np.all(below_subst, axis=1)
        return bool(np.any(row_white))

    def coated_substrate(self) -> npt.NDArray[np.bool_]:
        """Return the mask without undesired features, e.g. bath surface."""
        if not hasattr(self, "_coated_substrate"):
            # remove components that are not connected to the substrate
            _, img = cv2.connectedComponents(cv2.bitwise_not(self.binary_image()))
            x, y = (self.substrate_point() + self.substrate.nestled_points()).T
            labels = img[y, x]
            self._coated_substrate = np.isin(img, labels)
        return self._coated_substrate

    def extract_layer(self) -> npt.NDArray[np.bool_]:
        """Extract the coating layer as binary array from *self.image*."""
        if not hasattr(self, "_extracted_layer"):
            coated_mask = self.coated_substrate()
            # remove the substrate
            subst_mask = self.substrate.regions()[1].astype(bool)
            x0, y0 = self.substrate_point()
            ret = images_ANDXOR(coated_mask, subst_mask, (x0, y0))
            ret[:y0, :] = False
            self._extracted_layer = ret
        return self._extracted_layer

    def layer_contours(self) -> List[npt.NDArray[np.int32]]:
        """
        Return the contours of the coating layer regions.

        The contours include both the gas-liquid interface and the
        substrate-liquid interface of the coating layer. Each contour represents
        discrete region of coating layer.
        """
        if not hasattr(self, "_layer_contours"):
            contours, _ = cv2.findContours(
                self.extract_layer().astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            self._layer_contours = list(contours)
        return self._layer_contours

    def interfaces(self) -> List[npt.NDArray[np.int32]]:
        """
        Return indices which can be used to acquire substrate-liquid interface
        points from :meth:`layer_contours`.

        Returns
        -------
        indices
            Indices for the contours in :meth:`layer_contours`.

        Notes
        -----
        From the layer contours, this method finds the points that are adjacent
        to the substrate contours - i.e. the solid-liquid interface points on
        the liquid side.

        The return value is a list of `N` arrays, where `N` is the number of
        substrate regions in substrate image. `i`-th array represents the
        interface points on the substrate region which is labelled as `i + 1`
        by :meth:`SubstrateBase.regions`.

        Every array has `M` rows and 3 columns where `M` is the number of
        discrete interfaces on the substrate. For example if two discrete coating
        layers are on a substrate region, the array shape is `(2, 3)`.
        Rows are sorted in the order of appearance along the substrate contour.

        Each column describes the coating layer contour:

        1. Index of coating layer contour in :meth:`layer_contours`.
        2. Interface-starting index in the coating layer contour.
        3. Interface-ending index in the coating layer cnotour.

        The first column indicates the location of the coating layer contour in
        :meth:`layer_contours`. Combining this information with the row index
        allows sorting the contours. For example, if the first column is
        `array([[4], [2]])` this means the 5nd contour in :meth:`layer_contours`
        appears first and the 3st contour appears next along the substrate.

        The second and the third column locates the interface in each coating
        layer contour. For example, a row `array([[1, 100, 10]])` describes an
        interface which starts at `cnts[1][100]` and ends at `cnts[1][10]`
        (where `cnts` is the result of :meth:`layer_contours`).

        The direction from the starting point to the ending point is parallel to
        the direction of the layer contour. In the previous example, the indices
        of interface points are `[100, 101, ..., (last index), 0, 1, ..., 10]`,
        not `[100, 99, ..., 10]`.
        Note that this direction is opposite to the direction with respect to the
        substrate contour.

        See Also
        --------
        interface_points, surface_points
            Returns the points using the result of this method.

        """
        if not hasattr(self, "_interfaces"):
            ret = []
            reg_val, reg_img = self.substrate.regions()
            for v in range(1, reg_val):
                # From each layer contour, we want to find the points that are
                # adjacent to the substrate. For this we make a mask which
                # indicates the vicinity of the substrate.
                # XXX: MUST use substrate contour. In the future, substrate with
                # inner contour (i.e. hole) will be supported.
                subst = (reg_img == v) * np.uint8(255)
                (subst_cnt,), _ = cv2.findContours(
                    subst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                subst_cnt += self.substrate_point()
                cnt_img = np.zeros(self.image.shape[:2], dtype=np.uint8)
                cv2.drawContours(cnt_img, [subst_cnt], -1, 255)
                dil_cnt = cv2.dilate(cnt_img, np.ones((3, 3)))

                # Among every layer contour we find out which intersects with
                # the substrate vicinity and where the intersection is.
                interfaces = []
                for i, layer_cnt in enumerate(self.layer_contours()):
                    x, y = layer_cnt.transpose(2, 0, 1)
                    on_intf = dil_cnt[y, x].astype(bool)
                    interface = layer_cnt[on_intf]
                    if interface.size == 0:
                        continue

                    interface_idxs = np.arange(len(layer_cnt))[..., np.newaxis][on_intf]
                    # Case A) Contour starts outside of the interface. Here the
                    # interface will be in like [100, ..., 200] and the result
                    # index is [100, 200].
                    # Case B) Contour starts on the interface. Here the interface
                    # will be in like [0, ..., 10], [21, ...] where 11 to 20 is
                    # surface. The result index is [21, 10].
                    (jump,) = np.where(np.diff(interface_idxs) != 1)
                    if jump.size == 0:  # Case A
                        i0, i1 = interface_idxs[0], interface_idxs[-1]
                    else:
                        j = jump[-1]
                        i0, i1 = interface_idxs[j + 1], interface_idxs[j]
                    interfaces.append([i, i0, i1])

                if interfaces:
                    interfaces = np.array(interfaces, dtype=np.int32)
                    # We sort the interface patches along the substrate contour.
                    # We first select representing points from each interface
                    # patches and find the closest projection to substrate.
                    rep_pts = np.stack(
                        [self.layer_contours()[i][j] for i, j, _ in interfaces]
                    )
                    proj = find_polyline_projections(rep_pts, subst_cnt)
                    # Sort by projection's line segment index and then by t.
                    # Cheap hack: argsort (index + t)!
                    interfaces = interfaces[np.argsort(np.sum(proj, axis=1))]
                else:
                    interfaces = np.empty((0, 3), dtype=np.int32)
                ret.append(interfaces)

            self._interfaces = ret
        return self._interfaces

    def interface_points(self, substrate_label: int) -> List[npt.NDArray[np.int32]]:
        """
        Return the substrate-liquid interface points on a substrate.

        Parameters
        ----------
        substrate_label: positive int
            Label of the substrate region in :meth:`SubstrateBase.regions`.

        Returns
        -------
        points: list of point vector arrays
            Each array represents the interface points in each coating layer
            contour over the substrate. Arrays are sorted in the order of
            appearance along the substrate contour.

        See Also
        --------
        interfaces
        surface_points
        """
        if substrate_label < 1:
            return []

        indice_arr = self.interfaces()[substrate_label - 1]
        points = []
        for cnt_idx, i0, i1 in indice_arr:
            cnt = self.layer_contours()[cnt_idx]
            if i0 < i1:
                pt = cnt[i0 : i1 + 1]
            else:
                pt = np.concatenate([cnt[i0:], cnt[: i1 + 1]])
            points.append(pt)
        return points

    def surface_points(self, substrate_label: int) -> List[npt.NDArray[np.int32]]:
        """
        Return the surface points on a substrate.

        Surface points are the points from a layer region contour that are not
        adjacent to a substrate of interest. This includes not only the free
        surface (liquid-air interface) but also the interfaces with other
        substrate regions.

        Parameters
        ----------
        substrate_label: positive int
            Label of the substrate region in :meth:`SubstrateBase.regions`.

        Returns
        -------
        points: list of point vector arrays
            Each array represents the surface points in each coating layer
            contour over the substrate. Arrays are sorted in the order of
            appearance along the substrate contour.

        See Also
        --------
        interfaces
        interface_points
        """
        if substrate_label < 1:
            return []

        indice_arr = self.interfaces()[substrate_label - 1]
        points = []
        for cnt_idx, i0, i1 in indice_arr:
            cnt = self.layer_contours()[cnt_idx]
            if i0 < i1:
                pt = np.concatenate([cnt[i1 + 1 :], cnt[:i0]])
            else:
                pt = cnt[i1 + 1 : i0]
            points.append(pt)
        return points

    @abc.abstractmethod
    def examine(self) -> Optional[CoatingLayerError]:
        """
        Check the sanity of parameters.

        If the instance is invalid, return error instance.
        Else, return :obj:`None`.
        """

    def verify(self):
        """
        Verify if all parameters are suitably set by raising error on failure.

        To implement sanity check for concrete class, define :meth:`examine`.
        """
        err = self.examine()
        if err is not None:
            raise err

    def valid(self) -> bool:
        """
        Verify if all parameters are suitably set by returning boolean value.

        To implement sanity check for concrete class, define :meth:`examine`.
        """
        err = self.examine()
        ret = True
        if err is not None:
            ret = False
        return ret

    @abc.abstractmethod
    def draw(self) -> npt.NDArray[np.uint8]:
        """
        Decorate and return the coated substrate image as RGB format, using
        :meth:`draw_options` and :meth:`deco_options`.
        """

    @abc.abstractmethod
    def analyze_layer(self) -> Tuple:
        """Analyze the coated substrate image and return the data in tuple."""

    def analyze(self) -> DataType:
        """
        Return the result of :meth:`analyze_layer` as dataclass instance.
        """
        return self.Data(*self.analyze_layer())


@dataclasses.dataclass(frozen=True)
class LayerAreaParameters:
    """Additional parameters for :class:`LayerArea` instance."""

    pass


class BackgroundDrawMode(enum.Enum):
    """
    Option to determine how the background of the coating layer image is drawn.

    Attributes
    ==========

    ORIGINAL
        Show the original background.

    BINARY
        Show the background as binarized image.

    EMPTY
        Do not show the background. Only the layer will be drawn.

    """

    ORIGINAL = "ORIGINAL"
    BINARY = "BINARY"
    EMPTY = "EMPTY"


class SubtractionDrawMode(enum.Flag):
    """
    Option to determine how the template matching result will be displayed.

    Template matching result is shown by subtracting the pixels from the
    background.

    Attributes
    ==========

    NONE
        Do not show the template matching result.

    TEMPLATE
        Subtract the template ROI.

    SUBSTRRATE
        Subtract the substrate ROI.

    FULL
        Subtract both template and substrate ROIs.

    """

    NONE = 0
    TEMPLATE = 1
    SUBSTRATE = 2
    FULL = TEMPLATE | SUBSTRATE


@dataclasses.dataclass
class LayerAreaDrawOptions:
    """Basic drawing options for :class:`LayerArea` instance."""

    background: BackgroundDrawMode = BackgroundDrawMode.ORIGINAL
    subtract_mode: SubtractionDrawMode = SubtractionDrawMode.NONE


@dataclasses.dataclass
class LayerAreaDecoOptions:
    """
    Coating layer decorating options for :class:`LayerArea`.

    """

    layer: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(0, 0, 255), thickness=-1
    )


@dataclasses.dataclass
class LayerAreaData:
    """
    Analysis data for :class:`LayerArea`.

    Parameters
    ==========

    Area
        Number of the pixels in cross section image of coating layer.

    """

    Area: int


class LayerArea(
    CoatingLayerBase[
        SubstrateBase,
        LayerAreaParameters,
        LayerAreaDrawOptions,
        LayerAreaDecoOptions,
        LayerAreaData,
    ]
):
    """
    Class to analyze the cross section area of coating layer regions over
    substrate with arbitrary shape. Area unit is number of pixels.

    Examples
    ========

    Construct substrate reference instance first.

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import (SubstrateReference,
       ...     get_samples_path)
       >>> ref_path = get_samples_path("ref1.png")
       >>> ref_img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
       >>> tempROI = (200, 50, 1200, 200)
       >>> substROI = (400, 175, 1000, 500)
       >>> ref = SubstrateReference(ref_img, tempROI, substROI)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    Construct substrate instance from reference instance.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import Substrate
       >>> subst = Substrate(ref)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    Construct :class:`LayerArea` from substrate instance. :meth:`analyze` returns
    the number of pixels in coating area region.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import LayerArea
       >>> coat_path = get_samples_path("coat1.png")
       >>> coat_img = cv2.cvtColor(cv2.imread(coat_path), cv2.COLOR_BGR2RGB)
       >>> coat = LayerArea(coat_img, subst)
       >>> coat.analyze()
       LayerAreaData(Area=44348)
       >>> plt.imshow(coat.draw()) #doctest: +SKIP

    :attr:`draw_options` controls the overall visualization.

    .. plot::
       :include-source:
       :context: close-figs

       >>> coat.draw_options.subtract_mode = coat.SubtractionDrawMode.FULL
       >>> plt.imshow(coat.draw()) #doctest: +SKIP

    :attr:`deco_options` controls the decoration of coating layer reigon.

    .. plot::
       :include-source:
       :context: close-figs

       >>> coat.deco_options.layer.color.red = 255
       >>> plt.imshow(coat.draw()) #doctest: +SKIP

    """

    Parameters = LayerAreaParameters
    DrawOptions = LayerAreaDrawOptions
    DecoOptions = LayerAreaDecoOptions
    Data = LayerAreaData

    BackgroundDrawMode: TypeAlias = BackgroundDrawMode
    SubtractionDrawMode: TypeAlias = SubtractionDrawMode

    def examine(self) -> None:
        return None

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

        return image

    def analyze_layer(self) -> Tuple[int]:
        area = np.count_nonzero(self.extract_layer())
        return (area,)


def find_polyline_projections(points, lines) -> npt.NDArray[np.float64]:
    """
    Find the projection relation between points and a polyline.

    Parameters
    ----------
    points: ndarray
        Coordinate of the points.
        The shape must be `(N, 1, D)` where `N` is the number of points and `D`
        is the dimension.
    lines: ndarray
        Vertices of a polyline.
        The shape must be `(M, 1, D)` where `M` is the number of vertices and
        `D` is the dimension.

    Returns
    -------
    ndarray
        Projection relation between points and polyline segments.
        The shape is `(N, 2)`, where `N` is same as the number of points.
        Index of the row represents the index of point in *points*. Columns
        represent the index of segment where the point is projected to, and
        scalar to find the projection.
        For example, let `n`-th row is `[m, t]`. This means that `n`-th point in
        *point* is projected to the line between `m`-th vertex (A) and `m + 1`-th
        vertex (B) in *lines*. The projection point is `A + t*(B - A)`.
    """
    # TODO: if this is too slow, consider using numba.
    points = points.transpose(1, 0, 2)
    Ap = (points - lines)[:-1]
    AB = np.diff(lines, axis=0)
    t = np.sum(Ap * AB, axis=-1) / np.sum(AB * AB, axis=-1)
    Projs = lines[:-1] + (t[..., np.newaxis] * AB)
    dists = np.linalg.norm(Projs - points, axis=-1)

    valid_t = (0 <= t) & (t <= 1)
    valid_lines, valid_points = np.where(valid_t)
    # One point can be projected to multiple line segments. Therefore group the
    # projections by point and find the one with minimum distance.
    # https://stackoverflow.com/a/23161720/11501976
    pt_idxs = np.split(
        np.arange(len(valid_points))[np.argsort(valid_points)],
        np.cumsum(np.bincount(valid_points)[:-1]),
    )

    # line_t[N] : (M, t) -> Nth point is projected to Mth line with scalar t.
    line_t = np.zeros((len(pt_idxs), 2), dtype=np.float64)
    for idx in pt_idxs:
        pt = valid_points[idx]
        line = valid_lines[idx]
        mindist_idx = np.argmin(dists[line, pt])
        mindist_t = t[line[mindist_idx], pt[mindist_idx]]
        line_t[pt[mindist_idx]] = [line[mindist_idx], mindist_t]
    return line_t
