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
from .util.geometry import closest_in_polylines
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

    def regions(self) -> Tuple[int, npt.NDArray[np.int32]]:
        """
        Return the coated substrate image labelled by region indices.

        Returns
        -------
        retval
            Number of label values in *labels*.
        labels
            Labelled image.

        Notes
        -----
        Regions that are not connected to the substrate are considered as
        image artifacts, and are thus removed.
        """
        _, labels = cv2.connectedComponents(cv2.bitwise_not(self.binary_image()))
        pts = self.substrate_point() + self.substrate.region_points()
        subst_lab = np.unique(labels[pts[..., 1], pts[..., 0]])
        retval = len(subst_lab) + 1

        substrate_map = subst_lab.reshape(-1, 1, 1) == labels[np.newaxis, ...]
        labels[:] = 0
        for i in range(1, retval):
            labels[substrate_map[i - 1, ...]] = i

        return (retval, labels)

    def contour2(
        self, region_index: int
    ) -> Tuple[Tuple[npt.NDArray[np.int32]], npt.NDArray[np.int32]]:
        _, lab = self.regions()
        cnt = cv2.findContours(
            (lab == region_index) * np.uint8(255),
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_NONE,
        )
        return cnt

    def extract_layer(self) -> npt.NDArray[np.bool_]:
        """Extract the coating layer as binary array from *self.image*."""
        if not hasattr(self, "_extracted_layer"):
            _, regions = self.regions()
            coated_mask = regions.astype(bool)
            # remove the substrate
            subst_mask = self.substrate.regions()[1].astype(bool)
            x0, y0 = self.substrate_point()
            ret = images_ANDXOR(coated_mask, subst_mask, (x0, y0))
            ret[:y0, :] = False
            self._extracted_layer = ret
        return self._extracted_layer

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
            x, y = (self.substrate_point() + self.substrate.region_points()).T
            labels = img[y, x]
            self._coated_substrate = np.isin(img, labels)
        return self._coated_substrate

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

    def interfaces(self, substrate_region: int) -> List[List[npt.NDArray[np.float64]]]:
        r"""
        Find interfaces on each substrate contour with each layer contour.

        Parameters
        ----------
        substrate_region : int
            Index of the substrate region.

        Returns
        -------
        list
            List of list of arrays.
            - 1st-level list represents substrate contours.
            - 2nd-level list represents layer contours.
            - Array represents the interval for the interfaces.

        Notes
        -----
        Substrate can consist of many regions, each possibly having multiple
        contours. *substrate_region* decides which substrate region should the
        interfaces be searched from. This is same as the top-level index of
        :meth:`SubstrateBase.contours`.

        Once the substrate region is determined, interfaces of `j`-th layer
        contour on `i`-th substrate contour can be acquired by indexing the
        result with `[i][j]`. It will return an array whose shape is `(k, 2)`,
        where `k` is the number of interface intervals.

        Each interval describes continuous patch on the substrate contour covered
        by the layer. Two column values are the parameters that represent the
        starting point and ending point of the patch, respectively.
        Each parameter is a non-negative real number. The integer part is the
        index of polyline segment in substrate contour. The decimal part is the
        position of the point on the segment.

        For example, with parameter :math:`t` the point :math:`P(t)` is

        .. math::

            P(t) = P_{[t]} + (t - [t])(P_{[t] + 1} - P_{[t]})

        where :math:`P_i` is i-th point of the substrate contour.

        """
        ret = []
        for subst_cnt in self.substrate.contours(substrate_region)[0]:
            subst_cnt = subst_cnt + self.substrate_point()  # DON'T USE += !!
            subst_cnt = np.concatenate([subst_cnt, subst_cnt[:1]])  # closed line

            cnt_img = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.drawContours(cnt_img, [subst_cnt], -1, 255)
            dilated_cnt = cv2.dilate(cnt_img, np.ones((3, 3))).astype(bool)

            interfaces = []
            for layer_cnt in self.layer_contours():
                x, y = layer_cnt.transpose(2, 0, 1)
                mask = dilated_cnt[y, x]
                # Two reasons can be accounted for mask discontinuity:
                # (A) Starting point of contour lies on the interface, therefore
                # index jumps from the last to the first.
                # (B) Discrete interface.

                # First, roll the array to handle (A).
                noninterface_idx, _ = np.nonzero(~mask)

                if noninterface_idx.size == len(layer_cnt):
                    # layer is not adjacent to the substrate
                    interfaces.append(np.empty((0, 2), dtype=np.float64))
                    continue

                if noninterface_idx.size == 0:
                    # every point of the layer is interface
                    (interface_pts,) = layer_cnt.transpose(1, 0, 2)
                    interface_idxs = np.arange(len(layer_cnt))
                else:
                    SHIFT = len(layer_cnt) - noninterface_idx[-1] - 1
                    shifted_mask = np.roll(mask, SHIFT, axis=0)
                    interface_pts = np.roll(layer_cnt, SHIFT, axis=0)[shifted_mask]
                    interface_idxs = np.roll(
                        np.arange(len(layer_cnt))[..., np.newaxis], SHIFT, axis=0
                    )[shifted_mask]

                # Now, detect the discontinuity to handle (B)
                diff = np.diff(interface_idxs)
                (jumps,) = np.nonzero((diff != 1) & (diff != -(len(layer_cnt) - 1)))
                # Find projection from interface points (in layer contour)
                # onto substrate, and split by discontinuities.
                (prj,) = closest_in_polylines(
                    interface_pts[:, np.newaxis],
                    subst_cnt.transpose(1, 0, 2),
                ).T

                intervals = np.zeros((len(jumps) + 1, 2), dtype=np.float64)
                for i, prj in enumerate(np.split(prj, jumps, axis=0)):
                    # Store as interval. Points are reversed (-1 is the first)
                    # because prj is sorted by the direction of interface, which
                    # is opposite to the substrate contour.
                    intervals[i] = prj[[-1, 0]]

                # merge overlapping intervals
                # https://www.geeksforgeeks.org/merging-intervals/
                intervals = intervals[np.argsort(intervals[:, 0])]
                idx = 0
                for i in range(1, len(intervals)):
                    if intervals[idx][1] >= intervals[i][0]:
                        intervals[idx][1] = max(intervals[idx][1], intervals[i][1])
                    else:
                        idx += 1
                        intervals[idx] = intervals[i]
                interfaces.append(intervals[: idx + 1])

            ret.append(interfaces)

        return ret

    def interfaces2(self, substrate_region: int) -> List[List[npt.NDArray[np.int64]]]:
        r"""
        Find indices of solid-liquid interfaces on substrate contour.

        Parameters
        ----------
        substrate_region : int
            Index of the substrate region.

        Returns
        -------
        list
            List of list of arrays.
            - 1st-level list represents substrate contours.
            - 2nd-level list represents layer contours.
            - Array contains indices for the interface intervals.

        Notes
        -----
        Substrate can consist of many regions, each possibly having multiple
        contours. *substrate_region* decides which substrate region should the
        interfaces be searched from. This is equal to the top-level index of
        :meth:`SubstrateBase.contours2`.

        Once the substrate region is determined, interfaces of `j`-th layer
        contour on `i`-th substrate contour can be acquired by indexing the
        result with `[i][j]`. It will return an array whose shape is `(k, 2)`,
        where `k` is the number of interface intervals. If the layer does not
        touch the substrate, `k` is zero. On the other hand, `k` can be larger
        than 1 if the layer touches the substrate over several discontinuous
        regions.

        Each interval describes continuous patch on the substrate contour covered
        by the layer. Two column values are the indices for the starting index
        and ending index of the patch, respectively. To acquire the interface
        points, slice :meth:`SubstrateBase.contours2` with the indices.

        Examples
        --------

        .. plot::
           :include-source:
           :context: reset

           >>> import cv2
           >>> from dipcoatimage.finitedepth import (SubstrateReference,
           ...     Substrate, LayerArea, get_samples_path)
           >>> ref_path = get_samples_path("ref1.png")
           >>> ref_img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
           >>> tempROI = (200, 50, 1200, 200)
           >>> substROI = (400, 175, 1000, 500)
           >>> ref = SubstrateReference(ref_img, tempROI, substROI)
           >>> subst = Substrate(ref)
           >>> coat_path = get_samples_path("coat1.png")
           >>> coat_img = cv2.cvtColor(cv2.imread(coat_path), cv2.COLOR_BGR2RGB)
           >>> coat = LayerArea(coat_img, subst)
           >>> (R, S, L) = (0, 0, 0)
           >>> layer_contour = coat.layer_contours()[L] - coat.substrate_point()
           >>> import matplotlib.pyplot as plt  #doctest: +SKIP
           >>> for (i0, i1) in coat.interfaces2(R)[S][L]:  #doctest: +SKIP
           ...     subst_cnt = subst.contours2(R)[0][S]
           ...     plt.plot(*subst_cnt[i0:i1].transpose(2, 0, 1), "x")
           >>> plt.plot(*subst_cnt.transpose(2, 0, 1))  #doctest: +SKIP
           >>> plt.plot(*layer_contour.transpose(2, 0, 1))  #doctest: +SKIP

        """
        ret = []
        for subst_cnt in self.substrate.contours(substrate_region)[0]:
            subst_cnt = subst_cnt + self.substrate_point()  # DON'T USE += !!

            interfaces = []
            for layer_cnt in self.layer_contours():
                lcnt_img = np.zeros(self.image.shape[:2], dtype=np.uint8)
                lcnt_img[layer_cnt[..., 1], layer_cnt[..., 0]] = 255
                dilated_lcnt = cv2.dilate(lcnt_img, np.ones((3, 3))).astype(bool)

                x, y = subst_cnt.transpose(2, 0, 1)
                mask = dilated_lcnt[y, x]

                # Find indices of continuous True blocks
                idxs = np.where(
                    np.diff(np.concatenate(([False], mask[:, 0], [False]))) == 1
                )[0].reshape(-1, 2)

                interfaces.append(idxs)
            ret.append(interfaces)
        return ret

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

    layer: FeatureDrawingOptions = dataclasses.field(
        default_factory=lambda: FeatureDrawingOptions(
            color=Color(0, 0, 255), thickness=-1
        )
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
