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
            _, labels = cv2.connectedComponents(cv2.bitwise_not(self.binary_image()))
            points = self.substrate_point() + self.substrate.nestled_points()
            x, y = points.T
            subst_comps = np.unique(labels[y, x])

            self._coated_substrate = np.isin(labels, subst_comps)
        return self._coated_substrate

    def extract_layer(self) -> npt.NDArray[np.bool_]:
        """Extract the coating layer as binary array from *self.image*."""
        if not hasattr(self, "_extracted_layer"):
            coated_mask = self.coated_substrate()
            # remove the substrate
            subst_mask = ~self.substrate.binary_image().astype(bool)
            x0, y0 = self.substrate_point()
            ret = images_ANDXOR(coated_mask, subst_mask, (x0, y0))
            ret[:y0, :] = False
            self._extracted_layer = ret
        return self._extracted_layer

    def layer_contours(self) -> List[npt.NDArray[np.int32]]:
        """
        Return the contours of the coating layer surfaces.

        The contours include both gas-liquid interface and substrate-liquid
        interface of the coating layer. When the coating layer is discontinuous,
        multiple contours are returned.
        """
        if not hasattr(self, "_layer_contours"):
            contours, _ = cv2.findContours(
                self.extract_layer().astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            self._layer_contours = list(contours)
        return self._layer_contours

    def interfaces(
        self,
    ) -> Tuple[List[npt.NDArray[np.int32]], List[npt.NDArray[np.int32]]]:
        """
        Return substrate-liquid interface points and liquid-gas interface points.

        Two lists have same length, which is the number of discrete coating layers.
        Points on each list are sorted counter-clockwise in the image.
        """
        # along the substrate contour, find the points which belong to the layer
        layer_cnt = self.layer_contours()
        layer_map = np.zeros(self.image.shape[:2], dtype=np.uint8)
        for i, cnt in enumerate(layer_cnt):
            ((x, y),) = cnt.transpose(1, 2, 0)
            layer_map[y, x] = i + 1
        subst_dilated = cv2.dilate(
            cv2.bitwise_not(self.substrate.binary_image()), np.ones((3, 3))
        )
        (subst_cnt,), _ = cv2.findContours(
            subst_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        subst_cnt_in_self = self.substrate_point() + subst_cnt
        ((subst_x, subst_y),) = subst_cnt_in_self.transpose(1, 2, 0)
        subst_labels = layer_map[subst_y, subst_x]

        # for discrete layers, which appears first along the substrate contour?
        cnt_order = []
        for label in range(len(layer_cnt)):
            (lab_idx,) = np.where(subst_labels == label + 1)
            first_idx = lab_idx[0]
            cnt_order.append(first_idx)

        # separate each contour into substrate-layer and layer-gas
        subst_liq, liq_gas = [], []
        for label in subst_labels[np.sort(np.array(cnt_order))]:
            cnt = layer_cnt[label - 1]
            sl = subst_cnt_in_self[np.where(subst_labels == label)]
            (i0,), _ = np.where(np.all(cnt == sl[0], axis=-1))
            (i1,), _ = np.where(np.all(cnt == sl[-1], axis=-1))
            if i0 <= i1:
                lg = cnt[i0 + 1 : i1]
            else:
                lg = np.concatenate([cnt[i0 + 1 :], cnt[:i1]])

            subst_liq.append(sl)
            liq_gas.append(lg)
        return (subst_liq, liq_gas)

    def surface(self) -> npt.NDArray[np.int32]:
        """
        Return the free surface of the coating layer.

        The result is continuous points sorted counter-clockwise in the image.
        Discontinuous coating layer is connected by the substrate surface.

        See Also
        ========

        layer_contours
            Contours for each discrete coating layer region.

        interfaces
            Substrate-liquid interfaces and gas-liquid interfaces for each
            discrete coating layer region.
        """
        sl_interfaces, _ = self.interfaces()
        if len(sl_interfaces) == 0:
            return np.empty((0, 1, 2), dtype=np.int32)
        sl_points = np.concatenate(sl_interfaces)
        if len(sl_points) == 0:
            return np.empty((0, 1, 2), dtype=np.int32)
        p0, p1 = sl_points[0], sl_points[-1]
        (cnt,), _ = cv2.findContours(
            self.coated_substrate().astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        idx0 = np.argmin(np.linalg.norm(cnt - p0, axis=-1))
        idx1 = np.argmin(np.linalg.norm(cnt - p1, axis=-1))
        return cnt[int(idx0) : int(idx1 + 1)]

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
