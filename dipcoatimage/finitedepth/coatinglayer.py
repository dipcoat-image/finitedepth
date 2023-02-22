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
import numpy as np
import numpy.typing as npt
from .substrate import SubstrateBase
from .util import DataclassProtocol, BinaryImageDrawMode, binarize, colorize
from typing import TypeVar, Generic, Type, Optional, Tuple

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias


__all__ = [
    "CoatingLayerError",
    "match_template",
    "subtract_images",
    "CoatingLayerBase",
    "LayerAreaParameters",
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


def match_template(
    image: npt.NDArray[np.uint8], template: npt.NDArray[np.uint8]
) -> Tuple[float, Tuple[int, int]]:
    res = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
    score, _, loc, _ = cv2.minMaxLoc(res)
    return (score, loc)


def subtract_images(
    image1: npt.NDArray[np.uint8], image2: npt.NDArray[np.uint8], point: Tuple[int, int]
) -> npt.NDArray[np.uint8]:
    """Subtract *image2* from *image1* at *point*. Images must be binary."""
    H, W = image1.shape
    h, w = image2.shape
    x0, y0 = point

    x1, y1 = x0 + w, y0 + h
    img1_crop = image1[max(y0, 0) : min(y1, H), max(x0, 0) : min(x1, W)]
    img2_crop = image2[max(-y0, 0) : min(H - y0, h), max(-x0, 0) : min(W - x0, w)]
    xor = cv2.bitwise_xor(img1_crop, img2_crop)
    nxor = cv2.bitwise_not(xor)

    ret = image1.copy()
    ret[max(y0, 0) : min(y1, H), max(x0, 0) : min(x1, W)] = nxor
    return ret


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
        "_template_point",
        "_template_score",
        "_extracted_layer",
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

    def template_point(self) -> Tuple[int, int]:
        """
        Upper left point in ``(x, y)`` where the matched template is located.

        Template matching is performed with :obj:`cv2.TM_SQDIFF_NORMED`.

        Notes
        =====

        This method is cached. Calling this method for the first time updates
        :meth:`template_score` as well.

        """
        if not hasattr(self, "_template_point"):
            image = self.binary_image()
            x0, y0, x1, y1 = self.substrate.reference.templateROI
            template = self.substrate.reference.binary_image()[y0:y1, x0:x1]
            self._template_score, self._template_point = match_template(image, template)
        return self._template_point

    def template_score(self) -> float:
        """
        Normalized score in range ``[0, 1]`` for the matched template.

        Template matching is performed with :obj:`cv2.TM_SQDIFF_NORMED`. Zero
        score indicates complete match.

        Notes
        =====

        This method is cached. Calling this method for the first time updates
        :meth:`template_point` as well.

        """
        if not hasattr(self, "_template_score"):
            image = self.binary_image()
            x0, y0, x1, y1 = self.substrate.reference.templateROI
            template = self.substrate.reference.binary_image()[y0:y1, x0:x1]
            self._template_score, self._template_point = match_template(image, template)
        return self._template_score

    def substrate_point(self) -> Tuple[int, int]:
        """
        Upper left point in ``(x, y)`` where the substrate is located.

        """
        temp_point = self.template_point()
        temp2subst = self.substrate.reference.temp2subst()
        x0 = temp_point[0] + temp2subst[0]
        y0 = temp_point[1] + temp2subst[1]
        return (x0, y0)

    def capbridge_broken(self) -> bool:
        """
        Determines if the capillary bridge is broken in :attr:`self.image`.
        """
        _, y = self.substrate_point()
        below_subst = self.binary_image()[y:]
        # if any row is all-white, capillary bridge is broken
        row_white = np.all(below_subst, axis=1)
        return bool(np.any(row_white))

    def extract_layer(self) -> npt.NDArray[np.uint8]:
        """
        Extract the coating layer as binary array from *self.image*, where
        the substrate and any other undesired features removed.
        """
        if not hasattr(self, "_extracted_layer"):
            binimg = self.binary_image()
            H, W = binimg.shape

            neg_binimg = cv2.bitwise_not(binimg)
            _, labels = cv2.connectedComponents(neg_binimg)

            image = np.full((H, W), 255, dtype=binimg.dtype)
            x0, y0 = self.substrate_point()
            for x, y in self.substrate.nestled_points():
                label_point = (x0 + x, y0 + y)
                label = labels[label_point[::-1]]
                image[labels == label] = 0

            # remove the substrate
            substImg = self.substrate.binary_image()
            image = subtract_images(image, substImg, (x0, y0))

            # remove the area outside of the ROI
            h, w = substImg.shape
            x1, y1 = x0 + w, y0 + h
            image[:y0, :] = 255
            image[:y1, :x0] = 255
            image[:y1, x1:] = 255
            self._extracted_layer = image
        return self._extracted_layer

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


@dataclasses.dataclass
class LayerAreaDrawOptions:
    """
    Basic drawing options for :class:`LayerArea` instance.

    Parameters
    ==========

    draw_mode

    remove_substrate
        Flag whether to remove the substrate from the image.

    decorate
        Flag whether to decorate the coating layer.

    """

    draw_mode: BinaryImageDrawMode = BinaryImageDrawMode.ORIGINAL
    remove_substrate: bool = False
    decorate: bool = True


@dataclasses.dataclass
class LayerAreaDecoOptions:
    """
    Coating layer decorating options for :class:`LayerArea`.

    Parameters
    ==========

    layer_color
        RGB color to paint the coating layer.

    """

    layer_color: Tuple[int, int, int] = (0, 0, 255)


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
       >>> substROI = (400, 100, 1000, 500)
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

       >>> coat.draw_options.remove_substrate = True
       >>> plt.imshow(coat.draw()) #doctest: +SKIP

    :attr:`deco_options` controls the decoration of coating layer reigon.

    .. plot::
       :include-source:
       :context: close-figs

       >>> coat.deco_options.layer_color = (0, 255, 0)
       >>> plt.imshow(coat.draw()) #doctest: +SKIP

    """

    Parameters = LayerAreaParameters
    DrawOptions = LayerAreaDrawOptions
    DecoOptions = LayerAreaDecoOptions
    Data = LayerAreaData

    DrawMode: TypeAlias = BinaryImageDrawMode

    def examine(self) -> None:
        return None

    def draw(self) -> npt.NDArray[np.uint8]:
        draw_mode = self.draw_options.draw_mode
        if draw_mode == self.DrawMode.ORIGINAL:
            image = self.image.copy()
        elif draw_mode == self.DrawMode.BINARY:
            image = self.binary_image()
        else:
            raise TypeError("Unrecognized draw mode: %s" % draw_mode)

        mask = self.extract_layer().astype(bool)
        if self.draw_options.remove_substrate:
            image[mask] = 255

        ret = colorize(image)
        if self.draw_options.decorate:
            ret[~mask] = self.deco_options.layer_color
        return ret

    def analyze_layer(self) -> Tuple[int]:
        layer_img = self.extract_layer()
        area = layer_img.size - np.count_nonzero(layer_img)
        return (area,)
