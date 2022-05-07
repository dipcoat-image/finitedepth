"""
Coated Substrate Image
======================

:mod:`dipcoatimage.finitedepth.coatinglayer` provides class to extract the
coating layer region from coated substrate image.

Base class
----------

.. autoclass:: CoatingLayerError
   :members:

.. autoclass:: CoatingLayerBase
   :members:

Implementation
--------------

.. autoclass:: CoatingLayerParameters
   :members:

.. autoclass:: CoatingLayerDrawMode
   :members:

.. autoclass:: CoatingLayerDrawOptions
   :members:

.. autoclass:: CoatingLayerDecoOptions
   :members:

.. autoclass:: CoatingLayerData
   :members:

.. autoclass:: CoatingLayer
   :members:

"""


import abc
import cv2  # type: ignore
import dataclasses
import enum
import numpy as np
import numpy.typing as npt
from typing import TypeVar, Generic, Type, Optional, Tuple

try:
    from typing import TypeAlias  # type: ignore
except ImportError:
    from typing_extensions import TypeAlias  # type: ignore
from .substrate import SubstrateBase
from .util import DataclassProtocol, ThresholdParameters


__all__ = [
    "CoatingLayerError",
    "CoatingLayerBase",
    "CoatingLayerParameters",
    "CoatingLayerDrawMode",
    "CoatingLayerDrawOptions",
    "CoatingLayerDecoOptions",
    "CoatingLayerData",
    "CoatingLayer",
]


ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)
DecoOptionsType = TypeVar("DecoOptionsType", bound=DataclassProtocol)
DataType = TypeVar("DataType", bound=DataclassProtocol)


class CoatingLayerError(Exception):
    """Base class for error from :class:`CoatingLayerBase`."""

    pass


class CoatingLayerBase(
    abc.ABC, Generic[ParametersType, DrawOptionsType, DecoOptionsType, DataType]
):
    """
    Abstract base class for coating layer.

    Coating layer class extracts the coating layer region from coated substrate
    image and analyze it.

    .. rubric:: Constructor

    Constructor signature must not be modified because high-level API use factory
    to generate substrate instances. Additional parameters can be introduced by
    definig class attribute :attr:`Parameters`, :attr:`DrawOptions` and
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
    attr:`draw_options` and :attr:`deco_options`.

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
        "_extracted_layer",
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    DecoOptions: Type[DecoOptionsType]
    Data: Type[DataType]

    def __init_subclass__(cls) -> None:
        params = getattr(cls, "Parameters", None)
        if params is None:
            raise TypeError(f"{cls} has no attribute 'Parameters'.")
        elif not (isinstance(params, type) and dataclasses.is_dataclass(params)):
            raise TypeError(f"{params} is not dataclass type.")
        elif not params.__dataclass_params__.frozen:  # type: ignore
            raise TypeError(f"{params} is not frozen.")

        drawopts = getattr(cls, "DrawOptions", None)
        if drawopts is None:
            raise TypeError(f"{cls} has no attribute 'DrawOptions'.")
        elif not (isinstance(drawopts, type) and dataclasses.is_dataclass(drawopts)):
            raise TypeError(f"{drawopts} is not dataclass type.")

        decoopts = getattr(cls, "DecoOptions", None)
        if decoopts is None:
            raise TypeError(f"{cls} has no attribute 'DecoOptions'.")
        elif not (isinstance(decoopts, type) and dataclasses.is_dataclass(decoopts)):
            raise TypeError(f"{decoopts} is not dataclass type.")

        data = getattr(cls, "Data", None)
        if data is None:
            raise TypeError(f"{cls} has no attribute 'Data'.")
        elif not (isinstance(data, type) and dataclasses.is_dataclass(data)):
            raise TypeError(f"{data} is not dataclass type.")

        return super().__init_subclass__()

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        substrate: SubstrateBase,
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
    def substrate(self) -> SubstrateBase:
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
            if len(self.image.shape) == 2:
                gray = self.image
            elif len(self.image.shape) == 3:
                ch = self.image.shape[-1]
                if ch == 1:
                    gray = self.image
                elif ch == 3:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
                else:
                    raise TypeError(f"Image with invalid channel: {self.image.shape}")
            else:
                raise TypeError(f"Invalid image shape: {self.image.shape}")
            _, ret = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            if ret is None:
                ret = np.empty((0, 0))
            self._binary_image = ret
        return self._binary_image

    def template_point(self) -> Tuple[int, int]:
        """
        Upper left point in ``(x, y)`` where the matched template is located.

        Notes
        =====

        This method is cached.

        """
        if not hasattr(self, "_template_point"):
            x0, y0, x1, y1 = self.substrate.reference.templateROI
            template = self.substrate.reference.image[y0:y1, x0:x1]
            res = cv2.matchTemplate(self.image, template, cv2.TM_CCOEFF)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            self._template_point = tuple(max_loc)  # type: ignore
        return self._template_point  # type: ignore

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
            for (x, y) in self.substrate.nestled_points():
                label_point = (x0 + x, y0 + y)
                label = labels[label_point[::-1]]
                image[labels == label] = 0

            # remove the substrate
            h, w = self.substrate.image().shape[:2]
            x1 = x0 + w
            y1 = y0 + h

            cropped_img = image[max(y0, 0) : min(y1, H), max(x0, 0) : min(x1, W)]

            cropped_subs = self.substrate.binary_image()[
                max(-y0, 0) : min(H - y0, h), max(-x0, 0) : min(W - x0, w)
            ]
            xor = cv2.bitwise_xor(cropped_img, cropped_subs)
            nxor = cv2.bitwise_not(xor)
            image[max(y0, 0) : min(y1, H), max(x0, 0) : min(x1, W)] = nxor

            # remove the area outside of the ROI
            image[:y0, :] = 255
            image[:, :x0] = 255
            image[:, x1:] = 255
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
class CoatingLayerParameters:
    """Additional parameters for :class:`CoatingLayer` instance."""

    threshold: ThresholdParameters = ThresholdParameters()


class CoatingLayerDrawMode(enum.Enum):
    """
    Option for :class:`CoatingLayerDrawOptions` to determine how the coated
    substrate image is drawn.

    Attributes
    ==========

    ORIGINAL
        Show the original coated substrate image.

    BINARY
        Show the binarized coated substrate image.

    """

    ORIGINAL = "ORIGINAL"
    BINARY = "BINARY"


@dataclasses.dataclass
class CoatingLayerDrawOptions:
    """
    Drawing options for :class:`CoatingLayer()`.

    Parameters
    ==========

    draw_mode

    remove_substrate
        Flag whether to remove the substrate from the image.

    decorate
        Flag wheter to decorate the coating layer.

    """

    draw_mode: CoatingLayerDrawMode = CoatingLayerDrawMode.ORIGINAL
    remove_substrate: bool = False
    decorate: bool = True


@dataclasses.dataclass
class CoatingLayerDecoOptions:
    """
    Coating layer decorating options for :class:`CoatingLayer`.

    Parameters
    ==========

    layer_color
        RGB color to paint the coating layer.

    """

    layer_color: Tuple[int, int, int] = (0, 0, 255)


@dataclasses.dataclass
class CoatingLayerData:
    """
    Coating layer shape data for :class:`CoatingLayer`.

    Parameters
    ==========

    Area
        Number of the pixels in cross section image of coating layer.

    """

    Area: int


class CoatingLayer(
    CoatingLayerBase[
        CoatingLayerParameters,
        CoatingLayerDrawOptions,
        CoatingLayerDecoOptions,
        CoatingLayerData,
    ]
):
    Parameters: TypeAlias = CoatingLayerParameters
    DrawOptions: TypeAlias = CoatingLayerDrawOptions
    DecoOptions: TypeAlias = CoatingLayerDecoOptions
    Data: TypeAlias = CoatingLayerData

    DrawMode = CoatingLayerDrawMode
    Draw_Original = CoatingLayerDrawMode.ORIGINAL
    Draw_Binary = CoatingLayerDrawMode.BINARY

    def binary_image(self) -> npt.NDArray[np.uint8]:
        """
        Binarized :attr:`image` using :meth:`parameters`.

        Notes
        =====

        This method is cached. Do not modify its result.

        """
        if not hasattr(self, "_binary_image"):
            if len(self.image.shape) == 2:
                gray = self.image
            elif len(self.image.shape) == 3:
                ch = self.image.shape[-1]
                if ch == 1:
                    gray = self.image
                elif ch == 3:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
                else:
                    raise TypeError(f"Image with invalid channel: {self.image.shape}")
            else:
                raise TypeError(f"Invalid image shape: {self.image.shape}")
            _, ret = cv2.threshold(
                gray, **dataclasses.asdict(self.parameters.threshold)
            )
            if ret is None:
                ret = np.empty((0, 0))
            self._binary_image = ret
        return self._binary_image

    def examine(self) -> None:
        return None

    def decorate_layer(self, image: npt.NDArray[np.uint8]):
        """Decorate the coating layer in *image* by mutating."""
        decorated_layer = np.full(image.shape, 255, dtype=image.dtype)
        decorated_layer[
            ~self.extract_layer().astype(bool)
        ] = self.deco_options.layer_color
        bool_layer = np.all(decorated_layer.astype(bool), axis=2)
        mask = ~bool_layer
        image[mask] = decorated_layer[mask]

    def draw(self) -> npt.NDArray[np.uint8]:
        draw_mode = self.draw_options.draw_mode
        if self.draw_options.remove_substrate:
            ret = self.extract_layer()
        elif draw_mode == self.Draw_Original:
            image = self.image
        elif draw_mode == self.Draw_Binary:
            image = self.binary_image()
        else:
            raise TypeError("Unrecognized draw mode: %s" % draw_mode)

        if len(image.shape) == 2:
            ret = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            ch = image.shape[-1]
            if ch == 1:
                ret = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif ch == 3:
                ret = image.copy()
            else:
                raise TypeError(f"Image with invalid channel: {image.shape}")
        else:
            raise TypeError(f"Invalid image shape: {image.shape}")

        if self.draw_options.decorate:
            self.decorate_layer(ret)
        return ret

    def analyze_layer(self) -> Tuple[int]:
        layer_img = self.extract_layer()
        area = layer_img.size - np.count_nonzero(layer_img)
        return (area,)
