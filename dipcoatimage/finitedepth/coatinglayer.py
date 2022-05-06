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

"""


import abc
import cv2  # type: ignore
import dataclasses
import numpy as np
import numpy.typing as npt
from typing import TypeVar, Generic, Type, Optional, Tuple
from .substrate import SubstrateBase
from .util import DataclassProtocol


__all__ = [
    "CoatingLayerError",
    "CoatingLayerBase",
]


ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)
DecoOptionsType = TypeVar("DecoOptionsType", bound=DataclassProtocol)


class CoatingLayerError(Exception):
    """Base class for error from :class:`CoatingLayerBase`."""

    pass


class CoatingLayerBase(
    abc.ABC, Generic[ParametersType, DrawOptionsType, DecoOptionsType]
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

    Type of analysis result differs by type of :attr:`substrate`.
    :meth:`data_type` and :meth:`analyze_layer` must be compatibly implemented to
    return the analysis result of :attr:`image`, whatever analysis user may
    define.

    Parameters
    ==========

    image
        Reference image. May be grayscale or RGB.

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

        return super().__init_subclass__()

    @classmethod
    @abc.abstractmethod
    def data_type(cls, substtype: Type[SubstrateBase]) -> Type[DataclassProtocol]:
        """
        Type of data acquired by analyzing :attr:`image` with :attr:`substrate`.

        Result serves as type for the result of :meth:`analyze`, and as data
        header for exporting analysis result.
        """
        pass

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

    def analyze(self) -> DataclassProtocol:
        """
        Analyze the coated substrate image and return the data.

        Result is :meth:`analyze_layer` wrapped by :meth:`data_type`.
        """
        datatype = self.data_type(type(self.substrate))
        return datatype(*self.analyze_layer())
