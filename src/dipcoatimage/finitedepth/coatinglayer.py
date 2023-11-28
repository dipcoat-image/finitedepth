"""Analyze coating layer shape.

This module defines abstract class :class:`CoatingLayerBase` and its
implementation, :class:`CoatingLayer`.
"""


import abc
import dataclasses
import enum
from typing import TYPE_CHECKING, Generic, Optional, Tuple, Type, TypeVar

import cv2
import numpy as np
import numpy.typing as npt

from .cache import attrcache
from .parameters import PatchOptions
from .substrate import SubstrateBase

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "SubstTypeVar",
    "ParamTypeVar",
    "DrawOptTypeVar",
    "DecoOptTypeVar",
    "DataTypeVar",
    "CoatingLayerBase",
    "LayerParam",
    "SubtractionMode",
    "LayerDrawOpt",
    "LayerDecoOpt",
    "LayerData",
    "CoatingLayer",
    "images_XOR",
    "images_ANDXOR",
]


SubstTypeVar = TypeVar("SubstTypeVar", bound=SubstrateBase)
"""Type variable for the substrate type of :class:`CoatingLayerBase`."""
ParamTypeVar = TypeVar("ParamTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`CoatingLayerBase.ParamType`."""
DrawOptTypeVar = TypeVar("DrawOptTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`CoatingLayerBase.DrawOptType`."""
DecoOptTypeVar = TypeVar("DecoOptTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`CoatingLayerBase.DecoOptType`."""
DataTypeVar = TypeVar("DataTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`CoatingLayerBase.DataType`."""


class CoatingLayerBase(
    abc.ABC,
    Generic[SubstTypeVar, ParamTypeVar, DrawOptTypeVar, DecoOptTypeVar, DataTypeVar],
):
    """Abstract base class for coating layer instance.

    Coating layer instance stores a substrate instance and a target image, which is
    a binary image of coated substrate. The role of coating layer instance is to
    acquire coating layer region and analyze its shape.

    Coating layer instance can visualize its data and analyze the substrate image
    by the following methods:

    * :meth:`verify`: Sanity check before the analysis.
    * :meth:`draw`: Returns visualized result.
    * :meth:`analyze`: Returns analysis result.

    Concrete subclass must assign dataclasses types to the
    following class attributes:

    * :attr:`ParamType`: Type of :attr:`parameters`.
    * :attr:`DrawOptType`: Type of :attr:`draw_options`.
    * :attr:`DecoOptType`: Type of :attr:`deco_options`.
    * :attr:`DataType`: Type of :meth:`analyze`.

    Arguments:
        image: Binary target image.
        substrate: Substrate instance.
        parameters: Analysis parameters.
            If passed, must be an instance of :attr:`ParamType`.
            If not passed, attempts to construct :attr:`ParamType`
            instance without argument.
        draw_options: Visualization options to draw the target image.
            If passed, must be an instance of :attr:`DrawOptType`.
            If not passed, attempts to construct :attr:`DrawOptType`
            instance without argument.
        deco_options: Visualization options to draw the analysis result.
            If passed, must be an instance of :attr:`DecoOptType`.
            If not passed, attempts to construct :attr:`DecoOptType`
            instance without argument.
        tempmatch: Template matching result.
            Consists of template location in ``(x, y)`` and matching score.
            If passed, must be a template matching result betwen the
            template image from *substrate*'s reference instance and
            the target image.
            If not passed, template matching is automatically performed using
            :meth:`match_template`.
    """

    ParamType: Type[ParamTypeVar]
    """Type of :attr:`parameters.`

    This class attribute is defined but not set in :class:`CoatingLayerBase`.
    Concrete subclass must assign this attribute with frozen dataclass type.
    """
    DrawOptType: Type[DrawOptTypeVar]
    """Type of :attr:`draw_options.`

    This class attribute is defined but not set in :class:`CoatingLayerBase`.
    Concrete subclass must assign this attribute with dataclass type.
    """
    DecoOptType: Type[DecoOptTypeVar]
    """Type of :attr:`eco_options.`

    This class attribute is defined but not set in :class:`CoatingLayerBase`.
    Concrete subclass must assign this attribute with dataclass type.
    """
    DataType: Type[DataTypeVar]
    """Type of return value of :attr:`analyze.`

    This class attribute is defined but not set in :class:`CoatingLayerBase`.
    Concrete subclass must assign this attribute with dataclass type.
    """

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        substrate: SubstTypeVar,
        parameters: Optional[ParamTypeVar] = None,
        *,
        draw_options: Optional[DrawOptTypeVar] = None,
        deco_options: Optional[DecoOptTypeVar] = None,
        tempmatch: Optional[Tuple[Tuple[int, int], float]] = None,
    ):
        """Initialize the instance.

        - *image* is set to be immutable.
        - *substrate* is not type-checked in runtime.
        - *parameters* must be instance of :attr:`ParamType` or :obj:`None`.
          If :obj:`None`, a :attr:`ParamType` is attempted to be constructed.
        - *draw_options* must be instance of :attr:`DrawOptType` or :obj:`None`.
          If :obj:`None`, a :attr:`DrawOptType` is attempted to be constructed.
          If :attr:`DrawOptType`, the values are copied.
        - *deco_options* must be instance of :attr:`DecoOptType` or :obj:`None`.
          If :obj:`None`, a :attr:`DecoOptType` is attempted to be constructed.
          If :attr:`DecoOptType`, the values are copied.
        - *tempmatch* is expected to be passed by external constructor.
          If :obj:`None`, template matching is automatically performed.
        """
        super().__init__()
        self._image = image
        self._image.setflags(write=False)

        self._substrate = substrate

        if parameters is None:
            self._parameters = self.ParamType()
        else:
            if not isinstance(parameters, self.ParamType):
                raise TypeError(f"{parameters} is not instance of {self.ParamType}")
            self._parameters = parameters

        if draw_options is None:
            self._draw_options = self.DrawOptType()
        else:
            if not isinstance(draw_options, self.DrawOptType):
                raise TypeError(f"{draw_options} is not instance of {self.DrawOptType}")
            self._draw_options = dataclasses.replace(draw_options)

        if deco_options is None:
            self._deco_options = self.DecoOptType()
        else:
            if not isinstance(deco_options, self.DecoOptType):
                raise TypeError(f"{deco_options} is not instance of {self.DecoOptType}")
            self._deco_options = dataclasses.replace(deco_options)

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

        Note:
            This array is immutable to allow caching.
        """
        return self._image

    @property
    def substrate(self) -> SubstTypeVar:
        """Substrate instance for substrate and template images."""
        return self._substrate

    @property
    def parameters(self) -> ParamTypeVar:
        """Analysis parameters.

        This property returns a frozen dataclass instance.
        Its type is :attr:`ParamType`.

        Note:
            This dataclass must be frozen to allow caching.
        """
        return self._parameters

    @property
    def draw_options(self) -> DrawOptTypeVar:
        """Visualization options for targe image.

        This property returns a mutable dataclass instance.
        Its type is :attr:`DrawOptType`.

        Note:
            These options control how the target image itself is drawn
            in the visualization result. :meth:`deco_options` control
            how the analysis result is shown.
        """
        return self._draw_options

    @draw_options.setter
    def draw_options(self, options: DrawOptTypeVar):
        self._draw_options = options

    @property
    def deco_options(self) -> DecoOptTypeVar:
        """Visualization options for analysis result.

        This property returns a mutable dataclass instance.
        Its type is :attr:`DecpOptType`.

        Note:
            These options control how the analysis result is shown
            in the visualization result. :meth:`draw_options` control
            how the other features are drawn.
        """
        return self._deco_options

    @deco_options.setter
    def deco_options(self, options: DecoOptTypeVar):
        self._deco_options = options

    def match_template(
        self, image: npt.NDArray[np.uint8], template: npt.NDArray[np.uint8]
    ) -> Tuple[Tuple[int, int], float]:
        """Perform template matching between *image* and *template*.

        Template matching is performed using :func:`cv2.matchTemplate` with
        :obj:`cv2.TM_SQDIFF_NORMED`. Subclass may override this method to apply
        other algorithm.
        """
        res = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
        score, _, loc, _ = cv2.minMaxLoc(res)
        return (loc, score)

    @property
    def tempmatch(self) -> Tuple[Tuple[int, int], float]:
        """Template location and matching score."""
        return self._tempmatch

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
    def verify(self):
        """Sanity check before analysis.

        This method checks every intermediate step for analysis
        and raises error if anything is wrong. Passing this check
        should guarantee that :meth:`draw` and :meth:`analyze`
        returns without exception.
        """

    @abc.abstractmethod
    def draw(self) -> npt.NDArray[np.uint8]:
        """Return visualization result in RGB format.

        This method must always return without error. If visualization cannot be done,
        it should at least return original image.
        """

    @abc.abstractmethod
    def analyze(self) -> DataTypeVar:
        """Return analysis data of the target image.

        This method returns analysis result as a dataclass instance
        whose type is :attr:`DataType`. If analysis is impossible,
        error may be raised.
        """


@dataclasses.dataclass(frozen=True)
class LayerParam:
    """Analysis parameters for :class:`CoatingLayer`.

    This is an empty dataclass.
    """

    pass


class SubtractionMode(enum.Enum):
    """Option to determine how the template matching result will be displayed.

    Template matching result is shown by subtracting the pixels from the
    background.

    .. rubric:: **Members**

    - NONE: Do not show the template matching result.
    - TEMPLATE: Subtract the template ROI.
    - SUBSTRRATE: Subtract the substrate ROI.
    - FULL: Subtract both template and substrate ROIs.
    """

    NONE = "NONE"
    TEMPLATE = "TEMPLATE"
    SUBSTRATE = "SUBSTRATE"
    FULL = "FULL"


@dataclasses.dataclass
class LayerDrawOpt:
    """Drawing options for :class:`CoatingLayer`."""

    subtraction: SubtractionMode = SubtractionMode.NONE


@dataclasses.dataclass
class LayerDecoOpt:
    """Decorating options for :class:`CoatingLayer`.

    Arguments:
        layer: Determine how the coating layer is painted.
    """

    layer: PatchOptions = dataclasses.field(
        default_factory=lambda: PatchOptions(
            fill=True,
            edgecolor=(0, 0, 0),
            facecolor=(255, 255, 255),
            linewidth=2,
        )
    )


@dataclasses.dataclass
class LayerData:
    """Analysis data for :class:`CoatingLayer`.

    This is an empty dataclass.
    """

    pass


class CoatingLayer(
    CoatingLayerBase[
        SubstrateBase,
        LayerParam,
        LayerDrawOpt,
        LayerDecoOpt,
        LayerData,
    ]
):
    """Basic implementation of :class:`CoatingLayerBase`.

    Arguments:
        image
        substrate
        parameters (LayerParam, optional)
        draw_options (LayerDrawOpt, optional)
        deco_options (LayerDecoOpt, optional)
        tempmatch

    Examples:
        Construct substrate instance first.

        .. plot::
            :include-source:
            :context: reset

            >>> import cv2
            >>> from dipcoatimage.finitedepth import Reference, get_data_path
            >>> gray = cv2.imread(get_data_path("ref1.png"), cv2.IMREAD_GRAYSCALE)
            >>> _, im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            >>> tempROI = (200, 50, 1200, 200)
            >>> substROI = (400, 175, 1000, 500)
            >>> ref = Reference(im, tempROI, substROI)
            >>> from dipcoatimage.finitedepth import Substrate
            >>> subst = Substrate(ref)

        Construct coating layer instance from target image and the substrate instance.

        .. plot::
            :include-source:
            :context: close-figs

            >>> from dipcoatimage.finitedepth import CoatingLayer
            >>> gray = cv2.imread(get_data_path("coat1.png"), cv2.IMREAD_GRAYSCALE)
            >>> _, im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            >>> coat = CoatingLayer(im, subst)
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(coat.draw()) #doctest: +SKIP

        :attr:`draw_options` controls visualization of the substrate region.

        .. plot::
            :include-source:
            :context: close-figs

            >>> coat.draw_options.subtraction = coat.SubtractionMode.FULL
            >>> plt.imshow(coat.draw()) #doctest: +SKIP

        :attr:`deco_options` controls visualization of the coating layer region.

        .. plot::
            :include-source:
            :context: close-figs

            >>> coat.deco_options.layer.facecolor = (255, 0, 255)
            >>> plt.imshow(coat.draw()) #doctest: +SKIP
    """

    ParamType = LayerParam
    """Assigned with :class:`LayerParam`."""
    DrawOptType = LayerDrawOpt
    """Assigned with :class:`LayerDrawOpt`."""
    DecoOptType = LayerDecoOpt
    """Assigned with :class:`LayerDecoOpt`."""
    DataType = LayerData
    """Assigned with :class:`LayerData`."""

    SubtractionMode = SubtractionMode
    """Shortcut to :class:`SubtractionMode`."""

    def verify(self):
        """Implement :meth:`CoatingLayerBase.verify`."""
        pass

    def draw(self) -> npt.NDArray[np.uint8]:
        """Implement :meth:`CoatingLayerBase.draw`.

        #. Display the template matching result with :class:`SubtractionMode`.
        #. Draw the coating layer with :class:`PatchOptions`.
        """
        image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

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
            cnts, _ = cv2.findContours(
                self.extract_layer().astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            cv2.drawContours(
                image,
                cnts,
                -1,
                layer_opts.edgecolor,
                layer_opts.linewidth,
            )

        return image

    def analyze(self):
        """Implement :meth:`CoatingLayerBase.analyze`."""
        return self.DataType()


def images_XOR(
    img1: npt.NDArray[np.bool_],
    img2: npt.NDArray[np.bool_],
    point: Tuple[int, int] = (0, 0),
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
    point: Tuple[int, int] = (0, 0),
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
