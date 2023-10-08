"""Detect coating layer from coated substrate image."""


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
    "CoatingLayerBase",
    "CoatingLayer",
    "images_XOR",
    "images_ANDXOR",
]


SubstTypeVar = TypeVar("SubstTypeVar", bound=SubstrateBase)
ParamTypeVar = TypeVar("ParamTypeVar", bound="DataclassInstance")
DrawOptTypeVar = TypeVar("DrawOptTypeVar", bound="DataclassInstance")
DecoOptTypeVar = TypeVar("DecoOptTypeVar", bound="DataclassInstance")
DataTypeVar = TypeVar("DataTypeVar", bound="DataclassInstance")


class CoatingLayerBase(
    abc.ABC,
    Generic[SubstTypeVar, ParamTypeVar, DrawOptTypeVar, DecoOptTypeVar, DataTypeVar],
):
    """Abstract base class for coating layer."""

    ParamType: Type[ParamTypeVar]
    DrawOptType: Type[DrawOptTypeVar]
    DecoOptType: Type[DecoOptTypeVar]
    DataType: Type[DataTypeVar]

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
        """Initialize the instance."""
        super().__init__()
        self._image = image
        self._image.setflags(write=False)

        self._substrate = substrate

        if parameters is None:
            self._parameters = self.ParamType()
        else:
            if not isinstance(parameters, self.ParamType):
                raise TypeError(f"{parameters} is not instance of {self.ParamType}")
            self._parameters = dataclasses.replace(parameters)

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
            res = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
            score, _, loc, _ = cv2.minMaxLoc(res)
            self._tempmatch = (loc, score)
        else:
            self._tempmatch = tempmatch

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """Coated substrate image.

        This array is not writable to enable caching which requires immutability.
        """
        return self._image

    @property
    def substrate(self) -> SubstTypeVar:
        """Substrate instance passed to the constructor."""
        return self._substrate

    @property
    def parameters(self) -> ParamTypeVar:
        """Additional parameters for concrete class.

        Instance of :attr:`ParamType`, which must be a frozen dataclass.
        """
        return self._parameters

    @property
    def draw_options(self) -> DrawOptTypeVar:
        """Options to visualize the coated substrate image.

        Instance of :attr:`DrawOptType` dataclass.
        """
        return self._draw_options

    @draw_options.setter
    def draw_options(self, options: DrawOptTypeVar):
        self._draw_options = options

    @property
    def deco_options(self) -> DecoOptTypeVar:
        """Options to decorate the coating layer region.

        Instance of :attr:`DecoOptType` dataclass.
        """
        return self._deco_options

    @deco_options.setter
    def deco_options(self, options: DecoOptTypeVar):
        self._deco_options = options

    @property
    def tempmatch(self) -> Tuple[Tuple[int, int], float]:
        """Return template location and its objective function value."""
        return self._tempmatch

    def substrate_point(self) -> npt.NDArray[np.int32]:
        """Upper left point in ``(x, y)`` where the substrate is located."""
        temp_point, _ = self.tempmatch
        t = self.substrate.reference.templateROI[:2]
        s = self.substrate.reference.substrateROI[:2]
        return np.array(temp_point, dtype=np.int32) - t + s

    @attrcache("_coated_substrate")
    def coated_substrate(self) -> npt.NDArray[np.bool_]:
        """Remove image artifacts, e.g., bath surface."""
        _, img = cv2.connectedComponents(cv2.bitwise_not(self.image))
        x, y = (self.substrate_point() + self.substrate.region_points()).T
        return np.isin(img, img[y, x])

    @attrcache("_extracted_layer")
    def extract_layer(self) -> npt.NDArray[np.bool_]:
        """Extract the coating layer as binary array from *self.image*."""
        # remove the substrate
        x0, y0 = self.substrate_point()
        subst_mask = self.substrate.regions() >= 0
        ret = images_ANDXOR(self.coated_substrate(), subst_mask, (x0, y0))
        ret[:y0, :] = False
        return ret

    @abc.abstractmethod
    def verify(self):
        """Check to detect error and raise before analysis."""

    @abc.abstractmethod
    def draw(self) -> npt.NDArray[np.uint8]:
        """Decorate and return the coated substrate image.

        Result is in RGB format.

        Must return an image even if the instance is not valid.
        """

    @abc.abstractmethod
    def analyze(self) -> DataTypeVar:
        """Analyze the coated substrate image and return the data.

        May raise error if the instance is not valid.
        """


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Additional parameters for `CoatingLayer` instance."""

    pass


class SubtractionMode(enum.Enum):
    """Option to determine how the template matching result will be displayed.

    Template matching result is shown by subtracting the pixels from the
    background.

    Members
    -------
    NONE
        Do not show the template matching result.
    TEMPLATE
        Subtract the template ROI.
    SUBSTRRATE
        Subtract the substrate ROI.
    FULL
        Subtract both template and substrate ROIs.
    """

    NONE = "NONE"
    TEMPLATE = "TEMPLATE"
    SUBSTRATE = "SUBSTRATE"
    FULL = "FULL"


@dataclasses.dataclass
class DrawOptions:
    """Drawing options for `CoatingLayer` instance.

    Attributes
    ----------
    subtraction : SubtractionMode
    """

    subtraction: SubtractionMode = SubtractionMode.NONE


@dataclasses.dataclass
class DecoOptions:
    """Options to show the coating layer of `CoatingLayer`.

    Attributes
    ----------
    layer : PatchOptions
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
class Data:
    """Analysis data for `CoatingLayer`."""

    pass


class CoatingLayer(
    CoatingLayerBase[
        SubstrateBase,
        Parameters,
        DrawOptions,
        DecoOptions,
        Data,
    ]
):
    """Basic implementation of coating layer.

    Examples
    --------
    Construct substrate reference instance first.

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import Reference, get_data_path
       >>> gray = cv2.imread(get_data_path("ref1.png"), cv2.IMREAD_GRAYSCALE)
       >>> _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
       >>> tempROI = (200, 50, 1200, 200)
       >>> substROI = (400, 175, 1000, 500)
       >>> ref = Reference(img, tempROI, substROI)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    Construct substrate instance from reference instance.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import Substrate
       >>> subst = Substrate(ref)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    Construct `CoatingLayer` from substrate instance. :meth:`analyze` returns
    the number of pixels in coating area region.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import CoatingLayer
       >>> gray = cv2.imread(get_data_path("coat1.png"), cv2.IMREAD_GRAYSCALE)
       >>> _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
       >>> coat = CoatingLayer(img, subst)
       >>> plt.imshow(coat.draw()) #doctest: +SKIP

    :attr:`draw_options` controls the overall visualization.

    .. plot::
       :include-source:
       :context: close-figs

       >>> coat.draw_options.subtraction = coat.SubtractionMode.FULL
       >>> plt.imshow(coat.draw()) #doctest: +SKIP

    :attr:`deco_options` controls the decoration of coating layer reigon.

    .. plot::
       :include-source:
       :context: close-figs

       >>> coat.deco_options.layer.facecolor = (255, 0, 255)
       >>> plt.imshow(coat.draw()) #doctest: +SKIP
    """

    ParamType = Parameters
    DrawOptType = DrawOptions
    DecoOptType = DecoOptions
    DataType = Data

    SubtractionMode = SubtractionMode

    def verify(self):
        """Check error."""
        pass

    def draw(self) -> npt.NDArray[np.uint8]:
        """Return visualized image."""
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
        """Return analysis data."""
        return self.DataType()


def images_XOR(
    img1: npt.NDArray[np.bool_],
    img2: npt.NDArray[np.bool_],
    point: Tuple[int, int] = (0, 0),
) -> npt.NDArray[np.bool_]:
    """Subtract *img2* from *img1* at *point* by XOR operation.

    This function leaves the pixels that exist either in *img1* or *img2*. It
    can be used to visualize the template matching error.

    See Also
    --------
    images_ANDXOR
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
    """Subtract *img2* from *img1* at *point* by AND and XOR operation.

    This function leaves the pixels that exist in *img1* but not in *img2*. It
    can be used to extract the coating layer pixels.

    See Also
    --------
    images_XOR
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
