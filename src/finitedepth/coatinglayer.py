"""Analyze coating layer."""

import abc

import cv2
import numpy as np
import numpy.typing as npt

from .cache import attrcache
from .substrate import SubstrateBase

__all__ = [
    "CoatingLayerBase",
    "CoatingLayer",
    "RectLayerShape",
    "images_XOR",
    "images_ANDXOR",
]


class CoatingLayerBase(abc.ABC):
    """Abstract base class for coating layer object.

    Coating layer object stores :class:`SubstrateBase` object and target image, which is
    a binary image of coated substrate. The role of coating layer object is to acquire
    coating layer region by template matching and analyze its shape. :meth:`draw`
    returns visualized result.

    Arguments:
        image: Binary target image.
        substrate: Substrate instance.
        tempmatch: Pre-computed template matching result.
            External constructor can pass this argument to force the template matching
            result. If not passed, :meth:`match_template` performs matching.

    Attributes:
        image: Binary target image.
            This image is not writable for immutability.
        substrate: Substrate instance.
        tempmatch: Template matching location and score.
    """

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        substrate: SubstrateBase,
        *,
        tempmatch: tuple[tuple[int, ...], float] | None = None,
    ):
        """Initialize the instance.

        *image* is set to be immutable, and template matching is performed if
        *tempmatch* is ``None``.
        """
        self.image = image
        self.image.setflags(write=False)
        self.substrate = substrate

        if tempmatch is None:
            image = self.image
            x0, y0, x1, y1 = self.substrate.reference.templateROI
            template = self.substrate.reference.image[y0:y1, x0:x1]
            self.tempmatch = self.match_template(image, template)
        else:
            self.tempmatch = tempmatch

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
    def draw(self, *args, **kwargs) -> npt.NDArray[np.uint8]:
        """Return visualization result."""


class CoatingLayer(CoatingLayerBase):
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

    def draw(
        self,
        subtraction_mode: str = "none",
        layer_color: tuple[int, int, int] = (255, 0, 0),
        layer_thickness: int = -1,
    ) -> npt.NDArray[np.uint8]:
        """Subtract the template match result and paint the coating layer.

        Arguments:
            subtraction_mode (`{'none', 'template', 'substrate', 'full'}`): Subtraction
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


class RectLayerShape:
    """Coating layer over rectangular substrate."""


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
