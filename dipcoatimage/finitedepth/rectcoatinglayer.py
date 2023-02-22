"""
:mod:`dipcoatimage.finitedepth.rectcoatinglayer` provides class to analyze
the coating layer over rectangular substrate.

Base class
==========

.. autoclass:: LayerRegionFlag
   :members:

.. autoclass:: RectCoatingLayerBase
   :members:

Implementation
==============

.. autoclass:: RectLayerAreaDecoOptions
   :members:

.. autoclass:: RectLayerAreaData
   :members:

.. autoclass:: RectLayerArea
   :members:

"""

import cv2  # type: ignore
import dataclasses
import enum
import numpy as np
import numpy.typing as npt
from typing import TypeVar, Type, Tuple
from .rectsubstrate import RectSubstrate
from .coatinglayer import (
    CoatingLayerBase,
    LayerAreaParameters,
    LayerAreaDrawOptions,
)
from .util import (
    DataclassProtocol,
    BinaryImageDrawMode,
    MorphologyClosingParameters,
)

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias


__all__ = [
    "LayerRegionFlag",
    "RectCoatingLayerBase",
    "RectLayerAreaDecoOptions",
    "RectLayerAreaData",
    "RectLayerArea",
    "RectLayerShapeParameters",
    "RectLayerShapeDrawOptions",
    "RectLayerShapeDecoOptions",
    "RectLayerShapeData",
    "RectLayerShape",
    "get_extended_line",
]


class LayerRegionFlag(enum.IntFlag):
    """Label to classify the coating layer pixels by their regions."""

    BACKGROUND = 0
    LEFT = 1
    BOTTOM = 2
    RIGHT = 4


ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)
DecoOptionsType = TypeVar("DecoOptionsType", bound=DataclassProtocol)
DataType = TypeVar("DataType", bound=DataclassProtocol)


class RectCoatingLayerBase(
    CoatingLayerBase[
        RectSubstrate, ParametersType, DrawOptionsType, DecoOptionsType, DataType
    ]
):
    """
    Abstract base class for coating layer over rectangular substrate.

    :class:`RectCoatingLayerBase` is capable of classifying the coating layer
    pixels by their location relative to the substrate. To get the classification
    map, use :meth:`label_layer`.

    """

    __slots__ = ("_labelled_layer",)

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    DecoOptions: Type[DecoOptionsType]
    Data: Type[DataType]

    Region: TypeAlias = LayerRegionFlag

    def capbridge_broken(self) -> bool:
        x0, y0 = self.substrate_point()
        top = y0 + max(p[1] for (_, p) in self.substrate.vertex_points().items())
        bot, _ = self.binary_image().shape
        if top > bot:
            # substrate is located outside of the frame
            return False

        left = x0 + min(p[0] for (_, p) in self.substrate.vertex_points().items())
        right = x0 + max(p[0] for (_, p) in self.substrate.vertex_points().items())

        roi_binimg = self.binary_image()[top:bot, left:right]
        return bool(np.any(np.all(roi_binimg, axis=1)))

    def label_layer(self) -> npt.NDArray[np.uint8]:
        """
        Return the classification map of the pixels.

        Pixels are labelled with :class:`LayerRegionFlag` by their location
        relative to the substrate. The values can be combined to denote the pixel
        in the corner, i.e. ``LEFT | BOTTOM`` for the lower left region.

        """
        if not hasattr(self, "_labelled_layer"):
            h, w = self.image.shape[:2]
            ret = np.full((h, w), self.Region.BACKGROUND)

            mask = cv2.bitwise_not(self.extract_layer()).astype(bool)
            row, col = np.where(mask)
            points = np.stack([col, row], axis=1)

            p0 = np.array(self.substrate_point())
            A = p0 + np.array(
                self.substrate.vertex_points()[self.substrate.PointType.TOPLEFT]
            )
            B = p0 + np.array(
                self.substrate.vertex_points()[self.substrate.PointType.BOTTOMLEFT]
            )
            C = p0 + np.array(
                self.substrate.vertex_points()[self.substrate.PointType.BOTTOMRIGHT]
            )
            D = p0 + np.array(
                self.substrate.vertex_points()[self.substrate.PointType.TOPRIGHT]
            )

            left_of_AB = np.cross(B - A, points - A) >= 0
            under_BC = np.cross(C - B, points - B) >= 0
            right_of_CD = np.cross(D - C, points - C) >= 0

            left_x, left_y = points[left_of_AB].T
            ret[left_y, left_x] |= self.Region.LEFT

            bottom_x, bottom_y = points[under_BC].T
            ret[bottom_y, bottom_x] |= self.Region.BOTTOM

            right_x, right_y = points[right_of_CD].T
            ret[right_y, right_x] |= self.Region.RIGHT
            self._labelled_layer = ret

        return self._labelled_layer


@dataclasses.dataclass
class RectLayerAreaDecoOptions:
    """
    Decorating options for :class:`RectLayerArea`.

    Parameters
    ==========

    paint_Left
        Flag to paint the left-side region of the coating layer.

    Left_color
        RGB color to paint the left-side region of the coating layer.
        Ignored if *paint_Left* is false.

    paint_LeftCorner
        Flag to paint the left-side corner region of the coating layer.

    LeftCorner_color
        RGB color to paint the left-side corner region of the coating layer.
        Ignored if *paint_LeftCorner* is false.

    paint_Bottom
        Flag to paint the bottom region of the coating layer.

    Bottom_color
        RGB color to paint the bottom region of the coating layer.
        Ignored if *paint_Left* is false.

    paint_RightCorner
        Flag to paint the right-side corner region of the coating layer.

    RightCorner_color
        RGB color to paint the right-side corner region of the coating layer.
        Ignored if *paint_RightCorner* is false.

    paint_Right
        Flag to paint the right-side region of the coating layer.

    Right_color
        RGB color to paint the right-side region of the coating layer.
        Ignored if *paint_Right* is false.

    """

    paint_Left: bool = True
    Left_color: Tuple[int, int, int] = (0, 0, 255)
    paint_LeftCorner: bool = True
    LeftCorner_color: Tuple[int, int, int] = (0, 255, 0)
    paint_Bottom: bool = True
    Bottom_color: Tuple[int, int, int] = (0, 0, 255)
    paint_RightCorner: bool = True
    RightCorner_color: Tuple[int, int, int] = (0, 255, 0)
    paint_Right: bool = True
    Right_color: Tuple[int, int, int] = (0, 0, 255)


@dataclasses.dataclass
class RectLayerAreaData:
    """
    Analysis data for :class:`RectLayerArea`.

    Parameters
    ==========

    LeftArea, LeftCornerArea, BottomArea, RightCornerArea, RightArea
        Number of the pixels in each coating layer region.

    """

    LeftArea: int
    LeftCornerArea: int
    BottomArea: int
    RightCornerArea: int
    RightArea: int


class RectLayerArea(
    RectCoatingLayerBase[
        LayerAreaParameters,
        LayerAreaDrawOptions,
        RectLayerAreaDecoOptions,
        RectLayerAreaData,
    ]
):
    """
    Class to analyze the cross section area of coating layer regions over
    rectangular substrate. Area unit is number of pixels.

    Examples
    ========

    Construct substrate reference class first.

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

    Construct the parameters and substrate instance from reference instance.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import (CannyParameters,
       ...     HoughLinesParameters, RectSubstrate)
       >>> cparams = CannyParameters(50, 150)
       >>> hparams = HoughLinesParameters(1, 0.01, 100)
       >>> params = RectSubstrate.Parameters(cparams, hparams)
       >>> subst = RectSubstrate(ref, parameters=params)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    Construct :class:`RectLayerArea` from substrate class. :meth:`analyze`
    returns the number of pixels in coating area region.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import RectLayerArea
       >>> coat_path = get_samples_path("coat1.png")
       >>> coat_img = cv2.cvtColor(cv2.imread(coat_path), cv2.COLOR_BGR2RGB)
       >>> coat = RectLayerArea(coat_img, subst)
       >>> coat.analyze()
       RectLayerAreaData(LeftArea=8262, LeftCornerArea=173, BottomArea=27457,
                         RightCornerArea=193, RightArea=8263)
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

       >>> coat.deco_options.Bottom_color = (255, 0, 0)
       >>> plt.imshow(coat.draw()) #doctest: +SKIP

    """

    Parameters = LayerAreaParameters
    DrawOptions = LayerAreaDrawOptions
    DecoOptions = RectLayerAreaDecoOptions
    Data = RectLayerAreaData

    DrawMode: TypeAlias = BinaryImageDrawMode

    def examine(self) -> None:
        return None

    def draw(self) -> npt.NDArray[np.uint8]:
        draw_mode = self.draw_options.draw_mode
        if self.draw_options.remove_substrate:
            image = self.extract_layer()
        elif draw_mode == self.DrawMode.ORIGINAL:
            image = self.image
        elif draw_mode == self.DrawMode.BINARY:
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
            layer_label = self.label_layer()
            if self.deco_options.paint_Left:
                color = self.deco_options.Left_color
                ret[layer_label == self.Region.LEFT] = color
            if self.deco_options.paint_LeftCorner:
                color = self.deco_options.LeftCorner_color
                ret[layer_label == self.Region.LEFT | self.Region.BOTTOM] = color
            if self.deco_options.paint_Bottom:
                color = self.deco_options.Bottom_color
                ret[layer_label == self.Region.BOTTOM] = color
            if self.deco_options.paint_RightCorner:
                color = self.deco_options.RightCorner_color
                ret[layer_label == self.Region.BOTTOM | self.Region.RIGHT] = color
            if self.deco_options.paint_Right:
                color = self.deco_options.Right_color
                ret[layer_label == self.Region.RIGHT] = color
        return ret

    def analyze_layer(self) -> Tuple[int, int, int, int, int]:
        layer_label = self.label_layer()
        unique_count = dict(zip(*np.unique(layer_label, return_counts=True)))

        left_a = unique_count.get(self.Region.LEFT, 0)
        leftcorner_a = unique_count.get(self.Region.LEFT | self.Region.BOTTOM, 0)
        bottom_a = unique_count.get(self.Region.BOTTOM, 0)
        rightcorner_a = unique_count.get(self.Region.BOTTOM | self.Region.RIGHT, 0)
        right_a = unique_count.get(self.Region.RIGHT, 0)

        return (left_a, leftcorner_a, bottom_a, rightcorner_a, right_a)


@dataclasses.dataclass(frozen=True)
class RectLayerShapeParameters:
    """Analysis parameters for :class:`RectLayerShape` instance."""

    MorphologyClosing: MorphologyClosingParameters
    ReconstructRadius: int


@dataclasses.dataclass(frozen=True)
class RectLayerShapeDrawOptions:
    """Drawing options for :class:`RectLayerShape` instance."""

    pass


@dataclasses.dataclass(frozen=True)
class RectLayerShapeDecoOptions:
    """Decorating options for :class:`RectLayerShape` instance."""

    pass


@dataclasses.dataclass(frozen=True)
class RectLayerShapeData:
    """Analysis data for :class:`RectLayerShape` instance."""

    Area: int

    LayerLength_Left: float
    LayerLength_Right: float


class RectLayerShape(
    RectCoatingLayerBase[
        RectLayerShapeParameters,
        RectLayerShapeDrawOptions,
        RectLayerShapeDecoOptions,
        RectLayerShapeData,
    ]
):
    """
    Class for analyzing the shape and thickness of the coating layer over
    rectangular substrate.
    """

    __slots__ = (
        "_contactline_points",
        "_refined_layer",
    )

    Parameters = RectLayerShapeParameters
    DrawOptions = RectLayerShapeDrawOptions
    DecoOptions = RectLayerShapeDecoOptions
    Data = RectLayerShapeData

    def examine(self) -> None:
        return None

    def contactline_points(self) -> Tuple[int, int, int, int]:
        """
        Get the coordinates of the contact line points of the layer.

        Return value as ``(left x, left y, right x, right y)``.
        """
        if not hasattr(self, "_contactline_points"):
            # perform closing to remove error pixels
            img = self.extract_layer()
            closingParams = self.parameters.MorphologyClosing
            kernel = np.ones(closingParams.kernelSize)
            img_closed = cv2.morphologyEx(
                img,
                cv2.MORPH_CLOSE,
                kernel,
                anchor=closingParams.anchor,
                iterations=closingParams.iterations,
            )

            # reconstruct the remaining components around the lower corners
            # to remove large specks
            p0 = np.array(self.substrate_point())
            B = p0 + np.array(
                self.substrate.vertex_points()[self.substrate.PointType.BOTTOMLEFT]
            )
            C = p0 + np.array(
                self.substrate.vertex_points()[self.substrate.PointType.BOTTOMRIGHT]
            )
            comps, labels = cv2.connectedComponents(cv2.bitwise_not(img_closed))
            dist_thres = self.parameters.ReconstructRadius
            for i in range(1, comps):
                row, col = np.where(labels == i)
                points = np.stack([col, row], axis=1)
                left_dist = np.linalg.norm(points - B, axis=1)
                right_dist = np.linalg.norm(points - C, axis=1)
                if np.min(left_dist) > dist_thres and np.min(right_dist) > dist_thres:
                    labels[row, col] = 0
            labels[np.where(labels)] = 255  # binarize

            # get contact line points
            layer_label = self.label_layer().copy()
            layer_label[np.where(~labels.astype(bool))] = self.Region.BACKGROUND
            left_row, left_col = np.where(layer_label == self.Region.LEFT)
            left_points = np.stack([left_col, left_row], axis=1)
            if left_points.size != 0:
                left_x, left_y = left_points.T
                leftp_idx = np.argmin(left_y)
                leftp_x = int(left_x[leftp_idx])
                leftp_y = int(left_y[leftp_idx])
            else:
                leftp_x, leftp_y = [int(i) for i in B]
            right_row, right_col = np.where(layer_label == self.Region.RIGHT)
            right_points = np.stack([right_col, right_row], axis=1)
            if right_points.size != 0:
                right_x, right_y = right_points.T
                rightp_idx = np.argmin(right_y)
                rightp_x = int(right_x[rightp_idx])
                rightp_y = int(right_y[rightp_idx])
            else:
                rightp_x, rightp_y = [int(i) for i in C]

            self._contactline_points = (leftp_x, leftp_y, rightp_x, rightp_y)

        return self._contactline_points

    def refine_layer(self) -> npt.NDArray[np.uint8]:
        """Get the refined coating layer image without error pixels."""
        if not hasattr(self, "_refined_layer"):
            layer_img = self.extract_layer().copy()
            h, w = layer_img.shape[:2]
            x1, y1, x2, y2 = self.contactline_points()
            p1, p2 = (x1, y1), (x2, y2)
            ext_p1, ext_p2 = get_extended_line((h, w), p1, p2)
            pts = np.array([(0, 0), ext_p1, ext_p2, (w, 0)])
            # remove every pixels above the contact line
            cv2.fillPoly(layer_img, [pts], 255)
            layer_img = cv2.bitwise_not(layer_img)

            self._refined_layer = layer_img

        return self._refined_layer


def get_extended_line(
    frame_shape: Tuple[int, int], p1: Tuple[int, int], p2: Tuple[int, int]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Extends the line and return its cross points with frame edge.

    Parameters
    ==========

    frame_shape
        Shape of the frame in ``(height, width)``.

    p1, p2
        ``(x, y)`` coorindates of the two points of line.

    Returns
    =======

    ext_p1, ext_p2
        ``(x, y)`` coorindates of the two points of extended.

    Raises
    ======

    ZeroDivisionError
        *p1* and *p2* are same.

    Examples
    ========

    >>> from dipcoatimage.finitedepth.rectcoatinglayer import get_extended_line
    >>> get_extended_line((1080, 1920), (192, 108), (1920, 1080))
    ((0, 0), (1920, 1080))
    >>> get_extended_line((1080, 1920), (300, 500), (400, 500))
    ((0, 500), (1920, 500))
    >>> get_extended_line((1080, 1920), (900, 400), (900, 600))
    ((900, 0), (900, 1080))

    """
    h, w = frame_shape
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and y1 == y2:
        raise ZeroDivisionError("Duplicate points: %s and %s" % (p1, p2))

    elif x1 != x2 and y1 != y2:
        candidates = (
            (int((x2 - x1) / (y2 - y1) * (0 - y1) + x1), 0),
            (int((x2 - x1) / (y2 - y1) * (h - y1) + x1), h),
            (0, int((y2 - y1) / (x2 - x1) * (0 - x1) + y1)),
            (w, int((y2 - y1) / (x2 - x1) * (w - x1) + y1)),
        )

        ret = []
        for x, y in set(candidates):
            if 0 <= x <= w and 0 <= y <= h:
                ret.append((x, y))
        ret.sort()

        ext_p1, ext_p2 = ret

    elif x1 == x2 and y1 != y2:
        ext_p1 = (x1, 0)
        ext_p2 = (x1, h)

    elif x1 != x2 and y1 == y2:
        ext_p1 = (0, y1)
        ext_p2 = (w, y1)

    return ext_p1, ext_p2
