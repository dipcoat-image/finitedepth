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
    BasicCoatingLayerParameters,
    BasicCoatingLayerDrawOptions,
)
from .util import DataclassProtocol, BinaryImageDrawMode


__all__ = [
    "LayerRegionFlag",
    "RectCoatingLayerBase",
    "RectLayerAreaDecoOptions",
    "RectLayerAreaData",
    "RectLayerArea",
]


class LayerRegionFlag(enum.IntFlag):
    """
    Label for the coating layer regions.
    """

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
    """Abstract base class for coating layer over rectangular substrate."""

    __slots__ = ("_labelled_layer",)

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    DecoOptions: Type[DecoOptionsType]
    Data: Type[DataType]

    RegionFlag = LayerRegionFlag
    Region_Background = LayerRegionFlag.BACKGROUND
    Region_Left = LayerRegionFlag.LEFT
    Region_Bottom = LayerRegionFlag.BOTTOM
    Region_Right = LayerRegionFlag.RIGHT

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
        Return the array of coating layer divided and labelled.

        Returns
        =======

        ret
            :class:`numpy.ndarray` with :class:`LayerRegionFlag` elements.

        """
        h, w = self.image.shape[:2]
        ret = np.full((h, w), self.Region_Background)

        mask = cv2.bitwise_not(self.extract_layer()).astype(bool)
        row, col = np.where(mask)
        points = np.stack([col, row], axis=1)

        p0 = np.array(self.substrate_point())
        A = p0 + np.array(self.substrate.vertex_points()[self.substrate.Point_TopLeft])
        B = p0 + np.array(
            self.substrate.vertex_points()[self.substrate.Point_BottomLeft]
        )
        C = p0 + np.array(
            self.substrate.vertex_points()[self.substrate.Point_BottomRight]
        )
        D = p0 + np.array(self.substrate.vertex_points()[self.substrate.Point_TopRight])

        left_of_AB = np.cross(B - A, points - A) >= 0
        under_BC = np.cross(C - B, points - B) >= 0
        right_of_CD = np.cross(D - C, points - C) >= 0

        left_x, left_y = points[left_of_AB].T
        ret[left_y, left_x] |= self.Region_Left

        bottom_x, bottom_y = points[under_BC].T
        ret[bottom_y, bottom_x] |= self.Region_Bottom

        right_x, right_y = points[right_of_CD].T
        ret[right_y, right_x] |= self.Region_Right

        return ret


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
        Number of the pixels in cross section image of coating layer regions.

    """

    LeftArea: int
    LeftCornerArea: int
    BottomArea: int
    RightCornerArea: int
    RightArea: int


class RectLayerArea(
    RectCoatingLayerBase[
        BasicCoatingLayerParameters,
        BasicCoatingLayerDrawOptions,
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
       ...               RightCornerArea=193, RightArea=8263)
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
    Parameters = BasicCoatingLayerParameters
    DrawOptions = BasicCoatingLayerDrawOptions
    DecoOptions = RectLayerAreaDecoOptions
    Data = RectLayerAreaData

    DrawMode = BinaryImageDrawMode
    Draw_Original = BinaryImageDrawMode.ORIGINAL
    Draw_Binary = BinaryImageDrawMode.BINARY

    def examine(self) -> None:
        return None

    def draw(self) -> npt.NDArray[np.uint8]:
        draw_mode = self.draw_options.draw_mode
        if self.draw_options.remove_substrate:
            image = self.extract_layer()
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
            layer_label = self.label_layer()
            if self.deco_options.paint_Left:
                color = self.deco_options.Left_color
                ret[layer_label == self.Region_Left] = color
            if self.deco_options.paint_LeftCorner:
                color = self.deco_options.LeftCorner_color
                ret[layer_label == self.Region_Left | self.Region_Bottom] = color
            if self.deco_options.paint_Bottom:
                color = self.deco_options.Bottom_color
                ret[layer_label == self.Region_Bottom] = color
            if self.deco_options.paint_RightCorner:
                color = self.deco_options.RightCorner_color
                ret[layer_label == self.Region_Bottom | self.Region_Right] = color
            if self.deco_options.paint_Right:
                color = self.deco_options.Right_color
                ret[layer_label == self.Region_Right] = color
        return ret

    def analyze_layer(self) -> Tuple[int, int, int, int, int]:
        layer_label = self.label_layer()
        val, counts = np.unique(layer_label, return_counts=True)

        i = self.Region_Left
        count = counts[np.where(val == i)]
        (left_a,) = count if count.size > 0 else (0,)

        i = self.Region_Left | self.Region_Bottom
        count = counts[np.where(val == i)]
        (leftcorner_a,) = count if count.size > 0 else (0,)

        i = self.Region_Bottom
        count = counts[np.where(val == i)]
        (bottom_a,) = count if count.size > 0 else (0,)

        i = self.Region_Bottom | self.Region_Right
        count = counts[np.where(val == i)]
        (rightcorner_a,) = count if count.size > 0 else (0,)

        i = self.Region_Right
        count = counts[np.where(val == i)]
        (right_a,) = count if count.size > 0 else (0,)

        return (left_a, leftcorner_a, bottom_a, rightcorner_a, right_a)
