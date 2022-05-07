import dataclasses
from typing import Tuple
from .rectsubstrate import RectSubstrate
from .coatinglayer import CoatingLayerBase, CoatingLayerDrawOptions
from .util import BinaryImageDrawMode


@dataclasses.dataclass(frozen=True)
class RectCoatingLayerParameters:
    pass


@dataclasses.dataclass
class RectCoatingLayerDecoOptions:
    """
    Coating layer decorating options for :class:`RectCoatingLayer`.

    Parameters
    ==========

    paint_Left
        Flag to paint the left-hand side region.

    Left_color
        RGB color to paint the left-hand side region.
        Ignored if *paint_Left* is false.

    paint_LeftCorner
        Flag to paint the left-hand side corner region.

    LeftCorner_color
        RGB color to paint the left-hand side corner region.
        Ignored if *paint_LeftCorner* is false.

    paint_Bottom
        Flag to paint the bottom region.

    Bottom_color
        RGB color to paint the bottom region.
        Ignored if *paint_Left* is false.

    paint_RightCorner
        Flag to paint the right-hand side corner region.

    RightCorner_color
        RGB color to paint the right-hand side corner region.
        Ignored if *paint_RightCorner* is false.

    paint_Right
        Flag to paint the right-hand side region.

    Right_color
        RGB color to paint the right-hand side region.
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
class RectCoatingLayerData:
    """
    Coating layer shape data for :class:`RectCoatingLayer`.

    """

    LeftArea: int
    LeftCornerArea: int
    BottomArea: int
    RightCornerArea: int
    RightArea: int


class RectCoatingLayer(
    CoatingLayerBase[
        RectSubstrate,
        RectCoatingLayerParameters,
        CoatingLayerDrawOptions,
        RectCoatingLayerDecoOptions,
        RectCoatingLayerData,
    ]
):
    Parameters = RectCoatingLayerParameters
    DrawOptions = CoatingLayerDrawOptions
    DecoOptions = RectCoatingLayerDecoOptions
    Data = RectCoatingLayerData

    DrawMode = BinaryImageDrawMode
    Draw_Original = BinaryImageDrawMode.ORIGINAL
    Draw_Binary = BinaryImageDrawMode.BINARY
