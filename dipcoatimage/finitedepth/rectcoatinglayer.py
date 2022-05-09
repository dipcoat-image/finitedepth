"""
Coating Layer over Rectangular Substrate
========================================

:mod:`dipcoatimage.finitedepth.rectcoatinglayer` provides class to analyze
the coating layer over rectangular substrate.

Base class
----------

.. autoclass:: LayerRegionFlag
   :members:

.. autoclass:: RectCoatingLayerBase
   :members:

Implementation
--------------

"""

import cv2  # type: ignore
import enum
import numpy as np
import numpy.typing as npt
from typing import TypeVar, Type
from .rectsubstrate import RectSubstrate
from .coatinglayer import CoatingLayerBase
from .util import DataclassProtocol


__all__ = ["LayerRegionFlag", "RectCoatingLayerBase"]


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
