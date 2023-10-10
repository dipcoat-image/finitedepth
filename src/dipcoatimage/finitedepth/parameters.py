"""Common parameters."""

import dataclasses
import enum
from typing import Tuple

__all__ = [
    "LineOptions",
    "PatchOptions",
    "MarkerTypes",
    "MarkerOptions",
]


@dataclasses.dataclass
class LineOptions:
    """Parameters to draw a line in the image.

    Arguments:
        color: Color of the line in RGB.
        linewidth: Width of the line.
    """

    color: Tuple[int, int, int] = (0, 0, 0)
    linewidth: int = 1


@dataclasses.dataclass
class PatchOptions:
    """Parameters to draw a patch in the image.

    Arguments:
        fill: Whether to fill the patch with facecolor.
        edgecolor: Color of the edge in RGB.
        facecolor: Color of the face in RGB.
        linewidth: Width of the edge.
    """

    fill: bool = True
    edgecolor: Tuple[int, int, int] = (0, 0, 0)
    facecolor: Tuple[int, int, int] = (0, 0, 0)
    linewidth: int = 1


class MarkerTypes(enum.Enum):
    """Marker types for :func:`cv2.drawMarker`.

    .. rubric:: **Members**

    - CROSS: Use :obj:`cv2.MARKER_CROSS`.
    - TILTED_CROSS: Use :obj:`cv2.MARKER_TILTED_CROSS`.
    - STAR: Use :obj:`cv2.MARKER_STAR`.
    - DIAMOND: Use :obj:`cv2.MARKER_DIAMOND`.
    - SQUARE: Use :obj:`cv2.MARKER_SQUARE`.
    - TRIANGLE_UP: Use :obj:`cv2.MARKER_TRIANGLE_UP`.
    - TRIANGLE_DOWN: Use :obj:`cv2.TRIANGLE_DOWN`.
    """

    CROSS = "CROSS"
    TILTED_CROSS = "TILTED_CROSS"
    STAR = "STAR"
    DIAMOND = "DIAMOND"
    SQUARE = "SQUARE"
    TRIANGLE_UP = "TRIANGLE_UP"
    TRIANGLE_DOWN = "TRIANGLE_DOWN"


@dataclasses.dataclass
class MarkerOptions:
    """Parameters to draw a marker in the image.

    Arguments:
        color: Color of the marker in RGB.
        marker: Marker shape.
        linewidth: Width of the marker line.
        markersize: Size of the marker in pts.
    """

    color: Tuple[int, int, int] = (0, 0, 0)
    marker: MarkerTypes = MarkerTypes.CROSS
    linewidth: int = 1
    markersize: int = 20
