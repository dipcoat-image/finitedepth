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
    """Parameters to draw a line in the image."""

    color: Tuple[int, int, int] = (0, 0, 0)
    """Color of the line in RGB."""
    linewidth: int = 1
    """Width of the line."""


@dataclasses.dataclass
class PatchOptions:
    """Parameters to draw a patch in the image."""

    fill: bool = True
    """Whether to fill the patch with facecolor."""
    edgecolor: Tuple[int, int, int] = (0, 0, 0)
    """Color of the edge in RGB."""
    facecolor: Tuple[int, int, int] = (0, 0, 0)
    """Color of the face in RGB."""
    linewidth: int = 1
    """Width of the edge."""


class MarkerTypes(enum.Enum):
    """Marker types for :func:`cv2.drawMarker`."""

    CROSS = "CROSS"
    TILTED_CROSS = "TILTED_CROSS"
    STAR = "STAR"
    DIAMOND = "DIAMOND"
    SQUARE = "SQUARE"
    TRIANGLE_UP = "TRIANGLE_UP"
    TRIANGLE_DOWN = "TRIANGLE_DOWN"


@dataclasses.dataclass
class MarkerOptions:
    """Parameters to draw a marker in the image."""

    color: Tuple[int, int, int] = (0, 0, 0)
    """Color of the marker in RGB."""
    marker: MarkerTypes = MarkerTypes.CROSS
    linewidth: int = 1
    """Width of the marker line."""
    markersize: int = 20
    """Size of the marker in pts."""
