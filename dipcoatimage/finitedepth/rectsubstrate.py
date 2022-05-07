"""
Rectangular Substrate
=====================

:mod:`dipcoatimage.finitedepth.rectsubstrate` provides substrate image class to
analyze the substrate with rectangular cross-section shape.

Parameter classes
-----------------

.. autoclass:: RectSubstrateParameters
   :members:

Draw option classes
-------------------

.. autoclass:: RectSubstrateDrawMode
   :members:

.. autoclass:: RectSubstrateDrawOptions
   :members:

Error classes
-------------

.. autoclass:: RectSubstrateError
   :members:

.. autoclass:: RectSubstrateHoughLinesError
   :members:

.. autoclass:: RectSubstrateEdgeError
   :members:

Implementation
--------------

.. autoclass:: RectSubstrateLineType
   :members:

.. autoclass:: RectSubstratePointType
   :members:

.. autoclass:: RectSubstrate
   :members:

"""
import cv2  # type: ignore
import dataclasses
import enum
from math import isclose
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Dict
from .substrate import SubstrateError, SubstrateBase
from .util import intrsct_pt_polar, CannyParameters, HoughLinesParameters


__all__ = [
    "RectSubstrateParameters",
    "RectSubstrateDrawMode",
    "RectSubstrateDrawOptions",
    "RectSubstrateError",
    "RectSubstrateHoughLinesError",
    "RectSubstrateEdgeError",
    "RectSubstrateLineType",
    "RectSubstratePointType",
    "RectSubstrate",
]


@dataclasses.dataclass(frozen=True)
class RectSubstrateParameters:
    """Parameters for :class:`RectSubstrate`."""

    Canny: CannyParameters
    HoughLines: HoughLinesParameters


class RectSubstrateDrawMode(enum.Enum):
    """
    Option for :class:`RectSubstrateDrawOptions` to determine how the substrate
    image is drawn.

    Attributes
    ==========

    ORIGINAL
        Show the original substrate image.

    BINARY
        Show the binarized substrate image.

    EDGES
        Show the edges of the substrate image.

    """

    ORIGINAL = "ORIGINAL"
    BINARY = "BINARY"
    EDGES = "EDGES"


@dataclasses.dataclass
class RectSubstrateDrawOptions:
    """
    Drawing options for :class:`RectSubstrate`.

    Parameters
    ==========

    draw_mode

    draw_lines
        Flag to draw the detected straight lines on the edge of substrate image.

    line_color, line_thickness
        RGB color and thickness to draw the detected lines.
        Ignored if *draw_lines* is false.

    Draw_Edges
        Flag to draw the detected four edges of the substrate.

    edge_color, edge_thickness
        RGB color and thickness to draw the detected edges.
        Ignored if *Draw_Edges* is false.

    """

    draw_mode: RectSubstrateDrawMode = RectSubstrateDrawMode.ORIGINAL
    draw_lines: bool = True
    line_color: Tuple[int, int, int] = (0, 255, 0)
    line_thickness: int = 1
    Draw_Edges: bool = True
    edge_color: Tuple[int, int, int] = (0, 0, 255)
    edge_thickness: int = 5


class RectSubstrateError(SubstrateError):
    """Base class for error from :class:`RectSubstrate`."""

    pass


class RectSubstrateHoughLinesError(RectSubstrateError):
    """Error from Hough lines transformation in :class:`RectSubstrate`."""

    pass


class RectSubstrateEdgeError(RectSubstrateError):
    """Error from edge line classification in :class:`RectSubstrate`."""

    pass


class RectSubstrateLineType(enum.Enum):
    """
    Type of the line detected in :class:`RectSubstrate()`.

    Attributes
    ==========

    UNKNOWN
        Unknown line.

    LEFT
        Left-hand side edge line of the substrate.

    RIGHT
        Right-hand side edge line of the substrate.

    TOP
        Top edge line of the substrate.

    BOTTOM
        Bottom edge line of the substrate.

    """

    UNKNOWN = "UNKNOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    TOP = "TOP"
    BOTTOM = "BOTTOM"


class RectSubstratePointType(enum.Enum):
    """
    Type of the point detected in :class:`RectSubstrate()`.

    Attributes
    ==========

    UNKNOWN
        Unknown point.

    TOPLEFT
        Top left-hand side vertex point of the substrate.

    BOTTOMLEFT
        Bottom left-hand side vertex point of the substrate.

    BOTTOMRIGHT
        Bottom right-hand side vertex point of the substrate.

    TOPRIGHT
        Top right-hand side vertex point of the substrate.

    """

    UNKNOWN = "UNKNOWN"
    TOPLEFT = "TOPLEFT"
    BOTTOMLEFT = "BOTTOMLEFT"
    BOTTOMRIGHT = "BOTTOMRIGHT"
    TOPRIGHT = "TOPRIGHT"


class RectSubstrate(SubstrateBase[RectSubstrateParameters, RectSubstrateDrawOptions]):
    """
    Class for the substrate image in rectangular shape.

    Rectangular substrate is characterized by four edges and vertices,
    which are detected by :meth:`edge_lines` and :meth:`vertex_points`.

    Examples
    ========

    Construct substrate reference class first.

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import (SubstrateReference,
       ...     get_samples_path)
       >>> ref_path = get_samples_path('ref1.png')
       >>> img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
       >>> tempROI = (200, 50, 1200, 200)
       >>> substROI = (400, 100, 1000, 500)
       >>> ref = SubstrateReference(img, tempROI, substROI)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    Construct the parameters.

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

    """

    __slots__ = (
        "_cannyimage",
        "_lines",
        "_edge_lines",
        "_vertex_points",
    )

    Parameters = RectSubstrateParameters
    DrawOptions = RectSubstrateDrawOptions

    DrawMode = RectSubstrateDrawMode
    Draw_Original = RectSubstrateDrawMode.ORIGINAL
    Draw_Binary = RectSubstrateDrawMode.BINARY
    Draw_Edges = RectSubstrateDrawMode.EDGES

    LineType = RectSubstrateLineType
    Line_Unknown = RectSubstrateLineType.UNKNOWN
    Line_Left = RectSubstrateLineType.LEFT
    Line_Right = RectSubstrateLineType.RIGHT
    Line_Top = RectSubstrateLineType.TOP
    Line_Bottom = RectSubstrateLineType.BOTTOM

    PointType = RectSubstratePointType
    Point_Unknown = RectSubstratePointType.UNKNOWN
    Point_TopLeft = RectSubstratePointType.TOPLEFT
    Point_BottomLeft = RectSubstratePointType.BOTTOMLEFT
    Point_BottomRight = RectSubstratePointType.BOTTOMRIGHT
    Point_TopRight = RectSubstratePointType.TOPRIGHT

    def canny_image(self) -> npt.NDArray[np.uint8]:
        """
        Canny edge detection result on :meth:`binary_image`.

        Notes
        =====

        This property is cached. Do not mutate the result.

        """
        if not hasattr(self, "_cannyimage"):
            cparams = dataclasses.asdict(self.parameters.Canny)
            self._cannyimage = cv2.Canny(self.binary_image(), **cparams)
        return self._cannyimage  # type: ignore

    def lines(self) -> npt.NDArray[np.uint8]:
        """
        Feature vectors of straight lines in ``(r, theta)``, detected by
        :func:`cv2.HoughLines` on :meth:`canny_image`.

        Notes
        =====

        This property is cached. Do not mutate the result.

        """
        if not hasattr(self, "_lines"):
            hparams = dataclasses.asdict(self.parameters.HoughLines)
            lines = cv2.HoughLines(self.canny_image(), **hparams)
            if lines is None:
                lines = np.empty((0, 1, 2), dtype=np.float32)
            self._lines = lines
        return self._lines  # type: ignore

    def classify_line(self, r: float, theta: float) -> RectSubstrateLineType:
        """Classify a line by its distance *r* and angle *theta*."""
        h, w = self.image().shape[:2]
        r = abs(r)

        is_horizontal = any(
            isclose(theta, a, abs_tol=0.2) for a in (np.pi / 2, 3 * np.pi / 2)
        )
        is_vertical = any(isclose(theta, a, abs_tol=0.2) for a in (0, np.pi, 2 * np.pi))

        ret = self.Line_Unknown
        if is_horizontal:
            if r <= h / 2:
                ret = self.Line_Top
            else:
                ret = self.Line_Bottom
        elif is_vertical:
            if r <= w / 2:
                ret = self.Line_Left
            else:
                ret = self.Line_Right
        return ret

    def edge_lines(self) -> Dict[RectSubstrateLineType, Tuple[float, float]]:
        """
        Dictionary of rectangle edges detected from :meth:`lines` using
        :meth:`classify_line`.
        Values are ``(r, theta)`` of the edge line.

        Notes
        =====

        This property is cached. Do not mutate the result.

        """
        if not hasattr(self, "_edge_lines"):
            edge_lines = {}
            for line in self.lines():
                r, theta = line[0]
                line_type = self.classify_line(r, theta)
                if line_type in edge_lines:
                    continue
                else:
                    edge_lines[line_type] = (r, theta)
                if all(t in edge_lines for t in self.LineType):
                    break
            self._edge_lines = edge_lines  # type: ignore
        return self._edge_lines  # type: ignore

    def vertex_points(self) -> Dict[RectSubstratePointType, Tuple[int, int]]:
        """
        Dictionary of rectangle vertices from :meth:`edge_lines`.
        Values are ``(x, y)`` of the point.

        Notes
        =====

        This property is cached. Do not mutate the result.

        """
        if not hasattr(self, "_vertex_points"):
            left = self.edge_lines().get(self.Line_Left, None)
            right = self.edge_lines().get(self.Line_Right, None)
            top = self.edge_lines().get(self.Line_Top, None)
            bottom = self.edge_lines().get(self.Line_Bottom, None)
            points = {}
            if top and left:
                x, y = intrsct_pt_polar(*top, *left)
                points[self.Point_TopLeft] = (int(x), int(y))
            if top and right:
                x, y = intrsct_pt_polar(*top, *right)
                points[self.Point_TopRight] = (int(x), int(y))
            if bottom and left:
                x, y = intrsct_pt_polar(*bottom, *left)
                points[self.Point_BottomLeft] = (int(x), int(y))
            if bottom and right:
                x, y = intrsct_pt_polar(*bottom, *right)
                points[self.Point_BottomRight] = (int(x), int(y))
            self._vertex_points = points  # type: ignore
        return self._vertex_points  # type: ignore

    def examine(self) -> Optional[RectSubstrateError]:
        ret = None

        if len(self.lines()) == 0:
            msg = "No line detected from Hough transformation"
            ret = RectSubstrateHoughLinesError(msg)

        else:
            msg_tmpl = "Cannot detect %s of the substrate"
            missing = []
            if self.Line_Left not in self.edge_lines():
                missing.append("left wall")
            if self.Line_Right not in self.edge_lines():
                missing.append("right wall")
            if self.Line_Top not in self.edge_lines():
                missing.append("top wall")
            if self.Line_Bottom not in self.edge_lines():
                missing.append("bottom wall")

            if missing:
                msg = msg_tmpl % (", ".join(missing))
                ret = RectSubstrateEdgeError(msg)  # type: ignore

        return ret

    def draw(self) -> npt.NDArray[np.uint8]:
        h, w = self.image().shape[:2]

        draw_mode = self.draw_options.draw_mode
        if draw_mode is self.Draw_Original:
            image = self.image()
        elif draw_mode is self.Draw_Binary:
            image = self.binary_image()
        elif draw_mode is self.Draw_Edges:
            image = self.canny_image()
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

        if self.draw_options.draw_lines:
            color = self.draw_options.line_color
            thickness = self.draw_options.line_thickness
            for line in self.lines():
                r, theta = line[0]
                tx, ty = np.cos(theta), np.sin(theta)
                x0, y0 = tx * r, ty * r
                x1, y1 = int(x0 + w * (-ty)), int(y0 + h * tx)
                x2, y2 = int(x0 - w * (-ty)), int(y0 - h * tx)
                cv2.line(ret, (x1, y1), (x2, y2), color, thickness)

        if self.draw_options.Draw_Edges:
            vertex_points = self.vertex_points()
            topleft = vertex_points.get(self.Point_TopLeft, None)
            topright = vertex_points.get(self.Point_TopRight, None)
            bottomleft = vertex_points.get(self.Point_BottomLeft, None)
            bottomright = vertex_points.get(self.Point_BottomRight, None)

            color = self.draw_options.edge_color
            thickness = self.draw_options.edge_thickness
            if topleft and topright:
                cv2.line(ret, topleft, topright, color, thickness)
            if topright and bottomright:
                cv2.line(ret, topright, bottomright, color, thickness)
            if bottomright and bottomleft:
                cv2.line(ret, bottomright, bottomleft, color, thickness)
            if bottomleft and topleft:
                cv2.line(ret, bottomleft, topleft, color, thickness)

        return ret
