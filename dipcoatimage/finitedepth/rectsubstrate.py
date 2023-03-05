"""
Rectangular Substrate
=====================

:mod:`dipcoatimage.finitedepth.rectsubstrate` provides substrate image class to
analyze the substrate with rectangular cross-section shape.

Base class
----------

.. autoclass:: RectSubstrateError
   :members:

.. autoclass:: RectSubstrateParameters
   :members:

.. autoclass:: RectSubstrateHoughLinesError
   :members:

.. autoclass:: RectSubstrateEdgeError
   :members:

.. autoclass:: RectSubstrateLineType
   :members:

.. autoclass:: RectSubstratePointType
   :members:

.. autoclass:: RectSubstrateBase
   :members:

Implementation
--------------

.. autoclass:: RectSubstrateDrawMode
   :members:

.. autoclass:: RectSubstrateDrawOptions
   :members:

.. autoclass:: RectSubstrate
   :members:

"""
import cv2  # type: ignore
import dataclasses
import enum
from math import isclose
import numpy as np
from numpy.linalg import inv
import numpy.typing as npt
from typing import TypeVar, Tuple, Optional, Dict, Type
from .substrate import SubstrateError, SubstrateBase
from .util import (
    HoughLinesParameters,
    DataclassProtocol,
)

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias


__all__ = [
    "RectSubstrateError",
    "RectSubstrateParameters",
    "RectSubstrateHoughLinesError",
    "RectSubstrateEdgeError",
    "RectSubstrateLineType",
    "RectSubstratePointType",
    "RectSubstrateBase",
    "RectSubstrateDrawMode",
    "RectSubstrateDrawOptions",
    "RectSubstrate",
    "intrsct_pt_polar",
]


class RectSubstrateError(SubstrateError):
    """Base class for the errors from rectangular substrate class."""

    pass


@dataclasses.dataclass(frozen=True)
class RectSubstrateParameters:
    """
    Parameters for the rectangular substrate class to detect the substrate edges
    using Hough line transformation.
    """

    HoughLines: HoughLinesParameters


class RectSubstrateHoughLinesError(RectSubstrateError):
    """Error from Hough lines transformation in rectangular substrate."""

    pass


class RectSubstrateEdgeError(RectSubstrateError):
    """Error from edge line classification in rectangular substrate."""

    pass


class RectSubstrateLineType(enum.Enum):
    """
    Type of the line detected in rectangular substrate.

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
    Type of the point detected in rectangular substrate.

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


ParametersType = TypeVar("ParametersType", bound=RectSubstrateParameters)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)


class RectSubstrateBase(SubstrateBase[ParametersType, DrawOptionsType]):
    """
    Abstract base class for substrate with rectangular shape.

    Rectangular substrate is characterized by four edges and vertices,
    which are detected by :meth:`edge_lines` and :meth:`vertex_points`.

    """

    __slots__ = (
        "_gradient",
        "_lines",
        "_edge_lines",
        "_vertex_points",
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]

    LineType: TypeAlias = RectSubstrateLineType
    PointType: TypeAlias = RectSubstratePointType

    def gradient(self) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Return the pixel gradient in ``(G_x, G_y)`` using :func:`cv2.Sobel`.
        """
        if not hasattr(self, "_gradient"):
            Gx = cv2.Sobel(self.binary_image(), cv2.CV_32F, 1, 0)
            Gy = cv2.Sobel(self.binary_image(), cv2.CV_32F, 0, 1)
            self._gradient = (Gx, Gy)
        return self._gradient

    def edge_hull(self) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
        contours, _ = cv2.findContours(
            cv2.bitwise_not(self.binary_image()),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) != 1:
            raise NotImplementedError
        (cnt,) = contours
        hull = cv2.convexHull(cnt)
        # TODO: get more points by interpolating to `hull`
        tangent = np.gradient(hull, axis=0)
        # TODO: perform edge tangent flow to get smoother curve
        return hull, tangent

    def lines(self) -> npt.NDArray[np.uint8]:
        """
        Feature vectors of straight lines from :meth:`gradient` in
        ``(r, theta)``, detected by :func:`cv2.HoughLines`.

        Notes
        =====

        This property is cached. Do not mutate the result.

        """
        if not hasattr(self, "_lines"):
            Gx, Gy = self.gradient()
            G = Gx.astype(bool) | Gy.astype(bool)
            hparams = dataclasses.asdict(self.parameters.HoughLines)
            lines = cv2.HoughLines(G.astype(np.uint8), **hparams)
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

        ret = self.LineType.UNKNOWN
        if is_horizontal:
            if r <= h / 2:
                ret = self.LineType.TOP
            else:
                ret = self.LineType.BOTTOM
        elif is_vertical:
            if r <= w / 2:
                ret = self.LineType.LEFT
            else:
                ret = self.LineType.RIGHT
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
            h, w = self.image().shape[:2]

            left = self.edge_lines().get(self.LineType.LEFT, (0, 0))
            right = self.edge_lines().get(self.LineType.RIGHT, (w, 0))
            top = self.edge_lines().get(self.LineType.TOP, (0, np.pi / 2))
            bottom = self.edge_lines().get(self.LineType.BOTTOM, (h, np.pi / 2))
            points = {}
            if top and left:
                x, y = intrsct_pt_polar(*top, *left)
                points[self.PointType.TOPLEFT] = (int(x), int(y))
            if top and right:
                x, y = intrsct_pt_polar(*top, *right)
                points[self.PointType.TOPRIGHT] = (int(x), int(y))
            if bottom and left:
                x, y = intrsct_pt_polar(*bottom, *left)
                points[self.PointType.BOTTOMLEFT] = (int(x), int(y))
            if bottom and right:
                x, y = intrsct_pt_polar(*bottom, *right)
                points[self.PointType.BOTTOMRIGHT] = (int(x), int(y))
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
            if self.LineType.LEFT not in self.edge_lines():
                missing.append("left wall")
            if self.LineType.RIGHT not in self.edge_lines():
                missing.append("right wall")
            if self.LineType.TOP not in self.edge_lines():
                missing.append("top wall")
            if self.LineType.BOTTOM not in self.edge_lines():
                missing.append("bottom wall")

            if missing:
                msg = msg_tmpl % (", ".join(missing))
                ret = RectSubstrateEdgeError(msg)  # type: ignore

        return ret


class RectSubstrateDrawMode(enum.Enum):
    """
    Option to determine how the :class:`RectSubstrate` is drawn.

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

    draw_edges
        Flag to draw the detected four edges of the substrate.

    edge_color, edge_thickness
        RGB color and thickness to draw the detected edges.
        Ignored if *draw_edges* is false.

    """

    draw_mode: RectSubstrateDrawMode = RectSubstrateDrawMode.ORIGINAL
    draw_lines: bool = True
    line_color: Tuple[int, int, int] = (0, 255, 0)
    line_thickness: int = 1
    draw_edges: bool = True
    edge_color: Tuple[int, int, int] = (0, 0, 255)
    edge_thickness: int = 5


class RectSubstrate(
    RectSubstrateBase[RectSubstrateParameters, RectSubstrateDrawOptions]
):
    """
    Simplest implementation of :class:`RectSubstrateBase`.

    Examples
    ========

    Construct substrate reference instance first.

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import (SubstrateReference,
       ...     get_samples_path)
       >>> ref_path = get_samples_path("ref3.png")
       >>> img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
       >>> tempROI = (100, 50, 1200, 200)
       >>> substROI = (300, 100, 950, 600)
       >>> ref = SubstrateReference(img, tempROI, substROI)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    Construct the parameters and substrate instance from reference instance.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import (HoughLinesParameters,
       ...     RectSubstrate)
       >>> hparams = HoughLinesParameters(1, 0.01, 100)
       >>> params = RectSubstrate.Parameters(hparams)
       >>> subst = RectSubstrate(ref, parameters=params)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    Visualization can be controlled by modifying :attr:`draw_options`.

    .. plot::
       :include-source:
       :context: close-figs

       >>> subst.draw_options.draw_lines = False
       >>> subst.draw_options.edge_color = (255, 0, 0)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    """

    Parameters = RectSubstrateParameters
    DrawOptions = RectSubstrateDrawOptions

    DrawMode: TypeAlias = RectSubstrateDrawMode

    def draw(self) -> npt.NDArray[np.uint8]:
        h, w = self.image().shape[:2]

        draw_mode = self.draw_options.draw_mode
        if draw_mode is self.DrawMode.ORIGINAL:
            image = self.image()
        elif draw_mode is self.DrawMode.BINARY:
            image = self.binary_image()
        elif draw_mode is self.DrawMode.EDGES:
            Gx, Gy = self.gradient()
            image = (Gx.astype(bool) | Gy.astype(bool)) * np.uint8(255)
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

        if self.draw_options.draw_edges:
            vertex_points = self.vertex_points()
            topleft = vertex_points.get(self.PointType.TOPLEFT, None)
            topright = vertex_points.get(self.PointType.TOPRIGHT, None)
            bottomleft = vertex_points.get(self.PointType.BOTTOMLEFT, None)
            bottomright = vertex_points.get(self.PointType.BOTTOMRIGHT, None)

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


def intrsct_pt_polar(r1: float, t1: float, r2: float, t2: float) -> Tuple[float, float]:
    """
    Find the Cartesian coordinates of the intersecting point of two
    lines by their polar parameters.

    Parameters
    ==========

    r1, t1, r2, t2
        Radius and angle for the first and second line.

    Returns
    =======

    x, y
        Cartesian coordinates of the intersecting point.

    Examples
    ========

    >>> from dipcoatimage.finitedepth.rectsubstrate import intrsct_pt_polar
    >>> from numpy import pi
    >>> x, y = intrsct_pt_polar(10, pi/3, 5, pi/6)
    >>> round(x, 2)
    -1.34
    >>> round(y, 2)
    12.32

    """
    mat = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    vec = np.array([r1, r2])
    ret = inv(mat) @ vec
    return tuple(float(i) for i in ret)  # type: ignore
