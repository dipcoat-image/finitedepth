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
import numpy as np
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


class RectSubstrateLineType(enum.IntEnum):
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

    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2
    TOP = 3
    BOTTOM = 4


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

    def lines(self) -> npt.NDArray[np.float32]:
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

    def classify_lines(self, lines: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        """
        Classify *lines* from :func:`cv2.HoughLines`.

        Values are the members of :attr:`RectSubstrateLineType`.
        """
        TOL = 0.2

        r = lines[..., 0]
        h, w = self.image().shape[:2]
        is_upper = r <= h / 2
        is_left = r <= w / 2

        theta = lines[..., 1]
        is_horizontal = np.abs(np.cos(theta)) < np.cos(np.pi / 2 - TOL)
        is_vertical = np.abs(np.cos(theta)) > np.cos(TOL)

        ret = np.full(lines.shape[:2], self.LineType.UNKNOWN, dtype=np.uint8)
        ret[is_upper & is_horizontal] = self.LineType.TOP
        ret[~is_upper & is_horizontal] = self.LineType.BOTTOM
        ret[is_left & is_vertical] = self.LineType.LEFT
        ret[~is_left & is_vertical] = self.LineType.RIGHT
        return ret

    def edge_lines(self) -> Dict[RectSubstrateLineType, Tuple[np.float32, np.float32]]:
        """
        Dictionary of rectangle edges detected from :meth:`lines` using
        :meth:`classify_lines`.
        Values are ``(r, theta)`` of the edge line.

        Notes
        =====

        This property is cached. Do not mutate the result.

        """
        if not hasattr(self, "_edge_lines"):
            lines = self.lines()
            labels = self.classify_lines(lines)

            ret = {}
            for line_type in self.LineType:
                if line_type == self.LineType.UNKNOWN:
                    continue
                good_lines = lines[np.where(labels == line_type)]
                if good_lines.size != 0:
                    ret[line_type] = good_lines[0]

            self._edge_lines = ret

        return self._edge_lines

    def vertex_points(self) -> Dict[RectSubstratePointType, npt.NDArray[np.float32]]:
        """
        Dictionary of rectangle vertices from :meth:`edge_lines`.
        Values are ``(x, y)`` of the point.

        Notes
        =====

        This property is cached. Do not mutate the result.

        """
        if not hasattr(self, "_vertex_points"):
            h, w = self.image().shape[:2]

            edge_lines = self.edge_lines()
            left = edge_lines.get(
                self.LineType.LEFT,
                np.array([0, 0], dtype=np.float32),
            )
            right = edge_lines.get(
                self.LineType.RIGHT,
                np.array([w, 0], dtype=np.float32),
            )
            top = edge_lines.get(
                self.LineType.TOP,
                np.array([0, np.pi / 2], dtype=np.float32),
            )
            bottom = edge_lines.get(
                self.LineType.BOTTOM,
                np.array([h, np.pi / 2], dtype=np.float32),
            )

            def find_intersect(l1, l2):
                r1, t1 = l1
                r2, t2 = l2
                mat = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
                vec = np.array([r1, r2])
                return np.linalg.inv(mat) @ vec  # returns [x, y]

            points = {
                self.PointType.TOPLEFT: find_intersect(top, left).reshape(1, 2),
                self.PointType.TOPRIGHT: find_intersect(top, right).reshape(1, 2),
                self.PointType.BOTTOMLEFT: find_intersect(bottom, left).reshape(1, 2),
                self.PointType.BOTTOMRIGHT: find_intersect(bottom, right).reshape(1, 2),
            }

            self._vertex_points = points  # type: ignore

        return self._vertex_points  # type: ignore

    def examine(self) -> Optional[RectSubstrateError]:
        ret: Optional[RectSubstrateError] = None

        missing = []
        for pt in self.PointType:
            if pt == self.PointType.UNKNOWN:
                continue
            if pt not in self.vertex_points():
                missing.append(pt)

        if missing:
            msg = "Vertices missing: %s" % ", ".join([v.name for v in missing])
            ret = RectSubstrateEdgeError(msg)

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
            r, theta = np.transpose(self.lines(), (2, 0, 1))
            vec = np.dstack([np.cos(theta), np.sin(theta)])
            pts0 = vec * r[..., np.newaxis]
            pts1 = pts0 + np.tensordot(vec, np.array([[0, h], [-w, 0]]), axes=1)
            pts2 = pts0 + np.tensordot(vec, np.array([[0, -h], [w, 0]]), axes=1)

            for p0, p1 in zip(pts1, pts2):
                cv2.line(
                    ret,
                    p0.flatten().astype(np.int32),
                    p1.flatten().astype(np.int32),
                    self.draw_options.line_color,
                    self.draw_options.line_thickness,
                )

        if self.draw_options.draw_edges:
            vertex_points = self.vertex_points()
            topleft = vertex_points.get(
                self.PointType.TOPLEFT, np.empty((0, 2), dtype=np.float32)
            )
            has_topleft = topleft.size > 0
            topright = vertex_points.get(
                self.PointType.TOPRIGHT, np.empty((0, 2), dtype=np.float32)
            )
            has_topright = topright.size > 0
            bottomleft = vertex_points.get(
                self.PointType.BOTTOMLEFT, np.empty((0, 2), dtype=np.float32)
            )
            has_bottomleft = bottomleft.size > 0
            bottomright = vertex_points.get(
                self.PointType.BOTTOMRIGHT, np.empty((0, 2), dtype=np.float32)
            )
            has_bottomright = bottomright.size > 0

            if has_topleft and has_topright:
                cv2.line(
                    ret,
                    topleft.flatten().astype(np.int32),
                    topright.flatten().astype(np.int32),
                    self.draw_options.edge_color,
                    self.draw_options.edge_thickness,
                )
            if has_topright and has_bottomright:
                cv2.line(
                    ret,
                    topright.flatten().astype(np.int32),
                    bottomright.flatten().astype(np.int32),
                    self.draw_options.edge_color,
                    self.draw_options.edge_thickness,
                )
            if has_bottomright and has_bottomleft:
                cv2.line(
                    ret,
                    bottomright.flatten().astype(np.int32),
                    bottomleft.flatten().astype(np.int32),
                    self.draw_options.edge_color,
                    self.draw_options.edge_thickness,
                )
            if has_bottomleft and has_topleft:
                cv2.line(
                    ret,
                    bottomleft.flatten().astype(np.int32),
                    topleft.flatten().astype(np.int32),
                    self.draw_options.edge_color,
                    self.draw_options.edge_thickness,
                )

        return ret
