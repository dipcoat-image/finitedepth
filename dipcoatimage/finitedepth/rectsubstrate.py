"""
Rectangular Substrate
=====================

:mod:`dipcoatimage.finitedepth.rectsubstrate` provides substrate image class to
analyze the substrate with rectangular cross-section shape.

Base class
----------

.. autoclass:: RectSubstrateError
   :members:

.. autoclass:: HoughLinesParameters
   :members:

.. autoclass:: RectSubstrateParameters
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
from itertools import combinations
import numpy as np
import numpy.typing as npt
from typing import TypeVar, Tuple, Optional, Type
from .substrate import SubstrateError, SubstrateBase
from .util import (
    DataclassProtocol,
    colorize,
    FeatureDrawingOptions,
    Color,
)

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias


__all__ = [
    "RectSubstrateError",
    "HoughLinesParameters",
    "RectSubstrateParameters",
    "RectSubstrateBase",
    "RectSubstrateDrawMode",
    "RectSubstrateDrawOptions",
    "RectSubstrate",
]


class RectSubstrateError(SubstrateError):
    """Base class for the errors from rectangular substrate class."""

    pass


@dataclasses.dataclass(frozen=True)
class HoughLinesParameters:
    """Parameters for :func:`cv2.HoughLines`."""

    rho: float
    theta: float
    threshold: int
    srn: float = 0.0
    stn: float = 0.0
    min_theta: float = 0.0
    max_theta: float = np.pi


@dataclasses.dataclass(frozen=True)
class RectSubstrateParameters:
    """
    Parameters for the rectangular substrate class to detect the substrate edges
    using Hough line transformation.
    """

    HoughLines: HoughLinesParameters


ParametersType = TypeVar("ParametersType", bound=RectSubstrateParameters)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)


class RectSubstrateBase(SubstrateBase[ParametersType, DrawOptionsType]):
    """Abstract base class for substrate with quadrilateral shape."""

    __slots__ = (
        "_lines",
        "_intersect_points",
        "_vertex_points",
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]

    def contour(self) -> npt.NDArray[np.int32]:
        (cnt,), _ = cv2.findContours(
            cv2.bitwise_not(self.binary_image()),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        return cnt

    def edge(self) -> npt.NDArray[np.bool_]:
        """
        Return the substrate edge as boolean array.

        The edge locations are acquired from :meth:`contour`.
        """
        h, w = self.image().shape[:2]
        ret = np.zeros((h, w), bool)
        ((x, y),) = self.contour().transpose(1, 2, 0)
        ret[y, x] = True
        return ret

    def edge_hull(self) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
        hull = np.flip(cv2.convexHull(self.contour()), axis=0)
        # TODO: get more points by interpolating to `hull`
        tangent = np.gradient(hull, axis=0)
        # TODO: perform edge tangent flow to get smoother curve
        return hull, tangent

    def lines(self) -> npt.NDArray[np.float32]:
        """
        Get :func:`cv2.HoughLines` result from the binary image of *self*.

        This method first acquires the edge image from :meth:`edge`, and
        apply Hough line transformation with the parameters defined in
        :attr:`parameters`.

        If no line can be found, an empty array is returned.

        Notes
        =====

        This property is cached. Do not mutate the result.

        """
        if not hasattr(self, "_lines"):
            # TODO: find way to directly get lines from contour, not edge image
            hparams = dataclasses.asdict(self.parameters.HoughLines)
            lines = cv2.HoughLines(self.edge().astype(np.uint8), **hparams)
            if lines is None:
                lines = np.empty((0, 1, 2), dtype=np.float32)
            self._lines = lines
        return self._lines

    def edge_lines(self) -> npt.NDArray[np.float32]:
        """ "Return four edge lines of the substrate."""
        return self.lines()[:4]

    def intersect_points(self) -> npt.NDArray[np.float32]:
        """Return the intersection points of :meth:`edge_lines`."""
        if not hasattr(self, "_intersect_points"):
            mats, vecs = [], []
            for l1, l2 in combinations(self.edge_lines(), 2):
                ((r1, t1),) = l1
                ((r2, t2),) = l2
                mats.append(
                    np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
                )
                vecs.append(np.array([[r1], [r2]]))
            mat = np.array(mats)
            vec = np.array(vecs)
            sol_exists = np.linalg.det(mat) != 0
            intrsct = (
                np.linalg.inv(mat[np.where(sol_exists)]) @ vec[np.where(sol_exists)]
            )
            self._intersect_points = intrsct.astype(np.float32)
        return self._intersect_points

    def vertex_points(self) -> npt.NDArray[np.float32]:
        """
        Determine four vertices of the substrate from :meth:`intersect_points`.

        Points are sorted counterclockwise in the image.
        """
        if not hasattr(self, "_vertex_points"):
            intrsct = self.intersect_points()

            # get 4 points which make up a quadrilateral
            M = cv2.moments(self.contour())
            cent = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
            dist = np.linalg.norm(intrsct - cent[..., np.newaxis], axis=1)
            points = intrsct[np.argsort(dist, axis=0).flatten()][:4]

            # counterclockwise sort
            ((vec_x, vec_y),) = (points - cent[..., np.newaxis]).transpose(2, 1, 0)
            ccw_idx = np.flip(np.argsort(np.arctan2(vec_y, vec_x)))
            (points_ccw,) = points[ccw_idx].transpose(2, 0, 1)
            # sort as (A, B, C, D), where line BC is "bottom" of the substrate
            lower_points = np.argsort(points_ccw[..., 1])[-2:]
            B_idx = lower_points[np.argmin((lower_points + 1) % 4)]
            self._vertex_points = np.roll(points_ccw, 1 - B_idx, axis=0)

        return self._vertex_points

    def examine(self) -> Optional[RectSubstrateError]:
        ret: Optional[RectSubstrateError] = None

        l_num = len(self.lines())
        if l_num < 4:
            ret = RectSubstrateError(
                f"Insufficient lines from HoughLines (needs >= 4, got {l_num})"
            )

        p_num = len(self.intersect_points())
        if p_num < 4:
            ret = RectSubstrateError(
                f"Insufficient intersection points (needs >= 4, got {p_num})"
            )

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
    """Drawing options for :class:`RectSubstrate`."""

    draw_mode: RectSubstrateDrawMode = RectSubstrateDrawMode.BINARY
    lines: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(0, 255, 0), thickness=1
    )
    edges: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(0, 0, 255), thickness=5
    )
    hull: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(255, 0, 0), thickness=3
    )


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

       >>> from dipcoatimage.finitedepth import RectSubstrate, data_converter
       >>> param_val = dict(HoughLines=dict(rho=1.0, theta=0.01, threshold=100))
       >>> param = data_converter.structure(param_val, RectSubstrate.Parameters)
       >>> subst = RectSubstrate(ref, parameters=param)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    Visualization can be controlled by modifying :attr:`draw_options`.

    .. plot::
       :include-source:
       :context: close-figs

       >>> subst.draw_options.lines.thickness = 0
       >>> subst.draw_options.edges.color.red = 255
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    """

    Parameters = RectSubstrateParameters
    DrawOptions = RectSubstrateDrawOptions

    DrawMode: TypeAlias = RectSubstrateDrawMode

    def draw(self) -> npt.NDArray[np.uint8]:
        draw_mode = self.draw_options.draw_mode
        if draw_mode is self.DrawMode.ORIGINAL:
            image = self.image()
        elif draw_mode is self.DrawMode.BINARY:
            image = self.binary_image()
        elif draw_mode is self.DrawMode.EDGES:
            image = self.edge() * np.uint8(255)
        else:
            raise TypeError("Unrecognized draw mode: %s" % draw_mode)
        ret = colorize(image)

        line_opts = self.draw_options.lines
        if line_opts.thickness > 0:
            r, theta = np.transpose(self.lines(), (2, 0, 1))
            vec = np.dstack([np.cos(theta), np.sin(theta)])
            pts0 = vec * r[..., np.newaxis]
            h, w = ret.shape[:2]
            pts1 = pts0 + np.tensordot(vec, np.array([[0, h], [-w, 0]]), axes=1)
            pts2 = pts0 + np.tensordot(vec, np.array([[0, -h], [w, 0]]), axes=1)

            for p0, p1 in zip(pts1, pts2):
                cv2.line(
                    ret,
                    p0.flatten().astype(np.int32),
                    p1.flatten().astype(np.int32),
                    dataclasses.astuple(line_opts.color),
                    line_opts.thickness,
                )

        edge_opts = self.draw_options.edges
        if edge_opts.thickness > 0:
            tl, bl, br, tr = self.vertex_points().astype(np.int32)

            color = dataclasses.astuple(edge_opts.color)
            thickness = edge_opts.thickness
            cv2.line(ret, tl, tr, color, thickness)
            cv2.line(ret, tr, br, color, thickness)
            cv2.line(ret, br, bl, color, thickness)
            cv2.line(ret, bl, tl, color, thickness)

        hull_opts = self.draw_options.hull
        if hull_opts.thickness > 0:
            hull, _ = self.edge_hull()
            cv2.polylines(
                ret,
                [hull],
                isClosed=False,
                color=dataclasses.astuple(hull_opts.color),
                thickness=hull_opts.thickness,
            )

        return ret
