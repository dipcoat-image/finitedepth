"""Analyze rectangular substrate.

This module defines :class:`RectSubstrate`, which is an implementation of
:class:`PolySubstrateBase`.
"""
import dataclasses
import enum

import cv2
import numpy as np
import numpy.typing as npt

from .parameters import LineOptions, MarkerOptions
from .polysubstrate import PolySubstParam, PolySubstrateBase
from .reference import ReferenceBase

__all__ = [
    "PaintMode",
    "RectSubstDrawOpt",
    "RectSubstData",
    "RectSubstrate",
]


class PaintMode(enum.Enum):
    """Option to determine how the substrate image is painted.

    .. rubric:: **Members**

    - ORIGINAL: Show the original substrate image.
    - CONTOUR: Show the contour of the substrate.
    """

    ORIGINAL = "ORIGINAL"
    CONTOUR = "CONTOUR"


@dataclasses.dataclass
class RectSubstDrawOpt:
    """Drawing options for :class:`RectSubstrate`.

    Arguments:
        paint: Determine how the substrate image is painted
        vertices: Determine how the vertex points is be marked.
        sidelines: Determine how the side lines are drawn.
    """

    paint: PaintMode = PaintMode.ORIGINAL
    vertices: MarkerOptions = dataclasses.field(
        default_factory=lambda: MarkerOptions(color=(0, 255, 0))
    )
    sidelines: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(0, 0, 255))
    )


@dataclasses.dataclass
class RectSubstData:
    """Analysis data for :class:`RectSubstrate`.

    Arguments:
        Width: Width of the rectangular cross section in pixels.
    """

    Width: np.float32


class RectSubstrate(
    PolySubstrateBase[ReferenceBase, PolySubstParam, RectSubstDrawOpt, RectSubstData]
):
    """Simplest implementation of :class:`RectSubstrateBase`.

    Examples:
        Construct reference instance first.

        .. plot::
            :include-source:
            :context: reset

            >>> import cv2
            >>> from dipcoatimage.finitedepth import Reference, get_data_path
            >>> gray = cv2.imread(get_data_path("ref3.png"), cv2.IMREAD_GRAYSCALE)
            >>> _, im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            >>> tempROI = (13, 10, 1246, 200)
            >>> substROI = (100, 100, 1200, 500)
            >>> ref = Reference(im, tempROI, substROI)

        Construct substrate instance from the reference instance.

        .. plot::
            :include-source:
            :context: close-figs

            >>> from dipcoatimage.finitedepth import RectSubstrate
            >>> subst = RectSubstrate(
            ...     ref, RectSubstrate.ParamType(Sigma=3.0, Rho=1.0, Theta=0.01)
            ... )
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(subst.draw()) #doctest: +SKIP

        Visualization can be controlled by modifying :attr:`draw_options`.

        .. plot::
            :include-source:
            :context: close-figs

            >>> subst.draw_options.sidelines.linewidth = 3
            >>> plt.imshow(subst.draw()) #doctest: +SKIP
    """

    ParamType = PolySubstParam
    """Assigned with :class:`PolySubstParam`."""
    DrawOptType = RectSubstDrawOpt
    """Assigned with :class:`RectSubstDrawOpt`."""
    DataType = RectSubstData
    """Assigned with :class:`RectSubstData`."""
    SidesNum = 4

    PaintMode = PaintMode
    """Shortcut to :class:`PaintMode`."""

    def draw(self) -> npt.NDArray[np.uint8]:
        """Implements :meth:`SubstrateBase.draw`.

        #. Draw the substrate with by :class:`PaintMode`.
        #. Draw markers on vertices.
        #. Draw sidelines.
        """
        paint = self.draw_options.paint
        if paint is self.PaintMode.ORIGINAL:
            image = self.image()
        elif paint is self.PaintMode.CONTOUR:
            h, w = self.image().shape[:2]
            image = np.full((h, w), 255, np.uint8)
            cv2.drawContours(image, self.contour(), -1, 0, 1)  # type: ignore
        else:
            raise TypeError("Unrecognized draw mode: %s" % paint)
        ret = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        vert_opts = self.draw_options.vertices
        if vert_opts.linewidth > 0:
            color = vert_opts.color
            marker = getattr(cv2, "MARKER_" + vert_opts.marker.value)
            for (pt,) in self.contour()[self.vertices()]:
                cv2.drawMarker(
                    ret,
                    pt,
                    color=color,
                    markerType=marker,
                    markerSize=vert_opts.markersize,
                    thickness=vert_opts.linewidth,
                )

        side_opts = self.draw_options.sidelines
        if side_opts.linewidth > 0:
            try:
                tl, bl, br, tr = self.sideline_intersections().astype(np.int32)

                color = side_opts.color
                linewidth = side_opts.linewidth
                cv2.line(ret, tl, tr, color, linewidth)
                cv2.line(ret, tr, br, color, linewidth)
                cv2.line(ret, br, bl, color, linewidth)
                cv2.line(ret, bl, tl, color, linewidth)
            except ValueError:
                pass

        return ret

    def analyze(self):
        """Implements :meth:`SubstrateBase.analyze`."""
        _, B, C, _ = self.sideline_intersections()
        return self.DataType(np.linalg.norm(B - C))
