"""Rectangular substrate."""
import dataclasses
import enum

import cv2
import numpy as np
import numpy.typing as npt

from .parameters import LineOptions, MarkerOptions
from .polysubstrate import Parameters, PolySubstrateBase
from .reference import ReferenceBase

__all__ = [
    "RectSubstrate",
]


class PaintMode(enum.Enum):
    """Option to determine how the substrate image is painted.

    Members
    -------
    ORIGINAL
        Show the original substrate image.
    CONTOUR
        Show the contour of the substrate.
    """

    ORIGINAL = "ORIGINAL"
    CONTOUR = "CONTOUR"


@dataclasses.dataclass
class DrawOptions:
    """Drawing options for `RectSubstrate`.

    Attributes
    ----------
    paint : PaintMode
    vertices : MarkerOptions
    sidelines : LineOptions
    """

    paint: PaintMode = PaintMode.ORIGINAL
    vertices: MarkerOptions = dataclasses.field(
        default_factory=lambda: MarkerOptions(color=(0, 255, 0))
    )
    sidelines: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(0, 0, 255))
    )


@dataclasses.dataclass
class Data:
    """Analysis data for `RectSubstrate`.

    - Width: Number of the pixels between lower vertices of the substrate.
    """

    Width: np.float32


class RectSubstrate(PolySubstrateBase[ReferenceBase, Parameters, DrawOptions, Data]):
    """Simplest implementation of `RectSubstrateBase`.

    Examples
    --------
    Construct substrate reference instance first.

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import Reference, get_data_path
       >>> gray = cv2.imread(get_data_path("ref3.png"), cv2.IMREAD_GRAYSCALE)
       >>> _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
       >>> tempROI = (13, 10, 1246, 200)
       >>> substROI = (100, 100, 1200, 500)
       >>> ref = Reference(img, tempROI, substROI)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    Construct the parameters and substrate instance from reference instance.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import RectSubstrate, data_converter
       >>> param_val = dict(Sigma=3.0, Rho=1.0, Theta=0.01)
       >>> param = data_converter.structure(param_val, RectSubstrate.ParamType)
       >>> subst = RectSubstrate(ref, param)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    Visualization can be controlled by modifying :attr:`draw_options`.

    .. plot::
       :include-source:
       :context: close-figs

       >>> subst.draw_options.sidelines.linewidth = 3
       >>> subst.draw_options.sidelines.color = (255, 0, 255)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP
    """

    ParamType = Parameters
    DrawOptType = DrawOptions
    DataType = Data
    SidesNum = 4

    PaintMode = PaintMode

    def draw(self) -> npt.NDArray[np.uint8]:
        """Return visualized image."""
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
            tl, bl, br, tr = self.sideline_intersections().astype(np.int32)

            color = side_opts.color
            linewidth = side_opts.linewidth
            cv2.line(ret, tl, tr, color, linewidth)
            cv2.line(ret, tr, br, color, linewidth)
            cv2.line(ret, br, bl, color, linewidth)
            cv2.line(ret, bl, tl, color, linewidth)

        return ret

    def analyze(self):
        """Return analysis data."""
        _, B, C, _ = self.sideline_intersections()
        return self.DataType(np.linalg.norm(B - C))
