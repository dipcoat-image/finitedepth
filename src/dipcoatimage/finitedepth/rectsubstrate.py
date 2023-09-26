"""
Rectangular Substrate
=====================

:mod:`dipcoatimage.finitedepth.rectsubstrate` provides substrate image class to
analyze the substrate with rectangular cross-section shape.

.. autoclass:: RectSubstrate
   :members:

"""
from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt

from .polysubstrate import PolySubstrateBase
from .polysubstrate_param import Parameters as Parameters
from .rectsubstrate_param import Data, DrawOptions, PaintMode

__all__ = [
    "RectSubstrate",
]


class RectSubstrate(PolySubstrateBase[Parameters, DrawOptions, Data]):
    """
    Simplest implementation of :class:`RectSubstrateBase`.

    Examples
    ========

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
       >>> param = data_converter.structure(param_val, RectSubstrate.Parameters)
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

    Parameters = Parameters
    DrawOptions = DrawOptions
    Data = Data
    SidesNum = 4

    PaintMode = PaintMode

    def draw(self) -> npt.NDArray[np.uint8]:
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

    def analyze_substrate(self) -> Tuple[np.float64]:
        _, B, C, _ = self.sideline_intersections()
        return (np.linalg.norm(B - C),)
