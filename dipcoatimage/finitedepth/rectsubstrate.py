"""
Rectangular Substrate
=====================

:mod:`dipcoatimage.finitedepth.rectsubstrate` provides substrate image class to
analyze the substrate with rectangular cross-section shape.

.. autoclass:: RectSubstrate
   :members:

"""
import cv2  # type: ignore
import dataclasses
import numpy as np
import numpy.typing as npt
from .polysubstrate import PolySubstrateBase
from .polysubstrate_param import Parameters as Parameters
from .rectsubstrate_param import DrawOptions, PaintMode
from .util import colorize


__all__ = [
    "RectSubstrate",
]


class RectSubstrate(PolySubstrateBase[Parameters, DrawOptions]):
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
       >>> tempROI = (13, 10, 1246, 200)
       >>> substROI = (100, 100, 1200, 500)
       >>> ref = SubstrateReference(img, tempROI, substROI)
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
       >>> subst.draw_options.sidelines.color.red = 255
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    """

    Parameters = Parameters
    DrawOptions = DrawOptions
    SidesNum = 4

    PaintMode = PaintMode

    def draw(self) -> npt.NDArray[np.uint8]:
        paint = self.draw_options.paint
        if paint is self.PaintMode.ORIGINAL:
            image = self.image()
        elif paint is self.PaintMode.BINARY:
            image = self.binary_image()
        elif paint is self.PaintMode.EDGES:
            h, w = self.image().shape[:2]
            mask = np.zeros((h, w), bool)
            ((x, y),) = self.contour().transpose(1, 2, 0)
            mask[y, x] = True
            image = ~mask * np.uint8(255)
        else:
            raise TypeError("Unrecognized draw mode: %s" % paint)
        ret = colorize(image)

        vert_opts = self.draw_options.vertices
        if vert_opts.linewidth > 0:
            color = dataclasses.astuple(vert_opts.color)
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

            color = dataclasses.astuple(side_opts.color)
            linewidth = side_opts.linewidth
            cv2.line(ret, tl, tr, color, linewidth)
            cv2.line(ret, tr, br, color, linewidth)
            cv2.line(ret, br, bl, color, linewidth)
            cv2.line(ret, bl, tl, color, linewidth)

        return ret
