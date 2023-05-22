"""
Rectangular Substrate
=====================

:mod:`dipcoatimage.finitedepth.rectsubstrate` provides substrate image class to
analyze the substrate with rectangular cross-section shape.

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
from .polysubstrate import PolySubstrateParameters, PolySubstrateBase
from .util import (
    colorize,
    FeatureDrawingOptions,
    Color,
)

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias


__all__ = [
    "RectSubstrateDrawMode",
    "RectSubstrateDrawOptions",
    "RectSubstrate",
]


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
    sides: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(0, 0, 255), thickness=1
    )
    hull: FeatureDrawingOptions = FeatureDrawingOptions(
        color=Color(255, 0, 0), thickness=1
    )


class RectSubstrate(
    PolySubstrateBase[PolySubstrateParameters, RectSubstrateDrawOptions]
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

       >>> subst.draw_options.sides.thickness = 3
       >>> subst.draw_options.sides.color.red = 255
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    """

    Parameters = PolySubstrateParameters
    DrawOptions = RectSubstrateDrawOptions
    SidesNum = 4

    DrawMode: TypeAlias = RectSubstrateDrawMode

    def contour(self):
        # Flip s.t. the hull has same direction with the contour.
        # XXX: remove after implementing uniform layer for convex polyline
        return np.flip(cv2.convexHull(super().contour()), axis=0)

    def draw(self) -> npt.NDArray[np.uint8]:
        draw_mode = self.draw_options.draw_mode
        if draw_mode is self.DrawMode.ORIGINAL:
            image = self.image()
        elif draw_mode is self.DrawMode.BINARY:
            image = self.binary_image()
        elif draw_mode is self.DrawMode.EDGES:
            h, w = self.image().shape[:2]
            mask = np.zeros((h, w), bool)
            ((x, y),) = self.contour().transpose(1, 2, 0)
            mask[y, x] = True
            image = ~mask * np.uint8(255)
        else:
            raise TypeError("Unrecognized draw mode: %s" % draw_mode)
        ret = colorize(image)

        side_opts = self.draw_options.sides
        if side_opts.thickness > 0:
            tl, bl, br, tr = self.sideline_intersections().astype(np.int32)

            color = dataclasses.astuple(side_opts.color)
            thickness = side_opts.thickness
            cv2.line(ret, tl, tr, color, thickness)
            cv2.line(ret, tr, br, color, thickness)
            cv2.line(ret, br, bl, color, thickness)
            cv2.line(ret, bl, tl, color, thickness)

        hull_opts = self.draw_options.hull
        if hull_opts.thickness > 0:
            cv2.polylines(
                ret,
                [self.contour().astype(np.int32)],
                isClosed=False,
                color=dataclasses.astuple(hull_opts.color),
                thickness=hull_opts.thickness,
            )

        return ret
