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
from typing import Tuple
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

       >>> subst.draw_options.sides.thickness = 3
       >>> subst.draw_options.sides.color.red = 255
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    """

    Parameters = PolySubstrateParameters
    DrawOptions = RectSubstrateDrawOptions
    SidesNum = 4

    DrawMode: TypeAlias = RectSubstrateDrawMode

    def edge_hull(self) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
        hull = np.flip(cv2.convexHull(self.contour()), axis=0)
        # TODO: get more points by interpolating to `hull`
        tangent = np.gradient(hull, axis=0)
        # TODO: perform edge tangent flow to get smoother curve
        return hull, tangent

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
            image = mask * np.uint8(255)
        else:
            raise TypeError("Unrecognized draw mode: %s" % draw_mode)
        ret = colorize(image)

        side_opts = self.draw_options.sides
        if side_opts.thickness > 0:
            tl, bl, br, tr = self.vertex_points().astype(np.int32)

            color = dataclasses.astuple(side_opts.color)
            thickness = side_opts.thickness
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
