"""
Substrate geometry
==================

:mod:`dipcoatimage.finitedepth.substrate` provides substrate image class of
finite-depth dip coating process to recognize its geometry.

Base class
----------

.. autoclass:: SubstrateError
   :members:

.. autoclass:: SubstrateBase
   :members:

Implementation
--------------

.. autoclass:: Substrate
   :members:

.. automodule:: dipcoatimage.finitedepth.rectsubstrate

"""


import abc
import dataclasses
import cv2  # type: ignore
import numpy as np
import numpy.typing as npt
from .reference import SubstrateReferenceBase
from .substrate_param import Parameters, DrawOptions
from typing import TypeVar, Generic, Type, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "SubstrateError",
    "SubstrateBase",
    "Substrate",
]


class SubstrateError(Exception):
    """Base class for error from :class:`SubstrateBase`."""

    pass


ParametersType = TypeVar("ParametersType", bound="DataclassInstance")
DrawOptionsType = TypeVar("DrawOptionsType", bound="DataclassInstance")


class SubstrateBase(abc.ABC, Generic[ParametersType, DrawOptionsType]):
    """
    Abstract base class for substrate.

    Substrate class recognizes the geometry of substrate image from
    :class:`.SubstrateReferenceBase`.

    .. rubric:: Constructor

    Constructor signature must not be modified because high-level API use factory
    to generate substrate instances. Additional parameters can be introduced by
    definig class attribute :attr:`Parameters` and :attr:`DrawOptions`.

    .. rubric:: Parameters and DrawOptions

    Concrete class must have :attr:`Parameters` and :attr:`DrawOptions` which
    return dataclass types. Their instances are passed to the constructor at
    instance initialization, and can be accessed by :attr:`parameters` and
    :attr:`draw_options`.

    :attr:`Parameter` must be frozen to ensure immtability for caching. However,
    :attr:`DrawOptions` need not be frozen since visualization does not affect
    the identity of instance. Therefore methods affected by draw options must
    not be cached.

    .. rubric:: Sanity check

    Validity of the parameters can be checked by :meth:`verify` or :meth:`valid`.
    Their result can be implemented by defining :meth:`examine`.

    .. rubric:: Visualization

    :meth:`draw` defines the visualization logic for concrete class.
    Modifying :attr:`draw_options` changes the visualization result.

    Parameters
    ==========

    reference
        Substrate reference instance.

    parameters
        Additional parameters.

    draw_options
        Drawing options.

    """

    __slots__ = (
        "_ref",
        "_parameters",
        "_draw_options",
        "_regions",
        "_contours",
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]

    def __init__(
        self,
        reference: SubstrateReferenceBase,
        parameters: Optional[ParametersType] = None,
        *,
        draw_options: Optional[DrawOptionsType] = None,
    ):
        super().__init__()
        self._ref = reference

        if parameters is None:
            self._parameters = self.Parameters()
        else:
            self._parameters = dataclasses.replace(parameters)

        if draw_options is None:
            self._draw_options = self.DrawOptions()
        else:
            self._draw_options = dataclasses.replace(draw_options)

    @property
    def reference(self) -> SubstrateReferenceBase:
        """Substrate reference instance passed to constructor."""
        return self._ref

    @property
    def parameters(self) -> ParametersType:
        """
        Additional parameters for concrete class.

        Instance of :attr:`Parameters`, which must be a frozen dataclass.
        """
        return self._parameters

    @property
    def draw_options(self) -> DrawOptionsType:
        """
        Options to visualize the image.

        Instance of :attr:`DrawOptions` dataclass.
        """
        return self._draw_options

    @draw_options.setter
    def draw_options(self, options: DrawOptionsType):
        self._draw_options = options

    def image(self) -> npt.NDArray[np.uint8]:
        """Substrate image from :meth:`reference`."""
        # not property since it's not directly from the argument
        return self.reference.substrate_image()

    def binary_image(self) -> npt.NDArray[np.uint8]:
        """Binarized substrate image from :meth:`reference`."""
        x0, y0, x1, y1 = self.reference.substrateROI
        return self.reference.binary_image()[y0:y1, x0:x1]

    @abc.abstractmethod
    def region_points(self) -> npt.NDArray[np.int32]:
        """
        Points in `[x, y]` in each discrete substrate region.

        Returns
        -------
        ndarray
            `(N, 2)`-shaped array, where `N` is the number of discrete substrate
            regions.

        Notes
        -----
        These points are used to locate and distinguish each substrate region
        from foreground pixels. Every image analysis implementation is based on
        these points.

        Subclass should implement this method to return robust points. The
        implementation must be simple and non-dynamic. Returning just the center
        point of the image is a good example; selection of the substrate region
        from the reference image must be done to obey this rule (e.g., select
        s.t. the center point falls on the substrate region)
        """

    def regions(self) -> npt.NDArray[np.int8]:
        """
        Return image labelled by each discrete substrate regions.

        Returns
        -------
        ndarray
            Labelled image. `i`-th region in :meth:`region_points` is labelled
            with `i`. `-1` denotes background.

        Notes
        -----
        Maximum number of regions is 128.

        See Also
        --------
        region_points
        """
        if not hasattr(self, "_regions"):
            self._regions = np.full(self.image().shape[:2], -1, dtype=np.int8)
            _, labels = cv2.connectedComponents(cv2.bitwise_not(self.binary_image()))
            for i, pt in enumerate(self.region_points()):
                self._regions[labels == labels[pt[1], pt[0]]] = i
        return self._regions

    def contours(
        self, region: int
    ) -> Tuple[Tuple[npt.NDArray[np.int32], ...], npt.NDArray[np.int32]]:
        """
        Find contours of a substrate region identified by *region*.

        Parameters
        ----------
        region : int
            Label in :meth:`regions`.

        Returns
        -------
        tuple
            Tuple of the result of :func:`cv2.findContours`.

        Notes
        -----
        Contours are dense, i.e., no approximation is made.

        See Also
        --------
        regions
        """
        if not hasattr(self, "_contours"):
            contours = []
            for i in range(len(self.region_points())):
                reg = (self.regions() == i) * np.uint8(255)
                cnt = cv2.findContours(reg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                contours.append(cnt)
            self._contours = tuple(contours)
        return self._contours[region]

    @abc.abstractmethod
    def examine(self) -> Optional[SubstrateError]:
        """
        Check the sanity of parameters.

        If the instance is invalid, return error instance.
        Else, return :obj:`None`.
        """

    def verify(self):
        """
        Verify if all parameters are suitably set by raising error on failure.

        To implement sanity check for concrete class, define :meth:`examine`.
        """
        err = self.examine()
        if err is not None:
            raise err

    def valid(self) -> bool:
        """
        Verify if all parameters are suitably set by returning boolean value.

        To implement sanity check for concrete class, define :meth:`examine`.
        """
        err = self.examine()
        ret = True
        if err is not None:
            ret = False
        return ret

    @abc.abstractmethod
    def draw(self) -> npt.NDArray[np.uint8]:
        """Decorate and return the substrate image as RGB format."""


class Substrate(SubstrateBase[Parameters, DrawOptions]):
    """
    Simplest substrate class with no geometric information.

    Examples
    ========

    Construct substrate reference instance first.

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import (SubstrateReference,
       ...     get_data_path)
       >>> ref_path = get_data_path("ref1.png")
       >>> img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
       >>> tempROI = (200, 50, 1200, 200)
       >>> substROI = (400, 175, 1000, 500)
       >>> ref = SubstrateReference(img, tempROI, substROI)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    Construct :class:`Substrate` instance from reference instance.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import Substrate
       >>> subst = Substrate(ref)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    """

    Parameters = Parameters
    DrawOptions = DrawOptions

    def region_points(self) -> npt.NDArray[np.int32]:
        w = self.image().shape[1]
        return np.array([[w / 2, 0]], dtype=np.int32)

    def examine(self) -> None:
        return None

    def draw(self) -> npt.NDArray[np.uint8]:
        return self.image().copy()
