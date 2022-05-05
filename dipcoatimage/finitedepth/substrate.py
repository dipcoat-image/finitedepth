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

.. autoclass:: SubstrateParameters
   :members:

.. autoclass:: SubstrateDrawOptions
   :members:

.. autoclass:: Substrate
   :members:

.. automodule:: dipcoatimage.finitedepth.rectsubstrate

"""


import abc
import dataclasses
import numpy as np
import numpy.typing as npt
from typing import TypeVar, Generic, Type, Optional, List, Tuple
from .util import DataclassProtocol
from .reference import SubstrateReferenceBase


__all__ = [
    "SubstrateError",
    "SubstrateBase",
    "SubstrateParameters",
    "SubstrateDrawOptions",
    "Substrate",
]


class SubstrateError(Exception):
    """Base class for error from :class:`SubstrateBase`."""

    pass


ParametersType = TypeVar("ParametersType", bound=DataclassProtocol)
DrawOptionsType = TypeVar("DrawOptionsType", bound=DataclassProtocol)
T = TypeVar("T", bound="SubstrateBase")


class SubstrateBase(abc.ABC, Generic[ParametersType, DrawOptionsType]):
    """
    Abstract base class for substrate.

    Substrate class wraps substrate image from :class:`.SubstrateReferenceBase`
    and recognizes its geometry.

    .. rubric:: Constructor

    Constructor signature must not be modified because high-level API use factory
    to generate substrate instances. Additional parameters can be introduced by
    definig class attribute :attr:`Parameters` and :attr:`DrawOptions`.

    Instead of directly calling the constructor, use :meth:`from_reference` to
    construct the instance.

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

    Parameters
    ==========

    image
        Substrate image.

    parameters
        Additional parameters. Instance of :attr:`Parameters`, or :obj:`None`.

    draw_options
        Drawing options. Instance of :attr:`DrawOptions`, or :obj:`None`.

    """

    __slots__ = (
        "_image",
        "_parameters",
        "_draw_options",
    )

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]

    def __init_subclass__(cls) -> None:
        params = getattr(cls, "Parameters", None)
        if params is None:
            raise TypeError(f"{cls} has no attribute 'Parameters'.")
        elif not (isinstance(params, type) and dataclasses.is_dataclass(params)):
            raise TypeError(f"{params} is not dataclass type.")
        elif not params.__dataclass_params__.frozen:  # type: ignore
            raise TypeError(f"{params} is not frozen.")

        drawopts = getattr(cls, "DrawOptions", None)
        if drawopts is None:
            raise TypeError(f"{cls} has no attribute 'DrawOptions'.")
        elif not (isinstance(drawopts, type) and dataclasses.is_dataclass(drawopts)):
            raise TypeError(f"{drawopts} is not dataclass type.")
        return super().__init_subclass__()

    @classmethod
    def from_reference(
        cls: Type[T],
        ref: SubstrateReferenceBase,
        parameters: Optional[ParametersType] = None,
        *,
        draw_options: Optional[DrawOptionsType] = None,
    ) -> T:
        """Construct the substrate instance from reference instance."""
        return cls(ref.substrate_image, parameters, draw_options=draw_options)

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        parameters: Optional[ParametersType] = None,
        *,
        draw_options: Optional[DrawOptionsType] = None,
    ):
        super().__init__()
        self._image = image
        self._image = image
        self._image.setflags(write=False)

        if parameters is None:
            self._parameters = self.Parameters()
        else:
            self._parameters = dataclasses.replace(parameters)

        if draw_options is None:
            self._draw_options = self.DrawOptions()
        else:
            self._draw_options = dataclasses.replace(draw_options)

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """
        Substrate image passed to constructor.

        This array is not writable to enable caching which requires immutability.
        """
        return self._image

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

    @property
    def nestled_points(self) -> List[Tuple[int, int]]:
        """
        Find the points which are firmly nestled in the substrate.

        This method is used to eliminate the components in dip coating image
        which are not connected to the substrate.

        If the substrate is extremely concave or has holes in its image, this
        method may need to be reimplemented.

        Return value is a list of coordinates in ``(x, y)``, but for most cases
        this method returns a single point.
        If the substrate consists of components which are not connected, multiple
        points may be returned.

        Examples
        ========

        >>> import cv2
        >>> from dipcoatimage.finitedepth import (SubstrateReference, Substrate,
        ...     get_samples_path)
        >>> ref_path = get_samples_path('ref1.png')
        >>> img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
        >>> substROI = (400, 100, 1000, 500)
        >>> ref = SubstrateReference(img, substrateROI=substROI)
        >>> subst = Substrate.from_reference(ref)
        >>> subst.nestled_points
        [(300, 0)]

        """
        w = self.image.shape[1]
        return [(int(w / 2), 0)]

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


@dataclasses.dataclass(frozen=True)
class SubstrateParameters:
    """Additional parameters for :class:`Substrate` instance."""

    pass


@dataclasses.dataclass
class SubstrateDrawOptions:
    """Drawing options for :class:`Substrate`."""

    pass


class Substrate(SubstrateBase):
    """
    Simplest substrate class with no geometric information.

    Examples
    ========

    Construct substrate reference class first.

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import (SubstrateReference,
       ...     get_samples_path)
       >>> ref_path = get_samples_path('ref1.png')
       >>> img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
       >>> tempROI = (200, 50, 1200, 200)
       >>> substROI = (400, 100, 1000, 500)
       >>> ref = SubstrateReference(img, tempROI, substROI)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    Construct substrate class from reference class.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import Substrate
       >>> subst = Substrate.from_reference(ref)
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    """

    Parameters = SubstrateParameters
    DrawOptions = SubstrateDrawOptions

    def examine(self) -> None:
        return None

    def draw(self) -> npt.NDArray[np.uint8]:
        return self.image.copy()
