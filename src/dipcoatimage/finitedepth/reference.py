"""Reference class.

Reference class is a concrete subclass of :class:`ReferenceBase`.
"""


import abc
import dataclasses
from typing import TYPE_CHECKING, Generic, Optional, Tuple, Type, TypeVar

import cv2
import numpy as np
import numpy.typing as npt

from .parameters import LineOptions

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "OptionalROI",
    "IntROI",
    "ReferenceBase",
    "Reference",
    "sanitize_ROI",
]


ParametersType = TypeVar("ParametersType", bound="DataclassInstance")
DrawOptionsType = TypeVar("DrawOptionsType", bound="DataclassInstance")
DataType = TypeVar("DataType", bound="DataclassInstance")

OptionalROI = Tuple[int, int, Optional[int], Optional[int]]
"""Type annotation for ROI whose upper limits of x and y can be :obj:`None`.

:obj:`None` should be interpreted as a maximum value in the image.
"""
IntROI = Tuple[int, int, int, int]
"""Type annotation for ROI whose items are all integer.

:obj:`OptionalROI` can be converted to :obj:`IntROI` if the shape of the image
is given.
"""


class ReferenceBase(abc.ABC, Generic[ParametersType, DrawOptionsType, DataType]):
    """Abstract base class for reference instance."""

    Parameters: Type[ParametersType]
    DrawOptions: Type[DrawOptionsType]
    Data: Type[DataType]

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        templateROI: OptionalROI = (0, 0, None, None),
        substrateROI: OptionalROI = (0, 0, None, None),
        parameters: Optional[ParametersType] = None,
        *,
        draw_options: Optional[DrawOptionsType] = None,
    ):
        """Initialize the instance."""
        super().__init__()
        self._image = image
        self._image.setflags(write=False)

        h, w = image.shape[:2]
        self._templateROI = sanitize_ROI(templateROI, h, w)
        self._substrateROI = sanitize_ROI(substrateROI, h, w)

        if parameters is None:
            self._parameters = self.Parameters()
        else:
            if not isinstance(parameters, self.Parameters):
                raise TypeError(f"{parameters} is not instance of {self.Parameters}")
            self._parameters = dataclasses.replace(parameters)

        if draw_options is None:
            self._draw_options = self.DrawOptions()
        else:
            if not isinstance(draw_options, self.DrawOptions):
                raise TypeError(f"{draw_options} is not instance of {self.DrawOptions}")
            self._draw_options = dataclasses.replace(draw_options)

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """Reference image. Must be binary.

        This array is not writable to be immutable for caching.
        """
        return self._image

    @property
    def templateROI(self) -> IntROI:
        """Slice indices in ``(x0, y0, x1, y1)`` for template region."""
        return self._templateROI

    @property
    def substrateROI(self) -> IntROI:
        """Slice indices in ``(x0, y0, x1, y1)`` for substrate region."""
        return self._substrateROI

    @property
    def parameters(self) -> ParametersType:
        """Additional parameters for concrete class.

        Instance of :attr:`Parameters`, which must be a frozen dataclass.
        """
        return self._parameters

    @property
    def draw_options(self) -> DrawOptionsType:
        """Options to visualize the image.

        Instance of :attr:`DrawOptions` dataclass.
        """
        return self._draw_options

    @draw_options.setter
    def draw_options(self, options: DrawOptionsType):
        self._draw_options = options

    def substrate_image(self) -> npt.NDArray[np.uint8]:
        """:meth:`image` cropped by :meth:`substrateROI`."""
        x0, y0, x1, y1 = self.substrateROI
        return self.image[y0:y1, x0:x1]

    def temp2subst(self) -> npt.NDArray[np.int32]:
        """Vector from template region to substrate region."""
        x0, y0 = self.templateROI[:2]
        x1, y1 = self.substrateROI[:2]
        return np.array([x1 - x0, y1 - y0], dtype=np.int32)

    @abc.abstractmethod
    def verify(self):
        """Check to detect error and raise before analysis."""

    @abc.abstractmethod
    def draw(self) -> npt.NDArray[np.uint8]:
        """Decorate and return the reference image as RGB format."""

    @abc.abstractmethod
    def analyze(self) -> DataType:
        """Analyze the reference image and return the data.

        May raise error if the instance is not valid.
        """


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Additional parameters for `Reference` instance."""

    pass


@dataclasses.dataclass
class DrawOptions:
    """Drawing options for `Reference`.

    Attributes
    ----------
    templateROI, substrateROI : LineOptions
        Determines how the ROIs are drawn.
    """

    templateROI: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(0, 255, 0), linewidth=1)
    )
    substrateROI: LineOptions = dataclasses.field(
        default_factory=lambda: LineOptions(color=(255, 0, 0), linewidth=1)
    )


@dataclasses.dataclass
class Data:
    """Analysis data for `Reference`."""

    pass


class Reference(ReferenceBase[Parameters, DrawOptions, Data]):
    """Basic implementation of reference class.

    Examples
    --------
    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import Reference, get_data_path
       >>> gray = cv2.imread(get_data_path("ref1.png"), cv2.IMREAD_GRAYSCALE)
       >>> _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
       >>> tempROI = (200, 50, 1200, 200)
       >>> substROI = (400, 175, 1000, 500)
       >>> ref = Reference(img, tempROI, substROI)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    Visualization can be controlled by modifying :attr:`draw_options`.

    .. plot::
       :include-source:
       :context: close-figs

       >>> ref.draw_options.substrateROI.color = (0, 255, 255)
       >>> plt.imshow(ref.draw()) #doctest: +SKIP
    """

    Parameters = Parameters
    DrawOptions = DrawOptions
    Data = Data

    def verify(self):
        """Check error."""
        pass

    def draw(self) -> npt.NDArray[np.uint8]:
        """Return visualized result."""
        ret = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        substROI_opts = self.draw_options.substrateROI
        if substROI_opts.linewidth > 0:
            x0, y0, x1, y1 = self.substrateROI
            color = substROI_opts.color
            linewidth = substROI_opts.linewidth
            cv2.rectangle(ret, (x0, y0), (x1, y1), color, linewidth)

        tempROI_opts = self.draw_options.templateROI
        if tempROI_opts.linewidth > 0:
            x0, y0, x1, y1 = self.templateROI
            color = tempROI_opts.color
            linewidth = tempROI_opts.linewidth
            cv2.rectangle(ret, (x0, y0), (x1, y1), color, linewidth)
        return ret

    def analyze(self):
        """Return analysis data."""
        return self.Data()


def sanitize_ROI(roi: OptionalROI, h: int, w: int) -> IntROI:
    """Convert :obj:`OptionalROI` to :obj:`IntROI`."""
    full_roi = (0, 0, w, h)
    max_vars = (w, h, w, h)

    ret = list(roi)
    for i, var in enumerate(roi):
        if var is None:
            ret[i] = full_roi[i]
        elif var < 0:
            ret[i] = max_vars[i] + var
    return tuple(ret)  # type: ignore[return-value]
