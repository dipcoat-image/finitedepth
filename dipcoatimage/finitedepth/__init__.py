"""
DipCoatImage-FiniteDepth
========================

Image analysis package for the coating layer shape in finite-depth dip coating
process. Analysis is done by locating the bare substrate from coated substrate
image, and extracting the coating layer region.

This package provides

1. Handling substrate reference image and coated substrate image
2. Detecting bare substrate geometry
3. Extracting and analyzing coating layer
4. Saving analysis result

:mod:`dipcoatimage.finitedepth_gui` provides GUI for this package.

"""

from .version import __version__  # noqa

from .reference import (
    SubstrateReferenceError,
    SubstrateReferenceBase,
    SubstrateReference,
)
from .substrate import (
    SubstrateError,
    SubstrateBase,
    Substrate,
)
from .rectsubstrate import (
    RectSubstrateError,
    RectSubstrateHoughLinesError,
    RectSubstrateEdgeError,
    RectSubstrate,
)
from .coatinglayer import (
    CoatingLayerError,
    CoatingLayerBase,
    LayerArea,
)
from .rectcoatinglayer import RectCoatingLayerBase, RectLayerArea
from .experiment import (
    ExperimentError,
    ExperimentBase,
    Experiment
)
from .util import (
    get_samples_path,
    data_converter,
    CannyParameters,
    HoughLinesParameters,
)


__all__ = [
    "SubstrateReferenceError",
    "SubstrateReferenceBase",
    "SubstrateReference",
    "SubstrateError",
    "SubstrateBase",
    "Substrate",
    "CannyParameters",
    "HoughLinesParameters",
    "RectSubstrateError",
    "RectSubstrateHoughLinesError",
    "RectSubstrateEdgeError",
    "RectSubstrate",
    "CoatingLayerError",
    "CoatingLayerBase",
    "LayerArea",
    "RectCoatingLayerBase",
    "RectLayerArea",
    "ExperimentError",
    "ExperimentBase",
    "Experiment",
    "get_samples_path",
    "data_converter",
    "CannyParameters",
    "HoughLinesParameters",
]
