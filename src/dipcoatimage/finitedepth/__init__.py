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

from .analysis import Analysis, AnalysisBase, AnalysisError
from .coatinglayer import CoatingLayer, CoatingLayerBase, CoatingLayerError
from .experiment import Experiment, ExperimentBase, ExperimentError
from .polysubstrate import PolySubstrateBase, PolySubstrateError
from .rectcoatinglayer import RectCoatingLayerBase, RectLayerShape
from .rectsubstrate import RectSubstrate
from .reference import Reference, ReferenceBase, ReferenceError
from .serialize import Config, data_converter
from .substrate import Substrate, SubstrateBase, SubstrateError
from .util import get_data_path
from .version import __version__  # noqa

__all__ = [
    "ReferenceError",
    "ReferenceBase",
    "Reference",
    "SubstrateError",
    "SubstrateBase",
    "Substrate",
    "PolySubstrateError",
    "PolySubstrateBase",
    "RectSubstrate",
    "CoatingLayerError",
    "CoatingLayerBase",
    "CoatingLayer",
    "RectCoatingLayerBase",
    "RectLayerShape",
    "ExperimentError",
    "ExperimentBase",
    "Experiment",
    "AnalysisError",
    "AnalysisBase",
    "Analysis",
    "data_converter",
    "Config",
    "get_data_path",
]
