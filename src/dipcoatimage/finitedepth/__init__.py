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
    ReferenceError,
    ReferenceBase,
    Reference,
)
from .substrate import (
    SubstrateError,
    SubstrateBase,
    Substrate,
)
from .polysubstrate import (
    PolySubstrateError,
    PolySubstrateBase,
)
from .rectsubstrate import (
    RectSubstrate,
)
from .coatinglayer import (
    CoatingLayerError,
    CoatingLayerBase,
    CoatingLayer,
)
from .rectcoatinglayer import (
    RectCoatingLayerBase,
    RectLayerShape,
)
from .experiment import ExperimentError, ExperimentBase, Experiment
from .analysis import (
    ExperimentKind,
    experiment_kind,
    DataWriter,
    CSVWriter,
    AnalysisError,
    AnalysisBase,
    Analysis,
)
from .serialize import (
    data_converter,
    ImportArgs,
    ReferenceArgs,
    SubstrateArgs,
    CoatingLayerArgs,
    ExperimentArgs,
    AnalysisArgs,
    ExperimentData,
)
from .util import get_data_path


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
    "ExperimentKind",
    "experiment_kind",
    "DataWriter",
    "CSVWriter",
    "AnalysisError",
    "AnalysisBase",
    "Analysis",
    "data_converter",
    "ImportArgs",
    "ReferenceArgs",
    "SubstrateArgs",
    "CoatingLayerArgs",
    "ExperimentArgs",
    "AnalysisArgs",
    "ExperimentData",
    "get_data_path",
]
