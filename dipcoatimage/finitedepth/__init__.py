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
    RectSubstrateBase,
    RectSubstrateHoughLinesError,
    RectSubstrateEdgeError,
    RectSubstrate,
)
from .coatinglayer import (
    CoatingLayerError,
    CoatingLayerBase,
    LayerArea,
)
from .rectcoatinglayer import (
    RectCoatingLayerBase,
    RectLayerShape,
)
from .experiment import ExperimentError, ExperimentBase, Experiment
from .analysis import ExperimentKind, experiment_kind, Analyzer
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
from .util import (
    get_samples_path,
    HoughLinesParameters,
    MorphologyClosingParameters,
)


__all__ = [
    "SubstrateReferenceError",
    "SubstrateReferenceBase",
    "SubstrateReference",
    "SubstrateError",
    "SubstrateBase",
    "Substrate",
    "RectSubstrateError",
    "RectSubstrateBase",
    "RectSubstrateHoughLinesError",
    "RectSubstrateEdgeError",
    "RectSubstrate",
    "CoatingLayerError",
    "CoatingLayerBase",
    "LayerArea",
    "RectCoatingLayerBase",
    "RectLayerShape",
    "ExperimentError",
    "ExperimentBase",
    "Experiment",
    "ExperimentKind",
    "experiment_kind",
    "Analyzer",
    "data_converter",
    "ImportArgs",
    "ReferenceArgs",
    "SubstrateArgs",
    "CoatingLayerArgs",
    "ExperimentArgs",
    "AnalysisArgs",
    "ExperimentData",
    "get_samples_path",
    "HoughLinesParameters",
    "MorphologyClosingParameters",
]
