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

from importlib_resources import files

from .analysis import Analysis, AnalysisBase, AnalysisError
from .coatinglayer import CoatingLayer, CoatingLayerBase, CoatingLayerError
from .experiment import Experiment, ExperimentBase, ExperimentError
from .polysubstrate import PolySubstrateBase, PolySubstrateError
from .rectcoatinglayer import RectCoatingLayerBase, RectLayerShape
from .rectsubstrate import RectSubstrate
from .reference import Reference, ReferenceBase, ReferenceError
from .serialize import Config, data_converter
from .substrate import Substrate, SubstrateBase, SubstrateError
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


def get_data_path(*paths: str) -> str:
    """
    Get path to data file.

    Parameters
    ----------
    paths : str
        Subpaths under ``dipcoatimage/finitedepth/data/`` directory.

    Returns
    -------
    path
        Absolute path to the data.

    Examples
    ========

    >>> from dipcoatimage.finitedepth import get_data_path
    >>> get_data_path() # doctest: +SKIP
    'path/dipcoatimage/finitedepth/data'
    >>> get_data_path("coat1.png") # doctest: +SKIP
    'path/dipcoatimage/finitedepth/data/coat1.png'

    """
    data_path = files("dipcoatimage.finitedepth.data")
    if not paths:
        return str(data_path._paths[0])
    return str(data_path.joinpath(*paths))
