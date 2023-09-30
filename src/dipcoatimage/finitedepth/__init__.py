"""Image analysis for finit-depth dip coating.

DipCoatImage-FiniteDepth is an image analysis package for the coating layer
shape in finite-depth dip coating process. Analysis is done by locating the
bare substrate from coated substrate image, and extracting the coating layer
region.

This package provides

1. Handling substrate reference image and coated substrate image
2. Detecting bare substrate geometry
3. Extracting and analyzing coating layer
4. Saving analysis result
"""

import json
import os

import yaml
from importlib_resources import files

from .analysis import Analysis, AnalysisBase, AnalysisError
from .coatinglayer import CoatingLayer, CoatingLayerBase, CoatingLayerError
from .experiment import Experiment, ExperimentBase, ExperimentError
from .polysubstrate import PolySubstrateBase, PolySubstrateError
from .rectcoatinglayer import RectCoatingLayerBase, RectLayerShape
from .rectsubstrate import RectSubstrate
from .reference import Reference, ReferenceBase, ReferenceError
from .serialize import Config, ConfigBase, data_converter
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
    "ConfigBase",
    "Config",
    "get_data_path",
    "analyze_files",
]


def get_data_path(*paths: str) -> str:
    """Get path to data file.

    Parameters
    ----------
    paths : str
        Subpaths under ``dipcoatimage/finitedepth/data/`` directory.

    Returns
    -------
    path
        Absolute path to the data.

    Examples
    --------
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


def analyze_files(*paths: str):
    """Analyze from config files.

    Supported files are:
    - YAML
    - JSON
    """
    for path in paths:
        _, ext = os.path.splitext(path)
        ext = ext.lstrip(os.path.extsep).lower()
        try:
            with open(path, "r") as f:
                if ext == "yaml" or ext == "yml":
                    data = yaml.load(f, Loader=yaml.FullLoader)
                elif ext == "json":
                    data = json.load(f)
                else:
                    print(f"Skipping {path} ({ext} not supported)")
        except FileNotFoundError:
            print(f"Skipping {path} (path does not exist)")
            continue
        for k, v in data.items():
            try:
                config = data_converter.structure(v, Config)
                config.analyze(k)
            except Exception as err:
                print(f"Skipping {k} ({type(err).__name__}: {err})")
                continue


def main():
    """Entry point function."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="finitedepth",
        description="Finite depth dip coating analysis tool",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "data",
        description="Print path to sample data directory.",
        help="Print path to sample data directory.",
        epilog=(
            "In Python runtime, dipcoatimage.finitedepth.get_data_path() "
            "returns same result."
        ),
    )

    parser_analyze = subparsers.add_parser(
        "analyze",
        description="Parse configuration files and analyze.",
        help="Parse configuration files and analyze.",
        epilog="Supported file formats: YAML, JSON.",
    )
    parser_analyze.add_argument("file", type=str, nargs="+", help="target files")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(1)
    elif args.command == "data":
        print(get_data_path())
    elif args.command == "analyze":
        analyze_files(args.files)
