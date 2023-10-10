"""Image analysis for finite depth dip coating.

DipCoatImage-FiniteDepth provides an extensible framework to quantify the coating
layer shape in finite depth dip coating process. Using reference image of bare
substrate, coating layer region is acquired from target image of coated substrate
and analyzed.

The following abstract base classes constitute analysis workflow:

1. :class:`ReferenceBase`
2. :class:`SubstrateBase`
3. :class:`CoatingLayerBase`
4. :class:`ExperimentBase`
5. :class:`AnalysisBase`

Analysis parameters can be structured as :class:`Config` instance for
serialization. The serialized configuration file can be analyzed using
command line prompt:

.. code-block:: bash

    finitedepth analyze file [file ...]
"""

import argparse
import json
import os
import sys

import yaml
from importlib_resources import files

from .analysis import Analysis, AnalysisBase
from .coatinglayer import CoatingLayer, CoatingLayerBase
from .experiment import Experiment, ExperimentBase
from .polysubstrate import PolySubstrateBase
from .rectcoatinglayer import RectCoatingLayerBase, RectLayerShape
from .rectsubstrate import RectSubstrate
from .reference import Reference, ReferenceBase
from .serialize import Config, ConfigBase, data_converter
from .substrate import Substrate, SubstrateBase
from .version import __version__  # noqa

__all__ = [
    "ReferenceBase",
    "Reference",
    "SubstrateBase",
    "Substrate",
    "PolySubstrateBase",
    "RectSubstrate",
    "CoatingLayerBase",
    "CoatingLayer",
    "RectCoatingLayerBase",
    "RectLayerShape",
    "ExperimentBase",
    "Experiment",
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

    Arguments:
        paths: Subpaths under ``dipcoatimage/finitedepth/data/`` directory.

    Returns:
        Absolute path to the data.

    Examples:
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

    Supported files are: YAML, JSON

    Arguments:
        paths: Configuration file paths.
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
        analyze_files(*args.file)
