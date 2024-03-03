"""Package for image analysis of finite depth dip coating.

To analyze with command line, specify the parameters in configuration file(s) and run::

    finitedepth analyze <file1> [<file2> ...]
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
from importlib.metadata import entry_points
from importlib.resources import files

import yaml

from .reference import Reference, ReferenceBase

__all__ = [
    "get_sample_path",
    "analyze_files",
    "ReferenceBase",
    "Reference",
]


def get_sample_path(*paths: str) -> str:
    """Get path to sample file.

    Arguments:
        paths: Subpaths under ``finitedepth/samples/`` directory.

    Returns:
        Absolute path to the sample file.

    Examples:
        >>> from finitedepth import get_sample_path
        >>> get_sample_path() # doctest: +SKIP
        'path/finitedepth/samples'
        >>> get_sample_path("myfile") # doctest: +SKIP
        'path/finitedepth/samples/myfile'
    """
    return str(files("finitedepth").joinpath("samples", *paths))


def analyze_files(
    *paths: str, recursive: bool = False, entries: list[str] | None = None
) -> bool:
    """Perform analysis from configuration files.

    Supported formats:
        * YAML
        * JSON

    Each file can have multiple entries. Each entry must have ``type`` field which
    specifies the analyzer. For example, the following YAML file contains ``foo``
    entry which is analyzed by ``Foo`` analyzer.

    .. code-block:: yaml

        foo:
            type: Foo
            ...

    Analyzers are searched and loaded from entry point group
    ``"finitedepth.analyzers"``, and must have the following signature:

    * entry name (:obj:`str`)
    * entry fields (:obj:`dict`)

    Arguments:
        paths: Glob pattern for configuration file paths.
        recursive: If True, search *paths* recursively.
        entries: Regular expression for entries.
            If passed, only the matching entries are analyzed.

    Returns:
        Whether the analysis is finished without error.
    """
    # load analyzers
    ANALYZERS = {}
    for ep in entry_points(group="finitedepth.analyzers"):
        ANALYZERS[ep.name] = ep

    glob_paths = []
    for path in paths:
        glob_paths.extend(glob.glob(os.path.expandvars(path), recursive=recursive))

    if entries is not None:
        entry_patterns = [re.compile(e) for e in entries]
    else:
        entry_patterns = []

    ok = True
    for path in glob_paths:
        _, ext = os.path.splitext(path)
        ext = ext.lstrip(os.path.extsep).lower()
        try:
            with open(path, "r") as f:
                if ext == "yaml" or ext == "yml":
                    data = yaml.load(f, Loader=yaml.FullLoader)
                elif ext == "json":
                    data = json.load(f)
                else:
                    logging.error(f"Skipping file: '{path}' (format not supported)")
                    ok = False
                    continue
        except FileNotFoundError:
            logging.error(f"Skipping file: '{path}' (does not exist)")
            ok = False
            continue
        for k, v in data.items():
            if entry_patterns and all([p.fullmatch(k) is None for p in entry_patterns]):
                continue
            try:
                typename = v["type"]
                analyzer = ANALYZERS.get(typename, None)
                if analyzer is not None:
                    analyzer.load()(k, v)
                else:
                    logging.error(
                        f"Skipping entry: '{path}::{k}' (unknown type: '{typename}')"
                    )
            except Exception:
                logging.exception(f"Skipping entry: '{path}::{k}' (exception raised)")
                ok = False
                continue
    return ok


def main():
    """Entry point function."""
    parser = argparse.ArgumentParser(
        prog="finitedepth",
        description="Finite depth dip coating image analysis.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="set logging level",
    )
    subparsers = parser.add_subparsers(dest="command")

    samples = subparsers.add_parser(
        "samples",
        description="Print path to sample directory.",
        help="print path to sample directory",
    ).add_mutually_exclusive_group()
    samples.add_argument(
        "plugin",
        type=str,
        nargs="?",
        help="name of the plugin",
    )
    samples.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="list plugin names",
    )

    analyze = subparsers.add_parser(
        "analyze",
        description="Parse configuration files and analyze.",
        help="parse configuration files and analyze",
        epilog=(
            "Supported file formats: YAML, JSON.\n"
            "Refer to the package documentation for configuration file structure."
        ),
    )
    analyze.add_argument(
        "file", type=str, nargs="+", help="glob pattern for configuration files"
    )
    analyze.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="recursively find configuration files",
    )
    analyze.add_argument(
        "-e",
        "--entry",
        action="append",
        help="regex pattern for configuration file entries",
    )

    args = parser.parse_args()

    loglevel = args.log_level.upper()
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)8s] --- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=loglevel,
    )

    logging.debug(f"Input command: {' '.join(sys.argv)}")

    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(1)
    elif args.command == "samples":
        if args.list:
            header = [("PLUGIN", "PATH")]
            paths = [
                (ep.name, ep.load()())
                for ep in entry_points(group="finitedepth.samples")
            ]
            col0_max = max(len(p[0]) for p in header + paths)
            space = 3
            for col0, col1 in header + paths:
                line = col0.ljust(col0_max) + " " * space + col1
                print(line)
        elif args.plugin is None:
            print(get_sample_path())
        else:
            for ep in entry_points(group="finitedepth.samples"):
                if ep.name == args.plugin:
                    getter = ep.load()
                    print(getter())
                    break
            else:
                logging.error(
                    f"Unknown plugin: '{args.plugin}' (use '-l' option to list plugins)"
                )
                sys.exit(1)
    elif args.command == "analyze":
        ok = analyze_files(*args.file, recursive=args.recursive, entries=args.entry)
        if not ok:
            sys.exit(1)
