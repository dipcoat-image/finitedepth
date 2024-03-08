"""Package for image analysis of finite depth dip coating.

To analyze with command line, specify the parameters in configuration file(s) and run::

    finitedepth analyze <file1> [<file2> ...]
"""

import argparse
import csv
import dataclasses
import glob
import json
import logging
import os
import re
import sys
from importlib.metadata import entry_points
from importlib.resources import files

import cv2
import tqdm  # type: ignore
import yaml

from .coatinglayer import CoatingLayer, CoatingLayerBase, RectLayerShape
from .reference import Reference, ReferenceBase
from .substrate import PolySubstrateBase, RectSubstrate, Substrate, SubstrateBase

__all__ = [
    "get_sample_path",
    "analyze_files",
    "ReferenceBase",
    "Reference",
    "SubstrateBase",
    "Substrate",
    "PolySubstrateBase",
    "RectSubstrate",
    "CoatingLayerBase",
    "CoatingLayer",
    "RectLayerShape",
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


def coatingimage_analyzer(name, data):
    """Image analysis for a single image of coated substrate.

    Coating layer is analyzed by constructing :class:`CoatingLayerBase` implementation.

    The analyzer defines the following fields in configuration entry:

    - **referencePath** (`str`): Path to the reference image file.
        Image is converted to binary by Otsu's thresholding.
    - **targetPath** (`str`): Path to the target image file.
        Image is converted to binary by Otsu's thresholding.
    - **reference**, **substrate**, **layer** (`mapping`, optional):
        - **type** (`str`, optional): Registered class constructor.
          The constructors are found from entry point groups
          `'finitedepth.(references|substrates|coatinglayers)'`.
        - **parameters** (`mapping`, optional): Any additional parameter
          for class constructor.
        - **draw** (`mapping`, optional): Parameters for :meth:`draw`.
    - **output** (`mapping`, optional):
        - **(reference|substrate|layer)Data** (`str`, optional):
          Path to the output CSV file containing analysis result.
          The results are acquired from :meth:`analyze`. Invalid coating layer
          gives empty data.
        - **(reference|substrate|layer)Image** (`str`, optional):
          Path to the output image file containing visualization result.
          The results are acquired from :meth:`draw`.

    The default constructors used when **type** is not specified are :class:`Reference`,
    :class:`Substrate`, and :class:`CoatingLayer`.

    The following is the example for an entry in YAML configuration file:

    .. code-block:: yaml

        foo:
            type: CoatingImage
            referencePath: foo-ref.png
            targetPath: foo-target.png
            reference:
                parameters:
                    templateROI: [10, 10, 1250, 200]
                    substrateROI: [100, 100, 1200, 500]
            layer:
                draw:
                    layer_color: [0, 255, 0]
            output:
                layerImage: output/foo.jpg
                layerData: output/foo.csv
    """
    output = data.get("output", {})
    output_refdata = _makedir(output.get("referenceData", ""))
    output_refimg = _makedir(output.get("referenceImage", ""))
    output_substdata = _makedir(output.get("substrateData", ""))
    output_substimg = _makedir(output.get("substrateImage", ""))
    output_layerdata = _makedir(output.get("layerData", ""))
    output_layerimg = _makedir(output.get("layerImage", ""))

    RefType, SubstType, LayerType = _load_types(data)

    if output_refdata:
        csvwriter = _CsvWriter(output_refdata)
        next(csvwriter)
        csvwriter.send([d.name for d in dataclasses.fields(RefType.DataType)])

    try:
        refdata = data.get("reference", {})
        for path in tqdm.tqdm([data["referencePath"]], desc=f"{name} (ref)"):
            _, refimg = cv2.threshold(
                cv2.imread(os.path.expandvars(path), cv2.IMREAD_GRAYSCALE),
                0,
                255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU,
            )
            ref = RefType(refimg, **refdata.get("parameters", {}))
            if output_refdata:
                csvwriter.send(dataclasses.astuple(ref.analyze()))
            if output_refimg:
                cv2.imwrite(
                    output_refimg,
                    cv2.cvtColor(
                        ref.draw(**refdata.get("draw", {})), cv2.COLOR_BGR2RGB
                    ),
                )
    finally:
        if output_refdata:
            csvwriter.close()

    if output_substdata:
        csvwriter = _CsvWriter(output_substdata)
        next(csvwriter)
        csvwriter.send([d.name for d in dataclasses.fields(SubstType.DataType)])

    try:
        substdata = data.get("substrate", {})
        for reference in tqdm.tqdm([ref], desc=f"{name} (subst)"):
            subst = SubstType(reference, **substdata.get("parameters", {}))
            if output_substdata:
                csvwriter.send(dataclasses.astuple(subst.analyze()))
            if output_substimg:
                cv2.imwrite(
                    output_substimg,
                    cv2.cvtColor(
                        subst.draw(**substdata.get("draw", {})), cv2.COLOR_BGR2RGB
                    ),
                )
    finally:
        if output_substdata:
            csvwriter.close()

    if output_layerdata:
        csvwriter = _CsvWriter(output_layerdata)
        next(csvwriter)
        csvwriter.send([d.name for d in dataclasses.fields(LayerType.DataType)])

    try:
        layerdata = data.get("layer", {})
        for path in tqdm.tqdm([data["targetPath"]], desc=f"{name} (layer)"):
            _, tgtimg = cv2.threshold(
                cv2.imread(os.path.expandvars(path), cv2.IMREAD_GRAYSCALE),
                0,
                255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU,
            )
            layer = LayerType(tgtimg, subst, **layerdata.get("parameters", {}))
            if output_layerdata:
                if layer.valid():
                    data = dataclasses.astuple(layer.analyze())
                else:
                    data = ()
                csvwriter.send(data)
            if output_layerimg:
                cv2.imwrite(
                    output_layerimg,
                    cv2.cvtColor(
                        layer.draw(**layerdata.get("draw", {})), cv2.COLOR_BGR2RGB
                    ),
                )
    finally:
        if output_layerdata:
            csvwriter.close()


def coatingvideo_analyzer(name, data):
    """Image analysis for coated substrate video.

    Coating layer is analyzed by constructing :class:`CoatingLayerBase` implementation.
    The analyzer defines the following fields in configuration entry:

    - **referencePath** (`str`): Path to the reference image file.
        Image is converted to binary by Otsu's thresholding.
    - **targetPath** (`str`): Path to the target video file.
        Frames are converted to binary by Otsu's thresholding.
    - **reference**, **substrate**, **layer** (`mapping`, optional):
        - **type** (`str`, optional): Registered class constructor.
          The constructors are found from entry point groups
          `'finitedepth.(references|substrates|coatinglayers)'`.
        - **parameters** (`mapping`, optional): Any additional parameter
          for class constructor.
        - **draw** (`mapping`, optional): Parameters for :meth:`draw`.
    - **output** (`mapping`, optional):
        - **(reference|substrate|layer)Data** (`str`, optional):
          Path to the output CSV file containing analysis result.
          The results are acquired from :meth:`analyze`. Invalid coating layer
          gives empty data.
        - **(reference|substrate)Image** (`str`, optional):
          Path to the output image file containing visualization result.
          The results are acquired from :meth:`draw`.
        - **layerVideo** (`str`, optional):
          Path to the output video file containing visualization result.
          The frames are acquired from :meth:`draw`. The codec is set to be the same as
          the target video file.

    The default constructors used when **type** is not specified are :class:`Reference`,
    :class:`Substrate`, and :class:`CoatingLayer`.

    The following is the example for an entry in YAML configuration file:

    .. code-block:: yaml

        foo:
            type: CoatingImage
            referencePath: foo-ref.png
            targetPath: foo-target.mp4
            reference:
                parameters:
                    templateROI: [10, 10, 1250, 200]
                    substrateROI: [100, 100, 1200, 500]
            layer:
                draw:
                    layer_color: [0, 255, 0]
            output:
                layerVideo: output/foo.mp4
                layerData: output/foo.csv
    """
    output = data.get("output", {})
    output_refdata = _makedir(output.get("referenceData", ""))
    output_refimg = _makedir(output.get("referenceImage", ""))
    output_substdata = _makedir(output.get("substrateData", ""))
    output_substimg = _makedir(output.get("substrateImage", ""))
    output_layerdata = _makedir(output.get("layerData", ""))
    output_layervid = _makedir(output.get("layerVideo", ""))

    RefType, SubstType, LayerType = _load_types(data)

    if output_refdata:
        csvwriter = _CsvWriter(output_refdata)
        next(csvwriter)
        csvwriter.send([d.name for d in dataclasses.fields(RefType.DataType)])

    try:
        refdata = data.get("reference", {})
        for path in tqdm.tqdm([data["referencePath"]], desc=f"{name} (ref)"):
            _, refimg = cv2.threshold(
                cv2.imread(os.path.expandvars(path), cv2.IMREAD_GRAYSCALE),
                0,
                255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU,
            )
            ref = RefType(refimg, **refdata.get("parameters", {}))
            if output_refdata:
                csvwriter.send(dataclasses.astuple(ref.analyze()))
            if output_refimg:
                cv2.imwrite(
                    output_refimg,
                    cv2.cvtColor(
                        ref.draw(**refdata.get("draw", {})), cv2.COLOR_BGR2RGB
                    ),
                )
    finally:
        if output_refdata:
            csvwriter.close()

    if output_substdata:
        csvwriter = _CsvWriter(output_substdata)
        next(csvwriter)
        csvwriter.send([d.name for d in dataclasses.fields(SubstType.DataType)])

    try:
        substdata = data.get("substrate", {})
        for reference in tqdm.tqdm([ref], desc=f"{name} (subst)"):
            subst = SubstType(reference, **substdata.get("parameters", {}))
            if output_substdata:
                csvwriter.send(dataclasses.astuple(subst.analyze()))
            if output_substimg:
                cv2.imwrite(
                    output_substimg,
                    cv2.cvtColor(
                        subst.draw(**substdata.get("draw", {})), cv2.COLOR_BGR2RGB
                    ),
                )
    finally:
        if output_substdata:
            csvwriter.close()

    if output_layerdata:
        csvwriter = _CsvWriter(output_layerdata)
        next(csvwriter)
        csvwriter.send(
            ["time (s)"] + [d.name for d in dataclasses.fields(LayerType.DataType)]
        )

    try:
        cap = cv2.VideoCapture(os.path.expandvars(data["targetPath"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        layerdata = data.get("layer", {})
        for i in tqdm.tqdm(
            range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc=f"{name} (layer)"
        ):
            ok, frame = cap.read()
            if not ok:
                break
            _, tgtimg = cv2.threshold(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                0,
                255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU,
            )
            layer = LayerType(tgtimg, subst, **layerdata.get("parameters", {}))
            if output_layerdata:
                if layer.valid():
                    data = dataclasses.astuple(layer.analyze())
                else:
                    data = ()
                csvwriter.send([i / fps, *data])
            if output_layervid:
                frame = layer.draw(**layerdata.get("draw", {}))
                if i == 0:
                    H, W = frame.shape[:2]
                    vidwriter = cv2.VideoWriter(
                        output_layervid,
                        int(cap.get(cv2.CAP_PROP_FOURCC)),
                        fps,
                        (W, H),
                    )
                vidwriter.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
        if output_layerdata:
            csvwriter.close()
        if output_layervid:
            vidwriter.release()


def _makedir(path: str) -> str:
    path = os.path.expandvars(path)
    dirname, _ = os.path.split(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    return path


def _load_types(
    data: dict,
) -> tuple[type[ReferenceBase], type[SubstrateBase], type[CoatingLayerBase]]:
    REFERENCES = {}
    for ep in entry_points(group="finitedepth.references"):
        REFERENCES[ep.name] = ep
    SUBSTRATES = {}
    for ep in entry_points(group="finitedepth.substrates"):
        SUBSTRATES[ep.name] = ep
    LAYERS = {}
    for ep in entry_points(group="finitedepth.coatinglayers"):
        LAYERS[ep.name] = ep

    refdata = data.get("reference", {})
    reftype = refdata.get("type", "")
    if reftype:
        RefType = REFERENCES[reftype].load()
    else:
        RefType = Reference

    substdata = data.get("substrate", {})
    substtype = substdata.get("type", "")
    if substtype:
        SubstType = SUBSTRATES[substtype].load()
    else:
        SubstType = Substrate

    layerdata = data.get("layer", {})
    layertype = layerdata.get("type", "")
    if layertype:
        LayerType = LAYERS[layertype].load()
    else:
        LayerType = CoatingLayer

    return (RefType, SubstType, LayerType)


def _CsvWriter(path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        while True:
            row = yield
            writer.writerow(row)


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
        description="Show path to sample directory.",
        help="show path to sample directory",
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

    subparsers.add_parser(
        "analyzers",
        description="List installed analyzers.",
        help="list installed analyzers",
    )
    subparsers.add_parser(
        "references",
        description="List installed reference constructors.",
        help="list installed reference constructors",
    )
    subparsers.add_parser(
        "substrates",
        description="List installed substrate constructors.",
        help="list installed substrate constructors",
    )
    subparsers.add_parser(
        "coatinglayers",
        description="List installed coating layer constructors.",
        help="list installed coating layer constructors",
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
    elif args.command in ["analyzers", "references", "substrates", "coatinglayers"]:
        header = [("NAME", "SOURCE")]
        eps = [
            (ep.name, ep.value.split(":")[0])
            for ep in entry_points(group=f"finitedepth.{args.command}")
        ]
        col0_max = max(len(m[0]) for m in header + eps)
        space = 3
        for col0, col1 in header + eps:
            line = col0.ljust(col0_max) + " " * space + col1
            print(line)
    elif args.command == "analyze":
        ok = analyze_files(*args.file, recursive=args.recursive, entries=args.entry)
        if not ok:
            sys.exit(1)
