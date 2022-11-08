"""
Analysis
========

:mod:`dipcoatimage.finitedepth.analysis` provides classes to save the analysis
result from experiment, and classes to serialize the analysis parameters.

--------
Analysis
--------

.. autoclass:: ExperimentKind
   :members:

.. autofunction:: experiment_kind

.. autoclass:: DataWriter
   :members:

.. autoclass:: CSVWriter
   :members:

.. autoclass:: Analyzer
   :members:

------------------
Data serialization
------------------

.. autoclass:: ImportArgs
   :members:

.. autoclass:: ReferenceArgs
   :members:

.. autoclass:: SubstrateArgs
   :members:

.. autoclass:: CoatingLayerArgs
   :members:

.. autoclass:: ExperimentArgs
   :members:

.. autoclass:: AnalysisArgs
   :members:

.. autoclass:: ExperimentData
   :members:

"""

import abc
import csv
import cv2  # type: ignore
import dataclasses
import enum
import mimetypes
import numpy as np
import numpy.typing as npt
import os
import tqdm  # type: ignore
from typing import List, Type, Optional, Dict, Any, Tuple, Generator
from .reference import SubstrateReferenceBase
from .substrate import SubstrateBase
from .coatinglayer import CoatingLayerBase
from .experiment import ExperimentBase
from .util import import_variable, data_converter, OptionalROI, DataclassProtocol


__all__ = [
    "ExperimentKind",
    "experiment_kind",
    "DataWriter",
    "CSVWriter",
    "Analyzer",
    "ImportArgs",
    "ReferenceArgs",
    "SubstrateArgs",
    "CoatingLayerArgs",
    "ExperimentArgs",
    "AnalysisArgs",
    "ExperimentData",
]


class ExperimentKind(enum.Enum):
    """
    Enumeration of the experiment category by coated substrate files.

    NullExperiment
        Invalid file

    SingleImageExperiment
        Single image file

    MultiImageExperiment
        Multiple image files

    VideoExperiment
        Single video file

    """

    NullExperiment = "NullExperiment"
    SingleImageExperiment = "SingleImageExperiment"
    MultiImageExperiment = "MultiImageExperiment"
    VideoExperiment = "VideoExperiment"


def experiment_kind(paths: List[str]) -> ExperimentKind:
    """Get :class:`ExperimentKind` for given paths using MIME type."""
    INVALID = False
    video_count, image_count = 0, 0
    for p in paths:
        mtype, _ = mimetypes.guess_type(p)
        if mtype is None:
            INVALID = True
            break
        file_type, _ = mtype.split("/")
        if file_type == "video":
            video_count += 1
        elif file_type == "image":
            image_count += 1
        else:
            # unrecognized type
            INVALID = True
            break

        if video_count > 1:
            # video must be unique
            INVALID = True
            break
        elif video_count and image_count:
            # video cannot be combined with image
            INVALID = True
            break

    if INVALID:
        ret = ExperimentKind.NullExperiment
    elif video_count:
        ret = ExperimentKind.VideoExperiment
    elif image_count > 1:
        ret = ExperimentKind.MultiImageExperiment
    elif image_count:
        ret = ExperimentKind.SingleImageExperiment
    else:
        ret = ExperimentKind.NullExperiment
    return ret


class DataWriter(abc.ABC):
    """
    Abstract base class to write data file.

    Parameters
    ==========

    path
        Path to the data file.

    headers
        Headers for the data file.

    """

    def __init__(self, path: str, headers: List[str]):
        self.path = path
        self.headers = headers

    @abc.abstractmethod
    def prepare(self):
        """Prepare to write the data, e.g. create file or write headers."""
        pass

    @abc.abstractmethod
    def write_data(self, data: List[Any]):
        """Write *data* to the file."""
        pass

    @abc.abstractmethod
    def terminate(self):
        """Terminate the writing and close the file."""
        pass


class CSVWriter(DataWriter):
    """
    Writer for CSV file.

    Examples
    ========

    >>> from dipcoatimage.finitedepth.analysis import CSVWriter
    >>> writer = CSVWriter("data.csv", ["foo", "bar"])
    >>> def write_csv():
    ...     writer.prepare()
    ...     writer.write_data([10, 20])
    ...     writer.terminate()
    >>> write_csv() #doctest: +SKIP

    """

    def prepare(self):
        self.datafile = open(self.path, "w", newline="")
        self.writer = csv.writer(self.datafile)
        self.writer.writerow(self.headers)

    def write_data(self, data: List[Any]):
        self.writer.writerow(data)

    def terminate(self):
        self.datafile.close()


class Analyzer:
    """
    Class to save the analysis result.

    Parameters
    ==========

    paths
        Paths to the coated substrate image/video files.

    experiment
        Experiment instance to analyze the coated substrate images.

    Attributes
    ==========

    data_writers
        Dictionary of file extensions and their data writers.

    video_codecs
        Dictionary of video extensions and their FourCC values.

    """

    data_writers: Dict[str, Type[DataWriter]] = dict(csv=CSVWriter)
    video_codecs: Dict[str, int] = dict(mp4=cv2.VideoWriter_fourcc(*"mp4v"))

    def __init__(self, paths: List[str], experiment: ExperimentBase):
        self.paths = paths
        self.experiment = experiment

    def analyze(
        self,
        data_path: str = "",
        image_path: str = "",
        video_path: str = "",
        *,
        fps: Optional[float] = None,
        name: str = "",
    ):
        """
        Analyze :attr:`paths` with :attr:`experiment` and save the result.

        .. rubric:: Analysis data

        If *data_path* is passed, analysis result is saved as data file.
        Data writer is searched from :attr:`data_writers` using file extension.

        If FPS value is nonzero, time value for each coating layer is
        automatically prepended to the row.

        .. rubric:: Visualization image

        If *image_path* is passed, visualization results are saved as image
        files. Using formattable string (e.g. ``img_%02d.jpg``) to save multiple
        images.

        .. rubric:: Visualization video

        If *video_path* is passed, visualization results are saved as video file.
        FourCC codec value is searched from :attr:`video_codecs` using file
        extension.

        .. rubric:: FPS

        If *fps* is explicitly passed, it is used for data timestamps and
        visualization video FPS.

        .. rubric:: Progress bar

        Progress bar is printed. Pass *name* for the progress bar name.

        """
        expt_kind = experiment_kind(self.paths)

        # make image generator
        if (
            expt_kind == ExperimentKind.SingleImageExperiment
            or expt_kind == ExperimentKind.MultiImageExperiment
        ):
            img_gen = (cv2.imread(path) for path in self.paths)
            if fps is None:
                fps = 0

            total = len(self.paths)

        elif expt_kind == ExperimentKind.VideoExperiment:
            (path,) = self.paths
            cap = cv2.VideoCapture(path)
            fnum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            img_gen = (cap.read()[1] for _ in range(fnum))
            fps = cap.get(cv2.CAP_PROP_FPS)

            total = fnum

        else:
            raise TypeError(f"Unsupported experiment kind: {expt_kind}")

        analysis_gen = self.analysis_generator(
            data_path, image_path, video_path, fps=fps
        )
        next(analysis_gen)

        for img in tqdm.tqdm(img_gen, total=total, desc=name):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            analysis_gen.send(img)
        analysis_gen.send(None)

    def analysis_generator(
        self,
        data_path: str = "",
        image_path: str = "",
        video_path: str = "",
        *,
        fps: Optional[float] = None,
    ) -> Generator[None, Optional[npt.NDArray[np.uint8]], None]:
        """
        Send the coating layer image to this generator to analyze it.
        Sending ``None`` terminates the analysis.
        """
        self.experiment.substrate.reference.verify()
        self.experiment.substrate.verify()
        self.experiment.verify()

        # prepare for data writing
        if data_path:
            write_data = True

            dirname, _ = os.path.split(data_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            _, data_ext = os.path.splitext(data_path)
            data_ext = data_ext.lstrip(os.path.extsep).lower()
            writercls = self.data_writers.get(data_ext, None)
            if writercls is None:
                raise TypeError(f"Unsupported extension: {data_ext}")
            headers = [
                f.name for f in dataclasses.fields(self.experiment.layer_type.Data)
            ]
            if fps:
                headers = ["time (s)"] + headers
            datawriter = writercls(data_path, headers)
            datawriter.prepare()
        else:
            write_data = False

        # prepare for image writing
        if image_path:
            write_image = True

            dirname, _ = os.path.split(image_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            try:
                image_path % 0
                image_path_formattable = True
            except (TypeError, ValueError):
                image_path_formattable = False
        else:
            write_image = False

        # prepare for video writing
        if video_path:
            write_video = True

            _, video_ext = os.path.splitext(video_path)
            video_ext = video_ext.lstrip(os.path.extsep).lower()
            fourcc = self.video_codecs.get(video_ext, None)
            if fourcc is None:
                raise TypeError(f"Unsupported extension: {video_ext}")

            dirname, _ = os.path.split(video_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
        else:
            write_video = False

        layer_gen = self.experiment.layer_generator()
        i = 0
        try:
            while True:
                img = yield  # type: ignore
                if img is None:
                    break
                next(layer_gen)
                layer = layer_gen.send(img)
                valid = layer.valid()

                if write_data:
                    if valid:
                        data = list(dataclasses.astuple(layer.analyze()))
                        if fps:
                            data = [i / fps] + data
                    else:
                        data = []
                    datawriter.write_data(data)

                if write_image or write_video:
                    if valid:
                        visualized = layer.draw()
                    else:
                        visualized = img
                    visualized = cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR)

                    if write_image:
                        if image_path_formattable:
                            imgpath = image_path % i
                        else:
                            imgpath = image_path
                        cv2.imwrite(imgpath, visualized)

                    if write_video:
                        if i == 0:
                            h, w = img.shape[:2]
                            videowriter = cv2.VideoWriter(
                                video_path, fourcc, fps, (w, h)
                            )
                        videowriter.write(visualized)
                i += 1
        finally:
            if write_data:
                datawriter.terminate()
            if write_video:
                videowriter.release()
            yield


@dataclasses.dataclass
class ImportArgs:
    """Arguments to import the variable from module."""

    name: str = ""
    module: str = "dipcoatimage.finitedepth"


@dataclasses.dataclass
class ReferenceArgs:
    """
    Data for the concrete instance of :class:`SubstrateReferenceBase`.

    Parameters
    ==========

    type
        Information to import reference class.
        Class name defaults to ``SubstrateReference``.

    templateROI, substrateROI, parameters, draw_options
        Data for arguments of reference class.

    Examples
    ========

    .. plot::
       :include-source:

       >>> import cv2
       >>> from dipcoatimage.finitedepth import get_samples_path
       >>> from dipcoatimage.finitedepth.analysis import ReferenceArgs
       >>> refargs = ReferenceArgs(templateROI=(200, 100, 1200, 500),
       ...                         substrateROI=(300, 50, 1100, 600))
       >>> ref = refargs.as_reference(cv2.imread(get_samples_path("ref1.png")))
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    """

    type: ImportArgs = dataclasses.field(default_factory=ImportArgs)
    templateROI: OptionalROI = (0, 0, None, None)
    substrateROI: OptionalROI = (0, 0, None, None)
    parameters: dict = dataclasses.field(default_factory=dict)
    draw_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.type.name:
            self.type.name = "SubstrateReference"

    def as_structured_args(
        self,
    ) -> Tuple[Type[SubstrateReferenceBase], DataclassProtocol, DataclassProtocol]:
        """
        Structure the primitive data.

        Returns
        =======

        (refcls, params, drawopts)
            Type and arguments for reference class, structured from the data.

        """
        name = self.type.name
        module = self.type.module
        refcls = import_variable(name, module)
        if not (
            isinstance(refcls, type) and issubclass(refcls, SubstrateReferenceBase)
        ):
            raise TypeError(f"{refcls} is not substrate reference class.")

        params = data_converter.structure(
            self.parameters, refcls.Parameters  # type: ignore
        )
        drawopts = data_converter.structure(
            self.draw_options, refcls.DrawOptions  # type: ignore
        )
        return (refcls, params, drawopts)

    def as_reference(self, img: npt.NDArray[np.uint8]) -> SubstrateReferenceBase:
        """Construct the substrate reference instance."""
        refcls, params, drawopts = self.as_structured_args()

        ref = refcls(  # type: ignore
            img,
            self.templateROI,
            self.substrateROI,
            parameters=params,
            draw_options=drawopts,
        )
        return ref


@dataclasses.dataclass
class SubstrateArgs:
    """
    Data for the concrete instance of :class:`SubstrateBase`.

    Parameters
    ==========

    type
        Information to import substrate class.
        Class name defaults to ``Substrate``.

    parameters, draw_options
        Data for arguments of substrate class.

    Examples
    ========

    Construct substrate reference instance first.

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import get_samples_path
       >>> from dipcoatimage.finitedepth.analysis import ReferenceArgs
       >>> refargs = ReferenceArgs(templateROI=(200, 100, 1200, 500),
       ...                         substrateROI=(300, 50, 1100, 600))
       >>> ref = refargs.as_reference(cv2.imread(get_samples_path("ref1.png")))

    Construct substrate instance from the data.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import data_converter
       >>> from dipcoatimage.finitedepth.analysis import SubstrateArgs
       >>> params = {"Canny": {"threshold1": 50, "threshold2": 150},
       ...           "HoughLines": {"rho": 1, "theta": 0.01, "threshold": 100}}
       >>> arg = dict(type={"name": "RectSubstrate"}, parameters=params)
       >>> substargs = data_converter.structure(arg, SubstrateArgs)
       >>> subst = substargs.as_substrate(ref)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(subst.draw()) #doctest: +SKIP

    """

    type: ImportArgs = dataclasses.field(default_factory=ImportArgs)
    parameters: dict = dataclasses.field(default_factory=dict)
    draw_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.type.name:
            self.type.name = "Substrate"

    def as_structured_args(
        self,
    ) -> Tuple[Type[SubstrateBase], DataclassProtocol, DataclassProtocol]:
        """
        Structure the primitive data.

        Returns
        =======

        (substcls, params, drawopts)
            Type and arguments for substrate class, structured from the data.

        """
        name = self.type.name
        module = self.type.module
        substcls = import_variable(name, module)
        if not (isinstance(substcls, type) and issubclass(substcls, SubstrateBase)):
            raise TypeError(f"{substcls} is not substrate class.")

        params = data_converter.structure(
            self.parameters, substcls.Parameters  # type: ignore
        )
        drawopts = data_converter.structure(
            self.draw_options, substcls.DrawOptions  # type: ignore
        )
        return (substcls, params, drawopts)

    def as_substrate(self, ref: SubstrateReferenceBase) -> SubstrateBase:
        """Construct the substrate instance."""
        substcls, params, drawopts = self.as_structured_args()
        subst = substcls(  # type: ignore
            ref,
            parameters=params,
            draw_options=drawopts,
        )
        return subst


@dataclasses.dataclass
class CoatingLayerArgs:
    """
    Data for the concrete instance of :class:`CoatingLayerBase`.

    Parameters
    ==========

    type
        Information to import substrate class.
        Class name defaults to ``LayerArea``.

    parameters, draw_options, deco_options
        Data for arguments of coating layer class.

    Examples
    ========

    Construct substrate reference instance and substrate instance first.

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import get_samples_path, data_converter
       >>> from dipcoatimage.finitedepth.analysis import (ReferenceArgs,
       ...     SubstrateArgs)
       >>> refargs = ReferenceArgs(templateROI=(200, 100, 1200, 500),
       ...                         substrateROI=(300, 50, 1100, 600))
       >>> ref = refargs.as_reference(cv2.imread(get_samples_path("ref1.png")))
       >>> params = {"Canny": {"threshold1": 50, "threshold2": 150},
       ...           "HoughLines": {"rho": 1, "theta": 0.01, "threshold": 100}}
       >>> arg = dict(type={"name": "RectSubstrate"}, parameters=params)
       >>> substargs = data_converter.structure(arg, SubstrateArgs)
       >>> subst = substargs.as_substrate(ref)

    Construct coating layer instance from the data.

    .. plot::
       :include-source:
       :context: close-figs

       >>> import cv2
       >>> from dipcoatimage.finitedepth.analysis import CoatingLayerArgs
       >>> arg = dict(type={"name": "RectLayerArea"})
       >>> layerargs = data_converter.structure(arg, CoatingLayerArgs)
       >>> img_path = get_samples_path("coat1.png")
       >>> layer = layerargs.as_coatinglayer(cv2.imread(img_path), subst)
       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> plt.imshow(layer.draw()) #doctest: +SKIP

    """

    type: ImportArgs = dataclasses.field(default_factory=ImportArgs)
    parameters: dict = dataclasses.field(default_factory=dict)
    draw_options: dict = dataclasses.field(default_factory=dict)
    deco_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.type.name:
            self.type.name = "LayerArea"

    def as_structured_args(
        self,
    ) -> Tuple[
        Type[CoatingLayerBase], DataclassProtocol, DataclassProtocol, DataclassProtocol
    ]:
        """
        Structure the primitive data.

        Returns
        =======

        (layercls, params, drawopts, decoopts)
            Type and arguments for coating layer class, structured from the data.

        """
        name = self.type.name
        module = self.type.module
        layercls = import_variable(name, module)
        if not (isinstance(layercls, type) and issubclass(layercls, CoatingLayerBase)):
            raise TypeError(f"{layercls} is not coating layer class.")

        params = data_converter.structure(
            self.parameters, layercls.Parameters  # type: ignore
        )
        drawopts = data_converter.structure(
            self.draw_options, layercls.DrawOptions  # type: ignore
        )
        decoopts = data_converter.structure(
            self.deco_options, layercls.DecoOptions  # type: ignore
        )
        return (layercls, params, drawopts, decoopts)

    def as_coatinglayer(
        self, img: npt.NDArray[np.uint8], subst: SubstrateBase
    ) -> CoatingLayerBase:
        """Construct the coating layer instance."""
        layercls, params, drawopts, decoopts = self.as_structured_args()
        layer = layercls(  # type: ignore
            img, subst, parameters=params, draw_options=drawopts, deco_options=decoopts
        )
        return layer


@dataclasses.dataclass
class ExperimentArgs:
    """
    Data for the concrete instance of :class:`ExperimentBase`.

    Parameters
    ==========

    type
        Information to import substrate class.
        Class name defaults to ``Experiment``.

    parameters
        Data for arguments of experiment class.
    """

    type: ImportArgs = dataclasses.field(default_factory=ImportArgs)
    parameters: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.type.name:
            self.type.name = "Experiment"

    def as_structured_args(self) -> Tuple[Type[ExperimentBase], DataclassProtocol]:
        """
        Structure the primitive data.

        Returns
        =======

        (exptcls, params)
            Type and arguments for substrate class, structured from the data.

        """
        name = self.type.name
        module = self.type.module
        exptcls = import_variable(name, module)
        if not (isinstance(exptcls, type) and issubclass(exptcls, ExperimentBase)):
            raise TypeError(f"{exptcls} is not coating layer class.")

        params = data_converter.structure(
            self.parameters, exptcls.Parameters  # type: ignore
        )
        return (exptcls, params)

    def as_experiment(
        self,
        subst: SubstrateBase,
        layer_type: Type[CoatingLayerBase],
        layer_parameters: Optional[DataclassProtocol] = None,
        layer_drawoptions: Optional[DataclassProtocol] = None,
        layer_decooptions: Optional[DataclassProtocol] = None,
    ) -> ExperimentBase:
        """Construct the experiment instance."""
        exptcls, params = self.as_structured_args()
        expt = exptcls(  # type: ignore
            subst,
            layer_type,
            layer_parameters,
            layer_drawoptions,
            layer_decooptions,
            parameters=params,
        )
        return expt


@dataclasses.dataclass
class AnalysisArgs:
    """
    Arguments to save the analyis result.
    """

    data_path: str = ""
    image_path: str = ""
    video_path: str = ""
    fps: Optional[float] = None


@dataclasses.dataclass
class ExperimentData:
    """
    Class which wraps every information to construct and analyze the experiment.
    """

    ref_path: str = ""
    coat_paths: List[str] = dataclasses.field(default_factory=list)
    reference: ReferenceArgs = dataclasses.field(default_factory=ReferenceArgs)
    substrate: SubstrateArgs = dataclasses.field(default_factory=SubstrateArgs)
    coatinglayer: CoatingLayerArgs = dataclasses.field(default_factory=CoatingLayerArgs)
    experiment: ExperimentArgs = dataclasses.field(default_factory=ExperimentArgs)
    analysis: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)

    def analyze(self, name: str = ""):
        """Analyze and save the data."""
        refimg = cv2.cvtColor(cv2.imread(self.ref_path), cv2.COLOR_BGR2RGB)
        ref = self.reference.as_reference(refimg)
        subst = self.substrate.as_substrate(ref)

        layercls, params, drawopts, decoopts = self.coatinglayer.as_structured_args()

        expt = self.experiment.as_experiment(
            subst, layercls, params, drawopts, decoopts
        )
        analyzer = Analyzer(self.coat_paths, expt)

        analyzer.analyze(
            self.analysis.data_path,
            self.analysis.image_path,
            self.analysis.video_path,
            fps=self.analysis.fps,
            name=name,
        )
