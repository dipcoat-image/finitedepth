"""
Analysis
========

:mod:`dipcoatimage.finitedepth.analysis` provides classes to save the analysis
result from experiment.

"""

import abc
import csv
import cv2
import dataclasses
import enum
import mimetypes
import numpy as np
import numpy.typing as npt
import os
import tqdm  # type: ignore
from typing import List, Type, Optional, Dict, Any, Generator
from .experiment import ExperimentBase


__all__ = [
    "ExperimentKind",
    "experiment_kind",
    "DataWriter",
    "CSVWriter",
    "Analyzer",
]


class ExperimentKind(enum.Enum):
    """
    Enumeration of the experiment category by coated substrate files.

    NULL
        Invalid file

    SINGLE_IMAGE
        Single image file

    MULTI_IMAGE
        Multiple image files

    VIDEO
        Single video file

    """

    NULL = "NULL"
    SINGLE_IMAGE = "SINGLE_IMAGE"
    MULTI_IMAGE = "MULTI_IMAGE"
    VIDEO = "VIDEO"


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
        ret = ExperimentKind.NULL
    elif video_count:
        ret = ExperimentKind.VIDEO
    elif image_count > 1:
        ret = ExperimentKind.MULTI_IMAGE
    elif image_count:
        ret = ExperimentKind.SINGLE_IMAGE
    else:
        ret = ExperimentKind.NULL
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
    video_codecs: Dict[str, int] = dict(
        mp4=cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    )

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
        if (
            expt_kind == ExperimentKind.SINGLE_IMAGE
            or expt_kind == ExperimentKind.MULTI_IMAGE
        ):
            if fps is None:
                fps = 0.0
            analysis_gen = self.analysis_generator(
                data_path, image_path, video_path, fps=fps
            )
            next(analysis_gen)

            img_gen = (cv2.imread(path) for path in self.paths)
            for img in tqdm.tqdm(img_gen, total=len(self.paths), desc=name):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                analysis_gen.send(img)
            analysis_gen.send(None)

        elif expt_kind == ExperimentKind.VIDEO:
            (path,) = self.paths
            cap = cv2.VideoCapture(path)

            if fps is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
            analysis_gen = self.analysis_generator(
                data_path, image_path, video_path, fps=fps
            )
            next(analysis_gen)

            try:
                fnum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                img_gen = (cap.read()[1] for _ in range(fnum))
                for img in tqdm.tqdm(img_gen, total=fnum, desc=name):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    analysis_gen.send(img)
            finally:
                analysis_gen.send(None)
                cap.release()

        else:
            raise TypeError(f"Unsupported experiment kind: {expt_kind}")

    def analysis_generator(
        self,
        data_path: str = "",
        image_path: str = "",
        video_path: str = "",
        *,
        fps: float = 0.0,
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
                                video_path, fourcc, fps, (w, h)  # type: ignore
                            )
                        videowriter.write(visualized)
                i += 1
        finally:
            if write_data:
                datawriter.terminate()
            if write_video:
                videowriter.release()
            yield
