"""
Analysis
========

:mod:`dipcoatimage.finitedepth.analysis` provides classes to save the analysis
result from experiment.

"""

import abc
from collections.abc import Coroutine
import csv
import cv2
import dataclasses
import enum
import imageio.v2 as iio  # TODO: use PyAV
import mimetypes
import os
from .coatinglayer import CoatingLayerBase
from .analysis_param import Parameters
from typing import List, Type, Optional, Dict, Any, TypeVar, Generic, TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "ExperimentKind",
    "experiment_kind",
    "DataWriter",
    "CSVWriter",
    "AnalysisError",
    "AnalysisBase",
    "Analysis",
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


class AnalysisError(Exception):
    """Base class for error from :class:`AnalysisBase`."""

    pass


ParametersType = TypeVar("ParametersType", bound="DataclassInstance")


class AnalysisBase(Coroutine, Generic[ParametersType]):
    """
    Class to save the analysis result.

    Subclass must implement :meth:`__await__` which saves the analysis result.
    See :class:`Analyzer` for example.

    .. rubric:: Constructor

    Constructor signature must not be modified because high-level API use factory
    to generate experiment instances. Additional parameters can be introduced
    by definig class attribute :attr:`Parameters``.

    .. rubric:: Parameters

    Concrete class must have :attr:`Parameters` which returns dataclass type.
    Its instance is passed to the constructor at instance initialization, and can
    be accessed by :attr:`parameters`.

    .. rubric:: Sanity check

    Validity of the parameters can be checked by :meth:`verify` or :meth:`valid`.
    Their result can be implemented by defining :meth:`examine`.

    """

    __slots__ = (
        "_parameters",
        "_iterator",
    )

    Parameters: Type[ParametersType]

    def __init__(self, *, parameters: Optional[ParametersType] = None):
        if parameters is None:
            self._parameters = self.Parameters()
        else:
            self._parameters = dataclasses.replace(parameters)
        self._iterator = self.__await__()

    @property
    def parameters(self) -> ParametersType:
        return self._parameters

    def send(self, value: Optional[CoatingLayerBase]):
        if value is None:
            next(self._iterator)
        else:
            self._iterator.send(value)  # type: ignore[arg-type]

    def throw(self, type, value, traceback):
        self._iterator.throw(type, value, traceback)

    def close(self):
        self._iterator.close()

    @abc.abstractmethod
    def examine(self) -> Optional[AnalysisError]:
        """
        Check the sanity of parameters.

        If the instance is invalid, return error instance.
        Else, return :obj:`None`.
        """

    def verify(self):
        """
        Verify if all parameters are suitably set by raising error on failure.

        To implement sanity check for concrete class, define :meth:`examine`.
        """
        err = self.examine()
        if err is not None:
            raise err

    def valid(self) -> bool:
        """
        Verify if all parameters are suitably set by returning boolean value.

        To implement sanity check for concrete class, define :meth:`examine`.
        """
        return self.examine() is None


class Analysis(AnalysisBase[Parameters]):
    """
    Basic analysis class.

    Every coating layer instance sent to the coroutine is assumed to have same
    type and same substrate instance.
    """

    Parameters = Parameters

    data_writers: Dict[str, Type[DataWriter]] = dict(csv=CSVWriter)

    def examine(self):
        for path in [
            self.parameters.ref_data,
            self.parameters.subst_data,
            self.parameters.layer_data,
        ]:
            if path:
                _, ext = os.path.splitext(path)
                if ext.lstrip(os.path.extsep).lower() not in self.data_writers:
                    return AnalysisError(f"{path} has unsupported extension.")
        for path in [
            self.parameters.ref_visual,
            self.parameters.subst_visual,
        ]:
            if path:
                mtype, _ = mimetypes.guess_type(path)
                file_type, _ = mtype.split("/")
                if file_type != "image":
                    return AnalysisError(f"{path} is not image.")
        if self.parameters.layer_visual:
            mtype, _ = mimetypes.guess_type(self.parameters.layer_visual)
            file_type, _ = mtype.split("/")
            if file_type == "image":
                pass
            elif file_type == "video":
                if self.parameters.layer_fps is None or self.parameters.layer_fps <= 0:
                    return AnalysisError(
                        "layer_fps must be a positive number to write a video."
                    )
            else:
                return AnalysisError(f"{path} is not image nor video.")
        if self.parameters.layer_fps is not None and self.parameters.layer_fps <= 0:
            return AnalysisError("layer_fps must be None or a positive number.")
        return None

    def __await__(self):
        def makedir(path):
            dirname, _ = os.path.split(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

        def make_writercls(path):
            _, ext = os.path.splitext(path)
            writercls = self.data_writers[ext.lstrip(os.path.extsep).lower()]
            return writercls

        # prepare for analysis
        if self.parameters.ref_data:
            makedir(self.parameters.ref_data)
            rd_cls = make_writercls(self.parameters.ref_data)

        if self.parameters.ref_visual:
            makedir(self.parameters.ref_visual)

        if self.parameters.subst_data:
            makedir(self.parameters.subst_data)
            sd_cls = make_writercls(self.parameters.subst_data)

        if self.parameters.subst_visual:
            makedir(self.parameters.subst_visual)

        if self.parameters.layer_data:
            makedir(self.parameters.layer_data)
            ld_cls = make_writercls(self.parameters.layer_data)

        if self.parameters.layer_visual:
            makedir(self.parameters.layer_visual)
            mtype, _ = mimetypes.guess_type(self.parameters.layer_visual)
            lv_type, _ = mtype.split("/")
            if lv_type == "video":
                fps = self.parameters.layer_fps
                lv_writer = iio.get_writer(self.parameters.layer_visual, fps=fps)
            elif lv_type == "image":
                # TODO: implement wrapper for both video writer and image writer
                try:
                    self.parameters.layer_visual % 0
                    lv_formattable = True
                except (TypeError, ValueError):
                    lv_formattable = False

        # start analysis
        try:
            # Use first sent value
            layer = yield

            if self.parameters.ref_data:
                headers = []
                rd_writer = rd_cls(self.parameters.ref_data, headers)
                rd_writer.prepare()
                ...  # TODO: implement data for reference
                rd_writer.terminate()

            if self.parameters.ref_visual:
                img = layer.substrate.reference.draw()
                iio.imwrite(
                    self.parameters.ref_visual,
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                )

            if self.parameters.subst_data:
                headers = []
                sd_writer = sd_cls(self.parameters.subst_data, headers)
                sd_writer.prepare()
                ...  # TODO: implement data for substrate
                sd_writer.terminate()

            if self.parameters.subst_visual:
                img = layer.substrate.draw()
                iio.imwrite(
                    self.parameters.subst_visual,
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                )

            if self.parameters.layer_data:
                headers = [f.name for f in dataclasses.fields(layer.Data)]
                if self.parameters.layer_fps:
                    headers = ["time (s)"] + headers
                ld_writer = ld_cls(self.parameters.layer_data, headers)
                ld_writer.prepare()

            # Loop to analyze layers
            i = 0
            while True:
                if self.parameters.layer_data:
                    data = list(dataclasses.astuple(layer.analyze()))
                    if self.parameters.layer_fps:
                        data = [i / self.parameters.layer_fps] + data
                    ld_writer.write_data(data)

                if self.parameters.layer_visual:
                    img = cv2.cvtColor(layer.draw(), cv2.COLOR_BGR2RGB)
                    if lv_type == "video":
                        lv_writer.append_data(img)
                    elif lv_type == "image":
                        if lv_formattable:
                            path = self.parameters.layer_visual % i
                        else:
                            path = self.parameters.layer_visual
                        iio.imwrite(path, img)

                layer = yield
                i += 1

        finally:
            if self.parameters.layer_data:
                ld_writer.terminate()
            if self.parameters.layer_visual:
                if lv_type == "video":
                    lv_writer.close()
