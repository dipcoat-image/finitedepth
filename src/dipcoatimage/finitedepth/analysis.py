"""
Analysis
========

:mod:`dipcoatimage.finitedepth.analysis` provides classes to save the analysis
result from experiment.

"""

import abc
from collections.abc import Coroutine
import dataclasses
import enum
import mimetypes
import os
from .coatinglayer import CoatingLayerBase
from .analysis_param import ImageWriter, CSVWriter, Parameters
from typing import List, Type, Optional, TypeVar, Generic, TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "ExperimentKind",
    "experiment_kind",
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

    DataWriters = dict(csv=CSVWriter)

    def examine(self):
        for path in [
            self.parameters.ref_data,
            self.parameters.subst_data,
            self.parameters.layer_data,
        ]:
            if path:
                _, ext = os.path.splitext(path)
                if ext.lstrip(os.path.extsep).lower() not in self.DataWriters:
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

        def get_writercls(path):
            _, ext = os.path.splitext(path)
            writercls = self.DataWriters[ext.lstrip(os.path.extsep).lower()]
            return writercls

        fps = self.parameters.layer_fps
        # prepare for analysis as much as possible
        if self.parameters.ref_data:
            makedir(self.parameters.ref_data)
            rd_cls = get_writercls(self.parameters.ref_data)

        if self.parameters.ref_visual:
            makedir(self.parameters.ref_visual)
            rv_writer = ImageWriter(self.parameters.ref_visual, fps=fps)
            next(rv_writer)

        if self.parameters.subst_data:
            makedir(self.parameters.subst_data)
            sd_cls = get_writercls(self.parameters.subst_data)

        if self.parameters.subst_visual:
            makedir(self.parameters.subst_visual)
            sv_writer = ImageWriter(self.parameters.subst_visual, fps=fps)
            next(sv_writer)

        if self.parameters.layer_data:
            makedir(self.parameters.layer_data)
            ld_cls = get_writercls(self.parameters.layer_data)

        if self.parameters.layer_visual:
            makedir(self.parameters.layer_visual)
            lv_writer = ImageWriter(self.parameters.layer_visual, fps=fps)
            next(lv_writer)

        # start analysis
        try:
            # Use first sent value
            layer = yield

            if self.parameters.ref_data:
                headers = [
                    f.name for f in dataclasses.fields(layer.substrate.reference.Data)
                ]
                rd_writer = rd_cls(self.parameters.ref_data, headers)
                next(rd_writer)
                data = list(dataclasses.astuple(layer.substrate.reference.analyze()))
                rd_writer.send(data)
                rd_writer.close()

            if self.parameters.ref_visual:
                rv_writer.send(layer.substrate.reference.draw())
                rv_writer.close()

            if self.parameters.subst_data:
                headers = [f.name for f in dataclasses.fields(layer.substrate.Data)]
                sd_writer = sd_cls(self.parameters.subst_data, headers)
                next(sd_writer)
                data = list(dataclasses.astuple(layer.substrate.analyze()))
                sd_writer.send(data)
                sd_writer.close()

            if self.parameters.subst_visual:
                sv_writer.send(layer.substrate.draw())
                sv_writer.close()

            if self.parameters.layer_data:
                headers = [f.name for f in dataclasses.fields(layer.Data)]
                if fps:
                    headers = ["time (s)"] + headers
                ld_writer = ld_cls(self.parameters.layer_data, headers)
                next(ld_writer)

            # Loop to analyze layers
            i = 0
            while True:
                if self.parameters.layer_data:
                    data = list(dataclasses.astuple(layer.analyze()))
                    if self.parameters.layer_fps:
                        data = [i / fps] + data
                    ld_writer.send(data)

                if self.parameters.layer_visual:
                    lv_writer.send(layer.draw())

                layer = yield
                i += 1

        finally:
            if self.parameters.layer_data:
                ld_writer.close()
            if self.parameters.layer_visual:
                lv_writer.close()
