"""
Analysis
========

:mod:`dipcoatimage.finitedepth.analysis` provides classes to save the analysis
result from experiment.

"""

import abc
from collections.abc import Coroutine
import dataclasses
import mimetypes
import os
from .coatinglayer import CoatingLayerBase
from .analysis_param import ImageWriter, CSVWriter, Parameters
from typing import Type, Optional, TypeVar, Generic, TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "AnalysisError",
    "AnalysisBase",
    "Analysis",
]


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

    .. rubric:: FPS

    *FPS* can be set to tell the time interval between each coating layer image.

    .. rubric:: Sanity check

    Validity of the parameters can be checked by :meth:`verify`.

    """

    __slots__ = (
        "_parameters",
        "_iterator",
    )

    Parameters: Type[ParametersType]

    def __init__(
        self,
        parameters: Optional[ParametersType] = None,
        *,
        fps: Optional[float] = None,
    ):
        if parameters is None:
            self._parameters = self.Parameters()
        else:
            self._parameters = dataclasses.replace(parameters)
        self._fps = fps
        self._iterator = self.__await__()

    @property
    def parameters(self) -> ParametersType:
        return self._parameters

    @property
    def fps(self) -> Optional[float]:
        return self._fps

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
    def verify(self):
        """Check to detect error and raise before analysis."""


class Analysis(AnalysisBase[Parameters]):
    """
    Basic analysis class.

    Every coating layer instance sent to the coroutine is assumed to have same
    type and same substrate instance.
    """

    Parameters = Parameters

    DataWriters = dict(csv=CSVWriter)

    def verify(self):
        for path in [
            self.parameters.ref_data,
            self.parameters.subst_data,
            self.parameters.layer_data,
        ]:
            if path:
                _, ext = os.path.splitext(path)
                if ext.lstrip(os.path.extsep).lower() not in self.DataWriters:
                    raise AnalysisError(f"{path} has unsupported extension.")
        for path in [
            self.parameters.ref_visual,
            self.parameters.subst_visual,
        ]:
            if path:
                mtype, _ = mimetypes.guess_type(path)
                file_type, _ = mtype.split("/")
                if file_type != "image":
                    raise AnalysisError(f"{path} is not image.")
        if self.parameters.layer_visual:
            mtype, _ = mimetypes.guess_type(self.parameters.layer_visual)
            file_type, _ = mtype.split("/")
            if file_type == "image":
                pass
            elif file_type == "video":
                if self.fps is None or self.fps <= 0:
                    raise AnalysisError(
                        "fps must be a positive number to write a video."
                    )
            else:
                raise AnalysisError(f"{path} is not image nor video.")
        if self.fps is not None and self.fps <= 0:
            raise AnalysisError("fps must be None or a positive number.")

    def __await__(self):
        def makedir(path):
            dirname, _ = os.path.split(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

        def get_writercls(path):
            _, ext = os.path.splitext(path)
            writercls = self.DataWriters[ext.lstrip(os.path.extsep).lower()]
            return writercls

        fps = self.fps
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
                    if self.fps:
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
