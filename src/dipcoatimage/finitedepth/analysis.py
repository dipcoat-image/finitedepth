"""Analysis result writer."""

import abc
import csv
import dataclasses
import mimetypes
import os
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Generic, Optional, Type, TypeVar

import cv2

from .coatinglayer import CoatingLayerBase

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "AnalysisError",
    "AnalysisBase",
    "ImageWriter",
    "CSVWriter",
    "Analysis",
]


class AnalysisError(Exception):
    """Base class for error from `AnalysisBase`."""

    pass


ParametersType = TypeVar("ParametersType", bound="DataclassInstance")


class AnalysisBase(Coroutine, Generic[ParametersType]):
    """Class to save the analysis result.

    Subclass must implement :meth:`__await__` which saves the analysis result.
    See `Analyzer` for example.

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

    Parameters: Type[ParametersType]

    def __init__(
        self,
        parameters: Optional[ParametersType] = None,
        *,
        fps: Optional[float] = None,
    ):
        """Initialize the instance."""
        if parameters is None:
            self._parameters = self.Parameters()
        else:
            if not isinstance(parameters, self.Parameters):
                raise TypeError(f"{parameters} is not instance of {self.Parameters}")
            self._parameters = dataclasses.replace(parameters)
        self._fps = fps
        self._iterator = self.__await__()

    @property
    def parameters(self) -> ParametersType:
        """Analysis parameters."""
        return self._parameters

    @property
    def fps(self) -> Optional[float]:
        """Time interval between each coating layer image."""
        return self._fps

    def send(self, value: Optional[CoatingLayerBase]):
        """Analyze the coating layer."""
        if value is None:
            next(self._iterator)
        else:
            self._iterator.send(value)  # type: ignore[arg-type]

    def throw(self, type, value, traceback):
        """Throw exception into the analysis event loop."""
        self._iterator.throw(type, value, traceback)

    def close(self):
        """Terminate the analysis."""
        self._iterator.close()

    @abc.abstractmethod
    def verify(self):
        """Check to detect error and raise before analysis."""


def ImageWriter(path: str, fourcc: int, fps: float):
    """Write images to image files or a video file."""
    try:
        path % 0
        formattable = True
    except TypeError:
        formattable = False

    mtype, _ = mimetypes.guess_type(path)
    if mtype is None:
        raise TypeError(f"Invalid path: {path}.")
    ftype, _ = mtype.split("/")

    if ftype == "image" and not formattable:
        img = yield
        try:
            while True:
                img = yield
        finally:
            cv2.imwrite(path, img)
    else:
        if ftype == "video":
            pass
        elif ftype == "image":
            fourcc = 0
            fps = 0.0
        else:
            raise TypeError(f"Unsupported mimetype: {mtype}.")
        img = yield
        h, w = img.shape[:2]
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        try:
            while True:
                writer.write(img)
                img = yield
        finally:
            writer.release()


def CSVWriter(path: str):
    """Write data to a csv file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        while True:
            data = yield
            writer.writerow(data)


@dataclasses.dataclass(frozen=True)
class Parameters:
    """Parameters for `Analysis`.

    Attributes
    ----------
    ref_data, ref_visual : str
        Paths for data file and visualized file of reference image.
    subst_data, subst_visual : str
        Paths for data file and visualized file of substrate image.
    layer_data, layer_visual : str
        Paths for data file and visualized file of coating layer image(s).
        Pass formattable string (e.g. `img_%02d.jpg`) to save multiple images.
    layer_fourcc : str
        FourCC of codec to record analyzed layer video.
    """

    ref_data: str = ""
    ref_visual: str = ""
    subst_data: str = ""
    subst_visual: str = ""
    layer_data: str = ""
    layer_visual: str = ""
    layer_fourcc: str = ""


class Analysis(AnalysisBase[Parameters]):
    """Basic analysis class.

    Every coating layer instance sent to the coroutine is assumed to have same type and
    same substrate instance.
    """

    Parameters = Parameters

    DataWriters = dict(csv=CSVWriter)

    def verify(self):
        """Check file paths and fps in :meth:`parameters`."""
        for path in [
            os.path.expandvars(self.parameters.ref_data),
            os.path.expandvars(self.parameters.subst_data),
            os.path.expandvars(self.parameters.layer_data),
        ]:
            if path:
                _, ext = os.path.splitext(path)
                if ext.lstrip(os.path.extsep).lower() not in self.DataWriters:
                    raise AnalysisError(f"{path} has unsupported extension.")
        for path in [
            os.path.expandvars(self.parameters.ref_visual),
            os.path.expandvars(self.parameters.subst_visual),
        ]:
            if path:
                mtype, _ = mimetypes.guess_type(path)
                file_type, _ = mtype.split("/")
                if file_type != "image":
                    raise AnalysisError(f"{path} is not image.")
        if os.path.expandvars(self.parameters.layer_visual):
            mtype, _ = mimetypes.guess_type(
                os.path.expandvars(self.parameters.layer_visual)
            )
            file_type, _ = mtype.split("/")
            if file_type == "image":
                pass
            elif file_type == "video":
                if self.fps is None or self.fps < 0:
                    raise AnalysisError(
                        "fps must be a nonnegative number to write a video."
                    )
            else:
                raise AnalysisError(f"{path} is not image nor video.")
        if self.fps is not None and self.fps < 0:
            raise AnalysisError("fps must be None or a nonnegative number.")

    def __await__(self):
        """Analyze reference and substrate, then each sent coating layer."""

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
        ref_data = os.path.expandvars(self.parameters.ref_data)
        if ref_data:
            makedir(ref_data)
            rd_cls = get_writercls(ref_data)
            rd_writer = rd_cls(ref_data)
            next(rd_writer)

        ref_visual = os.path.expandvars(self.parameters.ref_visual)
        if ref_visual:
            makedir(ref_visual)
            rv_writer = ImageWriter(ref_visual, 0, 0.0)
            next(rv_writer)

        subst_data = os.path.expandvars(self.parameters.subst_data)
        if subst_data:
            makedir(subst_data)
            sd_cls = get_writercls(subst_data)
            sd_writer = sd_cls(subst_data)
            next(sd_writer)

        subst_visual = os.path.expandvars(self.parameters.subst_visual)
        if subst_visual:
            makedir(subst_visual)
            sv_writer = ImageWriter(subst_visual, 0, 0.0)
            next(sv_writer)

        layer_data = os.path.expandvars(self.parameters.layer_data)
        if layer_data:
            makedir(layer_data)
            ld_cls = get_writercls(layer_data)
            ld_writer = ld_cls(layer_data)
            next(ld_writer)

        layer_visual = os.path.expandvars(self.parameters.layer_visual)
        if layer_visual:
            makedir(layer_visual)
            if len(self.parameters.layer_fourcc) == 4:
                fourcc = cv2.VideoWriter_fourcc(*self.parameters.layer_fourcc)
            else:
                fourcc = 0
            lv_writer = ImageWriter(layer_visual, fourcc, fps)
            next(lv_writer)

        # start analysis
        try:
            # Use first sent value
            layer = yield
            layer.substrate.reference.verify()
            layer.substrate.verify()

            if ref_data:
                headers = [
                    f.name for f in dataclasses.fields(layer.substrate.reference.Data)
                ]
                rd_writer.send(headers)
                data = list(dataclasses.astuple(layer.substrate.reference.analyze()))
                rd_writer.send(data)

            if ref_visual:
                img = cv2.cvtColor(layer.substrate.reference.draw(), cv2.COLOR_RGB2BGR)
                rv_writer.send(img)

            if subst_data:
                headers = [f.name for f in dataclasses.fields(layer.substrate.Data)]
                sd_writer.send(headers)
                data = list(dataclasses.astuple(layer.substrate.analyze()))
                sd_writer.send(data)

            if subst_visual:
                img = cv2.cvtColor(layer.substrate.draw(), cv2.COLOR_RGB2BGR)
                sv_writer.send(img)

            if layer_data:
                headers = [f.name for f in dataclasses.fields(layer.Data)]
                if fps:
                    headers = ["time (s)"] + headers
                ld_writer.send(headers)

            # Loop to analyze layers
            i = 0
            while True:
                try:
                    layer.verify()
                    valid = True
                except Exception:
                    valid = False
                if layer_data:
                    if valid:
                        data = list(dataclasses.astuple(layer.analyze()))
                    else:
                        data = []
                    if self.fps:
                        data = [i / fps] + data
                    ld_writer.send(data)

                if layer_visual:
                    img = cv2.cvtColor(layer.draw(), cv2.COLOR_RGB2BGR)
                    lv_writer.send(img)

                layer = yield
                i += 1

        finally:
            if ref_data:
                rd_writer.close()
            if ref_visual:
                rv_writer.close()
            if subst_data:
                sd_writer.close()
            if subst_visual:
                sv_writer.close()
            if layer_data:
                ld_writer.close()
            if layer_visual:
                lv_writer.close()
