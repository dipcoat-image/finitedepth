"""Analysis result writer.

This module defines abstract class :class:`AnalysisBase` and its
implementation, :class:`Analysis`.
"""

import abc
import csv
import dataclasses
import mimetypes
import os
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Generator, Generic, Optional, Type, TypeVar

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image

from .coatinglayer import CoatingLayerBase

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "AnalysisBase",
    "ImageWriter",
    "CSVWriter",
    "Analysis",
]


ParamTypeVar = TypeVar("ParamTypeVar", bound="DataclassInstance")
"""Type variable for :attr:`AnalysisBase.ParamType`."""


class AnalysisBase(Coroutine, Generic[ParamTypeVar]):
    """Abstract base class for analysis instance.

    Analysis instance saves analysis results from multiple coating layer instances
    into file. It defines which data will be saved to which file with which format.

    To save the data, use the following methods:

    * :meth:`send`: Write the data into files.
        First call must pass :obj:`None`, which prepares the files.
        Subsequent calls must pass coating layer instances.
    * :meth:`close`: Close the files

    Concrete subclass must implement :meth:`object.__await__` which returns the
    file writing coroutine. Also, dataclass type must be assigned to class attribute
    :attr:`ParamType` which defines type of :attr:`parameters`.

    Arguments:
        parameters: File writing parameters.
            If passed, must be an instance of :attr:`ParamType`.
            If not passed, attempts to construct :attr:`ParamType`
            instance without argument.
        fps: Frame rate of multiple target images.
            If passed, resulting files can specify temporal data.
            For example, this value can be used for video file FPS and timestamps for
            numerical result.
    """

    ParamType: Type[ParamTypeVar]
    """Type of :attr:`parameters.`

    This class attribute is defined but not set in :class:`AnalysisBase`.
    Concrete subclass must assign this attribute with frozen dataclass type.
    """

    def __init__(
        self,
        parameters: Optional[ParamTypeVar] = None,
        *,
        fps: float = 0.0,
    ):
        """Initialize the instance.

        - *parameters* must be instance of :attr:`ParamType` or :obj:`None`.
          If :obj:`None`, a :attr:`ParamType` is attempted to be constructed.
        - *fps* must be positive.
        - Generator from :meth:`object.__await__` is internally saved to implement
          :meth:`send`, :meth:`throw` and :meth:`stop`.
        """
        if parameters is None:
            self._parameters = self.ParamType()
        else:
            if not isinstance(parameters, self.ParamType):
                raise TypeError(f"{parameters} is not instance of {self.ParamType}")
            self._parameters = parameters
        if fps < 0:
            raise ValueError("fps must not be negative.")
        self._fps = fps
        self._iterator = self.__await__()

    @property
    def parameters(self) -> ParamTypeVar:
        """File writing parameters.

        This property returns a frozen dataclass instance.
        Its type is :attr:`ParamType`.

        Note:
            This dataclass must be frozen to ensure reproducible results.
        """
        return self._parameters

    @property
    def fps(self) -> float:
        """FPS of the target images for incoming coating layer instances.

        This value is used by file writing implementation to determine the time interval
        between each coating layer instance. ``0.0`` indicates that FPS is unknown.
        """
        return self._fps

    def send(self, value: Optional[CoatingLayerBase]):
        """Write the coating layer data to file.

        Sends *value* to internal generator constructed from :meth:`object.__await__`.
        """
        if value is None:
            next(self._iterator)
        else:
            self._iterator.send(value)  # type: ignore[arg-type]

    def throw(self, type, value, traceback):
        """Throw exception into file writing coroutine."""
        self._iterator.throw(type, value, traceback)

    def close(self):
        """Terminate file writing coroutine."""
        self._iterator.close()

    @abc.abstractmethod
    def verify(self):
        """Sanity check before file writing.

        This method checks :attr:`parameters` and raises error if anything is wrong.
        """


def ImageWriter(
    path: str, fourcc: int, fps: float
) -> Generator[None, Optional[npt.NDArray[np.uint8]], None]:
    """Coroutine to write incoming RGB images into image file(s) or video file.

    This function supports several ways to write image data:

    #. Multiple single-page image files
        *path* is formattable path with image format (e.g., ``img%02d.jpg``).
    #. Single image file
        *path* is non-formattable path with image format (e.g., ``img.gif``).
        If the format supports multipage image, *fps* is used.
        If the format does not support multipage image, only the first image is written.
    #. Single video file
        *path* is non-formattable path with video format (e.g., ``img.mp4``).
        *fourcc* and *fps* is used to encode the video.

    Warning:
        When writing into a single image file, sending too many images will cause
        memory issue.

    Arguments:
        path: Resulting file path.
            Can have either image extension or video extension.
        fourcc: Result of :func:`cv2.VideoWriter_fourcc`.
            Specifies encoder to write the video. Ignored if *path* is not video.
        fps: Frame rate of incoming images.
            Specifies frame rate to write multipage image file or video file.
            Ignored if *path* is single-page image file(s).

    Note:
        Type of *path* (image vs video) is detected by :mod:`mimetypes`.
        Image file is written by :meth:`PIL`, and video file is written by
        :obj:`cv2.VideoWriter`.

    Examples:
        .. code-block:: python

            gen = ImageWriter(...)
            next(gen)  # Initialize the coroutine
            gen.send(img1)
            gen.send(img2)
            gen.close()  # Close the file
    """
    try:
        path % 0
        formattable = True
    except TypeError:
        formattable = False

    mtype, _ = mimetypes.guess_type(path)
    if mtype is None:
        raise TypeError(f"Invalid path: {path}.")
    ftype, _ = mtype.split("/")

    if ftype == "image":
        if formattable:
            i = 0
            while True:
                img = yield
                PIL.Image.fromarray(img).save(path % i)
                i += 1
        else:
            images = []
            img = yield
            try:
                while True:
                    images.append(PIL.Image.fromarray(img))
                    img = yield
            finally:
                if fps == 0.0:
                    try:
                        images[0].save(
                            path,
                            save_all=True,
                            append_images=images[1:],
                        )
                    except Exception:
                        images[-1].save(path)
                else:
                    try:
                        images[0].save(
                            path,
                            save_all=True,
                            append_images=images[1:],
                            duration=1000 / fps,
                        )
                    except Exception:
                        images[-1].save(path)
    elif ftype == "video":
        img = yield
        h, w = img.shape[:2]
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        try:
            while True:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                writer.write(img)
                img = yield
        finally:
            writer.release()
    else:
        raise TypeError(f"Unsupported mimetype: {mtype}.")


def CSVWriter(path: str):
    """Coroutine to write incoming data to CSV file.

    Arguments:
        path: Resulting file path.

    Examples:
        .. code-block:: python

            gen = CSVWriter("result.csv")
            next(gen)  # Initialize the coroutine
            gen.send([1, 2, 3])
            gen.send(["foo", "bar", "baz"])
            gen.close()  # Close the file
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        while True:
            data = yield
            writer.writerow(data)


@dataclasses.dataclass(frozen=True)
class AnalysisParam:
    """File writing parameters for :class:`Analysis`.

    Arguments:
        ref_data: Path for numerical data file of reference instance.
        ref_visual: Path for visualized file of reference instance.
        subst_data: Path for numerical data file of substrate instance.
        subst_visual: Path for visualized file of substrate instance.
        layer_data: Path for numerical data file of coating layer instances.
        layer_visual: Path for visualized file of coating layer instances.
        layer_fourcc: FourCC code to encode video file.
    """

    ref_data: str = ""
    ref_visual: str = ""
    subst_data: str = ""
    subst_visual: str = ""
    layer_data: str = ""
    layer_visual: str = ""
    layer_fourcc: str = ""


class Analysis(AnalysisBase[AnalysisParam]):
    """Implementation of :class:`Analysis`.

    Analysis data are stored in the following fashion:

    #. Reference instance of the first passed coating layer instance is analyzed.
    #. Substrate instance of the first passed coating layer instance is analyzed.
    #. All the passed coating layer instances are analyzed. If :attr:`fps` is not zero,
       timestamps are automatically prepended to the coating layer data.

    Image files are written using :func:`ImageWriter`. Data files are written using
    writers registered to class attribute :attr:`DataWriters`.

    Arguments:
        parameters (AnalysisParam)
        fps
    """

    ParamType = AnalysisParam
    """Assigned with :class:`AnalysisParam`."""

    DataWriters = dict(csv=CSVWriter)
    """Dictionary containing data writers.

    Keys are the file formats and values are the writers.
    """

    def verify(self):
        """Implement :meth:`AnalysisBase.verify`.

        #. If data paths are given in :attr:`parameters`, their extensions should be
           registered in :attr:`DataWriters`.
        #. MIME type of the visual files must be ``image`` or ``video``.
        """
        for path in [
            os.path.expandvars(self.parameters.ref_data),
            os.path.expandvars(self.parameters.subst_data),
            os.path.expandvars(self.parameters.layer_data),
        ]:
            if path:
                _, ext = os.path.splitext(path)
                if ext.lstrip(os.path.extsep).lower() not in self.DataWriters:
                    raise ValueError(f"{path} has unsupported extension.")
        for path in [
            os.path.expandvars(self.parameters.ref_visual),
            os.path.expandvars(self.parameters.subst_visual),
        ]:
            if path:
                mtype, _ = mimetypes.guess_type(path)
                file_type, _ = mtype.split("/")
                if file_type != "image":
                    raise ValueError(f"{path} is not image.")
        if os.path.expandvars(self.parameters.layer_visual):
            mtype, _ = mimetypes.guess_type(
                os.path.expandvars(self.parameters.layer_visual)
            )
            file_type, _ = mtype.split("/")
            if file_type not in ("image", "video"):
                raise ValueError(f"{path} is not image nor video.")

    def __await__(self):
        """Save the analysis result using incoming coating layer instances.

        If the directories of the files do not exist, they will be created.
        If FourCC string is empty, the FourCC value will be ``0``.
        Type and substrate instance of the coating layer instances must be homogeneous.
        """

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
                    f.name
                    for f in dataclasses.fields(layer.substrate.reference.DataType)
                ]
                rd_writer.send(headers)
                data = list(dataclasses.astuple(layer.substrate.reference.analyze()))
                rd_writer.send(data)

            if ref_visual:
                rv_writer.send(layer.substrate.reference.draw())

            if subst_data:
                headers = [f.name for f in dataclasses.fields(layer.substrate.DataType)]
                sd_writer.send(headers)
                data = list(dataclasses.astuple(layer.substrate.analyze()))
                sd_writer.send(data)

            if subst_visual:
                sv_writer.send(layer.substrate.draw())

            if layer_data:
                headers = [f.name for f in dataclasses.fields(layer.DataType)]
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
                    lv_writer.send(layer.draw())

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
