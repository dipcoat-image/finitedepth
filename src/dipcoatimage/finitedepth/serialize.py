"""Manage configuration for the analysis.

This module defines abstract class :class:`ConfigBase` and its
implementation, :class:`Config`.

Serialization and deserialization of arguments are performed by
:obj:`data_converter`. To handle custom classes, register
structure hook and unstructure hook to the converter.
"""

import abc
import dataclasses
import glob
import importlib
import mimetypes
import os
from typing import TYPE_CHECKING, Any, Generator, Tuple, Type

import cattrs
import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageSequence
import tqdm  # type: ignore

from .analysis import AnalysisBase
from .coatinglayer import CoatingLayerBase
from .experiment import ExperimentBase
from .reference import DynamicROI, ReferenceBase
from .substrate import SubstrateBase

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

__all__ = [
    "data_converter",
    "ImportArgs",
    "ReferenceArgs",
    "SubstrateArgs",
    "CoatingLayerArgs",
    "ExperimentArgs",
    "AnalysisArgs",
    "ConfigBase",
    "Config",
    "binarize",
]


data_converter = cattrs.Converter()
""":class:`cattrs.Converter` object for configuration serialization.

Examples:
    Constructing :class:`Config` objects from YAML-format configuration file:

    .. code-block:: python

        from dipcoatimage.finitedepth import data_converter, Config
        import yaml
        with open("path-to-config-file.yml", "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in data.items():
            config = data_converter.structure(v, Config)
"""


@dataclasses.dataclass
class ImportArgs:
    """Arguments to import variable from module.

    ``from foo import bar`` is represented by ``ImportArgs("bar", "foo")``.

    Arguments:
        name: Name of the variable.
        module: Module specification.
    """

    name: str = ""
    module: str = "dipcoatimage.finitedepth"

    def import_variable(self) -> Any:
        """Import the variable using its name and module."""
        if not self.name:
            raise ValueError("Empty variable name")

        SENTINEL = object()
        if self.module:
            module = importlib.import_module(self.module)
            ret = getattr(module, self.name, SENTINEL)
            if ret is SENTINEL:
                raise ImportError(f"cannot import name {self.name} from {module}")
        else:
            ret = eval(self.name)

        return ret


@dataclasses.dataclass
class ReferenceArgs:
    """Data to construct concrete instance of :class:`ReferenceBase`.

    Reference image is not specified in this dataclass.

    Arguments:
        type: Reference type.
        templateROI, substrateROI: ROIs for reference instance.
        parameters: Unstructured :attr:`ReferenceBase.parameters`.
        draw_options: Unstructured :attr:`ReferenceBase.draw_options`.
    """

    type: ImportArgs = dataclasses.field(
        default_factory=lambda: ImportArgs(name="Reference")
    )
    templateROI: DynamicROI = (0, 0, None, None)
    substrateROI: DynamicROI = (0, 0, None, None)
    parameters: dict = dataclasses.field(default_factory=dict)
    draw_options: dict = dataclasses.field(default_factory=dict)

    def as_objects(
        self,
    ) -> Tuple[Type[ReferenceBase], "DataclassInstance", "DataclassInstance"]:
        """Convert the data to Python objects.

        Returns:
            Type, parameters, and drawing options of reference instance.
                - Type is imported from :attr:`type`.
                - Parameters are structured from :attr:`parameters` to reference type's
                  :attr:`ReferenceBase.ParamType`, using :obj:`data_converter`.
                - Drawing options are structured from :attr:`draw_options` to
                  reference type's :attr:`ReferenceBase.DrawOptType`, using
                  :obj:`data_converter`.
        """
        refcls = self.type.import_variable()
        if not (isinstance(refcls, type) and issubclass(refcls, ReferenceBase)):
            raise TypeError(f"{refcls} is not substrate reference class.")

        params = data_converter.structure(
            self.parameters, refcls.ParamType  # type: ignore
        )
        drawopts = data_converter.structure(
            self.draw_options, refcls.DrawOptType  # type: ignore
        )
        return (refcls, params, drawopts)

    def as_reference(self, img: npt.NDArray[np.uint8]) -> ReferenceBase:
        """Construct the reference instance.

        Arguments:
            img: Reference image.

        Returns:
            Reference instance. Type and arguments are acquired from :meth:`as_objects`.
        """
        refcls, params, drawopts = self.as_objects()

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
    """Data to construct concrete instance of :class:`SubstrateBase`.

    Reference instance is not specified in this dataclass.

    Arguments:
        type: Substrate type.
        parameters: Unstructured :attr:`SubstrateBase.parameters`.
        draw_options: Unstructured :attr:`SubstrateBase.draw_options`.
    """

    type: ImportArgs = dataclasses.field(
        default_factory=lambda: ImportArgs(name="Substrate")
    )
    parameters: dict = dataclasses.field(default_factory=dict)
    draw_options: dict = dataclasses.field(default_factory=dict)

    def as_objects(
        self,
    ) -> Tuple[Type[SubstrateBase], "DataclassInstance", "DataclassInstance"]:
        """Convert the data to Python objects.

        Returns:
            Type, parameters, and drawing options of substrate instance.
                - Type is imported from :attr:`type`.
                - Parameters are structured from :attr:`parameters` to substrate type's
                  :attr:`SubstrateBase.ParamType`, using :obj:`data_converter`.
                - Drawing options are structured from :attr:`draw_options` to
                  substrate type's :attr:`SubstrateBase.DrawOptType`, using
                  :obj:`data_converter`.
        """
        substcls = self.type.import_variable()
        if not (isinstance(substcls, type) and issubclass(substcls, SubstrateBase)):
            raise TypeError(f"{substcls} is not substrate class.")

        params = data_converter.structure(
            self.parameters, substcls.ParamType  # type: ignore
        )
        drawopts = data_converter.structure(
            self.draw_options, substcls.DrawOptType  # type: ignore
        )
        return (substcls, params, drawopts)

    def as_substrate(self, ref: ReferenceBase) -> SubstrateBase:
        """Construct the substrate instance.

        Arguments:
            ref: Reference instance.

        Returns:
            Substrate instance. Type and arguments are acquired from :meth:`as_objects`.
        """
        substcls, params, drawopts = self.as_objects()
        subst = substcls(  # type: ignore
            ref,
            parameters=params,
            draw_options=drawopts,
        )
        return subst


@dataclasses.dataclass
class CoatingLayerArgs:
    """Data to construct concrete instance of :class:`CoatingLayerBase`.

    Target image and substrate instance are not specified in this dataclass.

    Arguments:
        type: Coating layer type.
        parameters: Unstructured :attr:`CoatingLayerBase.parameters`.
        draw_options: Unstructured :attr:`CoatingLayerBase.draw_options`.
        deco_options: Unstructured :attr:`CoatingLayerBase.deco_options`.
    """

    type: ImportArgs = dataclasses.field(
        default_factory=lambda: ImportArgs(name="CoatingLayer")
    )
    parameters: dict = dataclasses.field(default_factory=dict)
    draw_options: dict = dataclasses.field(default_factory=dict)
    deco_options: dict = dataclasses.field(default_factory=dict)

    def as_objects(
        self,
    ) -> Tuple[
        Type[CoatingLayerBase],
        "DataclassInstance",
        "DataclassInstance",
        "DataclassInstance",
    ]:
        """Convert the data to Python objects.

        Returns:
            Type, parameters, drawing and deco options of coating layer instance.
                - Type is imported from :attr:`type`.
                - Parameters are structured from :attr:`parameters` to coating layer
                  type's :attr:`CoatingLayerBase.ParamType`, using
                  :obj:`data_converter`.
                - Drawing options are structured from :attr:`draw_options` to
                  coating layer type's :attr:`CoatingLayerBase.DrawOptType`, using
                  :obj:`data_converter`.
                - Deco options are structured from :attr:`deco_options` to
                  coating layer type's :attr:`CoatingLayerBase.DecoOptType`, using
                  :obj:`data_converter`.
        """
        layercls = self.type.import_variable()
        if not (isinstance(layercls, type) and issubclass(layercls, CoatingLayerBase)):
            raise TypeError(f"{layercls} is not coating layer class.")

        params = data_converter.structure(
            self.parameters, layercls.ParamType  # type: ignore
        )
        drawopts = data_converter.structure(
            self.draw_options, layercls.DrawOptType  # type: ignore
        )
        decoopts = data_converter.structure(
            self.deco_options, layercls.DecoOptType  # type: ignore
        )
        return (layercls, params, drawopts, decoopts)

    def as_coatinglayer(
        self, img: npt.NDArray[np.uint8], subst: SubstrateBase
    ) -> CoatingLayerBase:
        """Construct the coating layer instance.

        Arguments:
            img: Target image.
            subst: Substrate instance.

        Returns:
            Coating layer instance. Type and arguments are acquired from
            :meth:`as_objects`.
        """
        layercls, params, drawopts, decoopts = self.as_objects()
        layer = layercls(  # type: ignore
            img, subst, parameters=params, draw_options=drawopts, deco_options=decoopts
        )
        return layer


@dataclasses.dataclass
class ExperimentArgs:
    """Data to construct concrete instance of :class:`ExperimentBase`.

    Arguments:
        type: Experiment type.
        parameters: Unstructured :attr:`ExperimentBase.parameters`.
    """

    type: ImportArgs = dataclasses.field(
        default_factory=lambda: ImportArgs(name="Experiment")
    )
    parameters: dict = dataclasses.field(default_factory=dict)

    def as_objects(self) -> Tuple[Type[ExperimentBase], "DataclassInstance"]:
        """Convert the data to Python objects.

        Returns:
            Type and parameters of experiment instance.
                - Type is imported from :attr:`type`.
                - Parameters are structured from :attr:`parameters` to experiment type's
                  :attr:`ExperimentBase.ParamType`, using :obj:`data_converter`.
        """
        exptcls = self.type.import_variable()
        if not (isinstance(exptcls, type) and issubclass(exptcls, ExperimentBase)):
            raise TypeError(f"{exptcls} is not experiment class.")

        params = data_converter.structure(
            self.parameters, exptcls.ParamType  # type: ignore
        )
        return (exptcls, params)

    def as_experiment(self) -> ExperimentBase:
        """Construct the experiment instance.

        Returns:
            Experiment instance. Type and arguments are acquired from
            :meth:`as_objects`.
        """
        exptcls, params = self.as_objects()
        expt = exptcls(parameters=params)
        return expt


@dataclasses.dataclass
class AnalysisArgs:
    """Data to construct concrete instance of :class:`AnalysisBase`.

    Arguments:
        type: Analysis type.
        parameters: Unstructured :attr:`AnalysisBase.parameters`.
    """

    type: ImportArgs = dataclasses.field(
        default_factory=lambda: ImportArgs(name="Analysis")
    )
    parameters: dict = dataclasses.field(default_factory=dict)
    fps: float = 0.0

    def as_objects(
        self,
    ) -> Tuple[Type[AnalysisBase], "DataclassInstance"]:
        """Convert the data to Python objects.

        Returns:
            Type and parameters of analysis instance.
                - Type is imported from :attr:`type`.
                - Parameters are structured from :attr:`parameters` to analysis type's
                  :attr:`AnalysisBase.ParamType`, using :obj:`data_converter`.
        """
        cls = self.type.import_variable()
        if not (isinstance(cls, type) and issubclass(cls, AnalysisBase)):
            raise TypeError(f"{cls} is not analysis class.")

        params = data_converter.structure(
            self.parameters, cls.ParamType  # type: ignore
        )
        return (cls, params)

    def as_analysis(self) -> AnalysisBase:
        """Construct the analysis instance.

        Returns:
            Analysis instance. Type and arguments are acquired from :meth:`as_objects`.
        """
        cls, params = self.as_objects()
        analysis = cls(parameters=params, fps=self.fps)
        return analysis


@dataclasses.dataclass(frozen=True)
class ConfigBase(abc.ABC):
    """Abstract base class for configuration instance.

    Configuration instance wraps all the data required for analysis; path to source
    images, analysis parameters, visualization options, and path to result files.
    It also implements methods to load reference and targe images from files, and
    provides methods to perform the analysis.

    To perform analysis, use :meth:`analyze`.

    Attributes
        ref_path: Path to reference image file.
        coat_path: Path to target image files or video file.
        reference: Arguments to construct reference instance.
        substrate: Arguments to construct substrate instance.
        coatinglayer: Arguments to construct coating layer instance.
        experiment: Arguments to construct experiment instance.
        analysis: Arguments to construct analysis instance.
    """

    ref_path: str
    coat_path: str
    reference: ReferenceArgs = dataclasses.field(default_factory=ReferenceArgs)
    substrate: SubstrateArgs = dataclasses.field(default_factory=SubstrateArgs)
    coatinglayer: CoatingLayerArgs = dataclasses.field(default_factory=CoatingLayerArgs)
    experiment: ExperimentArgs = dataclasses.field(default_factory=ExperimentArgs)
    analysis: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)

    @abc.abstractmethod
    def frame_count(self) -> int:
        """Return total number of images from :attr:`coat_paths`."""

    @abc.abstractmethod
    def reference_image(self) -> npt.NDArray[np.uint8]:
        """Return binarized image from :attr:`ref_path`."""

    @abc.abstractmethod
    def image_generator(self) -> Generator[npt.NDArray[np.uint8], None, None]:
        """Yield binarized images from :attr:`coat_path`."""

    @abc.abstractmethod
    def fps(self) -> float:
        """Find fps from :attr:`coat_path` and :attr:`analysis`."""

    def construct_reference(self) -> ReferenceBase:
        """Construct reference instance.

        This method provides quick construction for debugging.
        """
        return self.reference.as_reference(self.reference_image())

    def construct_substrate(self) -> SubstrateBase:
        """Construct substrate instance.

        This method provides quick construction for debugging.
        """
        return self.substrate.as_substrate(self.construct_reference())

    def construct_coatinglayer(self, i: int, sequential=True):
        """Construct *i*-th coating layer instance.

        If *sequential* is *True*, coating layer is sequentially constructed
        using experiment instance from :meth:`construct_experiment`.
        Else, the coating layer object is directly constructed.

        This method provides quick construction for debugging.

        Arguments:
            i: Index of the frame from *coat_path*
            sequential: Controls sequential construction.
        """
        if i + 1 > self.frame_count():
            raise ValueError("Index out of range.")
        img_gen = self.image_generator()
        subst = self.construct_substrate()
        layercls, params, drawopts, decoopts = self.coatinglayer.as_objects()
        if sequential:
            expt = self.construct_experiment()
            for _ in range(i + 1):
                img = next(img_gen)
                layer = expt.coatinglayer(
                    img,
                    subst,
                    layer_type=layercls,
                    layer_parameters=params,
                    layer_drawoptions=drawopts,
                    layer_decooptions=decoopts,
                )
        else:
            for _ in range(i + 1):
                img = next(img_gen)
            layer = layercls(
                img, subst, params, draw_options=drawopts, deco_options=decoopts
            )
        return layer

    def construct_experiment(self) -> ExperimentBase:
        """Construct experiment instance.

        This method provides quick construction for debugging.
        """
        return self.experiment.as_experiment()

    def construct_analysis(self) -> AnalysisBase:
        """Construct analysis instance.

        This method provides quick construction for debugging.

        :meth:`fps` is used for *fps* argument.
        """
        return dataclasses.replace(self.analysis, fps=self.fps()).as_analysis()

    def analyze(self, name: str = ""):
        """Analyze and save the data. Progress bar is shown.

        Arguments:
            name: Description for progress bar.
        """
        # Let Analyzer verify ref, subst, and layer to do whatever it wants.
        subst = self.construct_substrate()
        layercls, params, drawopts, decoopts = self.coatinglayer.as_objects()
        expt = self.construct_experiment()
        expt.verify()
        analysis = self.construct_analysis()
        analysis.verify()
        try:
            analysis.send(None)
            for img in tqdm.tqdm(
                self.image_generator(), total=self.frame_count(), desc=name
            ):
                layer = expt.coatinglayer(
                    img,
                    subst,
                    layer_type=layercls,
                    layer_parameters=params,
                    layer_drawoptions=drawopts,
                    layer_decooptions=decoopts,
                )
                analysis.send(layer)
        finally:
            analysis.close()


class Config(ConfigBase):
    """Basic implementation of :class:`ConfigBase`.

    This class implements abstract methods using :mod:`cv2` and :mod:`PIL`.
    Also, environment variables in :attr:`ref_path` and :attr:`coat_path` are expanded.
    """

    def frame_count(self) -> int:
        """Implement :meth:`ConfigBase.frame_count`."""
        i = 0
        files = glob.glob(os.path.expandvars(self.coat_path))
        for f in files:
            mtype, _ = mimetypes.guess_type(f)
            if mtype is None:
                continue
            mtype, _ = mtype.split("/")
            if mtype == "image":
                with PIL.Image.open(f) as img:
                    i += img.n_frames
            elif mtype == "video":
                cap = cv2.VideoCapture(f)
                i += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            else:
                continue
        return i

    def reference_image(self) -> npt.NDArray[np.uint8]:
        """Implement :meth:`ConfigBase.reference_image`."""
        with PIL.Image.open(os.path.expandvars(self.ref_path)) as img:
            ret = binarize(np.array(img.convert("L")), "rgb")
        return ret

    def image_generator(self) -> Generator[npt.NDArray[np.uint8], None, None]:
        """Implement :meth:`ConfigBase.image_generator`."""
        files = glob.glob(os.path.expandvars(self.coat_path))
        for f in files:
            mtype, _ = mimetypes.guess_type(f)
            if mtype is None:
                continue
            mtype, _ = mtype.split("/")
            if mtype == "image":
                with PIL.Image.open(f) as img:
                    for frame in PIL.ImageSequence.Iterator(img):
                        yield binarize(np.array(frame.convert("L")), "rgb")
            elif mtype == "video":
                cap = cv2.VideoCapture(f)
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield binarize(frame, "bgr")
            else:
                continue

    def fps(self) -> float:
        """Implement :meth:`ConfigBase.fps`.

        This method first checks :attr:`analysis` to get *fps*. If the value is ``0.0``,
        it then tries to read the frame rate from :attr:`coat_path`.
        """
        fps = self.analysis.fps
        if fps == 0.0:
            files = glob.glob(os.path.expandvars(self.coat_path))
            for f in files:
                mtype, _ = mimetypes.guess_type(f)
                if mtype is None:
                    continue
                mtype, _ = mtype.split("/")
                if mtype == "image":
                    with PIL.Image.open(f) as img:
                        duration = img.info.get("duration")
                    if duration is not None:
                        fps = float(1000 / duration)
                        break
                elif mtype == "video":
                    cap = cv2.VideoCapture(f)
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    cap.release()
                    break
                else:
                    continue
        return fps


def binarize(
    image: npt.NDArray[np.uint8],
    color: str,
) -> npt.NDArray[np.uint8]:
    """Binarize the image with Otsu's thresholding.

    Arguments:
        image: Input image.
        color ({"rgb", "bgr"}): Color convention. "rgb" indicates that 3-channel image
            should be interpreted as "RGB" and 4-channel be "RGBA". "bgr" indicates
            "BGR" and "BGRA".

    Note:
        Shape of *image* can be:
            - (H, W) or (H, W, 1) : grayscale image.
            - (H, W, 3) : RGB or BGR image.
            - (H, W, 4) : RGBA or BGRA image.
    """
    if color not in ["rgb", "bgr"]:
        raise TypeError(f"Invalid color convention: {color}")
    if image.size == 0:
        return np.empty((0, 0), dtype=np.uint8)
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3:
        ch = image.shape[-1]
        if ch == 1:
            gray = image
        elif ch == 3 and color == "rgb":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif ch == 3 and color == "bgr":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif ch == 4 and color == "rgb":
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif ch == 4 and color == "bgr":
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise TypeError(f"Image with invalid channel: {ch}")
    else:
        raise TypeError(f"Invalid image shape: {image}")
    _, ret = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if ret is None:
        ret = np.empty((0, 0), dtype=np.uint8)
    return ret
