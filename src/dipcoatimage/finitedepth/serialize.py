"""Read analysis configuration from file."""

import abc
import dataclasses
import glob
import importlib
import mimetypes
import os
from typing import TYPE_CHECKING, Any, Generator, Optional, Tuple, Type

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
from .reference import OptionalROI, ReferenceBase
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


@dataclasses.dataclass
class ImportArgs:
    """Arguments to import the variable from module."""

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
    """Data for the concrete instance of `ReferenceBase`.

    Parameters
    ----------
    type
        Information to import reference class.
        Class name defaults to ``Reference``.

    templateROI, substrateROI, parameters, draw_options
        Data for arguments of reference class.
    """

    type: ImportArgs = dataclasses.field(
        default_factory=lambda: ImportArgs(name="Reference")
    )
    templateROI: OptionalROI = (0, 0, None, None)
    substrateROI: OptionalROI = (0, 0, None, None)
    parameters: dict = dataclasses.field(default_factory=dict)
    draw_options: dict = dataclasses.field(default_factory=dict)

    def as_structured_args(
        self,
    ) -> Tuple[Type[ReferenceBase], "DataclassInstance", "DataclassInstance"]:
        """Structure the primitive data."""
        refcls = self.type.import_variable()
        if not (isinstance(refcls, type) and issubclass(refcls, ReferenceBase)):
            raise TypeError(f"{refcls} is not substrate reference class.")

        params = data_converter.structure(
            self.parameters, refcls.Parameters  # type: ignore
        )
        drawopts = data_converter.structure(
            self.draw_options, refcls.DrawOptions  # type: ignore
        )
        return (refcls, params, drawopts)

    def as_reference(self, img: npt.NDArray[np.uint8]) -> ReferenceBase:
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
    """Data for the concrete instance of `SubstrateBase`.

    Parameters
    ----------
    type
        Information to import substrate class.
        Class name defaults to ``Substrate``.

    parameters, draw_options
        Data for arguments of substrate class.
    """

    type: ImportArgs = dataclasses.field(
        default_factory=lambda: ImportArgs(name="Substrate")
    )
    parameters: dict = dataclasses.field(default_factory=dict)
    draw_options: dict = dataclasses.field(default_factory=dict)

    def as_structured_args(
        self,
    ) -> Tuple[Type[SubstrateBase], "DataclassInstance", "DataclassInstance"]:
        """Structure the primitive data."""
        substcls = self.type.import_variable()
        if not (isinstance(substcls, type) and issubclass(substcls, SubstrateBase)):
            raise TypeError(f"{substcls} is not substrate class.")

        params = data_converter.structure(
            self.parameters, substcls.Parameters  # type: ignore
        )
        drawopts = data_converter.structure(
            self.draw_options, substcls.DrawOptions  # type: ignore
        )
        return (substcls, params, drawopts)

    def as_substrate(self, ref: ReferenceBase) -> SubstrateBase:
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
    """Data for the concrete instance of `CoatingLayerBase`.

    Parameters
    ----------
    type
        Information to import substrate class.
        Class name defaults to ``CoatingLayer``.

    parameters, draw_options, deco_options
        Data for arguments of coating layer class.
    """

    type: ImportArgs = dataclasses.field(
        default_factory=lambda: ImportArgs(name="CoatingLayer")
    )
    parameters: dict = dataclasses.field(default_factory=dict)
    draw_options: dict = dataclasses.field(default_factory=dict)
    deco_options: dict = dataclasses.field(default_factory=dict)

    def as_structured_args(
        self,
    ) -> Tuple[
        Type[CoatingLayerBase],
        "DataclassInstance",
        "DataclassInstance",
        "DataclassInstance",
    ]:
        """Structure the primitive data."""
        layercls = self.type.import_variable()
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
    """Data for the concrete instance of `ExperimentBase`."""

    type: ImportArgs = dataclasses.field(
        default_factory=lambda: ImportArgs(name="Experiment")
    )
    parameters: dict = dataclasses.field(default_factory=dict)

    def as_structured_args(self) -> Tuple[Type[ExperimentBase], "DataclassInstance"]:
        """Structure the primitive data."""
        exptcls = self.type.import_variable()
        if not (isinstance(exptcls, type) and issubclass(exptcls, ExperimentBase)):
            raise TypeError(f"{exptcls} is not experiment class.")

        params = data_converter.structure(
            self.parameters, exptcls.Parameters  # type: ignore
        )
        return (exptcls, params)

    def as_experiment(self) -> ExperimentBase:
        """Construct the experiment instance."""
        exptcls, params = self.as_structured_args()
        expt = exptcls(parameters=params)
        return expt


@dataclasses.dataclass
class AnalysisArgs:
    """Data for the concrete instance of `AnalysisBase`."""

    type: ImportArgs = dataclasses.field(
        default_factory=lambda: ImportArgs(name="Analysis")
    )
    parameters: dict = dataclasses.field(default_factory=dict)
    fps: Optional[float] = None

    def as_structured_args(
        self,
    ) -> Tuple[Type[AnalysisBase], "DataclassInstance", Optional[float]]:
        """Structure the primitive data."""
        cls = self.type.import_variable()
        if not (isinstance(cls, type) and issubclass(cls, AnalysisBase)):
            raise TypeError(f"{cls} is not coating layer class.")

        params = data_converter.structure(
            self.parameters, cls.Parameters  # type: ignore
        )
        return (cls, params, self.fps)

    def as_analysis(self) -> AnalysisBase:
        """Construct the analysis instance."""
        cls, params, fps = self.as_structured_args()
        analysis = cls(parameters=params, fps=fps)
        return analysis


@dataclasses.dataclass(frozen=True)
class ConfigBase(abc.ABC):
    """Class which wraps every information to construct and analyze the experiment.

    Notes
    -----
    Environment variables are allowed in *ref_path* and *coat_path* fields.

    If *fps* is not passed to AnalysisArgs, try to determine it from input files.
    """

    ref_path: str = ""
    coat_path: str = ""
    reference: ReferenceArgs = dataclasses.field(default_factory=ReferenceArgs)
    substrate: SubstrateArgs = dataclasses.field(default_factory=SubstrateArgs)
    coatinglayer: CoatingLayerArgs = dataclasses.field(default_factory=CoatingLayerArgs)
    experiment: ExperimentArgs = dataclasses.field(default_factory=ExperimentArgs)
    analysis: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)

    @abc.abstractmethod
    def frame_count(self) -> int:
        """Return total number of images from *coat_paths*."""

    @abc.abstractmethod
    def reference_image(self) -> npt.NDArray[np.uint8]:
        """Return binarized image from :attr:`ref_path`."""

    @abc.abstractmethod
    def image_generator(self) -> Generator[npt.NDArray[np.uint8], None, None]:
        """Yield binarized images from :attr:`coat_path`."""

    @abc.abstractmethod
    def fps(self) -> Optional[float]:
        """Find fps."""

    def construct_reference(self) -> ReferenceBase:
        """Construct reference instance."""
        return self.reference.as_reference(self.reference_image())

    def construct_substrate(self) -> SubstrateBase:
        """Construct substrate instance."""
        return self.substrate.as_substrate(self.construct_reference())

    def construct_coatinglayer(self, i: int, sequential=True):
        """Construct *i*-th coating layer instance.

        If *sequential* is *True*, coating layer is sequentially constructed
        using `ExperimentBase` factory. Else, `CoatingLayerBase`
        object is directly constructed.

        Parameters
        ----------
        i : int
            Index of the frame from *coat_path*
        sequential : bool, default=True
            Controls sequential construction.
        """
        if i + 1 > self.frame_count():
            raise ValueError("Index out of range.")
        img_gen = self.image_generator()
        subst = self.construct_substrate()
        layercls, params, drawopts, decoopts = self.coatinglayer.as_structured_args()
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
        """Construct experiment instance."""
        return self.experiment.as_experiment()

    def construct_analysis(self) -> AnalysisBase:
        """Construct analysis instance.

        :meth:`fps` is used for *fps* parameter.
        """
        return dataclasses.replace(self.analysis, fps=self.fps()).as_analysis()

    def analyze(self, name: str = ""):
        """Analyze and save the data. Progress bar is shown.

        Parameters
        ----------
        name : str
            Description for progress bar.
        """
        # Let Analyzer verify ref, subst, and layer to do whatever it wants.
        subst = self.construct_substrate()
        layercls, params, drawopts, decoopts = self.coatinglayer.as_structured_args()
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
    """Analyze using cv2."""

    def frame_count(self) -> int:
        """Return total number of images from *coat_paths*."""
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
        """Return binarized image from :attr:`ref_path`."""
        with PIL.Image.open(os.path.expandvars(self.ref_path)) as img:
            ret = binarize(np.array(img.convert("L")), "rgb")
        return ret

    def image_generator(self) -> Generator[npt.NDArray[np.uint8], None, None]:
        """Yield binarized images from :attr:`coat_path`."""
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

    def fps(self) -> Optional[float]:
        """Find fps."""
        fps = self.analysis.fps
        if fps is None:
            files = glob.glob(os.path.expandvars(self.coat_path))
            for f in files:
                mtype, _ = mimetypes.guess_type(f)
                if mtype is None:
                    continue
                mtype, _ = mtype.split("/")
                if mtype == "image":
                    continue
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
    """Binarize *image* with Otsu's thresholding.

    Parameters
    ----------
    image : ndarray
        Input image.
    color : {"rgb", "bgr"}
        Color convention. For example, "rgb" indicates that 3-channel image should be
        interpreted as "RGB" and 4-channel be "RGBA".

    Notes
    -----
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
