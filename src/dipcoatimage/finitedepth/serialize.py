"""
Serialization
=============

Classes to serialize the analysis parameters into configuration files.
"""

import cattrs
import cv2
import dataclasses
import numpy as np
import numpy.typing as npt
import os
from .reference import SubstrateReferenceBase, OptionalROI
from .substrate import SubstrateBase
from .coatinglayer import CoatingLayerBase
from .experiment import ExperimentBase
from .analysis import ExperimentKind, experiment_kind, Analyzer
from .util.importing import import_variable
from typing import List, Type, Optional, Tuple, TYPE_CHECKING

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
    "ExperimentData",
]


data_converter = cattrs.Converter()


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
       >>> from dipcoatimage.finitedepth import ReferenceArgs, get_data_path
       >>> refargs = ReferenceArgs(templateROI=(13, 10, 1246, 200),
       ...                         substrateROI=(100, 100, 1200, 500))
       >>> ref = refargs.as_reference(cv2.imread(get_data_path("ref3.png")))
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
    ) -> Tuple[Type[SubstrateReferenceBase], "DataclassInstance", "DataclassInstance"]:
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
       >>> from dipcoatimage.finitedepth import ReferenceArgs, get_data_path
       >>> refargs = ReferenceArgs(templateROI=(13, 10, 1246, 200),
       ...                         substrateROI=(100, 100, 1200, 500))
       >>> ref = refargs.as_reference(cv2.imread(get_data_path("ref3.png")))

    Construct substrate instance from the data.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import SubstrateArgs
       >>> params = dict(Sigma=3.0, Rho=1.0, Theta=0.01)
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
    ) -> Tuple[Type[SubstrateBase], "DataclassInstance", "DataclassInstance"]:
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
        Class name defaults to ``CoatingLayer``.

    parameters, draw_options, deco_options
        Data for arguments of coating layer class.

    Examples
    ========

    Construct substrate reference instance and substrate instance first.

    .. plot::
       :include-source:
       :context: reset

       >>> import cv2
       >>> from dipcoatimage.finitedepth import (data_converter, ReferenceArgs,
       ...     SubstrateArgs, get_data_path)
       >>> refargs = ReferenceArgs(templateROI=(13, 10, 1246, 200),
       ...                         substrateROI=(100, 100, 1200, 500))
       >>> ref = refargs.as_reference(cv2.imread(get_data_path("ref3.png")))
       >>> params = dict(Sigma=3.0, Rho=1.0, Theta=0.01)
       >>> arg = dict(type={"name": "RectSubstrate"}, parameters=params)
       >>> substargs = data_converter.structure(arg, SubstrateArgs)
       >>> subst = substargs.as_substrate(ref)

    Construct coating layer instance from the data.

    .. plot::
       :include-source:
       :context: close-figs

       >>> from dipcoatimage.finitedepth import CoatingLayerArgs
       >>> params = dict(
       ...     KernelSize=(1, 1),
       ...     ReconstructRadius=50,
       ...     RoughnessMeasure="SDTW",
       ...     RoughnessSamples=100,
       ... )
       >>> arg = dict(type={"name": "RectLayerShape"}, parameters=params)
       >>> layerargs = data_converter.structure(arg, CoatingLayerArgs)
       >>> img_path = get_data_path("coat3.png")
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
            self.type.name = "CoatingLayer"

    def as_structured_args(
        self,
    ) -> Tuple[
        Type[CoatingLayerBase],
        "DataclassInstance",
        "DataclassInstance",
        "DataclassInstance",
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

    def as_structured_args(self) -> Tuple[Type[ExperimentBase], "DataclassInstance"]:
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
        layer_parameters: Optional["DataclassInstance"] = None,
        layer_drawoptions: Optional["DataclassInstance"] = None,
        layer_decooptions: Optional["DataclassInstance"] = None,
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

    Environment variables are allowed in *ref_path* and *coat_paths* fields.
    """

    ref_path: str = ""
    coat_paths: List[str] = dataclasses.field(default_factory=list)
    reference: ReferenceArgs = dataclasses.field(default_factory=ReferenceArgs)
    substrate: SubstrateArgs = dataclasses.field(default_factory=SubstrateArgs)
    coatinglayer: CoatingLayerArgs = dataclasses.field(default_factory=CoatingLayerArgs)
    experiment: ExperimentArgs = dataclasses.field(default_factory=ExperimentArgs)
    analysis: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)

    def __post_init__(self):
        self.ref_path = os.path.expandvars(self.ref_path)
        self.coat_paths = [os.path.expandvars(p) for p in self.coat_paths]

    def construct_reference(self) -> SubstrateReferenceBase:
        """
        Construct and return :class:`SubstrateReferenceBase` from the data.
        """
        refimg = cv2.cvtColor(cv2.imread(self.ref_path), cv2.COLOR_BGR2RGB)
        return self.reference.as_reference(refimg)

    def construct_substrate(self) -> SubstrateBase:
        """
        Construct and return :class:`SubstrateBase` from the data.
        """
        ref = self.construct_reference()
        return self.substrate.as_substrate(ref)

    def construct_experiment(self) -> ExperimentBase:
        """
        Construct and return :class:`ExperimentBase` from the data.
        """
        subst = self.construct_substrate()
        layercls, params, drawopts, decoopts = self.coatinglayer.as_structured_args()
        expt = self.experiment.as_experiment(
            subst, layercls, params, drawopts, decoopts
        )
        return expt

    def experiment_kind(self) -> ExperimentKind:
        return experiment_kind(self.coat_paths)

    def image_count(self) -> int:
        expt_kind = self.experiment_kind()
        if (
            expt_kind == ExperimentKind.SINGLE_IMAGE
            or expt_kind == ExperimentKind.MULTI_IMAGE
        ):
            ret = len(self.coat_paths)
        elif expt_kind == ExperimentKind.VIDEO:
            (path,) = self.coat_paths
            cap = cv2.VideoCapture(path)
            ret = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            ret = -1
        return ret

    def construct_coatinglayer(
        self, image_index: int = 0, sequential: bool = True
    ) -> CoatingLayerBase:
        """
        Construct and return :class:`CoatingLayerBase` from the data.

        Parameters
        ----------
        image_index : int
            Index of the image to construct the coating layer instance in
            multiframe experiment.

        sequential: bool
            If True, construction of instance is done by passing `n`-th image and
            `(n-1)`-th instance to :meth:`ExperimentBase.layer_generator`,
            recursively.

        Notes
        -----
        If the experiment consists of multiple frames (images or video), the
        index of image can be specified by *image_index*.

        For speed, you may want to explicitly pass `sequential=False`. This
        method first constructs :class:`ExperimentBase` to use it as a coating
        layer instance factory, and by default recursively generates the instance
        from the first frame to `image-index`-th frame. This approach honors the
        modification by :class:`ExperimentBase` implementation but can be
        extremely slow. `sequential=False` ignores the recursive generation and
        directly constructs the instance as if it were the first frame of the
        experiment.

        """
        layer_gen = self.construct_experiment().layer_generator()
        next(layer_gen)

        if image_index > self.image_count() - 1:
            raise ValueError("image_index exceeds image numbers.")

        expt_kind = self.experiment_kind()
        if (
            expt_kind == ExperimentKind.SINGLE_IMAGE
            or expt_kind == ExperimentKind.MULTI_IMAGE
        ):
            if sequential:
                img_gen = (cv2.imread(path) for path in self.coat_paths)
                for _ in range(image_index + 1):
                    img = next(img_gen)
                    layer = layer_gen.send(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                img = cv2.imread(self.coat_paths[image_index])
                layer = layer_gen.send(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        elif expt_kind == ExperimentKind.VIDEO:
            (path,) = self.coat_paths
            cap = cv2.VideoCapture(path)

            try:
                if sequential:
                    for _ in range(image_index + 1):
                        ok, img = cap.read()
                        if not ok:
                            raise ValueError("Failed to read frame.")
                        layer = layer_gen.send(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, image_index)
                    ok, img = cap.read()
                    if not ok:
                        raise ValueError("Failed to read frame.")
                    layer = layer_gen.send(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            finally:
                cap.release()
        else:
            raise TypeError("Invalid coating layer paths.")

        return layer

    def analyze(self, name: str = ""):
        """Analyze and save the data."""
        expt = self.construct_experiment()
        analyzer = Analyzer(self.coat_paths, expt)

        analyzer.analyze(
            self.analysis.data_path,
            self.analysis.image_path,
            self.analysis.video_path,
            fps=self.analysis.fps,
            name=name,
        )