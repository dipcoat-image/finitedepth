"""
Serialization
=============

Classes to serialize the analysis parameters into configuration files.
"""

import cattrs
import cv2  # type: ignore
import dataclasses
import numpy as np
import numpy.typing as npt
from typing import List, Type, Optional, Tuple
from .reference import SubstrateReferenceBase
from .substrate import SubstrateBase
from .coatinglayer import CoatingLayerBase
from .experiment import ExperimentBase
from .analysis import Analyzer
from .util import import_variable, OptionalROI, DataclassProtocol


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
       >>> from dipcoatimage.finitedepth import ReferenceArgs, get_samples_path
       >>> refargs = ReferenceArgs(templateROI=(100, 50, 1200, 200),
       ...                         substrateROI=(300, 100, 950, 600))
       >>> ref = refargs.as_reference(cv2.imread(get_samples_path("ref3.png")))
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
       >>> from dipcoatimage.finitedepth import ReferenceArgs, get_samples_path
       >>> refargs = ReferenceArgs(templateROI=(100, 50, 1200, 200),
       ...                         substrateROI=(300, 100, 950, 600))
       >>> ref = refargs.as_reference(cv2.imread(get_samples_path("ref3.png")))

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
       >>> from dipcoatimage.finitedepth import (data_converter, ReferenceArgs,
       ...     SubstrateArgs, get_samples_path)
       >>> refargs = ReferenceArgs(templateROI=(100, 50, 1200, 200),
       ...                         substrateROI=(300, 100, 950, 600))
       >>> ref = refargs.as_reference(cv2.imread(get_samples_path("ref3.png")))
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
       ...     MorphologyClosing=dict(kernelSize=(1, 1)),
       ...     ReconstructRadius=50,
       ...     RoughnessMeasure="SSFD",
       ... )
       >>> arg = dict(type={"name": "RectLayerShape"}, parameters=params)
       >>> layerargs = data_converter.structure(arg, CoatingLayerArgs)
       >>> img_path = get_samples_path("coat3.png")
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
