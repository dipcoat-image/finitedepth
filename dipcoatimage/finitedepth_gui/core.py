"""
GUI core
========

Core objects for GUI.

"""
import dataclasses
from dipcoatimage.finitedepth import (
    ExperimentBase,
    SubstrateReferenceBase,
    SubstrateBase,
    CoatingLayerBase,
)
from dipcoatimage.finitedepth.analysis import (
    ExperimentArgs,
    ReferenceArgs,
    SubstrateArgs,
    CoatingLayerArgs,
)
from dipcoatimage.finitedepth.util import OptionalROI, DataclassProtocol, Importer
import enum
from typing import Any, Optional, TypeVar, Type


__all__ = [
    "StructuredExperimentArgs",
    "StructuredReferenceArgs",
    "StructuredSubstrateArgs",
    "StructuredCoatingLayerArgs",
    "ClassSelection",
    "ExperimentMember",
    "VisualizationMode",
]


SEA = TypeVar("SEA", bound="StructuredExperimentArgs")


@dataclasses.dataclass
class StructuredExperimentArgs:
    """Structured data to construct experiment object."""

    type: Any
    parameters: Optional[DataclassProtocol]

    @classmethod
    def from_ExperimentArgs(cls: Type[SEA], args: ExperimentArgs) -> SEA:
        expttype, _ = Importer(args.type.name, args.type.module).try_import()
        if isinstance(expttype, type) and issubclass(expttype, ExperimentBase):
            try:
                param = expttype.Parameters()
            except (TypeError, ValueError):
                param = None
        else:
            param = None
        return cls(expttype, param)


SRA = TypeVar("SRA", bound="StructuredReferenceArgs")


@dataclasses.dataclass
class StructuredReferenceArgs:
    """Structured data to construct reference object."""

    type: Any
    templateROI: OptionalROI
    substrateROI: OptionalROI
    parameters: Optional[DataclassProtocol]
    draw_options: Optional[DataclassProtocol]

    @classmethod
    def from_ReferenceArgs(cls: Type[SRA], args: ReferenceArgs) -> SRA:
        reftype, _ = Importer(args.type.name, args.type.module).try_import()
        if isinstance(reftype, type) and issubclass(reftype, SubstrateReferenceBase):
            try:
                param = reftype.Parameters()
            except (TypeError, ValueError):
                param = None
            try:
                drawopt = reftype.DrawOptions()
            except (TypeError, ValueError):
                drawopt = None
        else:
            param = None
            drawopt = None
        return cls(reftype, args.templateROI, args.substrateROI, param, drawopt)


SSA = TypeVar("SSA", bound="StructuredSubstrateArgs")


@dataclasses.dataclass
class StructuredSubstrateArgs:
    """Structured data to construct substrate object."""

    type: Any
    parameters: Optional[DataclassProtocol]
    draw_options: Optional[DataclassProtocol]

    @classmethod
    def from_SubstrateArgs(cls: Type[SSA], args: SubstrateArgs) -> SSA:
        substtype, _ = Importer(args.type.name, args.type.module).try_import()
        if isinstance(substtype, type) and issubclass(substtype, SubstrateBase):
            try:
                param = substtype.Parameters()
            except (TypeError, ValueError):
                param = None
            try:
                drawopt = substtype.DrawOptions()
            except (TypeError, ValueError):
                drawopt = None
        else:
            param = None
            drawopt = None
        return cls(substtype, param, drawopt)


SCA = TypeVar("SCA", bound="StructuredCoatingLayerArgs")


@dataclasses.dataclass
class StructuredCoatingLayerArgs:
    """Structured data to construct coating layer object."""

    type: Any
    parameters: Optional[DataclassProtocol]
    draw_options: Optional[DataclassProtocol]
    deco_options: Optional[DataclassProtocol]

    @classmethod
    def from_CoatingLayerArgs(cls: Type[SCA], args: CoatingLayerArgs) -> SCA:
        layertype, _ = Importer(args.type.name, args.type.module).try_import()
        if isinstance(layertype, type) and issubclass(layertype, CoatingLayerBase):
            try:
                param = layertype.Parameters()
            except (TypeError, ValueError):
                param = None
            try:
                drawopt = layertype.DrawOptions()
            except (TypeError, ValueError):
                drawopt = None
            try:
                decoopt = layertype.DecoOptions()
            except (TypeError, ValueError):
                decoopt = None
        else:
            param = None
            drawopt = None
            decoopt = None
        return cls(layertype, param, drawopt, decoopt)


class ClassSelection(enum.IntFlag):
    """
    Enum to indicate class-specific selections, e.g. which tab widget or worker
    to choose.
    """

    UNKNOWN = 0
    REFERENCE = 1
    SUBSTRATE = 2
    EXPERIMENT = 4
    ANALYSIS = 8


class ExperimentMember(enum.Enum):
    """
    Enum to represent five components of the experiment data, plus ``UNKNOWN``
    for null value.

    The members of this class are used to indicate which tab widget is specified
    in experiment data view.
    """

    UNKNOWN = 0
    REFERENCE = 1
    SUBSTRATE = 2
    COATINGLAYER = 3
    EXPERIMENT = 4
    ANALYSIS = 5


class VisualizationMode(enum.IntEnum):
    """
    Option to determine how the image is shown.

    Attributes
    ==========

    OFF
        Do not visualize.

    FULL
        Full visualization. Reference and substrate are visualized as usual, and
        coating layer is visualized using coating layer decoration.

    FAST
        Fast visualization without coating layer decoration. Reference and
        substrate are visualized as usual, but coating layer is not decorated.

    """

    OFF = 0
    FULL = 1
    FAST = 2
