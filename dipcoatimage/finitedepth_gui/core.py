"""
GUI core
========

Core objects for GUI.

"""
import dataclasses
from dipcoatimage.finitedepth import (
    SubstrateReference,
    Substrate,
    LayerArea,
    Experiment,
)
from dipcoatimage.finitedepth.util import OptionalROI, DataclassProtocol
from typing import Any, Optional


__all__ = [
    "StructuredExperimentArgs",
    "StructuredReferenceArgs",
    "StructuredSubstrateArgs",
    "StructuredCoatingLayerArgs",
]


@dataclasses.dataclass
class StructuredExperimentArgs:
    """Structured data to construct experiment object."""

    type: Any = Experiment
    parameters: Optional[DataclassProtocol] = None


@dataclasses.dataclass
class StructuredReferenceArgs:
    """Structured data to construct reference object."""

    type: Any = SubstrateReference
    templateROI: OptionalROI = (0, 0, None, None)
    substrateROI: OptionalROI = (0, 0, None, None)
    parameters: Optional[DataclassProtocol] = None
    draw_options: Optional[DataclassProtocol] = None


@dataclasses.dataclass
class StructuredSubstrateArgs:
    """Structured data to construct substrate object."""

    type: Any = Substrate
    parameters: Optional[DataclassProtocol] = None
    draw_options: Optional[DataclassProtocol] = None


@dataclasses.dataclass
class StructuredCoatingLayerArgs:
    """Structured data to construct coating layer object."""

    type: Any = LayerArea
    parameters: Optional[DataclassProtocol] = None
    draw_options: Optional[DataclassProtocol] = None
    deco_options: Optional[DataclassProtocol] = None
