"""
Control Widgets
===============

This module provides widgets to control the analysis parameters.

"""

from .base import ControlWidget
from .exptwidget import ExperimentWidget
from .refwidget import ReferenceWidget
from .substwidget import SubstrateWidget
from .layerwidget import CoatingLayerWidget
from .analysiswidget import AnalysisWidget
from .controlwidget import MasterControlWidget

__all__ = [
    "ControlWidget",
    "ExperimentWidget",
    "ReferenceWidget",
    "SubstrateWidget",
    "CoatingLayerWidget",
    "EmptyDoubleValidator",
    "AnalysisWidget",
    "MasterControlWidget",
]
