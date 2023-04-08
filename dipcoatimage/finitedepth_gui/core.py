"""
GUI core
========

Core objects for GUI.

"""

import enum


__all__ = [
    "DataMember",
    "DataArgFlag",
    "VisualizationMode",
    "FrameSource",
    "ROIDrawMode",
]


class DataMember(enum.Enum):
    """
    Enum to represent five components (plus ``NULL``) which make up
    finite depth dip coating experiment analysis.

    The members of this class are used to indicate which tab widget is specified
    in experiment data view.
    """

    NULL = 0
    REFERENCE = 1
    SUBSTRATE = 2
    COATINGLAYER = 3
    EXPERIMENT = 4
    ANALYSIS = 5

    def displays(self) -> "DataMember":
        """
        Returns what the display shows when current data view is *self*.
        """
        if self == DataMember.REFERENCE:
            return DataMember.REFERENCE
        if self == DataMember.SUBSTRATE:
            return DataMember.SUBSTRATE
        return DataMember.EXPERIMENT


class DataArgFlag(enum.Flag):
    """
    Flag to represent the arguments of :class:`ExperimentData`.
    """

    NULL = 0
    REFPATH = 1
    COATPATHS = 2
    REFERENCE = 4
    SUBSTRATE = 8
    COATINGLAYER = 16
    EXPERIMENT = 32
    ANALYSIS = 64


class VisualizationMode(enum.Enum):
    """
    Option to determine how the coating layer instance is shown.

    Attributes
    ==========

    OFF
        Do not visualize.

    FULL
        Full visualization by constructing :class:`CoatingLayerBase` instance.

    FAST
        Fast visualization without constructing :class:`CoatingLayerBase`
        instance. Only the substrate region elimination is performed.

    """

    OFF = 0
    FULL = 1
    FAST = 2


class FrameSource(enum.Enum):
    NULL = 0
    FILE = 1
    CAMERA = 2


class ROIDrawMode(enum.Enum):
    NONE = 0
    TEMPLATE = 1
    SUBSTRATE = 2
