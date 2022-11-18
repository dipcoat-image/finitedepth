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
]


class DataMember(enum.Enum):
    """
    Enum to represent five components (plus ``NULL``) which makes up
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


class FrameSource(enum.Enum):
    NULL = 0
    FILE = 1
    CAMERA = 2
