"""
Analysis worker
===============

V2 for workers
"""

from dipcoatimage.finitedepth import ExperimentData
from PySide6.QtCore import QRunnable


__all__ = [
    "ExperimentWorker",
]


class ExperimentWorker(QRunnable):
    def __init__(self, parent):
        super().__init__(parent)

    def setExperimentData(self, exptData: ExperimentData):
        ...
