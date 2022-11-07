"""
Analysis worker
===============

V2 for workers
"""

import cv2  # type: ignore
import numpy as np
from dipcoatimage.finitedepth import (
    ExperimentData,
    SubstrateReferenceBase,
)
from dipcoatimage.finitedepth.analysis import AnalysisArgs
from dipcoatimage.finitedepth.util import Importer
from PySide6.QtCore import QRunnable


__all__ = [
    "ExperimentWorker",
]


class ExperimentWorker(QRunnable):
    def __init__(self, parent):
        super().__init__(parent)

        self.referenceImage = np.empty((0, 0, 0), dtype=np.uint8)
        self.reference = None
        self.substrateImage = np.empty((0, 0, 0), dtype=np.uint8)
        self.substrate = None
        self.coatingLayerImage = np.empty((0, 0, 0), dtype=np.uint8)
        self.coatingLayer = None
        self.experiment = None
        self.analysisArgs = AnalysisArgs()

    def setExperimentData(self, exptData: ExperimentData):
        refPath = exptData.ref_path
        if refPath:
            refImg = cv2.imread(exptData.ref_path)
        else:
            refImg = None
        if refImg is None:
            refImg = np.empty((0, 0, 0), dtype=np.uint8)
        else:
            refImg = cv2.cvtColor(refImg, cv2.COLOR_BGR2RGB)
        self.referenceImage = refImg

        refArgs = exptData.reference
        refType, _ = Importer(refArgs.type.name, refArgs.type.module).try_import()
        if isinstance(refType, type) and issubclass(refType, SubstrateReferenceBase):
            ref = refArgs.as_reference(refImg)
            if not ref.valid():
                ref = None
        else:
            ref = None
        self.reference = ref
