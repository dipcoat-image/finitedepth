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
    SubstrateBase,
    ExperimentBase,
)
from dipcoatimage.finitedepth.analysis import AnalysisArgs
from PySide6.QtCore import QRunnable
from typing import Optional


__all__ = [
    "ExperimentWorker",
]


class ExperimentWorker(QRunnable):
    def __init__(self, parent):
        super().__init__(parent)

        self.referenceImage = np.empty((0, 0, 0), dtype=np.uint8)
        self.reference: Optional[SubstrateReferenceBase] = None
        self.substrateImage = np.empty((0, 0, 0), dtype=np.uint8)
        self.substrate: Optional[SubstrateBase] = None
        self.experiment: Optional[ExperimentBase] = None
        self.analysisArgs = AnalysisArgs()

    def setExperimentData(self, exptData: ExperimentData):
        self.analysisArgs = exptData.analysis

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
        if self.referenceImage.size == 0:
            return

        refArgs = exptData.reference
        try:
            ref = refArgs.as_reference(self.referenceImage)
            if not ref.valid():
                self.reference = None
            else:
                self.reference = ref
        except TypeError:
            self.reference = None
        if self.reference is None:
            return

        substArgs = exptData.substrate
        try:
            subst = substArgs.as_substrate(self.reference)
            if not subst.valid():
                self.substrate = None
            else:
                self.substrate = subst
        except TypeError:
            self.substrate = None
        if self.substrate is None:
            return

        layerArgs = exptData.coatinglayer
        exptArgs = exptData.experiment
        try:
            structuredLayerArgs = layerArgs.as_structured_args()
            expt = exptArgs.as_experiment(self.substrate, *structuredLayerArgs)
            if not expt.valid():
                self.experiment = None
            else:
                self.experiment = expt
        except TypeError:
            self.experiment = None
        if self.experiment is None:
            return
