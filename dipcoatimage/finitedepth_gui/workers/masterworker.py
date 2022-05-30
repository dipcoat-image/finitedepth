import cv2  # type: ignore
from dipcoatimage.finitedepth.analysis import (
    ExperimentKind,
    ReferenceArgs,
    SubstrateArgs,
    CoatingLayerArgs,
    ExperimentArgs,
    AnalysisArgs,
)
from dipcoatimage.finitedepth_gui.core import (
    StructuredReferenceArgs,
    StructuredSubstrateArgs,
    StructuredCoatingLayerArgs,
    StructuredExperimentArgs,
    ClassSelection,
    VisualizationMode,
)
from dipcoatimage.finitedepth_gui.inventory import ExperimentItemModel
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Slot, Signal
from typing import Optional, List
from .refworker import ReferenceWorker
from .substworker import SubstrateWorker
from .exptworker import ExperimentWorker
from .analysisworker import AnalysisWorker


__all__ = [
    "MasterWorker",
]


class MasterWorker(QObject):
    """
    Object which contains subworkers. Detects every change which requires the
    display to be updated, and signals.

    When workers are updated, combination of their corresponding
    :class:`ClassSelection` members are emitted by :attr:`workersUpdated`.

    """

    referenceImageShapeChanged = Signal(int, int)
    workersUpdated = Signal(ClassSelection)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._exptitem_model = None
        self._currentExperimentRow = -1

        self._ref_worker = ReferenceWorker()
        self._subst_worker = SubstrateWorker()
        self._expt_worker = ExperimentWorker()
        self._anal_worker = AnalysisWorker()

        self.referenceWorker().imageShapeChanged.connect(
            self.referenceImageShapeChanged
        )
        self.setVisualizationMode(VisualizationMode.FULL)

    def experimentItemModel(self) -> Optional[ExperimentItemModel]:
        """Model which holds the experiment item data."""
        return self._exptitem_model

    def currentExperimentRow(self) -> int:
        """Currently activated row from :meth:`experimentItemModel`."""
        return self._currentExperimentRow

    def referenceWorker(self) -> ReferenceWorker:
        return self._ref_worker

    def substrateWorker(self) -> SubstrateWorker:
        return self._subst_worker

    def experimentWorker(self) -> ExperimentWorker:
        return self._expt_worker

    def analysisWorker(self) -> AnalysisWorker:
        return self._anal_worker

    def setReferenceImage(self, img: Optional[npt.NDArray[np.uint8]]):
        self.referenceWorker().setImage(img)
        self.referenceWorker().updateReference()
        self.substrateWorker().setReference(self.referenceWorker().reference())
        self.substrateWorker().updateSubstrate()
        self.experimentWorker().setSubstrate(self.substrateWorker().substrate())
        self.experimentWorker().updateExperiment()
        self.analysisWorker().setExperiment(self.experimentWorker().experiment())
        self.workersUpdated.emit(
            ClassSelection.REFERENCE
            | ClassSelection.SUBSTRATE
            | ClassSelection.EXPERIMENT
            | ClassSelection.ANALYSIS
        )

    def setExperimentItemModel(self, model: Optional[ExperimentItemModel]):
        """Set :meth:`experimentItemModel`."""
        old_model = self.experimentItemModel()
        if old_model is not None:
            self.disconnectModel(old_model)
        self._exptitem_model = model
        if model is not None:
            self.connectModel(model)

    def connectModel(self, model: ExperimentItemModel):
        model.coatPathsChanged.connect(self.onCoatPathsChange)
        model.referenceDataChanged.connect(self.onReferenceDataChange)
        model.substrateDataChanged.connect(self.onSubstrateDataChange)
        model.coatingLayerDataChanged.connect(self.onCoatingLayerDataChange)
        model.experimentDataChanged.connect(self.onExperimentDataChange)
        model.analysisDataChanged.connect(self.onAnalysisDataChange)

    def disconnectModel(self, model: ExperimentItemModel):
        model.coatPathsChanged.disconnect(self.onCoatPathsChange)
        model.referenceDataChanged.disconnect(self.onReferenceDataChange)
        model.substrateDataChanged.disconnect(self.onSubstrateDataChange)
        model.coatingLayerDataChanged.disconnect(self.onCoatingLayerDataChange)
        model.experimentDataChanged.disconnect(self.onExperimentDataChange)
        model.analysisDataChanged.disconnect(self.onAnalysisDataChange)

    @Slot(int, list, ExperimentKind)
    def onCoatPathsChange(self, row: int, paths: List[str], kind: ExperimentKind):
        if row == self.currentExperimentRow():
            self.analysisWorker().setPaths(paths, kind)
            self.workersUpdated.emit(ClassSelection.ANALYSIS)

    @Slot(int, ReferenceArgs, StructuredReferenceArgs)
    def onReferenceDataChange(
        self, row: int, refargs: ReferenceArgs, structrefargs: StructuredReferenceArgs
    ):
        if row == self.currentExperimentRow():
            self.referenceWorker().setStructuredReferenceArgs(structrefargs)
            self.referenceWorker().updateReference()
            self.substrateWorker().setReference(self.referenceWorker().reference())
            self.substrateWorker().updateSubstrate()
            self.experimentWorker().setSubstrate(self.substrateWorker().substrate())
            self.experimentWorker().updateExperiment()
            self.analysisWorker().setExperiment(self.experimentWorker().experiment())
            self.workersUpdated.emit(
                ClassSelection.REFERENCE
                | ClassSelection.SUBSTRATE
                | ClassSelection.EXPERIMENT
                | ClassSelection.ANALYSIS
            )

    @Slot(int, SubstrateArgs, StructuredSubstrateArgs)
    def onSubstrateDataChange(
        self,
        row: int,
        substargs: SubstrateArgs,
        structsubstargs: StructuredSubstrateArgs,
    ):
        if row == self.currentExperimentRow():
            self.substrateWorker().setStructuredSubstrateArgs(structsubstargs)
            self.substrateWorker().updateSubstrate()
            self.experimentWorker().setSubstrate(self.substrateWorker().substrate())
            self.experimentWorker().updateExperiment()
            self.analysisWorker().setExperiment(self.experimentWorker().experiment())
            self.workersUpdated.emit(
                ClassSelection.SUBSTRATE
                | ClassSelection.EXPERIMENT
                | ClassSelection.ANALYSIS
            )

    @Slot(int, CoatingLayerArgs, StructuredCoatingLayerArgs)
    def onCoatingLayerDataChange(
        self,
        row: int,
        layerargs: CoatingLayerArgs,
        structlayerargs: StructuredCoatingLayerArgs,
    ):
        if row == self.currentExperimentRow():
            self.experimentWorker().setStructuredCoatingLayerArgs(structlayerargs)
            self.experimentWorker().updateExperiment()
            self.analysisWorker().setExperiment(self.experimentWorker().experiment())
            self.workersUpdated.emit(
                ClassSelection.EXPERIMENT | ClassSelection.ANALYSIS
            )

    @Slot(int, ExperimentArgs, StructuredExperimentArgs)
    def onExperimentDataChange(
        self,
        row: int,
        exptargs: ExperimentArgs,
        structexptargs: StructuredExperimentArgs,
    ):
        if row == self.currentExperimentRow():
            self.experimentWorker().setStructuredExperimentArgs(structexptargs)
            self.experimentWorker().updateExperiment()
            self.analysisWorker().setExperiment(self.experimentWorker().experiment())
            self.workersUpdated.emit(
                ClassSelection.EXPERIMENT | ClassSelection.ANALYSIS
            )

    @Slot(int, AnalysisArgs)
    def onAnalysisDataChange(self, row: int, analargs: AnalysisArgs):
        if row == self.currentExperimentRow():
            self.analysisWorker().setAnalysisArgs(analargs)
            self.workersUpdated.emit(ClassSelection.ANALYSIS)

    @Slot(int)
    def setCurrentExperimentRow(self, row: int):
        self._currentExperimentRow = row
        model = self.experimentItemModel()
        if model is None:
            return
        refpath = model.data(model.index(row, model.Col_ReferencePath))
        if not refpath:
            img = None
        else:
            img = cv2.imread(refpath)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.referenceWorker().setImage(img)

        coatpaths = model.coatPaths(row)
        exptkind = model.experimentKind(row)
        self.analysisWorker().setPaths(coatpaths, exptkind)

        refargs = model.data(
            model.index(row, model.Col_Reference),
            model.Role_StructuredArgs,
        )
        self.referenceWorker().setStructuredReferenceArgs(refargs)
        substargs = model.data(
            model.index(row, model.Col_Substrate),
            model.Role_StructuredArgs,
        )
        self.substrateWorker().setStructuredSubstrateArgs(substargs)
        layerargs = model.data(
            model.index(row, model.Col_CoatingLayer),
            model.Role_StructuredArgs,
        )
        self.experimentWorker().setStructuredCoatingLayerArgs(layerargs)
        exptargs = model.data(
            model.index(row, model.Col_Experiment),
            model.Role_StructuredArgs,
        )
        self.experimentWorker().setStructuredExperimentArgs(exptargs)

        self.referenceWorker().updateReference()
        self.substrateWorker().setReference(self.referenceWorker().reference())
        self.substrateWorker().updateSubstrate()
        self.experimentWorker().setSubstrate(self.substrateWorker().substrate())
        self.experimentWorker().updateExperiment()
        self.analysisWorker().setExperiment(self.experimentWorker().experiment())
        self.workersUpdated.emit(
            ClassSelection.REFERENCE
            | ClassSelection.SUBSTRATE
            | ClassSelection.EXPERIMENT
            | ClassSelection.ANALYSIS
        )

    @Slot(VisualizationMode)
    def setVisualizationMode(self, mode: VisualizationMode):
        self.referenceWorker().setVisualizationMode(mode)
        self.substrateWorker().setVisualizationMode(mode)
        self.experimentWorker().setVisualizationMode(mode)
        self.workersUpdated.emit(
            ClassSelection.REFERENCE
            | ClassSelection.SUBSTRATE
            | ClassSelection.EXPERIMENT
        )

    @Slot()
    def onExperimentsRemove(self, rows: List[int]):
        if self.currentExperimentRow() in rows:
            self._currentExperimentRow = -1
            self.referenceWorker().clear()
            self.substrateWorker().clear()
            self.experimentWorker().clear()
            self.analysisWorker().clear()
