"""
Analysis worker
===============

"""

import cattrs
import cv2  # type: ignore
import enum
import numpy as np
import numpy.typing as npt
from dipcoatimage.finitedepth import (
    SubstrateReferenceBase,
    SubstrateBase,
    ExperimentBase,
    ExperimentKind,
    experiment_kind,
    Analyzer,
    ReferenceArgs,
    SubstrateArgs,
    AnalysisArgs,
    ExperimentData,
)
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool
from typing import List, Optional


__all__ = [
    "AnalysisState",
    "AnalysisWorkerSignals",
    "AnalysisWorker",
    "WorkerUpdateFlag",
    "ExperimentWorker",
]


class AnalysisState(enum.Enum):
    Running = 0
    Paused = 1
    Stopped = 2


class AnalysisWorkerSignals(QObject):
    stateChanged = Signal(AnalysisState)
    progressMaximumChanged = Signal(int)
    progressValueChanged = Signal(int)


class AnalysisWorker(QRunnable):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.coat_paths: List[str] = []
        self.experiment: Optional[ExperimentBase] = None
        self.analysisArgs = AnalysisArgs()

        self._state = AnalysisState.Stopped
        self.signals = AnalysisWorkerSignals()

    def state(self) -> AnalysisState:
        return self._state

    def setState(self, state: AnalysisState):
        if self._state != state:
            self._state = state
            self.signals.stateChanged.emit(state)

    def run(self):
        self.setState(AnalysisState.Running)

        if not self.coat_paths or self.experiment is None:
            self.setState(AnalysisState.Stopped)
            return

        exptKind = experiment_kind(self.coat_paths)
        dataPath = self.analysisArgs.data_path
        imagePath = self.analysisArgs.image_path
        videoPath = self.analysisArgs.video_path
        fps = self.analysisArgs.fps

        if (
            exptKind == ExperimentKind.SINGLE_IMAGE
            or exptKind == ExperimentKind.MULTI_IMAGE
        ):
            total = len(self.coat_paths)
            img_gen = (cv2.imread(path) for path in self.coat_paths)
            if fps is None:
                fps = 0

        elif exptKind == ExperimentKind.VIDEO:
            (path,) = self.coat_paths
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            img_gen = (cap.read()[1] for _ in range(total))
            fps = cap.get(cv2.CAP_PROP_FPS)

        else:
            self.setState(AnalysisState.Stopped)
            raise TypeError(f"Unsupported experiment kind: {exptKind}")

        self.signals.progressMaximumChanged.emit(total)

        analyzer = Analyzer(self.coat_paths, self.experiment)
        analysis_gen = analyzer.analysis_generator(
            dataPath, imagePath, videoPath, fps=fps
        )
        next(analysis_gen)

        for i, img in enumerate(img_gen):
            state = self.state()
            if state == AnalysisState.Paused:
                continue
            elif state == AnalysisState.Stopped:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            analysis_gen.send(img)
            self.signals.progressValueChanged.emit(i + 1)
        analysis_gen.send(None)
        self.setState(AnalysisState.Stopped)


class WorkerUpdateFlag(enum.IntFlag):
    """Flag to indicate how the worker should be updated."""

    NULL = 0
    REFIMAGE = 1
    REFERENCE = 2
    SUBSTRATE = 4
    EXPERIMENT = 8
    ANALYSIS = 16


class ExperimentWorker(QObject):
    """Stores structured experiment objects."""

    analysisStateChanged = Signal(AnalysisState)
    analysisProgressMaximumChanged = Signal(int)
    analysisProgressValueChanged = Signal(int)

    def __init__(self, parent):
        super().__init__(parent)

        self.exptData = ExperimentData()
        self.referenceImage = np.empty((0, 0, 0), dtype=np.uint8)
        self.reference: Optional[SubstrateReferenceBase] = None
        self.substrate: Optional[SubstrateBase] = None
        self.experiment: Optional[ExperimentBase] = None

        self.analysisWorker = AnalysisWorker()
        self._analysisState = AnalysisState.Stopped
        self._analysisProgressMaximum = 100
        self._analysisProgressValue = -1

        self.analysisWorker.setAutoDelete(False)
        self.analysisWorker.signals.stateChanged.connect(self._onAnalysisStateChange)
        self.analysisWorker.signals.progressMaximumChanged.connect(
            self._onAnalysisProgressMaximumChange
        )
        self.analysisWorker.signals.progressValueChanged.connect(
            self._onAnalysisProgressValueChange
        )

    def analysisState(self) -> AnalysisState:
        return self._analysisState

    def setAnalysisState(self, state: AnalysisState):
        if self.analysisState() == AnalysisState.Stopped:
            if state == AnalysisState.Running:
                QThreadPool.globalInstance().start(self.analysisWorker)
        else:
            self.analysisWorker.setState(state)

    def analysisProgressMaximum(self) -> int:
        return self._analysisProgressMaximum

    def analysisProgressValue(self) -> int:
        return self._analysisProgressValue

    def setExperimentData(self, exptData: ExperimentData, flag: WorkerUpdateFlag):
        self.exptData = exptData

        if flag & WorkerUpdateFlag.REFIMAGE:
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

        if flag & WorkerUpdateFlag.REFERENCE:
            self.reference = self.constructReference(
                self.referenceImage, exptData.reference
            )

        if flag & WorkerUpdateFlag.SUBSTRATE:
            self.substrate = self.constructSubstrate(self.reference, exptData.substrate)

        if flag & WorkerUpdateFlag.EXPERIMENT:
            if self.substrate is None:
                self.experiment = None
            else:
                layerArgs = exptData.coatinglayer
                exptArgs = exptData.experiment
                try:
                    structuredLayerArgs = layerArgs.as_structured_args()
                    expt = exptArgs.as_experiment(self.substrate, *structuredLayerArgs)
                    if not expt.valid():
                        self.experiment = None
                    else:
                        self.experiment = expt
                except (
                    TypeError,
                    ValueError,
                    ImportError,
                    ModuleNotFoundError,
                    cattrs.ClassValidationError,
                ):
                    self.experiment = None
            self.analysisWorker.experiment = self.experiment

        if flag & WorkerUpdateFlag.ANALYSIS:
            self.analysisWorker.coat_paths = exptData.coat_paths
            self.analysisWorker.analysisArgs = exptData.analysis

    @staticmethod
    def constructReference(
        refImg: npt.NDArray[np.uint8], refArgs: ReferenceArgs
    ) -> Optional[SubstrateReferenceBase]:
        if refImg.size == 0:
            ref = None
        else:
            try:
                ref = refArgs.as_reference(refImg)
                if not ref.valid():
                    ref = None
            except (
                TypeError,
                ValueError,
                ImportError,
                ModuleNotFoundError,
                cattrs.ClassValidationError,
            ):
                ref = None
        return ref

    @staticmethod
    def constructSubstrate(
        ref: Optional[SubstrateReferenceBase], substArgs: SubstrateArgs
    ) -> Optional[SubstrateBase]:
        if ref is None:
            subst = None
        else:
            try:
                subst = substArgs.as_substrate(ref)
                if not subst.valid():
                    subst = None
            except (
                TypeError,
                ValueError,
                ImportError,
                ModuleNotFoundError,
                cattrs.ClassValidationError,
            ):
                subst = None
        return subst

    def _onAnalysisStateChange(self, state: AnalysisState):
        self._analysisState = state
        self.analysisStateChanged.emit(state)

    def _onAnalysisProgressMaximumChange(self, maximum: int):
        self._analysisProgressMaximum = maximum
        self.analysisProgressMaximumChanged.emit(maximum)

    def _onAnalysisProgressValueChange(self, value: int):
        self._analysisProgressValue = value
        self.analysisProgressValueChanged.emit(value)
