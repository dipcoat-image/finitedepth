from dipcoatimage.finitedepth_gui.inventory import ExperimentItemModel
from PySide6.QtWidgets import QWidget
from typing import List


__all__ = [
    "ControlWidget",
]


class ControlWidget(QWidget):
    """Base class for all control widgets."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._exptitem_model = ExperimentItemModel()
        self._currentExperimentRow = -1

    def experimentItemModel(self) -> ExperimentItemModel:
        """Model which holds the experiment item data."""
        return self._exptitem_model

    def currentExperimentRow(self) -> int:
        """Currently activated row from :meth:`experimentItemModel`."""
        return self._currentExperimentRow

    def setExperimentItemModel(self, model: ExperimentItemModel):
        """Set :meth:`experimentItemModel`."""
        self._exptitem_model = model

    def setCurrentExperimentRow(self, row: int):
        self._currentExperimentRow = row

    def onExperimentsRemove(self, rows: List[int]):
        if self.currentExperimentRow() in rows:
            self._currentExperimentRow = -1
