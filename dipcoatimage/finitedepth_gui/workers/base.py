from dipcoatimage.finitedepth_gui.core import VisualizationMode
from PySide6.QtCore import QObject


__all__ = [
    "WorkerBase",
]


class WorkerBase(QObject):
    """Base class for all worker objects."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._visualize_mode = VisualizationMode.FULL

    def visualizationMode(self) -> VisualizationMode:
        return self._visualize_mode

    def setVisualizationMode(self, mode: VisualizationMode):
        self._visualize_mode = mode
