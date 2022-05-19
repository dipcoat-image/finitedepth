import dipcoatimage.finitedepth_gui
from dipcoatimage.finitedepth_gui.workers import (
    VisualizationMode,
)
import os
from PySide6.QtCore import Signal, QSize, Slot
from PySide6.QtGui import (
    QActionGroup,
    QAction,
    QIcon,
)
from PySide6.QtWidgets import QToolBar


__all__ = [
    "DisplayWidgetToolBar",
    "get_icons_path",
]


class DisplayWidgetToolBar(QToolBar):
    """Toolbar to controll the overall display."""

    visualizationModeChanged = Signal(VisualizationMode)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._visualizeActionGroup = QActionGroup(self)
        self._visualizeAction = QAction("Toggle visualization")
        self._fastVisualizeAction = QAction("Toggle fast visualization")

        self.visualizeActionGroup().triggered.connect(self.onVisualizeActionTrigger)
        self.visualizeActionGroup().setExclusionPolicy(
            QActionGroup.ExclusionPolicy.ExclusiveOptional
        )
        self.visualizeActionGroup().addAction(self.visualizeAction())
        self.visualizeActionGroup().addAction(self.fastVisualizeAction())

        self.visualizeAction().setCheckable(True)
        self.addAction(self.visualizeAction())
        self.fastVisualizeAction().setCheckable(True)
        self.addAction(self.fastVisualizeAction())

        self.initUI()

    def initUI(self):
        visIcon = QIcon()
        visIcon.addFile(get_icons_path("visualize.svg"), QSize(24, 24))
        self.visualizeAction().setIcon(visIcon)

        fastVisIcon = QIcon()
        fastVisIcon.addFile(get_icons_path("fastvisualize.svg"), QSize(24, 24))
        self.fastVisualizeAction().setIcon(fastVisIcon)

    def visualizeActionGroup(self) -> QActionGroup:
        return self._visualizeActionGroup

    def visualizeAction(self) -> QAction:
        """Action to toggle visualization mode."""
        return self._visualizeAction

    def fastVisualizeAction(self) -> QAction:
        """Action to toggle fast visualization mode."""
        return self._fastVisualizeAction

    @Slot(QAction)
    def onVisualizeActionTrigger(self, action: QAction):
        if action.isChecked() and action == self.visualizeAction():
            mode = VisualizationMode.FULL
        elif action.isChecked() and action == self.fastVisualizeAction():
            mode = VisualizationMode.FAST
        else:
            mode = VisualizationMode.OFF
        self.visualizationModeChanged.emit(mode)

    def setVisualizeActionToggleState(self, mode: VisualizationMode):
        if mode == VisualizationMode.OFF:
            self.visualizeAction().setChecked(False)
            self.fastVisualizeAction().setChecked(False)
        elif mode == VisualizationMode.FAST:
            self.visualizeAction().setChecked(False)
            self.fastVisualizeAction().setChecked(True)
        elif mode == VisualizationMode.FULL:
            self.fastVisualizeAction().setChecked(False)
            self.visualizeAction().setChecked(True)


def get_icons_path(*paths: str) -> str:
    """
    Get the absolute path to the directory where the icon files are
    stored.

    Parameters
    ==========

    paths
        Subpaths under ``dipcoatimage/finitedepth_gui/icons/`` directory.

    Returns
    =======

    path
        Absolute path to the icon depending on the user's system.

    """
    module_path = os.path.abspath(dipcoatimage.finitedepth_gui.__file__)
    module_path = os.path.split(module_path)[0]
    sample_dir = os.path.join(module_path, "icons")
    sample_dir = os.path.normpath(sample_dir)
    sample_dir = os.path.normcase(sample_dir)

    path = os.path.join(sample_dir, *paths)
    return path
