import dipcoatimage.finitedepth_gui
from dipcoatimage.finitedepth_gui.workers import VisualizationMode
import os
from PySide6.QtCore import Signal, QSize, Slot
from PySide6.QtGui import QActionGroup, QAction, QIcon
from PySide6.QtWidgets import QToolBar, QComboBox, QLineEdit, QToolButton, QMenu


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
        self._cameras_combo = QComboBox()
        self._streamAction = QAction()
        self._capturepath_ledit = QLineEdit()
        self._captureFormat_combo = QComboBox()
        self._captureButton = QToolButton()
        self._captureAndAddAction = QAction()
        self._recordpath_ledit = QLineEdit()
        self._recordFormat_combo = QComboBox()
        self._recordButton = QToolButton()
        self._recordAndAddAction = QAction()

        self.visualizeActionGroup().triggered.connect(self.onVisualizeActionTrigger)
        self.visualizeActionGroup().setExclusionPolicy(
            QActionGroup.ExclusionPolicy.ExclusiveOptional
        )
        self.visualizeActionGroup().addAction(self.visualizeAction())
        self.visualizeActionGroup().addAction(self.fastVisualizeAction())
        self.visualizeAction().setCheckable(True)
        self.fastVisualizeAction().setCheckable(True)

        visIcon = QIcon()
        visIcon.addFile(get_icons_path("visualize.svg"), QSize(24, 24))
        self.visualizeAction().setIcon(visIcon)

        fastVisIcon = QIcon()
        fastVisIcon.addFile(get_icons_path("fastvisualize.svg"), QSize(24, 24))
        self.fastVisualizeAction().setIcon(fastVisIcon)

        self.camerasComboBox().setPlaceholderText("Select camera")
        self.streamingAction().setToolTip("Toggle live streaming")
        streamActionIcon = QIcon()
        streamActionIcon.addFile(get_icons_path("camera.svg"), QSize(24, 24))
        self.streamingAction().setIcon(streamActionIcon)

        self.capturePathLineEdit().setPlaceholderText("Image capture path")
        self.captureFormatComboBox().setPlaceholderText("Image format")
        self.captureButton().setToolTip("Capture image")
        captureActionIcon = QIcon()
        captureActionIcon.addFile(get_icons_path("capture.svg"), QSize(24, 24))
        self.captureButton().setIcon(captureActionIcon)
        self.captureButton().setMenu(QMenu(self))
        self.captureAndAddAction().setText("Capture and add to data")
        self.captureAndAddAction().setCheckable(True)
        self.captureButton().menu().addAction(self.captureAndAddAction())
        self.captureButton().setPopupMode(QToolButton.MenuButtonPopup)

        self.recordPathLineEdit().setPlaceholderText("Video record path")
        self.recordFormatComboBox().setPlaceholderText("Video format")
        self.recordButton().setCheckable(True)
        self.recordButton().setToolTip("Record video")
        recordActionIcon = QIcon()
        recordActionIcon.addFile(get_icons_path("record.svg"), QSize(24, 24))
        self.recordButton().setIcon(recordActionIcon)
        self.recordButton().setMenu(QMenu(self))
        self.recordAndAddAction().setText("Record and add to data")
        self.recordAndAddAction().setCheckable(True)
        self.recordButton().menu().addAction(self.recordAndAddAction())
        self.recordButton().setPopupMode(QToolButton.MenuButtonPopup)

        self.addAction(self.visualizeAction())
        self.addAction(self.fastVisualizeAction())
        self.addSeparator()
        self.addSeparator()
        self.addWidget(self.camerasComboBox())
        self.addAction(self.streamingAction())
        self.addSeparator()
        self.addWidget(self.capturePathLineEdit())
        self.addWidget(self.captureFormatComboBox())
        self.addWidget(self.captureButton())
        self.addSeparator()
        self.addWidget(self.recordPathLineEdit())
        self.addWidget(self.recordFormatComboBox())
        self.addWidget(self.recordButton())

    def visualizeActionGroup(self) -> QActionGroup:
        return self._visualizeActionGroup

    def visualizeAction(self) -> QAction:
        """Action to toggle visualization mode."""
        return self._visualizeAction

    def fastVisualizeAction(self) -> QAction:
        """Action to toggle fast visualization mode."""
        return self._fastVisualizeAction

    def camerasComboBox(self) -> QComboBox:
        return self._cameras_combo

    def streamingAction(self) -> QAction:
        """Action to toggle live streaming mode."""
        return self._streamAction

    def capturePathLineEdit(self) -> QLineEdit:
        """Line edit to set the image capture path."""
        return self._capturepath_ledit

    def captureFormatComboBox(self) -> QComboBox:
        """Combo box to specify the captured image format."""
        return self._captureFormat_combo

    def captureButton(self) -> QToolButton:
        """Action to capture image from camera."""
        return self._captureButton

    def captureAndAddAction(self) -> QAction:
        """Checkable action to make captured image added to data."""
        return self._captureAndAddAction

    def recordPathLineEdit(self) -> QLineEdit:
        """Line edit to set the video record path."""
        return self._recordpath_ledit

    def recordFormatComboBox(self) -> QComboBox:
        """Combo box to specify the recorded video format."""
        return self._recordFormat_combo

    def recordButton(self) -> QToolButton:
        """Action to record video from camera."""
        return self._recordButton

    def recordAndAddAction(self) -> QAction:
        """Checkable action to make recorded video added to data."""
        return self._recordAndAddAction

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
