from dipcoatimage.finitedepth_gui.core import ClassSelection, VisualizationMode
from dipcoatimage.finitedepth_gui.roimodel import ROIModel
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from .toolbar import DisplayWidgetToolBar
from .roidisplay import NDArrayROILabel
from .videostream import MediaController


__all__ = [
    "MainDisplayWindow",
]


class MainDisplayWindow(QMainWindow):
    """Main window which includes various display widgets."""

    visualizationModeChanged = Signal(VisualizationMode)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._display_toolbar = DisplayWidgetToolBar()
        self._display_label = NDArrayROILabel()
        self._video_controller = MediaController()
        self._selectedClass = ClassSelection.EXPERIMENT

        self.displayToolBar().visualizationModeChanged.connect(
            self.visualizationModeChanged
        )

        self.addToolBar(self.displayToolBar())
        layout = QVBoxLayout()
        layout.addWidget(self.displayLabel())
        layout.addWidget(self.videoController())
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def displayToolBar(self) -> DisplayWidgetToolBar:
        """Toolbar to control display options."""
        return self._display_toolbar

    def displayLabel(self) -> NDArrayROILabel:
        """Label to display the visualization result."""
        return self._display_label

    def videoController(self) -> MediaController:
        return self._video_controller

    @Slot(ROIModel, bool)
    def toggleROIDraw(self, model: ROIModel, state: bool):
        if state:
            self.displayLabel().addROIModel(model)
        else:
            self.displayLabel().removeROIModel(model)

    @Slot(ClassSelection)
    def setSelectedClass(self, select: ClassSelection):
        self._selectedClass = select

    def setVisualizeActionToggleState(self, mode: VisualizationMode):
        self.displayToolBar().setVisualizeActionToggleState(mode)

    def hideVideoController(self, hide: bool):
        self.videoController().setVisible(not hide)
