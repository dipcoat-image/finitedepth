import os
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QMainWindow,
    QDockWidget,
    QPushButton,
    QFileDialog,
)
from .controlwidgets import (
    ExperimentWidget,
    ReferenceWidget,
    SubstrateWidget,
    CoatingLayerWidget,
    AnalysisWidget,
    MasterControlWidget,
)
from .display import MainDisplayWindow
from .inventory import ExperimentInventory
from .workers import MasterWorker


__all__ = ["AnalysisGUI"]


class AnalysisGUI(QMainWindow):
    """
    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui import AnalysisGUI
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     window = AnalysisGUI()
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP
    """

    refreshImage = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._main_display = MainDisplayWindow()
        self._expt_inv = ExperimentInventory()
        self._master_controlwidget = MasterControlWidget()
        self._master_worker = MasterWorker()
        self._cwd_button = QPushButton()

        self.masterControlWidget().setExperimentItemModel(
            self.experimentInventory().experimentItemModel()
        )
        self.experimentInventory().experimentListView().activated.connect(
            self.masterControlWidget().setCurrentExperimentIndex
        )

        self.masterControlWidget().drawROIToggled.connect(
            self.mainDisplayWindow().toggleROIDraw
        )
        self.masterControlWidget().selectedClassChanged.connect(
            self.mainDisplayWindow().exposeDisplayWidget
        )
        self.masterControlWidget().selectedClassChanged.connect(
            self.masterWorker().setVisualizingWorker
        )

        self.masterControlWidget().imageChanged.connect(
            self.masterWorker().setReferenceImage
        )
        self.masterWorker().setExperimentItemModel(
            self.experimentInventory().experimentItemModel()
        )
        self.experimentInventory().experimentListView().activated.connect(
            self.masterWorker().setCurrentExperimentIndex
        )
        self.masterWorker().visualizedImageChanged.connect(
            self.mainDisplayWindow().displayImage
        )

        self.cwdButton().clicked.connect(self.browseCWD)

        self.setWindowTitle("Coating layer analysis")
        self.setCentralWidget(self.mainDisplayWindow())

        expt_inv_dock = QDockWidget("Experiment inventory")
        expt_inv_dock.setWidget(self.experimentInventory())
        self.addDockWidget(Qt.LeftDockWidgetArea, expt_inv_dock)

        exptitem_dock = QDockWidget("Experiment item")
        exptitem_dock.setWidget(self.masterControlWidget())
        self.addDockWidget(Qt.BottomDockWidgetArea, exptitem_dock)

        self.cwdButton().setText("Browse")
        self.statusBar().addPermanentWidget(self.cwdButton())
        self.statusBar().showMessage(os.getcwd())

    def mainDisplayWindow(self) -> MainDisplayWindow:
        """Main window which includes all display widgets."""
        return self._main_display

    def experimentInventory(self) -> ExperimentInventory:
        """Widget to display the experiment items.."""
        return self._expt_inv

    def masterControlWidget(self) -> MasterControlWidget:
        """
        Tab widget which contains the widgets to control the data of activated
        experiment item.
        """
        return self._master_controlwidget

    def experimentWidget(self) -> ExperimentWidget:
        """Widget to manage data for experiment class."""
        return self.masterControlWidget().experimentWidget()

    def referenceWidget(self) -> ReferenceWidget:
        """Widget to manage data for substrate reference class."""
        return self.masterControlWidget().referenceWidget()

    def substrateWidget(self) -> SubstrateWidget:
        """Widget to manage data for substrate class."""
        return self.masterControlWidget().substrateWidget()

    def coatingLayerWidget(self) -> CoatingLayerWidget:
        """Widget to manage data for coating layer class."""
        return self.masterControlWidget().coatingLayerWidget()

    def analysisWidget(self) -> AnalysisWidget:
        """Widget to manage analysis."""
        return self.masterControlWidget().analysisWidget()

    def masterWorker(self) -> MasterWorker:
        """Object which contains workers for the experiment."""
        return self._master_worker

    def cwdButton(self) -> QPushButton:
        """Button to open file dialog to change current directory."""
        return self._cwd_button

    @Slot()
    def browseCWD(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select current working directory",
            "./",
            options=QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog,
        )
        if path:
            self.setCWD(path)

    def setCWD(self, path: str):
        os.chdir(path)
        self.statusBar().showMessage(path)
