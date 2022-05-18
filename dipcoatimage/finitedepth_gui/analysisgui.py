import os
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QScrollArea,
    QDockWidget,
    QPushButton,
    QWidget,
    QFileDialog,
)
from .controlwidgets import (
    ExperimentWidget,
    ReferenceWidget,
    SubstrateWidget,
    CoatingLayerWidget,
    AnalysisWidget,
)
from .display import (
    MainDisplayWindow,
)
from .inventory import (
    ExperimentInventory,
)
from .workers import (
    WorkerBase,
    MasterWorker,
)


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
        self._exptitem_tab = QTabWidget()
        self._expt_widget = ExperimentWidget()
        self._ref_widget = ReferenceWidget()
        self._subst_widget = SubstrateWidget()
        self._layer_widget = CoatingLayerWidget()
        self._anal_widget = AnalysisWidget()
        self._master_worker = MasterWorker()
        self._cwd_button = QPushButton()

        self.experimentItemTab().currentChanged.connect(self.onCurrentTabChange)
        self.experimentWidget().setExperimentItemModel(
            self.experimentInventory().experimentItemModel()
        )
        self.experimentInventory().experimentListView().activated.connect(
            self.experimentWidget().setCurrentExperimentIndex
        )
        self.referenceWidget().setExperimentItemModel(
            self.experimentInventory().experimentItemModel()
        )
        self.experimentInventory().experimentListView().activated.connect(
            self.referenceWidget().setCurrentExperimentIndex
        )
        self.substrateWidget().setExperimentItemModel(
            self.experimentInventory().experimentItemModel()
        )
        self.experimentInventory().experimentListView().activated.connect(
            self.substrateWidget().setCurrentExperimentIndex
        )
        self.coatingLayerWidget().setExperimentItemModel(
            self.experimentInventory().experimentItemModel()
        )
        self.experimentInventory().experimentListView().activated.connect(
            self.coatingLayerWidget().setCurrentExperimentIndex
        )
        self.analysisWidget().setExperimentItemModel(
            self.experimentInventory().experimentItemModel()
        )
        self.experimentInventory().experimentListView().activated.connect(
            self.analysisWidget().setCurrentExperimentIndex
        )

        self.referenceWidget().templateROIDrawButton().toggled.connect(
            self.onTemplateROIDrawButtonToggle
        )
        self.referenceWidget().substrateROIDrawButton().toggled.connect(
            self.onSubstrateROIDrawButtonToggle
        )

        self.referenceWidget().imageChanged.connect(
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
        expt_scroll = QScrollArea()
        expt_scroll.setWidgetResizable(True)
        expt_scroll.setWidget(self.experimentWidget())
        self.experimentItemTab().addTab(expt_scroll, "Experiment")
        ref_scroll = QScrollArea()
        ref_scroll.setWidgetResizable(True)
        ref_scroll.setWidget(self.referenceWidget())
        self.experimentItemTab().addTab(ref_scroll, "Reference")
        subst_scroll = QScrollArea()
        subst_scroll.setWidgetResizable(True)
        subst_scroll.setWidget(self.substrateWidget())
        self.experimentItemTab().addTab(subst_scroll, "Substrate")
        layer_scroll = QScrollArea()
        layer_scroll.setWidgetResizable(True)
        layer_scroll.setWidget(self.coatingLayerWidget())
        self.experimentItemTab().addTab(layer_scroll, "Coating Layer")
        analyze_scroll = QScrollArea()
        analyze_scroll.setWidgetResizable(True)
        analyze_scroll.setWidget(self.analysisWidget())
        self.experimentItemTab().addTab(analyze_scroll, "Analyze")
        exptitem_dock.setWidget(self.experimentItemTab())
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

    def experimentItemTab(self) -> QTabWidget:
        """Tab widget to display the data of activated experiment item."""
        return self._exptitem_tab

    def experimentWidget(self) -> ExperimentWidget:
        """Widget to manage data for experiment class."""
        return self._expt_widget

    def referenceWidget(self) -> ReferenceWidget:
        """Widget to manage data for substrate reference class."""
        return self._ref_widget

    def substrateWidget(self) -> SubstrateWidget:
        """Widget to manage data for substrate class."""
        return self._subst_widget

    def coatingLayerWidget(self) -> CoatingLayerWidget:
        """Widget to manage data for coating layer class."""
        return self._layer_widget

    def analysisWidget(self) -> AnalysisWidget:
        """Widget to manage analysis."""
        return self._anal_widget

    def masterWorker(self) -> MasterWorker:
        """Object which contains workers for the experiment."""
        return self._master_worker

    def cwdButton(self) -> QPushButton:
        """Button to open file dialog to change current directory."""
        return self._cwd_button

    @Slot(bool)
    def onTemplateROIDrawButtonToggle(self, state: bool):
        label = self.mainDisplayWindow().currentDisplayingLabel()
        if state:
            label.addROIModel(
                self.referenceWidget().templateROIWidget().roiModel()
            )
        else:
            label.removeROIModel(
                self.referenceWidget().templateROIWidget().roiModel()
            )

    @Slot(bool)
    def onSubstrateROIDrawButtonToggle(self, state: bool):
        label = self.mainDisplayWindow().currentDisplayingLabel()
        if state:
            label.addROIModel(
                self.referenceWidget().substrateROIWidget().roiModel()
            )
        else:
            label.removeROIModel(
                self.referenceWidget().substrateROIWidget().roiModel()
            )

    def determineDisplay(self, widget: QWidget) -> QWidget:
        return self.mainDisplayWindow().imageDisplayWidget()

    def determineWorker(self, widget: QWidget) -> WorkerBase:
        if not isinstance(widget, QScrollArea):
            ret: WorkerBase = self.masterWorker().experimentWorker()
        else:
            if widget.widget() == self.referenceWidget():
                ret = self.masterWorker().referenceWorker()
            elif widget.widget() == self.substrateWidget():
                ret = self.masterWorker().substrateWorker()
            else:
                ret = self.masterWorker().experimentWorker()
        return ret

    @Slot(int)
    def onCurrentTabChange(self, index: int):
        self.referenceWidget().templateROIDrawButton().setChecked(False)
        self.referenceWidget().substrateROIDrawButton().setChecked(False)
        self.mainDisplayWindow().videoDisplayWidget().videoPlayer().pause()

        widget = self.experimentItemTab().widget(index)
        self.mainDisplayWindow().exposeDisplayWidget(self.determineDisplay(widget))
        self.masterWorker().setVisualizingWorker(self.determineWorker(widget))

        self.masterWorker().emitImage()

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
