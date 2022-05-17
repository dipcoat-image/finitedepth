from dipcoatimage.finitedepth.analysis import ExperimentKind
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
from PySide6.QtGui import QAction
from PySide6.QtMultimedia import QMediaPlayer
from .controlwidgets import (
    ExperimentWidget,
    ReferenceWidget,
    SubstrateWidget,
    CoatingLayerWidget,
)
from .display import (
    PreviewableNDArrayVideoPlayer,
    ExperimentArrayProcessor,
    MainDisplayWindow,
)
from .inventory import (
    ExperimentInventory,
)
from .workers import (
    WorkerBase,
    ReferenceWorker,
    SubstrateWorker,
    ExperimentVisualizationMode,
    ExperimentWorker,
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

        self._exptvid_arrayprocessor = ExperimentArrayProcessor()
        self._main_display = MainDisplayWindow()
        self._expt_inv = ExperimentInventory()
        self._exptitem_tab = QTabWidget()
        self._prev_tab = None
        self._expt_scroll = QScrollArea()
        self._expt_widget = ExperimentWidget()
        self._ref_scroll = QScrollArea()
        self._ref_widget = ReferenceWidget()
        self._subst_scroll = QScrollArea()
        self._subst_widget = SubstrateWidget()
        self._layer_scroll = QScrollArea()
        self._layer_widget = CoatingLayerWidget()
        self._ref_worker = ReferenceWorker()
        self._subst_worker = SubstrateWorker()
        self._expt_worker = ExperimentWorker()
        self._cwd_button = QPushButton()

        self.setCentralWidget(self.mainDisplayWindow())
        self.mainDisplayWindow().visualizeActionGroup().triggered.connect(
            self.onVisualizeActionsTrigger
        )
        self.mainDisplayWindow().visualizeAction().setChecked(True)
        videoDisplayWidget = self.mainDisplayWindow().videoDisplayWidget()
        videoDisplayWidget.setVideoPlayer(PreviewableNDArrayVideoPlayer(self))
        videoDisplayWidget.setArrayProcessor(self.experimentVideoArrayProcessor())
        videoDisplayWidget.videoSlider().valueChanged.connect(
            self.onVideoSliderValueChange
        )
        self.experimentVideoArrayProcessor().setExperimentWorker(
            self.experimentWorker()
        )

        self.experimentWidget().setExperimentItemModel(
            self.experimentInventory().experimentItemModel()
        )
        self.experimentInventory().experimentListView().activated.connect(
            self.experimentWidget().setCurrentExperimentIndex
        )
        self.referenceWidget().imageChanged.connect(self.referenceWorker().setImage)
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

        self.referenceWorker().setExperimentItemModel(
            self.experimentInventory().experimentItemModel()
        )
        self.experimentInventory().experimentListView().activated.connect(
            self.referenceWorker().setCurrentExperimentIndex
        )
        self.substrateWorker().setExperimentItemModel(
            self.experimentInventory().experimentItemModel()
        )
        self.experimentInventory().experimentListView().activated.connect(
            self.substrateWorker().setCurrentExperimentIndex
        )
        self.experimentWorker().setExperimentItemModel(
            self.experimentInventory().experimentItemModel()
        )
        self.experimentInventory().experimentListView().activated.connect(
            self.experimentWorker().setCurrentExperimentIndex
        )

        self.cwdButton().clicked.connect(self.browseCWD)

        self.setWindowTitle("Coating layer analysis")

        expt_inv_dock = QDockWidget("Experiment inventory")
        expt_inv_dock.setWidget(self.experimentInventory())
        self.addDockWidget(Qt.LeftDockWidgetArea, expt_inv_dock)

        exptitem_dock = QDockWidget("Experiment item")
        self._expt_scroll.setWidgetResizable(True)
        self._expt_scroll.setWidget(self.experimentWidget())
        self.experimentItemTab().addTab(self._expt_scroll, "Experiment")
        self._ref_scroll.setWidgetResizable(True)
        self._ref_scroll.setWidget(self.referenceWidget())
        self.experimentItemTab().addTab(self._ref_scroll, "Reference")
        self._subst_scroll.setWidgetResizable(True)
        self._subst_scroll.setWidget(self.substrateWidget())
        self.experimentItemTab().addTab(self._subst_scroll, "Substrate")
        self._layer_scroll.setWidgetResizable(True)
        self._layer_scroll.setWidget(self.coatingLayerWidget())
        self.experimentItemTab().addTab(self._layer_scroll, "Coating Layer")
        exptitem_dock.setWidget(self.experimentItemTab())
        self.addDockWidget(Qt.BottomDockWidgetArea, exptitem_dock)

        self.cwdButton().setText("Browse")
        self.statusBar().addPermanentWidget(self.cwdButton())
        self.statusBar().showMessage(os.getcwd())

    def experimentVideoArrayProcessor(self) -> ExperimentArrayProcessor:
        """
        Array processor to visualize experiment video stream from local file.
        """
        return self._exptvid_arrayprocessor

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

    def referenceWorker(self) -> ReferenceWorker:
        """Worker for API with :class:`SubstrateReferenceBase`."""
        return self._ref_worker

    def substrateWorker(self) -> SubstrateWorker:
        """Worker for API with :class:`SubstrateBase`."""
        return self._subst_worker

    def experimentWorker(self) -> ExperimentWorker:
        """Worker for API with :class:`ExperimentBase`."""
        return self._expt_worker

    def cwdButton(self) -> QPushButton:
        """Button to open file dialog to change current directory."""
        return self._cwd_button

    @Slot(QAction)
    def onVisualizeActionsTrigger(self, act: QAction):
        state = act.isChecked()
        self.referenceWorker().setVisualizationMode(state)
        self.substrateWorker().setVisualizationMode(state)
        if state and act == self.mainDisplayWindow().visualizeAction():
            self.experimentWorker().setVisualizationMode(
                ExperimentVisualizationMode.FULL
            )
        elif state and act == self.mainDisplayWindow().fastVisualizeAction():
            self.experimentWorker().setVisualizationMode(
                ExperimentVisualizationMode.FAST
            )
        else:
            self.experimentWorker().setVisualizationMode(
                ExperimentVisualizationMode.OFF
            )
        self.experimentVideoArrayProcessor().setVisualizationMode(state)
        self.refreshImage.emit()

    @Slot()
    def onVideoSliderValueChange(self):
        player = self.mainDisplayWindow().videoDisplayWidget().videoPlayer()
        if player.playbackState() != QMediaPlayer.PlayingState:
            self.experimentWorker().updateLayerShapeGenerator()

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

    def determineDisplayWidget(
        self, tab: QScrollArea, exptkind: ExperimentKind
    ) -> QWidget:
        """
        Return a widget to display the visualization result when *tab*
        is activated for :meth:`experimentItemTab` and *exptkind* is
        set for :meth:`experimentWorker`.
        """
        if tab.widget() == self.referenceWidget():
            ret: QWidget = self.mainDisplayWindow().imageDisplayWidget()
        elif tab.widget() == self.substrateWidget():
            ret = self.mainDisplayWindow().imageDisplayWidget()
        else:
            if exptkind == ExperimentKind.SingleImageExperiment:
                ret = self.mainDisplayWindow().imageDisplayWidget()
            elif (
                exptkind == ExperimentKind.VideoExperiment
                or exptkind == ExperimentKind.MultiImageExperiment
            ):
                ret = self.mainDisplayWindow().videoDisplayWidget()
            else:
                ret = self.mainDisplayWindow().imageDisplayWidget()
        return ret

    def determineWorker(self, tab: QScrollArea) -> WorkerBase:
        """
        Return a worker corresponding to *tab* in :meth:`experimentItemTab`.
        """
        if tab.widget() == self.referenceWidget():
            ret: WorkerBase = self.referenceWorker()
        elif tab.widget() == self.substrateWidget():
            ret = self.substrateWorker()
        else:
            ret = self.experimentWorker()
        return ret

    def setCWD(self, path: str):
        os.chdir(path)
        self.statusBar().showMessage(path)
