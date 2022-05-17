from PySide6.QtCore import Qt, Slot, QModelIndex
from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QScrollArea,
    QDockWidget,
    QWidget,
)
from .controlwidgets import (
    ExperimentWidget,
    ReferenceWidget,
    SubstrateWidget,
    CoatingLayerWidget,
)
from .inventory import (
    ExperimentItemModel,
    ExperimentInventory,
)
from .workers import ReferenceWorker, SubstrateWorker, ExperimentWorker


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

    def __init__(self, parent=None):
        super().__init__(parent)

        self._expt_inv = ExperimentInventory()
        self._exptitem_tab = QTabWidget()
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

        self.setCentralWidget(QWidget())

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
        self.experimentInventory().experimentListView().activated.connect(
            self.onExperimentActivation
        )

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

    @Slot(QModelIndex)
    def onExperimentActivation(self, index: QModelIndex):
        """Update the experiment data to workers."""
        self.substrateWorker().clear()
        self.experimentWorker().clear()

        model = self.experimentInventory().experimentItemModel()
        self.substrateWorker().setStructuredSubstrateArgs(
            model.data(
                model.index(index.row(), ExperimentItemModel.Col_Substrate),
                Qt.UserRole,
            )[0]
        )
        self.experimentWorker().setStructuredCoatingLayerArgs(
            model.data(
                model.index(index.row(), ExperimentItemModel.Col_CoatingLayer),
                Qt.UserRole,
            )[0]
        )
        self.experimentWorker().setStructuredExperimentArgs(
            model.data(
                model.index(index.row(), ExperimentItemModel.Col_Experiment),
                Qt.UserRole,
            )[0]
        )
