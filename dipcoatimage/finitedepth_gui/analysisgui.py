from dipcoatimage.finitedepth.analysis import ExperimentArgs
from PySide6.QtCore import Qt, Slot, QModelIndex
from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QScrollArea,
    QDockWidget,
    QDataWidgetMapper,
    QWidget,
)
from .controlwidgets import (
    ExperimentWidgetData,
    ExperimentWidget,
    ReferenceWidget,
    SubstrateWidget,
    CoatingLayerWidget,
)
from .inventory import ExperimentItemModelColumns, ExperimentInventory


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
        self._exptdata_mapper = QDataWidgetMapper()
        self._exptitem_tab = QTabWidget()
        self._expt_scroll = QScrollArea()
        self._expt_widget = ExperimentWidget()
        self._ref_scroll = QScrollArea()
        self._ref_widget = ReferenceWidget()
        self._subst_scroll = QScrollArea()
        self._subst_widget = SubstrateWidget()
        self._layer_scroll = QScrollArea()
        self._layer_widget = CoatingLayerWidget()

        self.setCentralWidget(QWidget())

        self.experimentDataMapper().setModel(
            self.experimentInventory().experimentItemModel()
        )
        self.experimentDataMapper().addMapping(
            self.experimentWidget().experimentNameLineEdit(),
            ExperimentItemModelColumns.EXPERIMENT_NAME,
        )
        self.experimentDataMapper().addMapping(
            self.referenceWidget().pathLineEdit(),
            ExperimentItemModelColumns.REFERENCE_PATH,
        )
        self.experimentInventory().experimentListView().activated.connect(
            self.onExperimentActivation
        )
        self.experimentWidget().dataChanged.connect(self.onExperimentWidgetDataChange)

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

    def experimentDataMapper(self) -> QDataWidgetMapper:
        return self._exptdata_mapper

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

    @Slot(QModelIndex)
    def onExperimentActivation(self, index: QModelIndex):
        """Update the experiment data to widgets."""
        model = self.experimentInventory().experimentItemModel()
        self.experimentDataMapper().setCurrentIndex(index.row())
        self.experimentWidget().pathsView().setModel(model)
        self.experimentWidget().pathsView().setRootIndex(
            model.index(index.row(), ExperimentItemModelColumns.COAT_PATHS)
        )
        self.experimentWidget().setExperimentArgs(
            model.item(index.row(), ExperimentItemModelColumns.EXPERIMENT).data()[1]
        )

    @Slot(ExperimentWidgetData, ExperimentArgs)
    def onExperimentWidgetDataChange(
        self, widgetdata: ExperimentWidgetData, exptargs: ExperimentArgs
    ):
        """Update the data from :meth:`experimentWidget` to current model."""
        index = self.experimentInventory().experimentListView().currentIndex()
        if index.isValid():
            model = self.experimentInventory().experimentItemModel()
            item = model.item(index.row(), ExperimentItemModelColumns.EXPERIMENT)
            item.setData((widgetdata, exptargs))
