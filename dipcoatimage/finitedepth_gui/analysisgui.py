from dipcoatimage.finitedepth.analysis import (
    ReferenceArgs,
    SubstrateArgs,
    CoatingLayerArgs,
    ExperimentArgs,
)
from PySide6.QtCore import Qt, Slot, QModelIndex
from PySide6.QtGui import QStandardItem
from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QScrollArea,
    QDockWidget,
    QDataWidgetMapper,
    QWidget,
)
from .controlwidgets import (
    ExperimentWidget,
    ReferenceWidget,
    SubstrateWidget,
    CoatingLayerWidget,
)
from .inventory import (
    StructuredExperimentArgs,
    StructuredReferenceArgs,
    StructuredSubstrateArgs,
    StructuredCoatingLayerArgs,
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
        self._ref_worker = ReferenceWorker()
        self._subst_worker = SubstrateWorker()
        self._expt_worker = ExperimentWorker()

        self.setCentralWidget(QWidget())

        self.experimentDataMapper().setModel(
            self.experimentInventory().experimentItemModel()
        )
        self.experimentDataMapper().addMapping(
            self.experimentWidget().experimentNameLineEdit(),
            ExperimentItemModel.Col_ExperimentName,
        )
        self.experimentDataMapper().addMapping(
            self.referenceWidget().pathLineEdit(),
            ExperimentItemModel.Col_ReferencePath,
        )
        self.experimentInventory().experimentListView().activated.connect(
            self.onExperimentActivation
        )
        self.experimentWidget().dataChanged.connect(
            self.onStructuredExperimentArgsChange
        )
        self.referenceWidget().dataChanged.connect(self.onStructuredReferenceArgsChange)
        self.substrateWidget().dataChanged.connect(self.onStructuredSubstrateArgsChange)
        self.coatingLayerWidget().dataChanged.connect(
            self.onStructuredCoatingLayerArgsChange
        )

        self.referenceWidget().imageChanged.connect(self.referenceWorker().setImage)
        self.experimentInventory().experimentItemModel().itemChanged.connect(
            self.onExperimentItemChange
        )
        self.experimentInventory().experimentItemModel().rowsInserted.connect(
            self.onExperimentItemRowsChange
        )
        self.experimentInventory().experimentItemModel().rowsRemoved.connect(
            self.onExperimentItemRowsChange
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
        """Update the experiment data to widgets."""
        model = self.experimentInventory().experimentItemModel()

        self.experimentDataMapper().setCurrentIndex(index.row())
        self.experimentWidget().pathsView().setModel(model)
        self.experimentWidget().pathsView().setRootIndex(
            model.index(index.row(), ExperimentItemModel.Col_CoatPaths)
        )
        self.experimentWidget().setExperimentArgs(
            model.data(
                model.index(index.row(), ExperimentItemModel.Col_Experiment),
                Qt.UserRole,
            )[1]
        )
        self.referenceWidget().setReferenceArgs(
            model.data(
                model.index(index.row(), ExperimentItemModel.Col_Reference),
                Qt.UserRole,
            )[1]
        )
        self.substrateWidget().setSubstrateArgs(
            model.data(
                model.index(index.row(), ExperimentItemModel.Col_Substrate),
                Qt.UserRole,
            )[1]
        )
        self.coatingLayerWidget().setCoatingLayerArgs(
            model.data(
                model.index(index.row(), ExperimentItemModel.Col_CoatingLayer),
                Qt.UserRole,
            )[1]
        )

        self.referenceWorker().clear()
        self.substrateWorker().clear()
        self.experimentWorker().clear()

        self.referenceWorker().setStructuredReferenceArgs(
            model.data(
                model.index(index.row(), ExperimentItemModel.Col_Reference),
                Qt.UserRole,
            )[0]
        )
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

    @Slot(StructuredExperimentArgs, ExperimentArgs)
    def onStructuredExperimentArgsChange(
        self, widgetdata: StructuredExperimentArgs, exptargs: ExperimentArgs
    ):
        """Update the data from :meth:`experimentWidget` to current model."""
        index = self.experimentInventory().experimentListView().currentIndex()
        if index.isValid():
            model = self.experimentInventory().experimentItemModel()
            model.setData(
                model.index(index.row(), ExperimentItemModel.Col_Experiment),
                (widgetdata, exptargs),
                Qt.UserRole,  # type: ignore[arg-type]
            )

    @Slot(StructuredReferenceArgs, ReferenceArgs)
    def onStructuredReferenceArgsChange(
        self, widgetdata: StructuredReferenceArgs, refargs: ReferenceArgs
    ):
        """Update the data from :meth:`referenceWidget` to current model."""
        index = self.experimentInventory().experimentListView().currentIndex()
        if index.isValid():
            model = self.experimentInventory().experimentItemModel()
            model.setData(
                model.index(index.row(), ExperimentItemModel.Col_Reference),
                (widgetdata, refargs),
                Qt.UserRole,  # type: ignore[arg-type]
            )

    @Slot(StructuredSubstrateArgs, SubstrateArgs)
    def onStructuredSubstrateArgsChange(
        self, widgetdata: StructuredSubstrateArgs, substargs: SubstrateArgs
    ):
        """Update the data from :meth:`substrateWidget` to current model."""
        index = self.experimentInventory().experimentListView().currentIndex()
        if index.isValid():
            model = self.experimentInventory().experimentItemModel()
            model.setData(
                model.index(index.row(), ExperimentItemModel.Col_Substrate),
                (widgetdata, substargs),
                Qt.UserRole,  # type: ignore[arg-type]
            )

    @Slot(StructuredCoatingLayerArgs, CoatingLayerArgs)
    def onStructuredCoatingLayerArgsChange(
        self, widgetdata: StructuredCoatingLayerArgs, layerargs: CoatingLayerArgs
    ):
        """Update the data from :meth:`coatingLayerWidget` to current model."""
        index = self.experimentInventory().experimentListView().currentIndex()
        if index.isValid():
            model = self.experimentInventory().experimentItemModel()
            model.setData(
                model.index(index.row(), ExperimentItemModel.Col_CoatingLayer),
                (widgetdata, layerargs),
                Qt.UserRole,  # type: ignore[arg-type]
            )

    @Slot(QStandardItem)
    def onExperimentItemChange(self, item: QStandardItem):
        pass

    @Slot(QModelIndex, int, int)
    def onExperimentItemRowsChange(self, index: QModelIndex, first: int, last: int):
        """Apply the change of experiment file paths to experiment worker."""
        pass
