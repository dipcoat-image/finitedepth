"""
Experiment inventory
====================

Experiment item model and widget to view it.

"""
from dipcoatimage.finitedepth.analysis import (
    ExperimentKind,
    experiment_kind,
    ReferenceArgs,
    SubstrateArgs,
    CoatingLayerArgs,
    ExperimentArgs,
    AnalysisArgs,
)
import enum
from itertools import product
from PySide6.QtCore import Slot, Signal, Qt, QModelIndex
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (
    QWidget,
    QListView,
    QToolButton,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QFileDialog,
)
from typing import List, Optional
from .core import (
    StructuredExperimentArgs,
    StructuredReferenceArgs,
    StructuredSubstrateArgs,
    StructuredCoatingLayerArgs,
)

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias


__all__ = [
    "ExperimentItemModelColumns",
    "ExperimentItemModel",
    "ConfigFileTypeEnum",
    "ExperimentInventory",
]


class ExperimentItemModelColumns(enum.IntEnum):
    """
    Columns for :meth:`ExperimentInventory.experimentItemModel`. Data can make up
    :class:`ExperimentData` instance.

    0. EXPERIMENT_NAME
    1. REFERENCE_PATH
    2. COAT_PATHS
    3. REFERENCE
    4. SUBSTRATE
    5. COATINGLAYER
    6. EXPERIMENT
    7. ANALYSIS
    """

    EXPERIMENT_NAME = 0
    REFERENCE_PATH = 1
    COAT_PATHS = 2
    REFERENCE = 3
    SUBSTRATE = 4
    COATINGLAYER = 5
    EXPERIMENT = 6
    ANALYSIS = 7


class ExperimentItemModel(QStandardItemModel):
    """
    Model to store the data which makes up :class:`ExperimentData`.

    .. rubric:: Column names

    Following attributes are integer enums for colum names.

    0. Col_ExperimentName
        Name of the experiment.
    1. Col_ReferencePath
        Path to reference file. Corresponds to :attr:`ExperimentData.ref_path`.
    2. Col_CoatPaths
        Paths to coated substrate files. Experimnet kind is stored in data with
        default role, and each path is stored in children rows.
        Corresponds to :attr:`ExperimentData.coat_paths`.
    3. Col_Reference
        Data to construct reference object. Data is stored with :attr:`Role_Args`
        as :class:`ReferenceArgs` and with :attr:`Role_StructuredArgs` as
        :class:`StructuredReferenceArgs`.
        Corresponds to :attr:`ExperimentData.reference`.
    4. Col_Substrate
        Data to construct substrate object. Data is stored with :attr:`Role_Args`
        as :class:`SubstrateArgs` and with :attr:`Role_StructuredArgs` as
        :class:`StructuredSubstrateArgs`.
        Corresponds to :attr:`ExperimentData.substrate`.
    5. Col_CoatingLayer
        Data to construct coating layer object. Data is stored with
        :attr:`Role_Args` as :class:`CoatingLayerArgs` and with
        :attr:`Role_StructuredArgs` as :class:`StructuredCoatingLayerArgs`.
        Corresponds to :attr:`ExperimentData.coatinglayer`.
    6. Col_Experiment
        Data to construct experiment object. Data is stored with
        :attr:`Role_Args` as :class:`ExperimentArgs` and with
        :attr:`Role_StructuredArgs` as :class:`StructuredExperimentArgs`.
        Corresponds to :attr:`ExperimentData.experiment`.
    7. Col_Analysis
        Data to analyze experiment. Data is stored in :meth:`data` as
        :attr:`AnalysisArgs` with :attr:`Role_Args`.
        Corresponds to :attr:`ExperimentData.analysis`.
    """

    ColumnNames: TypeAlias = ExperimentItemModelColumns
    Col_ExperimentName = ExperimentItemModelColumns.EXPERIMENT_NAME
    Col_ReferencePath = ExperimentItemModelColumns.REFERENCE_PATH
    Col_CoatPaths = ExperimentItemModelColumns.COAT_PATHS
    Col_Reference = ExperimentItemModelColumns.REFERENCE
    Col_Substrate = ExperimentItemModelColumns.SUBSTRATE
    Col_CoatingLayer = ExperimentItemModelColumns.COATINGLAYER
    Col_Experiment = ExperimentItemModelColumns.EXPERIMENT
    Col_Analysis = ExperimentItemModelColumns.ANALYSIS

    Role_Args = Qt.UserRole
    Role_StructuredArgs = Qt.UserRole + 1  # type: ignore[operator]

    experimentsRemoved = Signal(list)
    coatPathsChanged = Signal(int, list, ExperimentKind)
    referenceDataChanged = Signal(int, ReferenceArgs, StructuredReferenceArgs)
    substrateDataChanged = Signal(int, SubstrateArgs, StructuredSubstrateArgs)
    coatingLayerDataChanged = Signal(int, CoatingLayerArgs, StructuredCoatingLayerArgs)
    experimentDataChanged = Signal(int, ExperimentArgs, StructuredExperimentArgs)
    analysisDataChanged = Signal(int, AnalysisArgs)

    def __init__(self, rows: int = 0, columns: int = len(ColumnNames), parent=None):
        super().__init__(rows, columns, parent)
        self._block_coatPathsChanged = False

        self.itemChanged.connect(self.onItemChange)  # type: ignore[attr-defined]
        self.rowsInserted.connect(self.onRowsChange)  # type: ignore[attr-defined]
        self.rowsMoved.connect(self.onRowsChange)  # type: ignore[attr-defined]
        self.rowsRemoved.connect(self.onRowsChange)  # type: ignore[attr-defined]

    def data(self, index, role=Qt.DisplayRole):
        ret = super().data(index, role)
        if ret is None and not index.parent().isValid():
            if index.column() == self.Col_CoatPaths:
                ret = ExperimentKind.NullExperiment
            if index.column() == self.Col_Reference:
                if role == self.Role_Args:
                    ret = ReferenceArgs()
                elif role == self.Role_StructuredArgs:
                    ret = StructuredReferenceArgs.from_ReferenceArgs(ReferenceArgs())
            elif index.column() == self.Col_Substrate:
                if role == self.Role_Args:
                    ret = SubstrateArgs()
                elif role == self.Role_StructuredArgs:
                    ret = StructuredSubstrateArgs.from_SubstrateArgs(SubstrateArgs())
            elif index.column() == self.Col_CoatingLayer:
                if role == self.Role_Args:
                    ret = CoatingLayerArgs()
                elif role == self.Role_StructuredArgs:
                    ret = StructuredCoatingLayerArgs.from_CoatingLayerArgs(
                        CoatingLayerArgs()
                    )
            elif index.column() == self.Col_Experiment:
                if role == self.Role_Args:
                    ret = ExperimentArgs()
                elif role == self.Role_StructuredArgs:
                    ret = StructuredExperimentArgs.from_ExperimentArgs(ExperimentArgs())
            elif index.column() == self.Col_Analysis:
                if role == self.Role_Args:
                    ret = AnalysisArgs()
        return ret

    @Slot(list)
    def removeExperiments(self, rows: List[int]):
        for i in reversed(sorted(rows)):
            self.removeRow(i)
        self.experimentsRemoved.emit(rows)

    @Slot(QStandardItem)
    def onItemChange(self, item: QStandardItem):
        parent = item.parent()
        if (
            parent is not None
            and parent.parent() is None
            and parent.column() == self.Col_CoatPaths
        ):  # coat path change
            if not self._block_coatPathsChanged:
                paths = []
                for i in range(self.rowCount(parent.index())):
                    index = self.index(i, 0, parent.index())
                    paths.append(self.data(index))
                kind = experiment_kind(paths)
                self.setData(parent.index(), kind)
                self.coatPathsChanged.emit(parent.row(), paths, kind)
        elif parent is None and item.column() == self.Col_Reference:  # ref change
            self.referenceDataChanged.emit(
                item.row(),
                self.data(item.index(), self.Role_Args),
                self.data(item.index(), self.Role_StructuredArgs),
            )
        elif parent is None and item.column() == self.Col_Substrate:  # subst change
            self.substrateDataChanged.emit(
                item.row(),
                self.data(item.index(), self.Role_Args),
                self.data(item.index(), self.Role_StructuredArgs),
            )
        elif parent is None and item.column() == self.Col_CoatingLayer:  # layer change
            self.coatingLayerDataChanged.emit(
                item.row(),
                self.data(item.index(), self.Role_Args),
                self.data(item.index(), self.Role_StructuredArgs),
            )
        elif parent is None and item.column() == self.Col_Experiment:  # expt change
            self.experimentDataChanged.emit(
                item.row(),
                self.data(item.index(), self.Role_Args),
                self.data(item.index(), self.Role_StructuredArgs),
            )
        elif parent is None and item.column() == self.Col_Analysis:  # analysis change
            self.analysisDataChanged.emit(
                item.row(),
                self.data(item.index(), self.Role_Args),
            )

    @Slot(QModelIndex, int, int)
    def onRowsChange(self, parent: QModelIndex, start: int, end: int):
        if (
            parent.isValid()
            and not parent.parent().isValid()
            and parent.column() == self.Col_CoatPaths
        ):
            if not self._block_coatPathsChanged:
                paths = []
                for i in range(self.rowCount(parent)):
                    index = self.index(i, 0, parent)
                    paths.append(self.data(index))
                kind = experiment_kind(paths)
                self.setData(parent, kind)
                self.coatPathsChanged.emit(parent.row(), paths, kind)

    def coatPaths(self, exptRow: int) -> List[str]:
        parent = self.index(exptRow, self.Col_CoatPaths)
        paths = []
        for i in range(self.rowCount(parent)):
            index = self.index(i, 0, parent)
            paths.append(self.data(index))
        return paths

    def experimentKind(self, exptRow: int) -> ExperimentKind:
        return self.data(self.index(exptRow, self.Col_CoatPaths))

    def setCoatPaths(self, exptRow: int, paths: List[str]):
        """
        Set the paths to coated substrate files of *exptRow*-th experiment to
        *paths*.

        This method signals *coatPathsChanged* only once.
        """
        self._block_coatPathsChanged = True
        parent = self.index(exptRow, self.Col_CoatPaths)
        self.removeRows(0, self.rowCount(parent), parent)
        for i, path in enumerate(paths):
            self.insertRow(i, parent)
            self.setData(self.index(i, 0, parent), path)
        kind = experiment_kind(paths)
        self.setData(parent, kind)
        self._block_coatPathsChanged = False
        self.coatPathsChanged.emit(parent.row(), paths, kind)

    def copyExperiment(self, row: int) -> List[QStandardItem]:
        def recursiveCopy(item: QStandardItem):
            ret = QStandardItem(item)
            for row, col in product(range(item.rowCount()), range(item.columnCount())):
                child = QStandardItem(item.child(row, col))
                ret.setChild(row, col, child)
            return ret

        items = [recursiveCopy(self.item(row, c)) for c in range(self.columnCount())]
        return items


class ConfigFileTypeEnum(enum.Enum):
    """Enum of supported file types. Values are file filters."""

    JSON = "JSON (*.json)"
    YAML = "YAML (*.yml)"


class ExperimentInventory(QWidget):

    experimentRowActivated = Signal(int)
    experimentsRemoved = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._exptcount = 0
        self._item_model = None
        self._list_view = QListView()
        self._add_button = QToolButton()
        self._delete_button = QPushButton()
        self._import_button = QPushButton()
        self._export_button = QPushButton()

        self.experimentListView().setSelectionMode(QListView.ExtendedSelection)
        self.experimentListView().setEditTriggers(QListView.SelectedClicked)
        self.experimentListView().activated.connect(self.onViewIndexActivated)
        self.addButton().setMenu(QMenu(self))
        copyAction = self.addButton().menu().addAction("")  # text set later
        self.addButton().setPopupMode(QToolButton.MenuButtonPopup)
        self.addButton().clicked.connect(self.addNewExperiment)
        copyAction.triggered.connect(self.copySelected)
        self.deleteButton().clicked.connect(self.deleteExperiment)

        layout = QVBoxLayout()
        layout.addWidget(self.experimentListView())
        button_layout = QHBoxLayout()
        self.addButton().setText("Add")
        self.addButton().setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        copyAction.setText("Copy selected items")
        self.deleteButton().setText("Delete")
        self.deleteButton().setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.importButton().setText("Import")
        self.importButton().clicked.connect(self.openImportDialog)
        self.exportButton().setText("Export")
        self.exportButton().clicked.connect(self.openExportDialog)
        button_layout.addWidget(self.addButton())
        button_layout.addWidget(self.deleteButton())
        layout.addLayout(button_layout)
        layout.addWidget(self.importButton())
        layout.addWidget(self.exportButton())
        self.setLayout(layout)

    def experimentItemModel(self) -> Optional[ExperimentItemModel]:
        return self._item_model

    def experimentListView(self) -> QListView:
        return self._list_view

    def addButton(self) -> QToolButton:
        return self._add_button

    def deleteButton(self) -> QPushButton:
        return self._delete_button

    def importButton(self) -> QPushButton:
        return self._import_button

    def exportButton(self) -> QPushButton:
        return self._export_button

    def setExperimentItemModel(self, model: Optional[ExperimentItemModel]):
        """Set :meth:`experimentItemModel`."""
        old_model = self.experimentItemModel()
        if old_model is not None:
            self.disconnectModel(old_model)
        self._item_model = model
        if model is not None:
            self.connectModel(model)

    def connectModel(self, model: ExperimentItemModel):
        model.experimentsRemoved.connect(self.experimentsRemoved)
        self.experimentListView().setModel(model)

    def disconnectModel(self, model: ExperimentItemModel):
        model.experimentsRemoved.disconnect(self.experimentsRemoved)
        self.experimentListView().setModel(None)  # type: ignore[arg-type]

    @Slot()
    def addNewExperiment(self):
        """Add new row to :meth:`experimentItemModel`."""
        model = self.experimentItemModel()
        if model is not None:
            items = [QStandardItem() for _ in range(model.columnCount())]
            items[model.Col_ExperimentName].setText(f"Experiment {self._exptcount}")
            model.appendRow(items)
            self._exptcount += 1

    @Slot()
    def deleteExperiment(self):
        model = self.experimentItemModel()
        if model is not None:
            rows = [idx.row() for idx in self.experimentListView().selectedIndexes()]
            model.removeExperiments(rows)

    def activateExperiment(self, index: int):
        self.experimentListView().setCurrentIndex(
            self.experimentListView().model().index(index, 0)
        )
        self.experimentListView().activated.emit(  # type: ignore[attr-defined]
            self.experimentListView().currentIndex()
        )

    @Slot(QModelIndex)
    def onViewIndexActivated(self, index: QModelIndex):
        if not index.parent().isValid():
            self.experimentRowActivated.emit(index.row())

    @Slot()
    def copySelected(self):
        model = self.experimentItemModel()
        if model is not None:
            rows = [idx.row() for idx in self.experimentListView().selectedIndexes()]
            for row in sorted(rows):
                items = model.copyExperiment(row)
                name = items[model.Col_ExperimentName].text()
                items[model.Col_ExperimentName].setText(f"{name} (copied)")
                model.appendRow(items)
                self._exptcount += 1

    @Slot()
    def openImportDialog(self):
        filters = ";;".join([e.value for e in ConfigFileTypeEnum])
        fileNames, selectedFilter = QFileDialog.getOpenFileNames(
            self,
            "Select configuration files",
            "./",
            filters,
            options=QFileDialog.DontUseNativeDialog,
        )
        if fileNames:
            self.importItems(fileNames, ConfigFileTypeEnum(selectedFilter))

    def importItems(self, fileNames: List[str], selectedFilter: ConfigFileTypeEnum):
        pass

    @Slot()
    def openExportDialog(self):
        filters = ";;".join([e.value for e in ConfigFileTypeEnum])
        fileName, selectedFilter = QFileDialog.getSaveFileName(
            self,
            "Save as configuration file",
            "./",
            filters,
            options=QFileDialog.DontUseNativeDialog,
        )
        if fileName:
            self.exportItems(fileName, ConfigFileTypeEnum(selectedFilter))

    def exportItems(self, fileName: str, selectedFilter: ConfigFileTypeEnum):
        pass
