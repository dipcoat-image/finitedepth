from dipcoatimage.finitedepth.analysis import (
    ReferenceArgs,
    SubstrateArgs,
    CoatingLayerArgs,
    ExperimentArgs,
)
import enum
from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (
    QWidget,
    QListView,
    QToolButton,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
)
from .controlwidgets import (
    ReferenceWidgetData,
    SubstrateWidgetData,
    CoatingLayerWidgetData,
    ExperimentWidgetData,
)

try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias


__all__ = ["ExperimentItemModelColumns", "ExperimentItemModel", "ExperimentInventory"]


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
    """

    EXPERIMENT_NAME = 0
    REFERENCE_PATH = 1
    COAT_PATHS = 2
    REFERENCE = 3
    SUBSTRATE = 4
    COATINGLAYER = 5
    EXPERIMENT = 6


class ExperimentItemModel(QStandardItemModel):
    """
    Model to store the data which makes up :class:`ExperimentData`.

    .. rubric:: ColumnNames

    :attr:`ColumnNames` is an integer enum which holds the column names.

    0. EXPERIMENT_NAME
        Name of the experiment.
    1. REFERENCE_PATH
        Path to reference file. Corresponds to :attr:`ExperimentData.ref_path`.
    2. COAT_PATHS
        Paths to coated substrate files. Each path is stored in children rows.
        Corresponds to :attr:`ExperimentData.coat_paths`.
    3. REFERENCE
        Data to construct reference object. Data are stored in :meth:`data` as
        tuple (:class:`ReferenceWidgetData`, :class:`ReferenceArgs`) with
        ``Qt.UserRole``. Corresponds to :attr:`ExperimentData.reference`.
    4. SUBSTRATE
        Data to construct substrate object. Data are stored in :meth:`data` as
        tuple (:class:`SubstrateWidgetData`, :class:`SubstrateArgs`) with
        ``Qt.UserRole``. Corresponds to :attr:`ExperimentData.substrate`.
    5. COATINGLAYER
        Data to construct coating layer object. Data are stored in :meth:`data`
        as tuple (:class:`CoatingLayerWidgetData`, :class:`CoatingLayerArgs`)
        with ``Qt.UserRole``. Corresponds to :attr:`ExperimentData.coatinglayer`.
    6. EXPERIMENT
        Data to construct experiment object. Data are stored in :meth:`data` as
        tuple (:class:`ExperimentWidgetData`, :class:`ExperimentArgs`) with
        ``Qt.UserRole``. Corresponds to :attr:`ExperimentData.experiment`.
    """

    ColumnNames: TypeAlias = ExperimentItemModelColumns
    Col_ExperimentName = ExperimentItemModelColumns.EXPERIMENT_NAME
    Col_ReferencePath = ExperimentItemModelColumns.REFERENCE_PATH
    Col_CoatPaths = ExperimentItemModelColumns.COAT_PATHS
    Col_Reference = ExperimentItemModelColumns.REFERENCE
    Col_Substrate = ExperimentItemModelColumns.SUBSTRATE
    Col_CoatingLayer = ExperimentItemModelColumns.COATINGLAYER
    Col_Experiment = ExperimentItemModelColumns.EXPERIMENT

    def __init__(self, rows: int = 0, columns: int = len(ColumnNames), parent=None):
        super().__init__(rows, columns, parent)

    def data(self, index, role=Qt.DisplayRole):
        ret = super().data(index, role)
        if role == Qt.UserRole and ret is None and not index.parent().isValid():
            if index.column() == self.Col_Reference:
                ret = (ReferenceWidgetData(), ReferenceArgs())
            elif index.column() == self.Col_Substrate:
                ret = (SubstrateWidgetData(), SubstrateArgs())
            elif index.column() == self.Col_CoatingLayer:
                ret = (CoatingLayerWidgetData(), CoatingLayerArgs())
            elif index.column() == self.Col_Experiment:
                ret = (ExperimentWidgetData(), ExperimentArgs())
        return ret


class ExperimentInventory(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._item_model = ExperimentItemModel()
        self._list_view = QListView()
        self._add_button = QToolButton()
        self._delete_button = QPushButton()

        self.experimentListView().setSelectionMode(QListView.ExtendedSelection)
        self.experimentListView().setEditTriggers(QListView.SelectedClicked)
        self.experimentListView().setModel(self.experimentItemModel())
        self.addButton().clicked.connect(self.onAddButtonClicked)

        layout = QVBoxLayout()
        layout.addWidget(self.experimentListView())
        button_layout = QHBoxLayout()
        self.addButton().setText("Add")
        self.addButton().setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.deleteButton().setText("Delete")
        self.deleteButton().setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button_layout.addWidget(self.addButton())
        button_layout.addWidget(self.deleteButton())
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def experimentItemModel(self) -> ExperimentItemModel:
        return self._item_model

    def experimentListView(self) -> QListView:
        return self._list_view

    def addButton(self) -> QToolButton:
        return self._add_button

    def deleteButton(self) -> QPushButton:
        return self._delete_button

    @Slot()
    def onAddButtonClicked(self):
        """Add new row to :meth:`experimentItemModel`."""
        items = [
            QStandardItem() for _ in range(self.experimentItemModel().columnCount())
        ]
        items[ExperimentItemModel.Col_ExperimentName].setText(
            f"Experiment {self.experimentItemModel().rowCount()}"
        )
        self.experimentItemModel().appendRow(items)
