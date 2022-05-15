import enum
from PySide6.QtCore import Slot
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


__all__ = ["ExperimentItemModelColumns", "ExperimentInventory"]


class ExperimentItemModelColumns(enum.IntEnum):
    """
    Columns for :meth:`ExperimentInventory.experimentItemModel`. Data can make up
    :class:`ExperimentData` instance.

    0. EXPERIMENT_NAME
        Name of the experiment.
    1. REFERENCE_PATH
        Path to reference file.
        Corresponds to :attr:`ExperimentData.ref_path`.
    2. COAT_PATHS
        Paths to coated substrate files. This is the root item, and each path is
        stored in children item.
        Corresponds to :attr:`ExperimentData.coat_paths`.
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


class ExperimentInventory(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._item_model = QStandardItemModel(0, len(ExperimentItemModelColumns))
        self._list_view = QListView()
        self._add_button = QToolButton()
        self._delete_button = QPushButton()

        self.experimentListView().setSelectionMode(QListView.ExtendedSelection)
        self.experimentListView().setEditTriggers(QListView.SelectedClicked)
        self.experimentListView().setModel(self.experimentItemModel())
        self.addButton().clicked.connect(self.addItem)

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

    def experimentItemModel(self) -> QStandardItemModel:
        """
        Model to store the data which makes up :class:`ExperimentData`.

        Columns are described in :class:`ExperimentItemModelColumns`.
        """
        return self._item_model

    def experimentListView(self) -> QListView:
        return self._list_view

    def addButton(self) -> QToolButton:
        return self._add_button

    def deleteButton(self) -> QPushButton:
        return self._delete_button

    @Slot()
    def addItem(self):
        items = [
            QStandardItem() for _ in range(self.experimentItemModel().columnCount())
        ]
        items[0].setText(f"Experiment {self.experimentItemModel().rowCount()}")
        self.experimentItemModel().appendRow(items)
