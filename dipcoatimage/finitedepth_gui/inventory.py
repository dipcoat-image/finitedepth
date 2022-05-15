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


__all__ = ["ExperimentInventory"]


class ExperimentInventory(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._item_model = QStandardItemModel(0, 3)
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

        Columns correspond to the members of :class:`ExperimentData`:

        0. Header (with experiment name)
        1. Reference path
        2. Coated substrate file paths (stored in child items)
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
