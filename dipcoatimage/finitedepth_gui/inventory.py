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

        The model consists of N columns:

        0. Header (with experiment name)
        ... (to be added)
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
        self.experimentItemModel().appendRow(
            QStandardItem(f"Experiment {self.experimentItemModel().rowCount()}")
        )
