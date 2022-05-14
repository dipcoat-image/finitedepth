from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QWidget,
    QListWidget,
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
        self._list_widget = QListWidget()
        self._add_button = QToolButton()
        self._delete_button = QPushButton()

        layout = QVBoxLayout()
        layout.addWidget(self.experimentListWidget())
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

    def experimentListWidget(self) -> QListWidget:
        return self._list_widget

    def addButton(self) -> QToolButton:
        return self._add_button

    def deleteButton(self) -> QPushButton:
        return self._delete_button
