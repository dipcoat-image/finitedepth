"""
Experiment data model
=====================

V2 for inventory.py
"""

import copy
import dataclasses
from dipcoatimage.finitedepth import ExperimentData
from dipcoatimage.finitedepth.util import DataclassProtocol
from PySide6.QtCore import (
    QAbstractItemModel,
    QModelIndex,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QWidget,
    QListView,
    QToolButton,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMenu,
    QSizePolicy,
)
from typing import Optional, Any, List, Type


__all__ = [
    "ExperimentDataItem",
    "ExperimentDataModel",
    "ExperimentListWidget",
]


class ExperimentDataItem(object):
    """
    Internal data item for :class:`ExperimentDataModel`.
    """

    def __init__(self):
        self._data = dict()
        self._children = []
        self._parent = None

    def data(self, role: Qt.ItemDataRole) -> Any:
        return self._data.get(role, None)

    def setData(self, role: Qt.ItemDataRole, data: Any):
        self._data[role] = data

    def children(self) -> List["ExperimentDataItem"]:
        return self._children

    def child(self, index):
        if len(self._children) > index:
            return self._children[index]
        return None

    def parent(self) -> Optional["ExperimentDataItem"]:
        return self._parent

    def setParent(self, parent: Optional["ExperimentDataItem"]):
        old_parent = self.parent()
        if old_parent is not None:
            old_parent._children.remove(self)
        if parent is not None:
            parent._children.append(self)
        self._parent = parent

    def remove(self, index: int):
        if index < len(self._children):
            orphan = self._children.pop(index)
            orphan._parent = None
            while orphan._children:
                orphan.remove(0)

    @classmethod
    def fromDataclass(cls, dcls: Type[DataclassProtocol]):
        inst = cls()
        for field in dataclasses.fields(dcls):
            t = field.type
            if dataclasses.is_dataclass(t):
                subItem = cls.fromDataclass(t)
            else:
                subItem = cls()
            subItem.setParent(inst)
        return inst

    def copyDataTo(self, other: "ExperimentDataItem"):
        other._data = copy.deepcopy(self._data)
        for (subSelf, subOther) in zip(self._children, other._children):
            subSelf.copyDataTo(subOther)


class ExperimentDataModel(QAbstractItemModel):
    """Model to store the data for :class:`ExperimentData`."""

    # https://stackoverflow.com/a/57129496/11501976

    dataClass = ExperimentData

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rootItem = ExperimentDataItem()

    def columnCount(self, index=QModelIndex()):
        if not index.isValid():
            return 1
        elif not index.parent().isValid():
            return 1
        elif not index.parent().parent().isValid():
            return 1
        return 0

    def rowCount(self, index=QModelIndex()):
        if not index.isValid():
            dataItem = self._rootItem
        else:
            dataItem = index.internalPointer()
        return len(dataItem._children)

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        dataItem = index.internalPointer()
        parentDataItem = dataItem.parent()
        if parentDataItem is self._rootItem:
            return QModelIndex()
        return self.createIndex(index.row(), index.column(), parentDataItem)

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        if not parent.isValid():
            parentItem = self._rootItem
        else:
            parentItem = parent.internalPointer()
        dataItem = parentItem.child(row)
        if dataItem is not None:
            return self.createIndex(row, column, dataItem)
        return QModelIndex()

    def data(self, index, role=Qt.DisplayRole):
        dataItem = index.internalPointer()
        if isinstance(dataItem, ExperimentDataItem):
            return dataItem.data(role)
        return None

    def setData(self, index, value, role=Qt.EditRole):
        dataItem = index.internalPointer()
        if isinstance(dataItem, ExperimentDataItem):
            dataItem.setData(role, value)
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def insertRows(self, row, count, parent=QModelIndex()):
        self.beginInsertRows(parent, row, row + count - 1)
        if not parent.isValid():
            for _ in range(count):
                newItem = ExperimentDataItem.fromDataclass(ExperimentData)
                newItem.setParent(self._rootItem)
        else:
            for _ in range(count):
                newItem = ExperimentDataItem()
                newItem.setParent(parent.internalPointer())
        self.endInsertRows()
        return True

    def copyRows(self, row, count, parent=QModelIndex()):
        """Copy and append the rows."""
        maxRow = self.rowCount()
        self.beginInsertRows(parent, maxRow, maxRow + count - 1)
        if not parent.isValid():
            for i in range(count):
                oldItem = self.index(row + i, 0, parent).internalPointer()
                newItem = ExperimentDataItem.fromDataclass(ExperimentData)
                newItem.setParent(self._rootItem)
                oldItem.copyDataTo(newItem)
        else:
            for i in range(count):
                oldItem = self.index(row + i, 0, parent)
                newItem = ExperimentDataItem()
                newItem.setParent(parent.internalPointer())
                oldItem.copyDataTo(newItem)
        self.endInsertRows()
        return True

    def removeRows(self, row, count, parent=QModelIndex()):
        self.beginRemoveRows(parent, row, row + count - 1)
        for _ in range(count):
            self._rootItem.remove(row)
        self.endRemoveRows()
        return True


class ExperimentListWidget(QWidget):
    """
    Widget to display the list of experiments.

    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import (ExperimentDataModel,
    ...     ExperimentListWidget)
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     window = ExperimentListWidget()
    ...     window.setModel(ExperimentDataModel())
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    activated = Signal(QModelIndex)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._listView = QListView()
        self._addButton = QToolButton()
        self._addButton.setMenu(QMenu(self))
        copyAction = self._addButton.menu().addAction("Copy selected items")
        self._deleteButton = QPushButton()

        self._listView.setSelectionMode(QListView.ExtendedSelection)
        self._listView.activated.connect(self.activated)
        self._addButton.clicked.connect(self.addNewExperiment)
        copyAction.triggered.connect(self.copySelectedExperiments)
        self._deleteButton.clicked.connect(self.deleteSelectedExperiments)

        self._addButton.setPopupMode(QToolButton.MenuButtonPopup)
        self._addButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._deleteButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._addButton.setText("Add")
        self._deleteButton.setText("Delete")

        layout = QVBoxLayout()
        layout.addWidget(self._listView)
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self._addButton)
        buttonLayout.addWidget(self._deleteButton)
        layout.addLayout(buttonLayout)
        self.setLayout(layout)

    def model(self) -> Optional[ExperimentDataModel]:
        return self._listView.model()

    def setModel(self, model: Optional[ExperimentDataModel]):
        self._listView.setModel(model)

    @Slot()
    def addNewExperiment(self):
        model = self.model()
        if model is not None:
            rowNum = model.rowCount()
            success = model.insertRow(model.rowCount())
            if success:
                index = model.index(rowNum, 0)
                model.setData(index, "New Experiment", role=Qt.DisplayRole)

    @Slot()
    def copySelectedExperiments(self):
        model = self.model()
        if model is not None:
            rows = [idx.row() for idx in self._listView.selectedIndexes()]
            for i in sorted(rows):
                model.copyRows(i, 1)

    @Slot()
    def deleteSelectedExperiments(self):
        model = self.model()
        if model is not None:
            rows = [idx.row() for idx in self._listView.selectedIndexes()]
            for i in reversed(sorted(rows)):
                model.removeRow(i)
