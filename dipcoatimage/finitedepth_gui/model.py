"""
Experiment data model
=====================

V2 for inventory.py
"""

import dataclasses
from dipcoatimage.finitedepth import ExperimentData
from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt


__all__ = [
    "ExperimentDataItem",
    "ExperimentDataModel",
]


class ExperimentDataItem(object):
    """
    Internal data item for :class:`ExperimentDataModel`.
    """

    def __init__(self, parent=None):
        self._data = dict()
        self._children = []
        if parent is not None:
            parent._children.append(self)
        self._parent = parent

    def data(self, role):
        return self._data.get(role, None)

    def setData(self, role, data):
        self._data[role] = data

    def children(self):
        return self._children

    def child(self, index):
        if len(self._children) > index:
            return self._children[index]
        return None

    def parent(self):
        return self._parent

    def remove(self, index):
        if index < len(self._children):
            orphan = self._children.pop(index)
            orphan._parent = None
            while orphan._children:
                orphan.remove(0)


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
            return True
        return False

    def insertRows(self, row, count, parent=QModelIndex()):
        if not parent.isValid():
            self.beginInsertRows(parent, row, row + count - 1)
            for _ in range(count):
                newItem = ExperimentDataItem(self._rootItem)
                for field in dataclasses.fields(ExperimentData):
                    subItem = ExperimentDataItem(newItem)
                    if dataclasses.is_dataclass(field.type):
                        for _ in dataclasses.fields(field.type):
                            ExperimentDataItem(subItem)
            self.endInsertRows()
            return True
        else:
            self.beginInsertRows(parent, row, row + count - 1)
            for _ in range(count):
                ExperimentDataItem(parent.internalPointer())
            self.endInsertRows()
            return True

    def removeRows(self, row, count, parent=QModelIndex()):
        self.beginRemoveRows(parent, row, row + count - 1)
        for _ in range(count):
            self._rootItem.remove(row)
        self.endRemoveRows()
        return True
