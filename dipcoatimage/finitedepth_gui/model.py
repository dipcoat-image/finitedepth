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
)
from typing import Optional, Any, List, Type


__all__ = [
    "ExperimentDataItem",
    "ExperimentDataModel",
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

    def setParent(self, parent: Optional["ExperimentDataItem"], insertIndex: int = -1):
        old_parent = self.parent()
        if old_parent is not None:
            old_parent._children.remove(self)
        if parent is not None:
            if insertIndex == -1:
                parent._children.append(self)
            else:
                parent._children.insert(insertIndex, self)
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

    ROW_COATPATHS = 1

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
        if not parent.isValid():
            self.beginInsertRows(parent, row, row + count - 1)
            for _ in range(count):
                newItem = ExperimentDataItem.fromDataclass(ExperimentData)
                newItem.setParent(self._rootItem)
            self.endInsertRows()
            return True
        elif not parent.parent().isValid() and parent.row() == self.ROW_COATPATHS:
            self.beginInsertRows(parent, row, row + count - 1)
            for _ in range(count):
                newItem = ExperimentDataItem()
                newItem.setParent(parent.internalPointer())
            self.endInsertRows()
            return True
        return False

    def copyRows(
        self,
        sourceParent: QModelIndex,
        sourceRow: int,
        count: int,
        destinationParent: QModelIndex,
        destinationChild: int,
    ) -> bool:
        if sourceParent != destinationParent:
            return False
        if not sourceParent.isValid():
            newItems = []
            for i in range(count):
                oldItem = self.index(sourceRow + i, 0, sourceParent).internalPointer()
                newItem = ExperimentDataItem.fromDataclass(ExperimentData)
                oldItem.copyDataTo(newItem)
                newItems.append(newItem)
            self.beginInsertRows(
                sourceParent, destinationChild, destinationChild + count - 1
            )
            for item in reversed(newItems):
                item.setParent(self._rootItem, destinationChild)
            self.endInsertRows()
            return True
        elif (
            not sourceParent.parent().isValid()
            and sourceParent.row() == self.ROW_COATPATHS
        ):
            newItems = []
            for i in range(count):
                oldItem = self.index(sourceRow + i, 0, sourceParent).internalPointer()
                newItem = ExperimentDataItem()
                oldItem.copyDataTo(newItem)
                newItems.append(newItem)
            self.beginInsertRows(
                sourceParent, destinationChild, destinationChild + count - 1
            )
            parentDataItem = destinationParent.internalPointer()
            for item in reversed(newItems):
                item.setParent(parentDataItem, destinationChild)
            self.endInsertRows()
        return False

    def removeRows(self, row, count, parent=QModelIndex()):
        if not parent.isValid():
            self.beginRemoveRows(parent, row, row + count - 1)
            for _ in range(count):
                self._rootItem.remove(row)
            self.endRemoveRows()
            return True
        elif not parent.parent().isValid() and parent.row() == self.ROW_COATPATHS:
            self.beginRemoveRows(parent, row, row + count - 1)
            dataItem = parent.internalPointer()
            for _ in range(count):
                dataItem.remove(row)
            self.endRemoveRows()
            return True
        return False
