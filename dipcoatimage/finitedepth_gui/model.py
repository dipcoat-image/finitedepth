"""
Experiment data model
=====================

V2 for inventory.py
"""

import copy
from dipcoatimage.finitedepth import ExperimentData
from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt, Signal
from typing import Optional, Any, Union


__all__ = [
    "ExperimentDataItem",
    "ExperimentDataModel",
]


class ExperimentDataItem(object):
    """
    Internal data node for :class:`ExperimentDataModel` which represents tree
    structure with one column.

    Data for the node can be get and set by :meth:`data` and :meth:`setData`.
    Tree structure can be accessed by :meth:`child` and :meth:`parent`, which
    are modified by :meth:`setParent` and :meth:`remove`.

    """

    __slots__ = ("_data", "_children", "_parent")

    def __init__(self):
        self._data = dict()
        self._children = []
        self._parent = None

    def data(self, role: Union[Qt.ItemDataRole, int]) -> Any:
        if isinstance(role, int):
            role = Qt.ItemDataRole(role)
        if role == Qt.EditRole:
            role = Qt.DisplayRole
        return self._data.get(role, None)

    def setData(self, role: Union[Qt.ItemDataRole, int], data: Any):
        if isinstance(role, int):
            role = Qt.ItemDataRole(role)
        if role == Qt.EditRole:
            role = Qt.DisplayRole
        self._data[role] = data

    def columnCount(self) -> int:
        return 1

    def rowCount(self) -> int:
        return len(self._children)

    def child(self, index: int) -> Optional["ExperimentDataItem"]:
        """Get *index*-th subitem, or None if the index is invalid."""
        if len(self._children) > index:
            return self._children[index]
        return None

    def parent(self) -> Optional["ExperimentDataItem"]:
        return self._parent

    def setParent(self, parent: Optional["ExperimentDataItem"], insertIndex: int = -1):
        """
        Set *self* as *insertIndex*-th child of *parent*.

        If *insertIndex* is -1, *self* becomes the last child of *parent*.
        If *parent* is None, *self* is no longer a child of another node and
        becomes the top node of its tree structure.
        """
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
        """
        Destroy all tree structure under *index*-th child.
        """
        if index < len(self._children):
            orphan = self._children.pop(index)
            orphan._parent = None
            while orphan._children:
                orphan.remove(0)

    def copyDataTo(self, other: "ExperimentDataItem"):
        """
        Copy the data of *self* and its children to *other* and its children.
        """
        other._data = copy.deepcopy(self._data)
        for (subSelf, subOther) in zip(self._children, other._children):
            subSelf.copyDataTo(subOther)


class ExperimentDataModel(QAbstractItemModel):
    """
    Model to store the data for :class:`ExperimentData`.

    Each row on the top level has vertical tree structure which can be used to
    construct :class:`ExperimentData`.

    Subtree structure under each top-level row is not modifiable, except the rows
    which represents the coating layer image paths.
    """

    # https://stackoverflow.com/a/57129496/11501976

    Role_RefPath = Qt.DisplayRole
    Role_CoatPath = Qt.DisplayRole
    Role_RefArgs = Qt.UserRole
    Role_SubstArgs = Qt.UserRole
    Role_LayerArgs = Qt.UserRole
    Role_ExptArgs = Qt.UserRole
    Role_AnalysisArgs = Qt.UserRole

    ROW_REFPATH = 0
    ROW_COATPATHS = 1
    ROW_REFERENCE = 2
    ROW_SUBSTRATE = 3
    ROW_COATINGLAYER = 4
    ROW_EXPERIMENT = 5
    ROW_ANALYSIS = 6

    activatedIndexChanged = Signal(QModelIndex)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rootItem = ExperimentDataItem()
        self._activatedIndex = QModelIndex()

    @classmethod
    def _itemFromExperimentData(cls, exptData: ExperimentData) -> ExperimentDataItem:
        item = ExperimentDataItem()

        refPathItem = ExperimentDataItem()
        refPathItem.setData(cls.Role_RefPath, exptData.ref_path)
        refPathItem.setParent(item)

        coatPathsItem = ExperimentDataItem()
        for path in exptData.coat_paths:
            coatPathItem = ExperimentDataItem()
            coatPathItem.setData(cls.Role_CoatPath, path)
            coatPathItem.setParent(coatPathsItem)
        coatPathsItem.setParent(item)

        refArgs = exptData.reference
        refArgsItem = ExperimentDataItem()
        refArgsItem.setData(cls.Role_RefArgs, refArgs)
        refArgsItem.setParent(item)

        substArgs = exptData.substrate
        substArgsItem = ExperimentDataItem()
        substArgsItem.setData(cls.Role_SubstArgs, substArgs)
        substArgsItem.setParent(item)

        layerArgs = exptData.coatinglayer
        layerArgsItem = ExperimentDataItem()
        layerArgsItem.setData(cls.Role_LayerArgs, layerArgs)
        layerArgsItem.setParent(item)

        exptArgs = exptData.experiment
        exptArgsItem = ExperimentDataItem()
        exptArgsItem.setData(cls.Role_ExptArgs, exptArgs)
        exptArgsItem.setParent(item)

        analysisArgs = exptData.analysis
        analysisArgsItem = ExperimentDataItem()
        analysisArgsItem.setData(cls.Role_AnalysisArgs, analysisArgs)
        analysisArgsItem.setParent(item)

        return item

    def columnCount(self, index=QModelIndex()):
        if not index.isValid():
            dataItem = self._rootItem
        else:
            dataItem = index.internalPointer()
        return dataItem.columnCount()

    def rowCount(self, index=QModelIndex()):
        if not index.isValid():
            dataItem = self._rootItem
        else:
            dataItem = index.internalPointer()
        return dataItem.rowCount()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        dataItem = index.internalPointer()
        parentDataItem = dataItem.parent()
        if parentDataItem is None or parentDataItem is self._rootItem:
            return QModelIndex()
        grandparentDataItem = parentDataItem.parent()
        if grandparentDataItem is None:
            return QModelIndex()
        row = grandparentDataItem._children.index(parentDataItem)
        return self.createIndex(row, 0, parentDataItem)

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

    def flags(self, index):
        ret = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if index.parent().isValid():
            ret |= Qt.ItemIsEditable
        return ret

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
            activatedIndex = self.activatedIndex()
            activatedRow = activatedIndex.row()
            activatedColumn = activatedIndex.column()
            reactivate = (parent == activatedIndex.parent()) and row <= activatedRow

            self.beginInsertRows(parent, row, row + count - 1)
            for _ in range(count):
                newItem = self._itemFromExperimentData(ExperimentData())
                newItem.setParent(self._rootItem)

            if reactivate:
                newRow = activatedRow + count
                newIndex = self.index(newRow, activatedColumn, parent)
                self.setActivatedIndex(newIndex)
            self.endInsertRows()
            return True
        elif (
            not parent.parent().parent().isValid()
            and parent.row() == self.ROW_COATPATHS
        ):
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
        """
        Copy *count* rows starting with *sourceRow* under parent *sourceParent*
        to row *destinationChild* under parent *destinationParent*.

        Every node of the tree and its data is copied.

        Returns True on successs; otherwise return False.

        """
        if sourceParent != destinationParent:
            return False
        if not sourceParent.isValid():
            activatedIndex = self.activatedIndex()
            activatedRow = activatedIndex.row()
            activatedColumn = activatedIndex.column()
            reactivate = (
                sourceParent == activatedIndex.parent()
            ) and destinationChild <= activatedRow

            newItems = []
            for i in range(count):
                oldItem = self.index(sourceRow + i, 0, sourceParent).internalPointer()
                newItem = self._itemFromExperimentData(ExperimentData())
                oldItem.copyDataTo(newItem)
                newItems.append(newItem)
            self.beginInsertRows(
                sourceParent, destinationChild, destinationChild + count - 1
            )
            for item in reversed(newItems):
                item.setParent(self._rootItem, destinationChild)

            if reactivate:
                newRow = activatedRow + count
                newIndex = self.index(newRow, activatedColumn, sourceParent)
                self.setActivatedIndex(newIndex)
            self.endInsertRows()
            return True
        elif (
            not sourceParent.parent().parent().isValid()
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
            # to avoid reference issue, decide whether activated index should be
            # changed before destroying data structure
            activatedIndex = self.activatedIndex()
            activatedRow = activatedIndex.row()
            activatedColumn = activatedIndex.column()
            reactivate = (parent == activatedIndex.parent()) and row <= activatedRow

            self.beginRemoveRows(parent, row, row + count - 1)
            for _ in range(count):
                self._rootItem.remove(row)

            if reactivate:
                if activatedRow < row + count:
                    newIndex = QModelIndex()
                else:
                    newRow = activatedRow - count
                    newIndex = self.index(newRow, activatedColumn, parent)
                self.setActivatedIndex(newIndex)
            self.endRemoveRows()
            return True
        elif (
            not parent.parent().parent().isValid()
            and parent.row() == self.ROW_COATPATHS
        ):
            self.beginRemoveRows(parent, row, row + count - 1)
            dataItem = parent.internalPointer()
            for _ in range(count):
                dataItem.remove(row)
            self.endRemoveRows()
            return True
        return False

    def activatedIndex(self) -> QModelIndex:
        return self._activatedIndex

    def setActivatedIndex(self, index: QModelIndex):
        if index.parent().isValid():
            index = QModelIndex()
        self._activatedIndex = index
        self.activatedIndexChanged.emit(index)
