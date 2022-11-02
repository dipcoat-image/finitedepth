"""
Experiment data model
=====================

V2 for inventory.py
"""

import copy
import enum
from dipcoatimage.finitedepth import ExperimentData
from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt, Signal
from typing import Optional, Any, Union, Tuple


__all__ = [
    "ExperimentDataItem",
    "IndexRole",
    "ExperimentDataModel",
]


class ExperimentDataItem(object):
    """
    Internal data node for :class:`ExperimentDataModel` which represents tree
    structure with one column.

    Data for the node can be get and set by :meth:`data` and :meth:`setData` with
    ``Qt.ItemDataRole``.

    Tree structure can be accessed by :meth:`child` and :meth:`parent`, which
    are modified by :meth:`setParent` and :meth:`remove`.

    """

    __slots__ = ("_data", "_children", "_parent")

    def __init__(self):
        self._data = dict()
        self._children = []
        self._parent = None

    def data(self, role: Union[Qt.ItemDataRole, int]) -> Any:
        """
        Get the data stored with *role*.

        ``EditRole`` and ``DisplayRole`` are treated as identical keys.
        """
        if isinstance(role, int):
            role = Qt.ItemDataRole(role)
        if role == Qt.EditRole:
            role = Qt.DisplayRole
        return self._data.get(role, None)

    def setData(self, role: Union[Qt.ItemDataRole, int], data: Any):
        """
        Set the data with *role*.

        ``EditRole`` and ``DisplayRole`` are treated as identical keys.
        """
        if isinstance(role, int):
            role = Qt.ItemDataRole(role)
        if role == Qt.EditRole:
            role = Qt.DisplayRole
        self._data[role] = data

    def columnCount(self) -> int:
        """Get the number of sub-columns, which is always 1."""
        return 1

    def rowCount(self) -> int:
        """Get the number of sub-rows, which is the number of children."""
        return len(self._children)

    def child(self, index: int) -> Optional["ExperimentDataItem"]:
        """Get *index*-th subitem, or ``None`` if the index is invalid."""
        if len(self._children) > index:
            return self._children[index]
        return None

    def childIndex(self, child: "ExperimentDataItem") -> Tuple[int, int]:
        """
        Return the row and the column of *child* in *self*.

        Column value is always 0 because the tree structure is one-dimensional.
        """
        row = self._children.index(child)
        col = 0
        return (row, col)

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


class IndexRole(enum.Enum):
    """Role of the ``QModelIndex`` of :class:`ExperimentDataModel`."""

    UNKNOWN = 0
    EXPTDATA = 1
    REFPATH = 2
    COATPATHS = 3
    COATPATH = 4
    REFARGS = 5
    SUBSTARGS = 6
    LAYERARGS = 7
    EXPTARGS = 8
    ANALYSISARGS = 9


class ExperimentDataModel(QAbstractItemModel):
    """
    Model to store the data for :class:`ExperimentData`.

    This model has row-based tree structure. Every level has only one column.

    Structure of the model is strictly defined and each index has its role to
    store certain data. The roles are defined in :class:`IndexRole` and can be
    queried by :meth:`whatsThisIndex`. To get the index with certain role, use
    :meth:`getIndexFor`.

    There can be multiple indices with :obj:`IndexRole.EXPTDATA`, each storing
    the data for a single :class:`ExperimentData` instance under its children.
    The structure is defined as follows:

    * EXPTDATA (can be inserted/removed)
        * REFPATH
        * COATPATHS
            * COATPATH (can be inserted/removed)
        * REFARGS
        * SUBSTARGS
        * LAYERARGS
        * EXPTARGS
        * ANALYSISARGS

    Only the rows with :obj:`IndexRole.EXPTDATA` and :obj:`IndexRole.COATPATH`
    can be inserted/removed. In other words, number of rows under the index with
    :obj:`IndexRole.EXPTDATA` is always 7.

    Class attributes with ``Role_[...]`` represents the item data role which is
    used to store the data. Therefore the data can be successfully retrieved by
    using :class:`IndexRole` and ``Role_[...]``. For example, the reference path
    is stored in the index with :obj:`IndexRole.REFPATH` by :attr:`Role_RefPath`.

    A single index with :obj:`IndexRole.EXPTDATA` can be activated to be shown
    by the views. Currently activated index can be get by :meth:`activatedIndex`.

    """

    # https://stackoverflow.com/a/57129496/11501976

    Role_RefPath = Qt.DisplayRole
    Role_CoatPath = Qt.DisplayRole
    Role_RefArgs = Qt.UserRole
    Role_SubstArgs = Qt.UserRole
    Role_LayerArgs = Qt.UserRole
    Role_ExptArgs = Qt.UserRole
    Role_AnalysisArgs = Qt.UserRole

    Row_RefPath = 0
    Row_CoatPaths = 1
    Row_RefArgs = 2
    Row_SubstArgs = 3
    Row_LayerArgs = 4
    Row_ExptArgs = 5
    Row_AnalysisArgs = 6

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
        row, col = grandparentDataItem.childIndex(parentDataItem)
        return self.createIndex(row, col, parentDataItem)

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
        if self.whatsThisIndex(index) != IndexRole.EXPTDATA:
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
        elif self.whatsThisIndex(parent) == IndexRole.COATPATHS:
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
        elif self.whatsThisIndex(sourceParent) == IndexRole.COATPATHS:
            parentDataItem = destinationParent.internalPointer()
            if not isinstance(parentDataItem, ExperimentDataItem):
                return False
            newItems = []
            for i in range(count):
                oldItem = self.index(sourceRow + i, 0, sourceParent).internalPointer()
                newItem = ExperimentDataItem()
                oldItem.copyDataTo(newItem)
                newItems.append(newItem)
            self.beginInsertRows(
                sourceParent, destinationChild, destinationChild + count - 1
            )
            for item in reversed(newItems):
                item.setParent(parentDataItem)
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
        elif self.whatsThisIndex(parent) == IndexRole.COATPATHS:
            self.beginRemoveRows(parent, row, row + count - 1)
            dataItem = parent.internalPointer()
            for _ in range(count):
                dataItem.remove(row)
            self.endRemoveRows()
            return True
        return False

    def activatedIndex(self) -> QModelIndex:
        """
        Currently activated item.

        Only the item with :obj:`IndexRole.EXPTDATA` can be activated. If no item
        is activated, returns invalid index.

        Can be set by :meth:`setActivatedIndex`.
        """
        return self._activatedIndex

    def setActivatedIndex(self, index: QModelIndex):
        """
        Changes the activated index.

        Only the item with :obj:`IndexRole.EXPTDATA` can be activated. If *index*
        cannot be activated, an invalid index is set instead.

        Emits :attr:`activatedIndexChanged` signal.
        """
        if self.whatsThisIndex(index) != IndexRole.EXPTDATA:
            index = QModelIndex()
        self._activatedIndex = index
        self.activatedIndexChanged.emit(index)

    @classmethod
    def whatsThisIndex(cls, index: QModelIndex) -> IndexRole:
        """Return the role of *index* in the model."""
        if not isinstance(index.model(), cls):
            return IndexRole.UNKNOWN

        indexLevel = -1
        _index = index
        while _index.isValid():
            _index = _index.parent()
            indexLevel += 1
        row, col = index.row(), index.column()

        if indexLevel == 0 and col == 0:
            return IndexRole.EXPTDATA
        if indexLevel == 1 and col == 0:
            row = index.row()
            if row == cls.Row_RefPath:
                return IndexRole.REFPATH
            if row == cls.Row_CoatPaths:
                return IndexRole.COATPATHS
            if row == cls.Row_RefArgs:
                return IndexRole.REFARGS
            if row == cls.Row_SubstArgs:
                return IndexRole.SUBSTARGS
            if row == cls.Row_LayerArgs:
                return IndexRole.LAYERARGS
            if row == cls.Row_ExptArgs:
                return IndexRole.EXPTARGS
            if row == cls.Row_AnalysisArgs:
                return IndexRole.ANALYSISARGS
        if (
            indexLevel == 2
            and col == 0
            and cls.whatsThisIndex(index.parent()) is IndexRole.COATPATHS
        ):
            return IndexRole.COATPATH
        return IndexRole.UNKNOWN

    def getIndexFor(self, indexRole: IndexRole, parent: QModelIndex) -> QModelIndex:
        """
        Return the index with *indexRole* under *parent*.

        If the index cannot be specified, returns an invalid index.
        """
        if self.whatsThisIndex(parent) != IndexRole.EXPTDATA:
            return QModelIndex()

        if indexRole == IndexRole.EXPTDATA:
            return parent
        elif indexRole == IndexRole.REFPATH:
            return self.index(self.Row_RefPath, 0, parent)
        elif indexRole == IndexRole.COATPATHS:
            return self.index(self.Row_CoatPaths, 0, parent)
        elif indexRole == IndexRole.REFARGS:
            return self.index(self.Row_RefArgs, 0, parent)
        elif indexRole == IndexRole.SUBSTARGS:
            return self.index(self.Row_SubstArgs, 0, parent)
        elif indexRole == IndexRole.LAYERARGS:
            return self.index(self.Row_LayerArgs, 0, parent)
        elif indexRole == IndexRole.EXPTARGS:
            return self.index(self.Row_ExptArgs, 0, parent)
        elif indexRole == IndexRole.ANALYSISARGS:
            return self.index(self.Row_AnalysisArgs, 0, parent)
        return QModelIndex()
