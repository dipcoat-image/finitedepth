"""
Experiment data model
=====================

V2 for inventory.py
"""

import copy
import enum
from functools import reduce
from dipcoatimage.finitedepth import (
    ExperimentData,
    SubstrateReferenceBase,
    SubstrateBase,
    CoatingLayerBase,
    ExperimentBase,
    ImportArgs,
)
from dipcoatimage.finitedepth.util import Importer
from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt, Signal
from .core import DataArgFlag
from .worker import AnalysisState, WorkerUpdateFlag, ExperimentWorker
from typing import Optional, Any, Union, Tuple, List, Dict


__all__ = [
    "ExperimentDataItem",
    "IndexRole",
    "ExperimentDataModel",
    "getTopLevelIndex",
    "ExperimentSignalBlocker",
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
        self._data: Dict[Qt.ItemDataRole, Any] = dict()
        self._children = []
        self._parent = None

    def data(self, role: Union[Qt.ItemDataRole, int]) -> Any:
        """
        Get the data stored with *role*.

        ``EditRole`` and ``DisplayRole`` are treated as identical keys.
        """
        if isinstance(role, int):
            role = Qt.ItemDataRole(role)
        if role == Qt.ItemDataRole.EditRole:
            role = Qt.ItemDataRole.DisplayRole
        return self._data.get(role, None)

    def setData(self, role: Union[Qt.ItemDataRole, int], data: Any):
        """
        Set the data with *role*.

        ``EditRole`` and ``DisplayRole`` are treated as identical keys.
        """
        if isinstance(role, int):
            role = Qt.ItemDataRole(role)
        if role == Qt.ItemDataRole.EditRole:
            role = Qt.ItemDataRole.DisplayRole
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


class IndexRole(enum.Enum):
    """Role of the ``QModelIndex`` of :class:`ExperimentDataModel`."""

    UNKNOWN = enum.auto()
    EXPTDATA = enum.auto()
    REFPATH = enum.auto()
    COATPATHS = enum.auto()
    REFARGS = enum.auto()
    SUBSTARGS = enum.auto()
    LAYERARGS = enum.auto()
    EXPTARGS = enum.auto()
    ANALYSISARGS = enum.auto()

    COATPATH = enum.auto()

    REF_TYPE = enum.auto()
    REF_TEMPLATEROI = enum.auto()
    REF_SUBSTRATEROI = enum.auto()
    REF_PARAMETERS = enum.auto()
    REF_DRAWOPTIONS = enum.auto()

    SUBST_PARAMETERS = enum.auto()
    SUBST_DRAWOPTIONS = enum.auto()

    LAYER_TYPE = enum.auto()
    LAYER_PARAMETERS = enum.auto()
    LAYER_DRAWOPTIONS = enum.auto()
    LAYER_DECOOPTIONS = enum.auto()

    EXPT_TYPE = enum.auto()
    EXPT_PARAMETERS = enum.auto()


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
            * REF_TYPE
            * REF_TEMPLATEROI
            * REF_SUBSTRATEROI
            * REF_PARAMETERS
            * REF_DRAWOPTIONS
        * SUBSTARGS
            * SUBST_PARAMETERS
            * SUBST_DRAWOPTIONS
        * LAYERARGS
            * LAYER_TYPE
            * LAYER_PARAMETERS
            * LAYER_DRAWOPTIONS
            * LAYER_DECOOPTIONS
        * EXPTARGS
            * EXPT_TYPE
            * EXPT_PARAMETERS
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

    Role_ExptName = Qt.ItemDataRole.DisplayRole
    Role_RefPath = Qt.ItemDataRole.DisplayRole
    Role_CoatPath = Qt.ItemDataRole.DisplayRole
    Role_AnalysisArgs = Qt.ItemDataRole.UserRole

    Role_ImportArgs = Qt.ItemDataRole.UserRole
    Role_DataclassType = Qt.ItemDataRole.UserRole
    Role_DataclassData = Qt.ItemDataRole.UserRole + 1
    Role_ROI = Qt.ItemDataRole.UserRole

    Row_RefPath = 0
    Row_CoatPaths = 1
    Row_RefArgs = 2
    Row_SubstArgs = 3
    Row_LayerArgs = 4
    Row_ExptArgs = 5
    Row_AnalysisArgs = 6

    Row_RefType = 0
    Row_RefTemplateROI = 1
    Row_RefSubstrateROI = 2
    Row_RefParameters = 3
    Row_RefDrawOptions = 4

    Row_SubstParameters = 0
    Row_SubstDrawOptions = 1

    Row_LayerType = 0
    Row_LayerParameters = 1
    Row_LayerDrawOptions = 2
    Row_LayerDecoOptions = 3

    Row_ExptType = 0
    Row_ExptParameters = 1

    experimentDataChanged = Signal(QModelIndex, DataArgFlag)
    activatedIndexChanged = Signal(QModelIndex)
    analysisStateChanged = Signal(AnalysisState)
    analysisProgressMaximumChanged = Signal(int)
    analysisProgressValueChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rootItem = ExperimentDataItem()
        self._dataCache: Dict[QModelIndex, Dict[Qt.ItemDataRole, Any]] = dict()
        self._workers: List[ExperimentWorker] = []
        self._activatedIndex = QModelIndex()
        self._blockExperimentSignals = False

    @classmethod
    def _itemFromExperimentData(cls, exptData: ExperimentData) -> ExperimentDataItem:
        item = ExperimentDataItem()

        refPathItem = ExperimentDataItem()
        refPathItem.setData(cls.Role_RefPath, exptData.ref_path)
        refPathItem.setParent(item)

        coatPathsItem = ExperimentDataItem()
        coatPathsItem.setParent(item)
        for path in exptData.coat_paths:
            coatPathItem = ExperimentDataItem()
            coatPathItem.setData(cls.Role_CoatPath, path)
            coatPathItem.setParent(coatPathsItem)

        refArgs = exptData.reference
        refArgsItem = ExperimentDataItem()
        refArgsItem.setParent(item)
        refTypeItem = ExperimentDataItem()
        refTypeItem.setData(cls.Role_ImportArgs, refArgs.type)
        refTypeItem.setParent(refArgsItem)
        templateROIItem = ExperimentDataItem()
        templateROIItem.setData(cls.Role_ROI, refArgs.templateROI)
        templateROIItem.setParent(refArgsItem)
        substrateROIItem = ExperimentDataItem()
        substrateROIItem.setData(cls.Role_ROI, refArgs.substrateROI)
        substrateROIItem.setParent(refArgsItem)
        refType, _ = Importer(refArgs.type.name, refArgs.type.module).try_import()
        refParametersItem = ExperimentDataItem()
        refDrawOptionsItem = ExperimentDataItem()
        if isinstance(refType, type) and issubclass(refType, SubstrateReferenceBase):
            refParametersItem.setData(cls.Role_DataclassType, refType.Parameters)
            refDrawOptionsItem.setData(cls.Role_DataclassType, refType.DrawOptions)
        refParametersItem.setData(cls.Role_DataclassData, refArgs.parameters)
        refParametersItem.setParent(refArgsItem)
        refDrawOptionsItem.setData(cls.Role_DataclassData, refArgs.draw_options)
        refDrawOptionsItem.setParent(refArgsItem)

        substArgs = exptData.substrate
        substArgsItem = ExperimentDataItem()
        substArgsItem.setParent(item)
        substArgsItem.setData(cls.Role_ImportArgs, substArgs.type)
        substType, _ = Importer(substArgs.type.name, substArgs.type.module).try_import()
        substParametersItem = ExperimentDataItem()
        substDrawOptionsItem = ExperimentDataItem()
        if isinstance(substType, type) and issubclass(substType, SubstrateBase):
            substParametersItem.setData(cls.Role_DataclassType, substType.Parameters)
            substDrawOptionsItem.setData(cls.Role_DataclassType, substType.DrawOptions)
        substParametersItem.setData(cls.Role_DataclassData, substArgs.parameters)
        substParametersItem.setParent(substArgsItem)
        substDrawOptionsItem.setData(cls.Role_DataclassData, substArgs.draw_options)
        substDrawOptionsItem.setParent(substArgsItem)

        layerArgs = exptData.coatinglayer
        layerArgsItem = ExperimentDataItem()
        layerArgsItem.setParent(item)
        layerTypeItem = ExperimentDataItem()
        layerTypeItem.setData(cls.Role_ImportArgs, layerArgs.type)
        layerTypeItem.setParent(layerArgsItem)
        layerType, _ = Importer(layerArgs.type.name, layerArgs.type.module).try_import()
        layerParametersItem = ExperimentDataItem()
        layerDrawOptionsItem = ExperimentDataItem()
        layerDecoOptionsItem = ExperimentDataItem()
        if isinstance(layerType, type) and issubclass(layerType, CoatingLayerBase):
            layerParametersItem.setData(cls.Role_DataclassType, layerType.Parameters)
            layerDrawOptionsItem.setData(cls.Role_DataclassType, layerType.DrawOptions)
            layerDecoOptionsItem.setData(cls.Role_DataclassType, layerType.DecoOptions)
        layerParametersItem.setData(cls.Role_DataclassData, layerArgs.parameters)
        layerParametersItem.setParent(layerArgsItem)
        layerDrawOptionsItem.setData(cls.Role_DataclassData, layerArgs.draw_options)
        layerDrawOptionsItem.setParent(layerArgsItem)
        layerDecoOptionsItem.setData(cls.Role_DataclassData, layerArgs.deco_options)
        layerDecoOptionsItem.setParent(layerArgsItem)

        exptArgs = exptData.experiment
        exptArgsItem = ExperimentDataItem()
        exptArgsItem.setParent(item)
        exptTypeItem = ExperimentDataItem()
        exptTypeItem.setData(cls.Role_ImportArgs, exptArgs.type)
        exptTypeItem.setParent(exptArgsItem)
        exptType, _ = Importer(exptArgs.type.name, exptArgs.type.module).try_import()
        exptParametersItem = ExperimentDataItem()
        if isinstance(exptType, type) and issubclass(exptType, ExperimentBase):
            exptParametersItem.setData(cls.Role_DataclassType, exptType.Parameters)
        exptParametersItem.setData(cls.Role_DataclassData, exptArgs.parameters)
        exptParametersItem.setParent(exptArgsItem)

        analysisArgs = exptData.analysis
        analysisArgsItem = ExperimentDataItem()
        analysisArgsItem.setData(cls.Role_AnalysisArgs, analysisArgs)
        analysisArgsItem.setParent(item)

        return item

    def indexToExperimentData(self, index: QModelIndex) -> ExperimentData:
        data = ExperimentData()
        if self.whatsThisIndex(index) != IndexRole.EXPTDATA:
            return data

        refPathIdx = self.getIndexFor(IndexRole.REFPATH, index)
        refPath = refPathIdx.data(self.Role_RefPath)
        data.ref_path = refPath

        coatPathsIdx = self.getIndexFor(IndexRole.COATPATHS, index)
        coatPaths = [
            self.index(row, 0, coatPathsIdx).data(self.Role_CoatPath)
            for row in range(self.rowCount(coatPathsIdx))
        ]
        data.coat_paths = coatPaths

        refArgsIdx = self.getIndexFor(IndexRole.REFARGS, index)
        refTypeIdx = self.getIndexFor(IndexRole.REF_TYPE, refArgsIdx)
        refType = refTypeIdx.data(self.Role_ImportArgs)
        data.reference.type = refType
        tempROIIdx = self.getIndexFor(IndexRole.REF_TEMPLATEROI, refArgsIdx)
        tempROI = tempROIIdx.data(self.Role_ROI)
        data.reference.templateROI = tempROI
        substROIIdx = self.getIndexFor(IndexRole.REF_SUBSTRATEROI, refArgsIdx)
        substROI = substROIIdx.data(self.Role_ROI)
        data.reference.substrateROI = substROI
        refParamsIdx = self.getIndexFor(IndexRole.REF_PARAMETERS, refArgsIdx)
        refParams = refParamsIdx.data(self.Role_DataclassData)
        data.reference.parameters = refParams
        refDrawOptIdx = self.getIndexFor(IndexRole.REF_DRAWOPTIONS, refArgsIdx)
        refDrawOpts = refDrawOptIdx.data(self.Role_DataclassData)
        data.reference.draw_options = refDrawOpts

        substArgsIdx = self.getIndexFor(IndexRole.SUBSTARGS, index)
        substType = substArgsIdx.data(self.Role_ImportArgs)
        data.substrate.type = substType
        substParamsIdx = self.getIndexFor(IndexRole.SUBST_PARAMETERS, substArgsIdx)
        substParams = substParamsIdx.data(self.Role_DataclassData)
        data.substrate.parameters = substParams
        substDrawOptIdx = self.getIndexFor(IndexRole.SUBST_DRAWOPTIONS, substArgsIdx)
        substDrawOpts = substDrawOptIdx.data(self.Role_DataclassData)
        data.substrate.draw_options = substDrawOpts

        layerArgsIdx = self.getIndexFor(IndexRole.LAYERARGS, index)
        layerTypeIdx = self.getIndexFor(IndexRole.LAYER_TYPE, layerArgsIdx)
        layerType = layerTypeIdx.data(self.Role_ImportArgs)
        data.coatinglayer.type = layerType
        layerParamsIdx = self.getIndexFor(IndexRole.LAYER_PARAMETERS, layerArgsIdx)
        layerParams = layerParamsIdx.data(self.Role_DataclassData)
        data.coatinglayer.parameters = layerParams
        layerDrawOptIdx = self.getIndexFor(IndexRole.LAYER_DRAWOPTIONS, layerArgsIdx)
        layerDrawOpts = layerDrawOptIdx.data(self.Role_DataclassData)
        data.coatinglayer.draw_options = layerDrawOpts
        layerDecoOptIdx = self.getIndexFor(IndexRole.LAYER_DECOOPTIONS, layerArgsIdx)
        layerDecoOpts = layerDecoOptIdx.data(self.Role_DataclassData)
        data.coatinglayer.deco_options = layerDecoOpts

        exptArgsIdx = self.getIndexFor(IndexRole.EXPTARGS, index)
        exptTypeIdx = self.getIndexFor(IndexRole.EXPT_TYPE, exptArgsIdx)
        exptType = exptTypeIdx.data(self.Role_ImportArgs)
        data.experiment.type = exptType
        exptParamsIdx = self.getIndexFor(IndexRole.EXPT_PARAMETERS, exptArgsIdx)
        exptParams = exptParamsIdx.data(self.Role_DataclassData)
        data.experiment.parameters = exptParams

        analysisArgsIdx = self.getIndexFor(IndexRole.ANALYSISARGS, index)
        analysisArgs = analysisArgsIdx.data(self.Role_AnalysisArgs)
        data.analysis = analysisArgs
        return data

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

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole) -> bool:
        indexRole = self.whatsThisIndex(index)
        ret = self._setData(index, indexRole, value, role)
        if not ret:
            return False

        # update worker and emit signals
        if indexRole in [
            IndexRole.REFPATH,
        ]:
            dataArgs = DataArgFlag.REFPATH
            workerUpdateFlag = (
                WorkerUpdateFlag.REFIMAGE
                | WorkerUpdateFlag.REFERENCE
                | WorkerUpdateFlag.SUBSTRATE
                | WorkerUpdateFlag.EXPERIMENT
            )
        elif indexRole in [
            IndexRole.REF_TYPE,
            IndexRole.REF_TEMPLATEROI,
            IndexRole.REF_SUBSTRATEROI,
            IndexRole.REF_PARAMETERS,
        ]:
            dataArgs = DataArgFlag.REFERENCE
            workerUpdateFlag = (
                WorkerUpdateFlag.REFERENCE
                | WorkerUpdateFlag.SUBSTRATE
                | WorkerUpdateFlag.EXPERIMENT
            )
        elif indexRole in [
            IndexRole.REF_DRAWOPTIONS,
        ]:
            dataArgs = DataArgFlag.REFERENCE
            workerUpdateFlag = WorkerUpdateFlag.REFERENCE
        elif indexRole in [
            IndexRole.SUBSTARGS,
            IndexRole.SUBST_PARAMETERS,
        ]:
            dataArgs = DataArgFlag.SUBSTRATE
            workerUpdateFlag = WorkerUpdateFlag.SUBSTRATE | WorkerUpdateFlag.EXPERIMENT
        elif indexRole in [
            IndexRole.SUBST_DRAWOPTIONS,
        ]:
            dataArgs = DataArgFlag.SUBSTRATE
            workerUpdateFlag = WorkerUpdateFlag.SUBSTRATE
        elif indexRole in [
            IndexRole.LAYER_TYPE,
            IndexRole.LAYER_PARAMETERS,
            IndexRole.LAYER_DRAWOPTIONS,
            IndexRole.LAYER_DECOOPTIONS,
        ]:
            dataArgs = DataArgFlag.COATINGLAYER
            workerUpdateFlag = WorkerUpdateFlag.EXPERIMENT
        elif indexRole in [
            IndexRole.EXPT_TYPE,
            IndexRole.EXPT_PARAMETERS,
        ]:
            dataArgs = DataArgFlag.EXPERIMENT
            workerUpdateFlag = WorkerUpdateFlag.EXPERIMENT
        elif indexRole in [
            IndexRole.COATPATH,
        ]:
            dataArgs = DataArgFlag.COATPATHS
            workerUpdateFlag = WorkerUpdateFlag.ANALYSIS
        elif indexRole in [
            IndexRole.ANALYSISARGS,
        ]:
            dataArgs = DataArgFlag.ANALYSIS
            workerUpdateFlag = WorkerUpdateFlag.ANALYSIS
        else:
            dataArgs = DataArgFlag.NULL
            workerUpdateFlag = WorkerUpdateFlag.NULL
        topLevelIndex = getTopLevelIndex(index)
        self.updateWorker(topLevelIndex, workerUpdateFlag)
        self.emitExperimentDataChanged(topLevelIndex, dataArgs)
        return True

    def cacheData(self, index: QModelIndex, value: Any, role: Qt.ItemDataRole) -> bool:
        if index.model() != self:
            return False
        if not index.isValid():
            return False
        idxCache = self._dataCache.get(index)
        if idxCache is None:
            idxCache = dict()
            self._dataCache[index] = idxCache
        idxCache[role] = value
        return True

    def submit(self) -> bool:
        for index, data in self._dataCache.items():
            indexRole = self.whatsThisIndex(index)
            for dataRole, value in data.items():
                self._setData(index, indexRole, value, dataRole)
        self._dataCache = dict()
        return True

    def _setData(
        self,
        index: QModelIndex,
        indexRole: IndexRole,
        value: Any,
        dataRole: Qt.ItemDataRole,
    ) -> bool:
        subDataclassIndices = [
            IndexRole.SUBST_PARAMETERS,
            IndexRole.SUBST_DRAWOPTIONS,
        ]
        if indexRole in subDataclassIndices and dataRole == self.Role_DataclassType:
            return False
        dataItem = index.internalPointer()
        if not isinstance(dataItem, ExperimentDataItem):
            return False
        dataItem.setData(dataRole, value)
        self.dataChanged.emit(index, index, [dataRole])

        # update subitems
        if (
            indexRole == IndexRole.SUBSTARGS
            and dataRole == self.Role_ImportArgs
            and isinstance(value, ImportArgs)
        ):
            substType, _ = Importer(value.name, value.module).try_import()
            if isinstance(substType, type) and issubclass(substType, SubstrateBase):
                paramType = substType.Parameters
                drawOptType = substType.DrawOptions
            else:
                paramType = None
                drawOptType = None
            typeRole = self.Role_DataclassType
            paramIdxRole = IndexRole.SUBST_PARAMETERS
            paramIdx = self.getIndexFor(paramIdxRole, index)
            paramItem = paramIdx.internalPointer()
            if isinstance(paramItem, ExperimentDataItem):
                paramItem.setData(typeRole, paramType)
                self.dataChanged.emit(paramIdx, paramIdx, [typeRole])
            drawOptIdxRole = IndexRole.SUBST_DRAWOPTIONS
            drawOptIdx = self.getIndexFor(drawOptIdxRole, index)
            drawOptItem = drawOptIdx.internalPointer()
            if isinstance(drawOptItem, ExperimentDataItem):
                drawOptItem.setData(typeRole, drawOptType)
                self.dataChanged.emit(drawOptIdx, drawOptIdx, [typeRole])
        return True

    def revert(self):
        self._dataCache = dict()

    def emitExperimentDataChanged(self, index: QModelIndex, dataArgs: DataArgFlag):
        if not dataArgs:
            return
        if self._blockExperimentSignals:
            return
        self.experimentDataChanged.emit(index, dataArgs)

    def worker(self, index: QModelIndex) -> Optional[ExperimentWorker]:
        if self.whatsThisIndex(index) != IndexRole.EXPTDATA:
            return None
        row = index.row()
        if row + 1 > len(self._workers):
            return None
        return self._workers[row]

    def updateWorker(self, index: QModelIndex, flag: WorkerUpdateFlag) -> bool:
        if not flag:
            return False
        if self._blockExperimentSignals:
            return False
        worker = self.worker(index)
        if worker is None:
            return False
        exptData = self.indexToExperimentData(index)
        worker.setExperimentData(exptData, flag)
        return True

    def insertRows(self, row, count, parent=QModelIndex()):
        if not parent.isValid():
            activatedIndex = self.activatedIndex()
            activatedRow = activatedIndex.row()
            activatedColumn = activatedIndex.column()
            reactivate = (parent == activatedIndex.parent()) and row <= activatedRow

            self.beginInsertRows(parent, row, row + count - 1)
            for _ in reversed(range(count)):
                exptData = ExperimentData()
                newItem = self._itemFromExperimentData(exptData)
                newItem.setParent(self._rootItem, row)
                worker = ExperimentWorker(self)
                self._workers.insert(row, worker)
                worker.setExperimentData(
                    exptData, reduce(lambda x, y: x | y, WorkerUpdateFlag)
                )

            if reactivate:
                newRow = activatedRow + count
                newIndex = self.index(newRow, activatedColumn, parent)
                self.setActivatedIndex(newIndex)
            self.endInsertRows()
            return True
        elif self.whatsThisIndex(parent) == IndexRole.COATPATHS:
            self.beginInsertRows(parent, row, row + count - 1)
            for _ in reversed(range(count)):
                newItem = ExperimentDataItem()
                newItem.setParent(parent.internalPointer(), row)
            self.endInsertRows()

            topLevelIndex = getTopLevelIndex(parent)
            self.updateWorker(topLevelIndex, WorkerUpdateFlag.EXPERIMENT)
            self.emitExperimentDataChanged(topLevelIndex, DataArgFlag.COATPATHS)
            return True
        return False

    def insertExperimentDataRows(
        self, row: int, count: int, names: List[str], exptData: List[ExperimentData]
    ) -> bool:
        activatedIndex = self.activatedIndex()
        activatedRow = activatedIndex.row()
        activatedColumn = activatedIndex.column()
        reactivate = row <= activatedRow

        self.beginInsertRows(QModelIndex(), row, row + count - 1)
        for i in reversed(range(count)):
            data = exptData[i]
            newItem = self._itemFromExperimentData(data)
            newItem.setData(self.Role_ExptName, names[i])
            newItem.setParent(self._rootItem, row)
            worker = ExperimentWorker(self)
            self._workers.insert(row, worker)
            worker.setExperimentData(data, reduce(lambda x, y: x | y, WorkerUpdateFlag))

        if reactivate:
            newRow = activatedRow + count
            newIndex = self.index(newRow, activatedColumn, QModelIndex())
            self.setActivatedIndex(newIndex)
        self.endInsertRows()
        return True

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
            newWorkers = []
            for i in range(count):
                oldIdx = self.index(sourceRow + i, 0, sourceParent)
                oldItem = oldIdx.internalPointer()
                oldData = self.indexToExperimentData(oldIdx)
                newItem = self._itemFromExperimentData(copy.deepcopy(oldData))
                newItem.setData(self.Role_ExptName, oldItem.data(self.Role_ExptName))
                newItems.append(newItem)
                newWorker = ExperimentWorker(self)
                newWorkers.append(newWorker)
                newWorker.setExperimentData(
                    oldData, reduce(lambda x, y: x | y, WorkerUpdateFlag)
                )
            self.beginInsertRows(
                sourceParent, destinationChild, destinationChild + count - 1
            )
            for item, worker in zip(reversed(newItems), reversed(newWorkers)):
                item.setParent(self._rootItem, destinationChild)
                self._workers.insert(destinationChild, worker)

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
                newItem.setData(self.Role_ExptName, oldItem.data(self.Role_ExptName))
                newItems.append(newItem)
            self.beginInsertRows(
                sourceParent, destinationChild, destinationChild + count - 1
            )
            for item in reversed(newItems):
                item.setParent(parentDataItem, destinationChild)
            self.endInsertRows()

            topLevelIndex = getTopLevelIndex(sourceParent)
            self.updateWorker(topLevelIndex, WorkerUpdateFlag.EXPERIMENT)
            self.emitExperimentDataChanged(topLevelIndex, DataArgFlag.COATPATHS)
        return False

    def removeRows(self, row, count, parent=QModelIndex()):
        if not parent.isValid():
            # to avoid reference issue, decide whether activated index should be
            # changed before destroying data structure
            activatedIndex = self.activatedIndex()
            activatedRow = activatedIndex.row()
            activatedColumn = activatedIndex.column()
            reactivate = (parent == activatedIndex.parent()) and row <= activatedRow
            if reactivate:
                self.setActivatedIndex(QModelIndex())

            self.beginRemoveRows(parent, row, row + count - 1)
            for _ in range(count):
                self._rootItem.remove(row)
                worker = self._workers.pop(row)
                worker.setAnalysisState(AnalysisState.Stopped)
            if reactivate:
                if activatedRow >= row + count:
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

            topLevelIndex = getTopLevelIndex(parent)
            self.updateWorker(topLevelIndex, WorkerUpdateFlag.EXPERIMENT)
            self.emitExperimentDataChanged(topLevelIndex, DataArgFlag.COATPATHS)
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
        oldIndex = self.activatedIndex()
        oldWorker = self.worker(oldIndex)
        if oldWorker is not None:
            oldWorker.analysisStateChanged.disconnect(self.analysisStateChanged)
            oldWorker.analysisProgressMaximumChanged.disconnect(
                self.analysisProgressMaximumChanged
            )
            oldWorker.analysisProgressValueChanged.disconnect(
                self.analysisProgressValueChanged
            )
        if self.whatsThisIndex(index) != IndexRole.EXPTDATA:
            index = QModelIndex()
        self._activatedIndex = index
        newWorker = self.worker(index)
        if newWorker is not None:
            newWorker.analysisStateChanged.connect(self.analysisStateChanged)
            newWorker.analysisProgressMaximumChanged.connect(
                self.analysisProgressMaximumChanged
            )
            newWorker.analysisProgressValueChanged.connect(
                self.analysisProgressValueChanged
            )
            self.analysisStateChanged.emit(newWorker.analysisState())
            self.analysisProgressMaximumChanged.emit(
                newWorker.analysisProgressMaximum()
            )
            self.analysisProgressValueChanged.emit(newWorker.analysisProgressValue())
        else:
            self.analysisStateChanged.emit(AnalysisState.Stopped)
            self.analysisProgressMaximumChanged.emit(0)
            self.analysisProgressValueChanged.emit(0)
        self.activatedIndexChanged.emit(index)

    @classmethod
    def whatsThisIndex(cls, index: QModelIndex) -> IndexRole:
        """Return the role of *index* in the model."""
        if not isinstance(index.model(), cls):
            return IndexRole.UNKNOWN
        if not index.parent().isValid():
            return IndexRole.EXPTDATA

        parentRole = cls.whatsThisIndex(index.parent())
        row = index.row()

        if parentRole == IndexRole.EXPTDATA:
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

        if parentRole == IndexRole.COATPATHS:
            return IndexRole.COATPATH

        if parentRole == IndexRole.REFARGS:
            if row == cls.Row_RefType:
                return IndexRole.REF_TYPE
            if row == cls.Row_RefTemplateROI:
                return IndexRole.REF_TEMPLATEROI
            if row == cls.Row_RefSubstrateROI:
                return IndexRole.REF_SUBSTRATEROI
            if row == cls.Row_RefParameters:
                return IndexRole.REF_PARAMETERS
            if row == cls.Row_RefDrawOptions:
                return IndexRole.REF_DRAWOPTIONS

        if parentRole == IndexRole.SUBSTARGS:
            if row == cls.Row_SubstParameters:
                return IndexRole.SUBST_PARAMETERS
            if row == cls.Row_SubstDrawOptions:
                return IndexRole.SUBST_DRAWOPTIONS

        if parentRole == IndexRole.LAYERARGS:
            if row == cls.Row_LayerType:
                return IndexRole.LAYER_TYPE
            if row == cls.Row_LayerParameters:
                return IndexRole.LAYER_PARAMETERS
            if row == cls.Row_LayerDrawOptions:
                return IndexRole.LAYER_DRAWOPTIONS
            if row == cls.Row_LayerDecoOptions:
                return IndexRole.LAYER_DECOOPTIONS

        if parentRole == IndexRole.EXPTARGS:
            if row == cls.Row_ExptType:
                return IndexRole.EXPT_TYPE
            if row == cls.Row_ExptParameters:
                return IndexRole.EXPT_PARAMETERS

        return IndexRole.UNKNOWN

    def getIndexFor(self, indexRole: IndexRole, parent: QModelIndex) -> QModelIndex:
        """
        Return the index with *indexRole* under *parent*.

        If the index cannot be specified, returns an invalid index.
        """
        parentRole = self.whatsThisIndex(parent)

        if parentRole == IndexRole.EXPTDATA:
            if indexRole == IndexRole.REFPATH:
                return self.index(self.Row_RefPath, 0, parent)
            if indexRole == IndexRole.COATPATHS:
                return self.index(self.Row_CoatPaths, 0, parent)
            if indexRole == IndexRole.REFARGS:
                return self.index(self.Row_RefArgs, 0, parent)
            if indexRole == IndexRole.SUBSTARGS:
                return self.index(self.Row_SubstArgs, 0, parent)
            if indexRole == IndexRole.LAYERARGS:
                return self.index(self.Row_LayerArgs, 0, parent)
            if indexRole == IndexRole.EXPTARGS:
                return self.index(self.Row_ExptArgs, 0, parent)
            if indexRole == IndexRole.ANALYSISARGS:
                return self.index(self.Row_AnalysisArgs, 0, parent)

        if parentRole == IndexRole.REFARGS:
            if indexRole == IndexRole.REF_TYPE:
                return self.index(self.Row_RefType, 0, parent)
            if indexRole == IndexRole.REF_TEMPLATEROI:
                return self.index(self.Row_RefTemplateROI, 0, parent)
            if indexRole == IndexRole.REF_SUBSTRATEROI:
                return self.index(self.Row_RefSubstrateROI, 0, parent)
            if indexRole == IndexRole.REF_PARAMETERS:
                return self.index(self.Row_RefParameters, 0, parent)
            if indexRole == IndexRole.REF_DRAWOPTIONS:
                return self.index(self.Row_RefDrawOptions, 0, parent)

        if parentRole == IndexRole.SUBSTARGS:
            if indexRole == IndexRole.SUBST_PARAMETERS:
                return self.index(self.Row_SubstParameters, 0, parent)
            if indexRole == IndexRole.SUBST_DRAWOPTIONS:
                return self.index(self.Row_SubstDrawOptions, 0, parent)

        if parentRole == IndexRole.LAYERARGS:
            if indexRole == IndexRole.LAYER_TYPE:
                return self.index(self.Row_LayerType, 0, parent)
            if indexRole == IndexRole.LAYER_PARAMETERS:
                return self.index(self.Row_LayerParameters, 0, parent)
            if indexRole == IndexRole.LAYER_DRAWOPTIONS:
                return self.index(self.Row_LayerDrawOptions, 0, parent)
            if indexRole == IndexRole.LAYER_DECOOPTIONS:
                return self.index(self.Row_LayerDecoOptions, 0, parent)

        if parentRole == IndexRole.EXPTARGS:
            if indexRole == IndexRole.EXPT_TYPE:
                return self.index(self.Row_ExptType, 0, parent)
            if indexRole == IndexRole.EXPT_PARAMETERS:
                return self.index(self.Row_ExptParameters, 0, parent)

        return QModelIndex()


def getTopLevelIndex(index: QModelIndex) -> QModelIndex:
    if not index.isValid():
        return QModelIndex()
    while index.parent().isValid():
        index = index.parent()
    return index


class ExperimentSignalBlocker:
    def __init__(self, model: ExperimentDataModel):
        self.model = model

    def __enter__(self):
        self.model._blockExperimentSignals = True
        return self

    def __exit__(self, type, value, traceback):
        self.model._blockExperimentSignals = False
