"""
Experiment data list view
=========================

"""

import enum
from itertools import groupby
from operator import itemgetter
from PySide6.QtCore import (
    QModelIndex,
    Qt,
    Slot,
)
from PySide6.QtWidgets import (
    QStyledItemDelegate,
    QWidget,
    QListView,
    QToolButton,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMenu,
    QSizePolicy,
    QFileDialog,
)
from dipcoatimage.finitedepth import data_converter, ExperimentData
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
import json
import yaml  # type: ignore[import]
import os
from typing import Optional, List


__all__ = [
    "DataFileTypeEnum",
    "ExperimentDataListView",
    "ExperimentNameDelegate",
]


class DataFileTypeEnum(enum.Enum):
    """
    Enum of supported file types for experiment data. Values are file filters.
    """

    YAML = "YAML (*.yml)"
    JSON = "JSON (*.json)"

    def asExtensions(self) -> List[str]:
        s = self.value
        patterns = s[s.find("(") + 1 : s.find(")")].split(" ")
        exts = [p[p.find(".") + 1 :] for p in patterns]
        return exts


class ExperimentDataListView(QWidget):
    """
    Widget to display the list of experiment data.

    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
    >>> from dipcoatimage.finitedepth_gui.views import ExperimentDataListView
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     exptListWidget = ExperimentDataListView()
    ...     exptListWidget.setModel(model)
    ...     exptListWidget.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None
        self._listView = QListView()
        self._addButton = QToolButton()
        self._addButton.setMenu(QMenu(self))
        copyAction = self._addButton.menu().addAction("Copy selected items")
        self._deleteButton = QPushButton()
        self._importButton = QPushButton()
        self._exportButton = QPushButton()

        self._listView.setItemDelegate(ExperimentNameDelegate())
        self._listView.setSelectionMode(QListView.ExtendedSelection)
        self._listView.activated.connect(self._onIndexActivated)
        self._addButton.clicked.connect(self.appendNewRow)
        copyAction.triggered.connect(self.copySelectedRows)
        self._deleteButton.clicked.connect(self.deleteSelectedRows)
        self._importButton.clicked.connect(self.openImportDialog)
        self._exportButton.clicked.connect(self.openExportDialog)

        self._addButton.setPopupMode(QToolButton.MenuButtonPopup)
        self._addButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._deleteButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._addButton.setText("Add")
        self._deleteButton.setText("Delete")
        self._importButton.setText("Import")
        self._exportButton.setText("Export")

        layout = QVBoxLayout()
        layout.addWidget(self._listView)
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self._addButton)
        buttonLayout.addWidget(self._deleteButton)
        layout.addLayout(buttonLayout)
        layout.addWidget(self._importButton)
        layout.addWidget(self._exportButton)
        self.setLayout(layout)

    def model(self) -> Optional[ExperimentDataModel]:
        return self._model

    def setModel(self, model: Optional[ExperimentDataModel]):
        oldModel = self.model()
        if oldModel is not None:
            oldModel.activatedIndexChanged.disconnect(self._listView.viewport().update)
        self._model = model
        self._listView.setModel(model)
        if model is not None:
            model.activatedIndexChanged.connect(self._listView.viewport().update)

    @Slot()
    def appendNewRow(self):
        model = self.model()
        if model is None:
            return
        model.insertExperimentDataRows(
            model.rowCount(), 1, ["New Experiment"], [ExperimentData()]
        )

    @Slot()
    def copySelectedRows(self):
        model = self.model()
        if model is None:
            return
        rows = [idx.row() for idx in self._listView.selectedIndexes()]
        continuous_rows = [
            list(map(itemgetter(1), g))
            for k, g in groupby(enumerate(sorted(rows)), lambda i_x: i_x[0] - i_x[1])
        ]
        parent = self._listView.rootIndex()
        for row_list in continuous_rows:
            model.copyRows(
                parent, row_list[0], len(row_list), parent, model.rowCount(parent)
            )

    @Slot()
    def deleteSelectedRows(self):
        model = self.model()
        if model is None:
            return
        rows = [idx.row() for idx in self._listView.selectedIndexes()]
        continuous_rows = [
            list(map(itemgetter(1), g))
            for k, g in groupby(enumerate(sorted(rows)), lambda i_x: i_x[0] - i_x[1])
        ]
        parent = self._listView.rootIndex()
        for row_list in reversed(continuous_rows):
            model.removeRows(row_list[0], len(row_list), parent)

    def _onIndexActivated(self, index: QModelIndex):
        model = self.model()
        if model is None:
            return
        model.setActivatedIndex(index)

    @Slot()
    def openImportDialog(self):
        filters = ";;".join([e.value for e in DataFileTypeEnum])
        fileNames, selectedFilter = QFileDialog.getOpenFileNames(
            self,
            "Select configuration files",
            "./",
            filters,
            options=QFileDialog.Options.DontUseNativeDialog,
        )
        if fileNames:
            self.importItems(fileNames, DataFileTypeEnum(selectedFilter))

    def importItems(self, fileNames: List[str], selectedFilter: DataFileTypeEnum):
        model = self.model()
        if model is None:
            return
        count, names, exptData = 0, [], []
        for filename in fileNames:
            with open(filename, "r") as f:
                if selectedFilter == DataFileTypeEnum.JSON:
                    dataDict = json.load(f)
                elif selectedFilter == DataFileTypeEnum.YAML:
                    dataDict = yaml.load(f, Loader=yaml.FullLoader)
            for name, data in dataDict.items():
                count += 1
                names.append(name)
                exptData.append(data_converter.structure(data, ExperimentData))
        model.insertExperimentDataRows(model.rowCount(), count, names, exptData)

    @Slot()
    def openExportDialog(self):
        filters = ";;".join([e.value for e in DataFileTypeEnum])
        fileName, selectedFilter = QFileDialog.getSaveFileName(
            self,
            "Save as configuration file",
            "./",
            filters,
            options=QFileDialog.Options.DontUseNativeDialog,
        )
        selectedFilter = DataFileTypeEnum(selectedFilter)
        if fileName:
            path, ext = os.path.splitext(fileName)
            if not ext:
                fileName = f"{path}{os.extsep}{selectedFilter.asExtensions()[0]}"
            self.exportItems(fileName, selectedFilter)

    def exportItems(self, fileName: str, selectedFilter: DataFileTypeEnum):
        model = self.model()
        if model is None:
            return
        indices = self._listView.selectedIndexes()
        data = {}
        for index in indices:
            name = model.data(index, model.Role_ExptName)
            exptData = model.indexToExperimentData(index)
            data[name] = data_converter.unstructure(exptData)
        with open(fileName, "w") as f:
            if selectedFilter == DataFileTypeEnum.JSON:
                json.dump(data, f, indent=2)
            elif selectedFilter == DataFileTypeEnum.YAML:
                yaml.dump(data, f)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.deleteSelectedRows()
        super().keyPressEvent(event)


class ExperimentNameDelegate(QStyledItemDelegate):
    """Delegate to mark activated experiment."""

    ACTIVATED_INDENT = 10
    ACTIVATED_MARKER_RADIUS = 2

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            if index == model.activatedIndex():
                option.font.setBold(True)
                option.rect.adjust(self.ACTIVATED_INDENT, 0, 0, 0)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            if index == model.activatedIndex():
                markerSpace = option.rect.adjusted(
                    0, 0, -option.rect.width() + self.ACTIVATED_INDENT, 0
                )
                w, h = markerSpace.width(), markerSpace.height()
                dx = w // 2 - self.ACTIVATED_MARKER_RADIUS
                dy = h // 2 - self.ACTIVATED_MARKER_RADIUS
                markerRect = markerSpace.adjusted(dx, dy, -dx, -dy)

                painter.save()
                painter.setBrush(Qt.black)
                painter.drawEllipse(markerRect)
                painter.restore()
