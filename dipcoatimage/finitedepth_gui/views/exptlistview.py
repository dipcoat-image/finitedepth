"""
Experiment data list view
=========================

V2 for inventory.py
"""

import enum
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
import yaml
from typing import Optional, List


__all__ = [
    "DataFileTypeEnum",
    "ExperimentListView",
    "ExperimentNameDelegate",
]


class DataFileTypeEnum(enum.Enum):
    """
    Enum of supported file types for experiment data. Values are file filters.
    """

    JSON = "JSON (*.json)"
    YAML = "YAML (*.yml)"

    def asExtensions(self) -> List[str]:
        s = self.value
        patterns = s[s.find("(") + 1 : s.find(")")].split(" ")
        exts = [p[p.find(".") + 1 :] for p in patterns]
        return exts


class ExperimentListView(QWidget):
    """
    Widget to display the list of experiment data.

    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
    >>> from dipcoatimage.finitedepth_gui.views import ExperimentListView
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     exptListWidget = ExperimentListView()
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
        if model None:
            return
        rowNum = model.rowCount()
        model.insertExperimentDataRows(
            model.rowCount(), 1, ["New Experiment"], [ExperimentData()]
        )

    @Slot()
    def copySelectedRows(self):
        model = self.model()
        if model is None:
            return
        for index in self._listView.selectedIndexes():
            parent = index.parent()
            model.copyRows(parent, index.row(), 1, parent, model.rowCount(parent))

    @Slot()
    def deleteSelectedRows(self):
        model = self.model()
        if model is not None:
            rows = [idx.row() for idx in self._listView.selectedIndexes()]
            for i in reversed(sorted(rows)):
                model.removeRow(i)

    def _onIndexActivated(self, index: QModelIndex):
        model = self.model()
        if model is not None:
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
