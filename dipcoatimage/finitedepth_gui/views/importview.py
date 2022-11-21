"""
Import data view
================

"""

from dipcoatimage.finitedepth import ImportArgs
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLineEdit,
    QVBoxLayout,
    QStyledItemDelegate,
)
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel


__all__ = [
    "ImportDataView",
    "ImportArgsDelegate",
]


class ImportDataView(QGroupBox):
    """Widget to display import data."""

    editingFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._varNameLineEdit = QLineEdit()
        self._moduleNameLineEdit = QLineEdit()

        self._varNameLineEdit.editingFinished.connect(self.editingFinished)
        self._moduleNameLineEdit.editingFinished.connect(self.editingFinished)

        self._varNameLineEdit.setPlaceholderText("Variable name")
        self._moduleNameLineEdit.setPlaceholderText("Module")

        layout = QVBoxLayout()
        layout.addWidget(self._varNameLineEdit)
        layout.addWidget(self._moduleNameLineEdit)
        self.setLayout(layout)

    def variableName(self) -> str:
        return self._varNameLineEdit.text()

    def setVariableName(self, name: str):
        self._varNameLineEdit.setText(name)

    def moduleName(self) -> str:
        return self._moduleNameLineEdit.text()

    def setModuleName(self, name: str):
        self._moduleNameLineEdit.setText(name)

    def clear(self):
        self._varNameLineEdit.clear()
        self._moduleNameLineEdit.clear()


class ImportArgsDelegate(QStyledItemDelegate):
    def setModelData(self, editor, model, index):
        if isinstance(model, ExperimentDataModel) and isinstance(
            editor, ImportDataView
        ):
            importArgs = ImportArgs(editor.variableName(), editor.moduleName())
            model.cacheData(index, importArgs, model.Role_ImportArgs)
        super().setModelData(editor, model, index)

    def setEditorData(self, editor, index):
        model = index.model()
        if isinstance(model, ExperimentDataModel) and isinstance(
            editor, ImportDataView
        ):
            importArgs = model.data(index, role=model.Role_ImportArgs)
            if isinstance(importArgs, ImportArgs):
                editor.setVariableName(importArgs.name)
                editor.setModuleName(importArgs.module)
        super().setEditorData(editor, index)
