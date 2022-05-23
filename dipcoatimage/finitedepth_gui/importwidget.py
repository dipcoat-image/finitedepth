"""
Importing Widget
================

This module provides widget to import the variable.

"""

import dataclasses
from dipcoatimage.finitedepth.util import ImportStatus, Importer
import enum
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (
    QWidget,
    QTableView,
    QHeaderView,
    QComboBox,
    QPushButton,
    QLineEdit,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
)
from typing import Any


__all__ = [
    "RegistryItemData",
    "RegistryModelColumns",
    "RegistryWidget",
    "ImportWidget",
]


@dataclasses.dataclass(frozen=True)
class RegistryItemData:
    """
    Dataclass for registered variable of :class:`ImportWidget`.

    Parameters
    ==========

    variables
        Imported variable. If import failed, it is :attr:`ImportWidget.INVALID`.

    status
        Status from import attempt.

    """

    variable: Any
    status: ImportStatus


class RegistryModelColumns(enum.IntEnum):
    """
    Columns for :meth:`RegistryWidget.registryModel`.

    0. ITEM_NAME
        Name of the item displayed in combo box.
        This column stores :class:`RegistryItemData` instance resulting from
        import attempt with *VARIABLE_NAME* and *MODULE_NAME*.
    1. VARIABLE_NAME
        Name of the variable that imports the object.
    2. MODULE_NAME
        Name of the module that imports the object.
    """

    ITEM_NAME = 0
    VARIABLE_NAME = 1
    MODULE_NAME = 2


class RegistryWidget(QWidget):
    """Widget to control the variable registry of :class:`ImportWidget`."""

    rowAdded = Signal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._registry_view = QTableView()
        self._add_button = QPushButton()
        self._remove_button = QPushButton()

        self.registryView().horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.registryView().setSelectionBehavior(QTableView.SelectRows)
        self.addButton().setText("Add")
        self.addButton().clicked.connect(self.addRow)
        self.removeButton().setText("Remove")
        self.removeButton().clicked.connect(self.removeItem)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.addButton())
        buttons_layout.addWidget(self.removeButton())
        layout = QVBoxLayout()
        layout.addWidget(self.registryView())
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def registryView(self) -> QTableView:
        """View for registered items."""
        return self._registry_view

    def addButton(self) -> QPushButton:
        return self._add_button

    def removeButton(self) -> QPushButton:
        return self._remove_button

    @Slot()
    def addRow(self):
        self.rowAdded.emit("New item", "", "")

    @Slot()
    def removeItem(self):
        indices = self.registryView().selectionModel().selectedRows()
        rows = sorted([index.row() for index in indices])
        for i in reversed(rows):
            self.registryView().model().removeRow(i)


class ImportWidget(QWidget):
    """
    Widget which imports and returns the variable.

    .. rubric:: Getting the object

    User can get the object by one of the two ways.

    1. Registering and selecting
    2. Passing import information

    Registration imports the object, stores it and returns when selected.
    Passing import information stores the object temporarily - if the variable
    changes, previous one is removed.

    Either way, :attr:`variableChanged` signal is emitted and :meth:`variable`
    returns the object.
    If importing fails, :obj:`ImportWidget.INVALID` is returned as sentinel.
    Validity can be checked by :meth:`isValid` or :meth:`importStatus`.

    .. rubric:: Import information

    Import information consists of variable name and module name. The variable
    must be importable by ``from {module name} import {variable name}`` syntax.

    Examples
    ========

    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.importwidget import ImportWidget
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     widget = ImportWidget()
    ...     widget.registerVariable(
    ...         "MyItem", "SubstrateReference", "dipcoatimage.finitedepth"
    ...     )
    ...     widget.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    INVALID = Importer.INVALID
    variableChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._registry_model = QStandardItemModel(0, len(RegistryModelColumns))
        self._registry_widget = RegistryWidget()
        self._var_cbox = QComboBox()
        self._hideshow_button = QPushButton()
        self._varname_ledit = QLineEdit()
        self._module_ledit = QLineEdit()
        self._msg_box = QLabel()
        self._registry_button = QPushButton()
        self._variable = self.INVALID
        self._status = ImportStatus.NO_MODULE

        self.registryModel().setHorizontalHeaderLabels(
            ["Item name", "Variable name", "Module name"]
        )
        self.registryModel().itemChanged.connect(self.onItemChange)
        self.registryWidget().registryView().setModel(self.registryModel())
        self.registryWidget().rowAdded.connect(self.registerVariable)
        self.variableComboBox().setPlaceholderText(
            "Select variable or specify import information"
        )
        self.variableComboBox().setModel(self.registryModel())
        self.variableComboBox().currentIndexChanged.connect(self.onSelectionChange)
        self.hideShowButton().setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.hideShowButton().setCheckable(True)
        self.hideShowButton().toggled.connect(self.toggleHideShow)
        self.hideShowButton().toggle()
        self.variableNameLineEdit().setPlaceholderText(
            self.registryModel()
            .horizontalHeaderItem(RegistryModelColumns.VARIABLE_NAME)
            .text()
        )
        self.variableNameLineEdit().editingFinished.connect(self.onInformationEdit)
        self.moduleNameLineEdit().setPlaceholderText(
            self.registryModel()
            .horizontalHeaderItem(RegistryModelColumns.MODULE_NAME)
            .text()
        )
        self.moduleNameLineEdit().editingFinished.connect(self.onInformationEdit)
        self.registryWindowButton().setText("Open registry window")
        self.registryWindowButton().clicked.connect(self.registryWidget().show)

        cbox_layout = QHBoxLayout()
        cbox_layout.addWidget(self.variableComboBox())
        cbox_layout.addWidget(self.hideShowButton())
        layout = QVBoxLayout()
        layout.addLayout(cbox_layout)
        layout.addWidget(self.variableNameLineEdit())
        layout.addWidget(self.moduleNameLineEdit())
        layout.addWidget(self.messageBox())
        layout.addWidget(self.registryWindowButton())
        self.setLayout(layout)

    def registryModel(self) -> QStandardItemModel:
        """
        Model to store the registered objects.

        Columns are described in :class:`RegistryModelColumns`.
        """
        return self._registry_model

    def registryWidget(self) -> RegistryWidget:
        """Widget to control the registry items."""
        return self._registry_widget

    def variableComboBox(self) -> QComboBox:
        """Combo box to select the registered object."""
        return self._var_cbox

    def hideShowButton(self) -> QPushButton:
        """Button to hide/show detailed widgets."""
        return self._hideshow_button

    def variableNameLineEdit(self) -> QLineEdit:
        """Line edit for variable name used to import the object."""
        return self._varname_ledit

    def moduleNameLineEdit(self) -> QLineEdit:
        """Line edit for module name used to import the object."""
        return self._module_ledit

    def messageBox(self) -> QLineEdit:
        """Informs if the import information is valid."""
        return self._msg_box

    def registryWindowButton(self) -> QPushButton:
        """Button to open registry window."""
        return self._registry_button

    def variable(self) -> Any:
        """
        Return current variable.

        If variable is invalid, :attr:`INVAID` is returned.
        """
        return self._variable

    def importStatus(self) -> ImportStatus:
        """Return import status of current variable."""
        return self._status

    def isValid(self) -> bool:
        """Return if current variable is valid."""
        return self.variable() is not self.INVALID

    @Slot(QStandardItem)
    def onItemChange(self, item: QStandardItem):
        if item.column() != RegistryModelColumns.ITEM_NAME:
            model = self.registryModel()
            index = item.row()
            varname = model.item(index, RegistryModelColumns.VARIABLE_NAME).text()
            modname = model.item(index, RegistryModelColumns.MODULE_NAME).text()
            var, status = Importer(varname, modname).try_import()
            data = RegistryItemData(var, status)
            model.item(index, RegistryModelColumns.ITEM_NAME).setData(data)
        if item.row() == self.variableComboBox().currentIndex():
            self.onSelectionChange(item.row())

    @Slot(str, str, str)
    def registerVariable(self, itemText: str, varName: str, modName: str):
        """Register the information and variable to combo box."""
        var, status = Importer(varName, modName).try_import()
        data = RegistryItemData(var, status)
        item0 = QStandardItem(itemText)
        item0.setData(data)
        item1 = QStandardItem(varName)
        item2 = QStandardItem(modName)
        self.registryModel().appendRow([item0, item1, item2])

    @Slot(int)
    def onSelectionChange(self, index: int):
        """Apply the data from combo box selection."""
        if index != -1:
            varname = (
                self.registryModel()
                .item(index, RegistryModelColumns.VARIABLE_NAME)
                .text()
            )
            modname = (
                self.registryModel()
                .item(index, RegistryModelColumns.MODULE_NAME)
                .text()
            )
            data = (
                self.registryModel().item(index, RegistryModelColumns.ITEM_NAME).data()
            )
            self.variableNameLineEdit().setText(varname)
            self.moduleNameLineEdit().setText(modname)
            self._applyVariable(data.variable, data.status)

    @Slot()
    def onInformationEdit(self):
        """Import and apply variable with current texts."""
        self.variableComboBox().setCurrentIndex(-1)
        varname = self.variableNameLineEdit().text()
        modname = self.moduleNameLineEdit().text()
        var, status = Importer(varname, modname).try_import()
        self._applyVariable(var, status)

    def setImportInformation(self, varName: str, modName: str):
        """Import and apply variable with passed informations."""
        self.variableComboBox().setCurrentIndex(-1)
        self.variableNameLineEdit().setText(varName)
        self.moduleNameLineEdit().setText(modName)
        var, status = Importer(varName, modName).try_import()
        self._applyVariable(var, status)

    def _applyVariable(self, var: Any, status: ImportStatus):
        self._variable = var
        self.variableChanged.emit()

        if status is ImportStatus.VALID:
            txt = "Import successful."
        elif status is ImportStatus.NO_MODULE:
            txt = "Invalid module name!"
        elif status is ImportStatus.NO_VARIABLE:
            txt = "Invalid variable name!"
        self.messageBox().setText(txt)
        self._status = status

    @Slot(bool)
    def toggleHideShow(self, state: bool):
        """Hide or show detailed widgets."""
        txt = "Show details" if state else "Hide details"
        self.hideShowButton().setText(txt)
        self.variableNameLineEdit().setVisible(not state)
        self.moduleNameLineEdit().setVisible(not state)
        self.messageBox().setVisible(not state)
        self.registryWindowButton().setVisible(not state)

    def clear(self):
        self.variableComboBox().setCurrentIndex(-1)
        self.variableNameLineEdit().clear()
        self.moduleNameLineEdit().clear()
        self._applyVariable(self.INVALID, ImportStatus.NO_MODULE)
