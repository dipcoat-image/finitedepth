"""
Importing Widget
================

This module provides widget to import the variable.

"""

import dataclasses
from dipcoatimage.finitedepth.util import import_variable
import enum
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QComboBox,
    QPushButton,
    QLineEdit,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
)
from typing import Any


__all__ = ["ImportStatus", "VariableItemData", "ImportWidget"]


class ImportStatus(enum.Enum):
    """Constants for variable import status of :class:`ImportWidget`."""

    VALID = 0
    NO_MODULE = 1
    NO_VARIABLE = 2


@dataclasses.dataclass(frozen=True)
class VariableItemData:
    """Dataclass for registered variable of :class:`ImportWidget`."""

    varname: str
    modname: str
    variable: Any
    status: ImportStatus


class ImportWidget(QWidget):
    """
    Widget which imports the variable.

    .. rubric:: Specifying the object

    Object can be specified by one of the two ways.

    1. Registering and selecting
    2. Passing import information

    Registeration imports the object once, caches it and returns when selected.
    Passing import information caches temporarily - if the variable changes,
    previous one is removed.

    Either way, :attr:`variableChanged` signal emits the specified object and
    :meth:`variable` returns the object.
    If importing fails, :obj:`ImportWidget.INVALID` is emitted and returned.

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
    ...     widget.registerVariable("Invalid", "foo", "bar")
    ...     widget.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    INVALID = object()
    variableChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._var_cbox = QComboBox()
        self._hideshow_button = QPushButton()
        self._varname_ledit = QLineEdit()
        self._module_ledit = QLineEdit()
        self._msg_box = QLabel()
        self._registry_button = QPushButton()
        self._variable = self.INVALID
        self._status = ImportStatus.NO_MODULE

        self.variableComboBox().setPlaceholderText(
            "Select variable or specify import information"
        )
        self.variableComboBox().currentIndexChanged.connect(self.onSelectionChange)
        self.hideShowButton().setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.hideShowButton().setCheckable(True)
        self.hideShowButton().toggled.connect(self.toggleHideShow)
        self.hideShowButton().toggle()
        self.variableNameLineEdit().setPlaceholderText("Variable name")
        self.variableNameLineEdit().editingFinished.connect(self.onInformationEdit)
        self.moduleNameLineEdit().setPlaceholderText("Module name")
        self.moduleNameLineEdit().editingFinished.connect(self.onInformationEdit)
        self.registryWindowButton().setText("Open registry window")
        self.registryWindowButton().clicked.connect(self.openRegistryWindow)

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

        If variable is invalid, :attr:`INVAID` isreturned.
        """
        return self._variable

    def importStatus(self) -> ImportStatus:
        """Return import status of current variable."""
        return self._status

    def isValid(self) -> bool:
        """Return if current variable is valid."""
        return self.variable() is not self.INVALID

    @Slot(str, str, str)
    def registerVariable(self, itemText: str, varName: str, modName: str):
        """Register the information and variable to combo box."""
        try:
            var = import_variable(varName, modName)
            status = ImportStatus.VALID
        except ModuleNotFoundError:
            var = self.INVALID
            status = ImportStatus.NO_MODULE
        except (ImportError, NameError):
            var = self.INVALID
            status = ImportStatus.NO_VARIABLE
        data = VariableItemData(varName, modName, var, status)
        self.variableComboBox().addItem(itemText, data)

    @Slot(int)
    def onSelectionChange(self, index: int):
        """Apply the data from combo box selection."""
        if index != -1:
            data = self.variableComboBox().itemData(index)
            self.variableNameLineEdit().setText(data.varname)
            self.moduleNameLineEdit().setText(data.modname)
            self._applyVariable(data.variable, data.status)

    @Slot()
    def onInformationEdit(self):
        self.variableComboBox().setCurrentIndex(-1)
        varname = self.variableNameLineEdit().text()
        modname = self.moduleNameLineEdit().text()
        try:
            var = import_variable(varname, modname)
            status = ImportStatus.VALID
        except ModuleNotFoundError:
            var = self.INVALID
            status = ImportStatus.NO_MODULE
        except (ImportError, NameError):
            var = self.INVALID
            status = ImportStatus.NO_VARIABLE
        self._applyVariable(var, status)

    def _applyVariable(self, var: Any, status: ImportStatus):
        self._variable = var
        self.variableChanged.emit(var)

        if status is ImportStatus.VALID:
            txt = "Import successful."
        elif status is ImportStatus.NO_MODULE:
            txt = "Invalid module name."
        elif status is ImportStatus.NO_VARIABLE:
            txt = "Invalid variable name."
        self.messageBox().setText(txt)

    @Slot(bool)
    def toggleHideShow(self, state: bool):
        """Hide or show detailed widgets."""
        txt = "Show details" if state else "Hide details"
        self.hideShowButton().setText(txt)
        self.variableNameLineEdit().setVisible(not state)
        self.moduleNameLineEdit().setVisible(not state)
        self.messageBox().setVisible(not state)
        self.registryWindowButton().setVisible(not state)

    def openRegistryWindow(self):
        self.registrywindow = QWidget()  # do not pass self as parent
        self.registrywindow.show()
