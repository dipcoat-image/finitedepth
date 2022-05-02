"""
Variable Widget
===============

This module provides widget to specify the variable with import information.

"""

import dataclasses
from dipcoatimage.finitedepth.util import import_variable
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


__all__ = ["VariableItemData", "VariableWidget"]


@dataclasses.dataclass(frozen=True)
class VariableItemData:
    varname: str
    modname: str
    variable: Any


class VariableWidget(QWidget):
    """
    Widget to specify the variable with import information.

    .. rubric:: Import information

    Import information consists of variable name and module name. The variable
    must be importable by ``from {module name} import {variable name}`` syntax.

    .. rubric:: Specifying the object

    Object can be specified by one of the two ways.

    1. Registering and selecting
    2. Passing import information

    Registeration caches the imported object and returns when selected. Passing
    import information caches only once - if the variable changes, previous one
    is removed.

    Either way, :attr:`variableChanged` signal emits the specified object and
    :meth:`variable` returns the object.
    If importing fails, :obj:`VariableWidget.INVALID` is emitted and returned.

    Examples
    ========

    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.variablewidget import VariableWidget
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     widget = VariableWidget()
    ...     widget.registerVariable(
    ...         "MyItem", "SubstrateReference", "dipcoatimage.finitedepth"
    ...     )
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

        self.variableComboBox().setPlaceholderText(
            "Select variable or specify import information"
        )
        self.hideShowButton().setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.hideShowButton().setCheckable(True)
        self.hideShowButton().toggled.connect(self.toggleHideShow)
        self.hideShowButton().toggle()
        self.variableNameLineEdit().setPlaceholderText("Variable name")
        self.moduleNameLineEdit().setPlaceholderText("Module name")
        self.registryWindowButton().setText("Open registry window")

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
        """Informs if the information is valid."""
        return self._msg_box

    def registryWindowButton(self) -> QPushButton:
        """Button to open registry window."""
        return self._registry_button

    @Slot(bool)
    def toggleHideShow(self, state: bool):
        """Hide or show detailed widgets."""
        txt = "Show details" if state else "Hide details"
        self.hideShowButton().setText(txt)
        self.variableNameLineEdit().setVisible(not state)
        self.moduleNameLineEdit().setVisible(not state)
        self.messageBox().setVisible(not state)
        self.registryWindowButton().setVisible(not state)

    @Slot(str, str, str)
    def registerVariable(self, itemText: str, varName: str, modName: str):
        """Register the information and variable to combo box."""
        try:
            var = import_variable(varName, modName)
        except (NameError, ModuleNotFoundError):
            var = self.INVALID
        data = VariableItemData(varName, modName, var)
        self.variableComboBox().addItem(itemText, data)
