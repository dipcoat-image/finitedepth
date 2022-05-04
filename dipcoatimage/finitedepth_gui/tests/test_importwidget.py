from dipcoatimage.finitedepth import SubstrateReference
from dipcoatimage.finitedepth_gui.importwidget import ImportWidget, ImportStatus
from PySide6.QtCore import Qt
import pytest


@pytest.fixture
def importwidget(qtbot):
    widget = ImportWidget()
    widget.registerVariable("Valid", "SubstrateReference", "dipcoatimage.finitedepth")
    widget.registerVariable("NoModule", "SubstrateReference", "foo")
    widget.registerVariable("NoVariable", "bar", "dipcoatimage.finitedepth")
    return widget


def test_ImportWidget_getByRegistry(qtbot, importwidget):
    with qtbot.waitSignal(importwidget.variableChanged):
        importwidget.variableComboBox().setCurrentIndex(0)
    assert importwidget.variable() is SubstrateReference
    assert importwidget.importStatus() is ImportStatus.VALID
    assert importwidget.isValid()

    with qtbot.waitSignal(importwidget.variableChanged):
        importwidget.variableComboBox().setCurrentIndex(1)
    assert importwidget.variable() is ImportWidget.INVALID
    assert importwidget.importStatus() is ImportStatus.NO_MODULE
    assert not importwidget.isValid()

    with qtbot.waitSignal(importwidget.variableChanged):
        importwidget.variableComboBox().setCurrentIndex(2)
    assert importwidget.variable() is ImportWidget.INVALID
    assert importwidget.importStatus() is ImportStatus.NO_VARIABLE
    assert not importwidget.isValid()


def test_ImportWidget_getByImportInformation(qtbot):
    importwidget = ImportWidget()

    with qtbot.waitSignal(importwidget.variableChanged):
        importwidget.moduleNameLineEdit().setText("foo")
        qtbot.keyPress(importwidget.moduleNameLineEdit(), Qt.Key_Return)
        importwidget.variableNameLineEdit().setText("bar")
        qtbot.keyPress(importwidget.variableNameLineEdit(), Qt.Key_Return)
    assert importwidget.variable() is ImportWidget.INVALID
    assert importwidget.importStatus() is ImportStatus.NO_MODULE
    assert not importwidget.isValid()

    with qtbot.waitSignal(importwidget.variableChanged):
        importwidget.moduleNameLineEdit().setText("dipcoatimage.finitedepth")
        qtbot.keyPress(importwidget.moduleNameLineEdit(), Qt.Key_Return)
    assert importwidget.variable() is ImportWidget.INVALID
    assert importwidget.importStatus() is ImportStatus.NO_VARIABLE
    assert not importwidget.isValid()

    with qtbot.waitSignal(importwidget.variableChanged):
        importwidget.variableNameLineEdit().setText("SubstrateReference")
        qtbot.keyPress(importwidget.variableNameLineEdit(), Qt.Key_Return)
    assert importwidget.variable() is SubstrateReference
    assert importwidget.importStatus() is ImportStatus.VALID
    assert importwidget.isValid()
