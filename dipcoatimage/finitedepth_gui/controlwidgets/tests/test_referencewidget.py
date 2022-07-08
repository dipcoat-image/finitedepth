"""Test for reference widget."""

from dipcoatimage.finitedepth import get_samples_path, SubstrateReference
from dipcoatimage.finitedepth_gui.controlwidgets import ReferenceWidget
from PySide6.QtCore import Qt
import pytest


REF_PATH = get_samples_path("ref1.png")


@pytest.fixture
def refwidget(qtbot):
    widget = ReferenceWidget()
    widget.typeWidget().registerVariable(
        "SubstrateReference", "SubstrateReference", "dipcoatimage.finitedepth"
    )
    return widget


def test_ReferenceWidget_updateROIMaximum(qtbot):
    refwidget = ReferenceWidget()
    assert refwidget.templateROIWidget().x1SpinBox().maximum() == 0
    assert refwidget.templateROIWidget().y1SpinBox().maximum() == 0
    assert refwidget.templateROIWidget().x2SpinBox().maximum() == 0
    assert refwidget.templateROIWidget().y2SpinBox().maximum() == 0
    assert refwidget.templateROIWidget().x1SpinBox().value() == 0
    assert refwidget.templateROIWidget().y1SpinBox().value() == 0
    assert refwidget.templateROIWidget().x2SpinBox().value() == 0
    assert refwidget.templateROIWidget().y2SpinBox().value() == 0
    assert refwidget.substrateROIWidget().x1SpinBox().maximum() == 0
    assert refwidget.substrateROIWidget().y1SpinBox().maximum() == 0
    assert refwidget.substrateROIWidget().x2SpinBox().maximum() == 0
    assert refwidget.substrateROIWidget().y2SpinBox().maximum() == 0
    assert refwidget.substrateROIWidget().x1SpinBox().value() == 0
    assert refwidget.substrateROIWidget().y1SpinBox().value() == 0
    assert refwidget.substrateROIWidget().x2SpinBox().value() == 0
    assert refwidget.substrateROIWidget().y2SpinBox().value() == 0

    with qtbot.waitSignals(
        [
            refwidget.templateROIWidget().roiMaximumChanged,
            refwidget.substrateROIWidget().roiMaximumChanged,
        ]
    ):
        refwidget.pathLineEdit().setText(REF_PATH)
        qtbot.keyPress(refwidget.pathLineEdit(), Qt.Key_Return)

    assert refwidget.templateROIWidget().x1SpinBox().maximum() == 1407
    assert refwidget.templateROIWidget().y1SpinBox().maximum() == 1125
    assert refwidget.templateROIWidget().x2SpinBox().maximum() == 1407
    assert refwidget.templateROIWidget().y2SpinBox().maximum() == 1125
    assert refwidget.templateROIWidget().x1SpinBox().value() == 0
    assert refwidget.templateROIWidget().y1SpinBox().value() == 0
    assert refwidget.templateROIWidget().x2SpinBox().value() == 1407
    assert refwidget.templateROIWidget().y2SpinBox().value() == 1125
    assert refwidget.substrateROIWidget().x1SpinBox().maximum() == 1407
    assert refwidget.substrateROIWidget().y1SpinBox().maximum() == 1125
    assert refwidget.substrateROIWidget().x2SpinBox().maximum() == 1407
    assert refwidget.substrateROIWidget().y2SpinBox().maximum() == 1125
    assert refwidget.substrateROIWidget().x1SpinBox().value() == 0
    assert refwidget.substrateROIWidget().y1SpinBox().value() == 0
    assert refwidget.substrateROIWidget().x2SpinBox().value() == 1407
    assert refwidget.substrateROIWidget().y2SpinBox().value() == 1125


def test_ReferenceWidget_onReferenceTypeChange(qtbot, refwidget):
    signals = [
        refwidget.parametersWidget().currentChanged,
        refwidget.drawOptionsWidget().currentChanged,
    ]

    with qtbot.waitSignals(signals):
        refwidget.typeWidget().variableComboBox().setCurrentIndex(0)
    assert (
        refwidget.parametersWidget().currentWidget().dataclassType()
        == SubstrateReference.Parameters
    )
    assert (
        refwidget.drawOptionsWidget().currentWidget().dataclassType()
        == SubstrateReference.DrawOptions
    )

    with qtbot.waitSignals(signals):
        refwidget.typeWidget().setImportInformation("foo", "bar")
    assert refwidget.parametersWidget().currentIndex() == 0
    assert refwidget.drawOptionsWidget().currentIndex() == 0


def test_ReferenceWidget_exclusiveButtons(qtbot):
    refwidget = ReferenceWidget()
    qtbot.mouseClick(refwidget.substrateROIDrawButton(), Qt.LeftButton)
    assert refwidget.substrateROIDrawButton().isChecked()
    assert not refwidget.templateROIDrawButton().isChecked()

    qtbot.mouseClick(refwidget.templateROIDrawButton(), Qt.LeftButton)
    assert not refwidget.substrateROIDrawButton().isChecked()
    assert refwidget.templateROIDrawButton().isChecked()

    qtbot.mouseClick(refwidget.substrateROIDrawButton(), Qt.LeftButton)
    assert refwidget.substrateROIDrawButton().isChecked()
    assert not refwidget.templateROIDrawButton().isChecked()
