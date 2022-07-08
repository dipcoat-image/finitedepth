"""Test for substrate widget."""

from dipcoatimage.finitedepth import Substrate, RectSubstrate
from dipcoatimage.finitedepth_gui.controlwidgets import (
    SubstrateWidget,
)
import pytest


@pytest.fixture
def substwidget(qtbot):
    widget = SubstrateWidget()
    widget.typeWidget().registerVariable(
        "Substrate", "Substrate", "dipcoatimage.finitedepth"
    )
    widget.typeWidget().registerVariable(
        "RectSubstrate", "RectSubstrate", "dipcoatimage.finitedepth"
    )
    return widget


def test_SubstrateWidget_onSubstrateTypeChange(qtbot, substwidget):
    signals = [
        substwidget.parametersWidget().currentChanged,
        substwidget.drawOptionsWidget().currentChanged,
    ]

    with qtbot.waitSignals(signals):
        substwidget.typeWidget().variableComboBox().setCurrentIndex(0)
    assert (
        substwidget.parametersWidget().currentWidget().dataclassType()
        == Substrate.Parameters
    )
    assert (
        substwidget.drawOptionsWidget().currentWidget().dataclassType()
        == Substrate.DrawOptions
    )

    with qtbot.waitSignals(signals):
        substwidget.typeWidget().variableComboBox().setCurrentIndex(1)
    assert (
        substwidget.parametersWidget().currentWidget().dataclassType()
        == RectSubstrate.Parameters
    )
    assert (
        substwidget.drawOptionsWidget().currentWidget().dataclassType()
        == RectSubstrate.DrawOptions
    )

    with qtbot.waitSignals(signals):
        substwidget.typeWidget().setImportInformation("foo", "bar")
    assert substwidget.parametersWidget().currentIndex() == 0
    assert substwidget.drawOptionsWidget().currentIndex() == 0
