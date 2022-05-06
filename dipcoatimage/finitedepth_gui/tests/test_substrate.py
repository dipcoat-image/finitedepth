"""Test for substrate widget and substrate worker."""

from dipcoatimage.finitedepth import Substrate, RectSubstrate, data_converter
from dipcoatimage.finitedepth.experiment import SubstrateArgs, ImportArgs
from dipcoatimage.finitedepth_gui.controlwidgets import (
    SubstrateWidget,
)
import pytest


def dict_includes(sup, sub):
    for key, value in sub.items():
        if key not in sup:
            return False
        if isinstance(value, dict):
            if not dict_includes(sup[key], value):
                return False
        else:
            if not value == sup[key]:
                return False
    return True


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
        substwidget.dataChanged,
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


def test_SubstrateWidget_setSubstrateArgs(qtbot, substwidget):
    substargs = SubstrateArgs(
        type=ImportArgs(name="RectSubstrate"),
        parameters=dict(
            Canny=dict(threshold1=50.0, threshold2=150.0),
            HoughLines=dict(rho=1.0, theta=0.01, threshold=100),
        ),
        draw_options=dict(draw_lines=False),
    )
    substwidget.setSubstrateArgs(data_converter.unstructure(substargs))

    assert substwidget.typeWidget().variableComboBox().currentIndex() == -1
    assert substwidget.typeWidget().variableNameLineEdit().text() == substargs.type.name
    assert substwidget.typeWidget().moduleNameLineEdit().text() == substargs.type.module

    assert dict_includes(
        data_converter.unstructure(
            substwidget.parametersWidget().currentWidget().dataValue()
        ),
        substargs.parameters,
    )

    assert dict_includes(
        data_converter.unstructure(
            substwidget.drawOptionsWidget().currentWidget().dataValue()
        ),
        substargs.draw_options,
    )


def test_SubstrateWidget_dataChanged_count(qtbot):
    """
    Test that substwidget.dataChanged do not trigger signals multiple times.
    """
    widget = SubstrateWidget()

    class Counter:
        def __init__(self):
            self.i = 0

        def count(self, _):
            self.i += 1

    counter = Counter()
    widget.dataChanged.connect(counter.count)

    substargs = data_converter.unstructure(SubstrateArgs())
    widget.setSubstrateArgs(substargs)
    assert counter.i == 0
