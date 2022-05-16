"""Test for substrate widget and substrate worker."""

import cv2  # type: ignore
from dipcoatimage.finitedepth import (
    get_samples_path,
    SubstrateReference,
    Substrate,
    RectSubstrate,
    data_converter,
)
from dipcoatimage.finitedepth.analysis import SubstrateArgs, ImportArgs
from dipcoatimage.finitedepth_gui.controlwidgets import (
    SubstrateWidget,
    SubstrateWidgetData,
)
from dipcoatimage.finitedepth.util import dict_includes
from dipcoatimage.finitedepth_gui.workers import SubstrateWorker
import numpy as np
import pytest


REF_PATH = get_samples_path("ref1.png")
REF_IMG = cv2.imread(REF_PATH)
if REF_IMG is None:
    raise TypeError("Invalid reference image sample.")
REF_IMG = cv2.cvtColor(REF_IMG, cv2.COLOR_BGR2RGB)
REF = SubstrateReference(REF_IMG, substrateROI=(400, 100, 1000, 500))
params = data_converter.structure(
    dict(
        Canny=dict(threshold1=50.0, threshold2=150.0),
        HoughLines=dict(rho=1.0, theta=0.01, threshold=100),
    ),
    RectSubstrate.Parameters,
)
SUBST = RectSubstrate(REF, parameters=params)


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
    substwidget.setSubstrateArgs(substargs)

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

    widget.setSubstrateArgs(SubstrateArgs())
    assert counter.i == 0


def test_SubstrateWorker_setSubstrateWidgetData(qtbot):
    worker = SubstrateWorker()
    assert worker.substrateType() is None
    assert worker.reference() is None
    assert worker.parameters() is None
    assert worker.drawOptions() is None

    valid_data1 = SubstrateWidgetData(Substrate, None, None)
    worker.setSubstrateWidgetData(valid_data1)
    assert worker.substrateType() == valid_data1.type
    assert worker.parameters() == worker.substrateType().Parameters()
    assert worker.drawOptions() == worker.substrateType().DrawOptions()

    valid_data2 = SubstrateWidgetData(
        Substrate, Substrate.Parameters(), Substrate.DrawOptions()
    )
    worker.setSubstrateWidgetData(valid_data2)
    assert worker.substrateType() == valid_data1.type
    assert worker.parameters() == worker.substrateType().Parameters()
    assert worker.drawOptions() == worker.substrateType().DrawOptions()

    type_invalid_data = SubstrateWidgetData(
        type, Substrate.Parameters(), Substrate.DrawOptions()
    )
    worker.setSubstrateWidgetData(type_invalid_data)
    assert worker.substrateType() is None
    assert worker.parameters() is None
    assert worker.drawOptions() is None

    options_invalid_data = SubstrateWidgetData(RectSubstrate, None, None)
    # invalid data can be converted to default (if exists)
    worker.setSubstrateWidgetData(options_invalid_data)
    assert worker.substrateType() == options_invalid_data.type
    assert worker.parameters() is None
    assert worker.drawOptions() == worker.substrateType().DrawOptions()


def test_SubstrateWorker_visualizedImage(qtbot):
    worker = SubstrateWorker()
    assert worker.visualizedImage().size == 0

    worker.setReference(REF)
    worker.setSubstrateWidgetData(
        SubstrateWidgetData(type(SUBST), SUBST.parameters, SUBST.draw_options)
    )
    worker.updateSubstrate()
    assert np.all(worker.visualizedImage() == SUBST.draw())

    worker.setVisualizationMode(False)
    assert np.all(worker.visualizedImage() == SUBST.image())

    worker.setSubstrateWidgetData(SubstrateWidgetData(None, None, None))
    worker.updateSubstrate()
    assert np.all(worker.visualizedImage() == REF.substrate_image())

    worker.setReference(None)
    assert worker.visualizedImage().size == 0
