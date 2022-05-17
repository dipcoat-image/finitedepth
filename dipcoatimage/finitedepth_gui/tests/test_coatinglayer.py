"""Test for coating layer widget."""


import cv2  # type: ignore
from dipcoatimage.finitedepth import (
    get_samples_path,
    SubstrateReference,
    RectSubstrate,
    LayerArea,
    RectLayerArea,
    data_converter,
)
from dipcoatimage.finitedepth_gui.controlwidgets import (
    CoatingLayerWidget,
)
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
LAYER_PATH = get_samples_path("coat1.png")
LAYER_IMG = cv2.imread(LAYER_PATH)
if LAYER_IMG is None:
    raise TypeError("Invalid coated substrate image sample.")
LAYER = RectLayerArea(LAYER_IMG, SUBST)


@pytest.fixture
def layerwidget(qtbot):
    widget = CoatingLayerWidget()
    widget.typeWidget().registerVariable(
        "LayerArea", "LayerArea", "dipcoatimage.finitedepth"
    )
    widget.typeWidget().registerVariable(
        "RectLayerArea", "RectLayerArea", "dipcoatimage.finitedepth"
    )
    return widget


def test_CoatingLayerWidget_onCoatingLayerTypeChange(qtbot, layerwidget):
    signals = [
        layerwidget.parametersWidget().currentChanged,
        layerwidget.drawOptionsWidget().currentChanged,
        layerwidget.decoOptionsWidget().currentChanged,
    ]

    with qtbot.waitSignals(signals):
        layerwidget.typeWidget().variableComboBox().setCurrentIndex(0)
    assert (
        layerwidget.parametersWidget().currentWidget().dataclassType()
        == LayerArea.Parameters
    )
    assert (
        layerwidget.drawOptionsWidget().currentWidget().dataclassType()
        == LayerArea.DrawOptions
    )
    assert (
        layerwidget.decoOptionsWidget().currentWidget().dataclassType()
        == LayerArea.DecoOptions
    )

    with qtbot.waitSignals(
        [
            layerwidget.decoOptionsWidget().currentChanged,
        ]
    ):
        layerwidget.typeWidget().variableComboBox().setCurrentIndex(1)
    assert (
        layerwidget.parametersWidget().currentWidget().dataclassType()
        == RectLayerArea.Parameters
    )
    assert (
        layerwidget.drawOptionsWidget().currentWidget().dataclassType()
        == RectLayerArea.DrawOptions
    )
    assert (
        layerwidget.decoOptionsWidget().currentWidget().dataclassType()
        == RectLayerArea.DecoOptions
    )

    with qtbot.waitSignals(signals):
        layerwidget.typeWidget().setImportInformation("foo", "bar")
    assert layerwidget.parametersWidget().currentIndex() == 0
    assert layerwidget.drawOptionsWidget().currentIndex() == 0
    assert layerwidget.decoOptionsWidget().currentIndex() == 0
