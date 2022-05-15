"""Test for reference widget and reference worker."""

import cv2  # type: ignore
from dipcoatimage.finitedepth import (
    get_samples_path,
    SubstrateReference,
    data_converter,
)
from dipcoatimage.finitedepth.analysis import ReferenceArgs
from dipcoatimage.finitedepth_gui.controlwidgets import (
    ReferenceWidget,
    ReferenceWidgetData,
)
from dipcoatimage.finitedepth.util import dict_includes
from dipcoatimage.finitedepth_gui.workers import ReferenceWorker
from PySide6.QtCore import Qt
import pytest


REF_PATH = get_samples_path("ref1.png")
REF_IMG = cv2.imread(REF_PATH)
if REF_IMG is None:
    raise TypeError("Invalid reference image sample.")
REF_IMG = cv2.cvtColor(REF_IMG, cv2.COLOR_BGR2RGB)


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
        refwidget.dataChanged,
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


def test_ReferenceWidget_setReferenceArgs(qtbot, refwidget):
    refargs = ReferenceArgs(
        templateROI=(50, 50, 100, 100),
        substrateROI=(100, 100, 200, 200),
        draw_options=dict(substrateROI_thickness=2),
    )
    refwidget.setReferencePath(REF_PATH)
    refwidget.setReferenceArgs(data_converter.unstructure(refargs))

    assert refwidget.typeWidget().variableComboBox().currentIndex() == -1
    assert refwidget.typeWidget().variableNameLineEdit().text() == refargs.type.name
    assert refwidget.typeWidget().moduleNameLineEdit().text() == refargs.type.module

    assert refwidget.templateROIWidget().roiMaximum() == (1407, 1125)
    assert refwidget.templateROIWidget().roiModel().roi() == refargs.templateROI
    assert refwidget.templateROIWidget().x1SpinBox().value() == refargs.templateROI[0]
    assert refwidget.templateROIWidget().y1SpinBox().value() == refargs.templateROI[1]
    assert refwidget.templateROIWidget().x2SpinBox().value() == refargs.templateROI[2]
    assert refwidget.templateROIWidget().y2SpinBox().value() == refargs.templateROI[3]

    assert refwidget.substrateROIWidget().roiMaximum() == (1407, 1125)
    assert refwidget.substrateROIWidget().roiModel().roi() == refargs.substrateROI
    assert refwidget.substrateROIWidget().x1SpinBox().value() == refargs.substrateROI[0]
    assert refwidget.substrateROIWidget().y1SpinBox().value() == refargs.substrateROI[1]
    assert refwidget.substrateROIWidget().x2SpinBox().value() == refargs.substrateROI[2]
    assert refwidget.substrateROIWidget().y2SpinBox().value() == refargs.substrateROI[3]

    assert dict_includes(
        data_converter.unstructure(
            refwidget.parametersWidget().currentWidget().dataValue()
        ),
        refargs.parameters,
    )

    assert dict_includes(
        data_converter.unstructure(
            refwidget.drawOptionsWidget().currentWidget().dataValue()
        ),
        refargs.draw_options,
    )


def test_ReferenceWidget_dataChanged_count(qtbot):
    """
    Test that refwidget.dataChanged do not trigger signals multiple times.
    """

    class Counter:
        def __init__(self):
            self.i = 0

        def count(self, _):
            self.i += 1

    widget = ReferenceWidget()
    counter = Counter()
    widget.dataChanged.connect(counter.count)
    widget.onPathEditFinished()
    assert counter.i == 1

    widget = ReferenceWidget()
    counter = Counter()
    widget.dataChanged.connect(counter.count)
    refargs = data_converter.unstructure(ReferenceArgs())
    widget.setReferenceArgs(refargs)
    assert counter.i == 0


def test_ReferenceWorker_setReferenceWidgetData(qtbot):
    worker = ReferenceWorker()
    assert worker.referenceType() is None
    assert worker.image().size == 0
    assert worker.parameters() is None
    assert worker.drawOptions() is None

    valid_data1 = ReferenceWidgetData(
        SubstrateReference, (0, 0, None, None), (0, 0, None, None), None, None
    )
    worker.setReferenceWidgetData(valid_data1)
    assert worker.referenceType() == valid_data1.type
    assert worker.parameters() == worker.referenceType().Parameters()
    assert worker.drawOptions() == worker.referenceType().DrawOptions()

    valid_data2 = ReferenceWidgetData(
        SubstrateReference,
        (0, 0, None, None),
        (0, 0, None, None),
        SubstrateReference.Parameters(),
        SubstrateReference.DrawOptions(),
    )
    worker.setReferenceWidgetData(valid_data2)
    assert worker.referenceType() == valid_data2.type
    assert worker.parameters() == valid_data2.parameters
    assert worker.drawOptions() == valid_data2.draw_options

    type_invalid_data = ReferenceWidgetData(
        type,
        (0, 0, None, None),
        (0, 0, None, None),
        SubstrateReference.Parameters(),
        SubstrateReference.DrawOptions(),
    )
    worker.setReferenceWidgetData(type_invalid_data)
    assert worker.referenceType() is None
    assert worker.parameters() is None
    assert worker.drawOptions() is None
