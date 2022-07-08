"""Test for substrate worker."""

import cv2  # type: ignore
from dipcoatimage.finitedepth import (
    get_samples_path,
    SubstrateReference,
    Substrate,
    RectSubstrate,
    data_converter,
)
from dipcoatimage.finitedepth_gui.inventory import StructuredSubstrateArgs
from dipcoatimage.finitedepth_gui.workers import SubstrateWorker
import numpy as np


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


def test_SubstrateWorker_setStructuredSubstrateArgs(qtbot):
    worker = SubstrateWorker()
    assert worker.substrateType() is None
    assert worker.reference() is None
    assert worker.parameters() is None
    assert worker.drawOptions() is None

    valid_data1 = StructuredSubstrateArgs(Substrate, None, None)
    worker.setStructuredSubstrateArgs(valid_data1)
    assert worker.substrateType() == valid_data1.type
    assert worker.parameters() == worker.substrateType().Parameters()
    assert worker.drawOptions() == worker.substrateType().DrawOptions()

    valid_data2 = StructuredSubstrateArgs(
        Substrate, Substrate.Parameters(), Substrate.DrawOptions()
    )
    worker.setStructuredSubstrateArgs(valid_data2)
    assert worker.substrateType() == valid_data1.type
    assert worker.parameters() == worker.substrateType().Parameters()
    assert worker.drawOptions() == worker.substrateType().DrawOptions()

    type_invalid_data = StructuredSubstrateArgs(
        type, Substrate.Parameters(), Substrate.DrawOptions()
    )
    worker.setStructuredSubstrateArgs(type_invalid_data)
    assert worker.substrateType() is None
    assert worker.parameters() is None
    assert worker.drawOptions() is None

    options_invalid_data = StructuredSubstrateArgs(RectSubstrate, None, None)
    # invalid data can be converted to default (if exists)
    worker.setStructuredSubstrateArgs(options_invalid_data)
    assert worker.substrateType() == options_invalid_data.type
    assert worker.parameters() is None
    assert worker.drawOptions() == worker.substrateType().DrawOptions()


def test_SubstrateWorker_visualizedImage(qtbot):
    worker = SubstrateWorker()
    assert worker.visualizedImage().size == 0

    worker.setReference(REF)
    worker.setStructuredSubstrateArgs(
        StructuredSubstrateArgs(type(SUBST), SUBST.parameters, SUBST.draw_options)
    )
    worker.updateSubstrate()
    assert np.all(worker.visualizedImage() == SUBST.draw())

    worker.setVisualizationMode(False)
    assert np.all(worker.visualizedImage() == SUBST.image())

    worker.setStructuredSubstrateArgs(StructuredSubstrateArgs(None, None, None))
    worker.updateSubstrate()
    assert np.all(worker.visualizedImage() == REF.substrate_image())

    worker.setReference(None)
    assert worker.visualizedImage().size == 0
